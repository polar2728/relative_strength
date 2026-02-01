import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import warnings
import time
import threading
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
st.set_page_config(page_title="NSE RS Leaders Scanner PRO", layout="wide")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
RS_LOOKBACK_6M = 126
RS_LOOKBACK_3M = 63
MIN_RS_RANK = 80
MIN_LIQUIDITY_CR = 5

BENCHMARK_CANDIDATES = {
    "Nifty 50": "NIFTY 50",
    "Nifty Next 50": "NIFTY NEXT 50",
    "Nifty 100": "NIFTY 100",
    "Nifty 200": "NIFTY 200",
    "Nifty 500": "NIFTY 500",
    "Nifty Midcap 150": "NIFTY MIDCAP 150",
    "Nifty Total Market": "NIFTY TOTAL MARKET",
}

# ─────────────────────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────────────────────
class RateLimiter:
    """Thread-safe rate limiter for Kite API (3 requests/second)"""
    def __init__(self, max_calls=3, period=1.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                self.calls = [call for call in self.calls if call > now - self.period]
                
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.period - (now - self.calls[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                self.calls.append(time.time())
            
            return func(*args, **kwargs)
        return wrapper

# ─────────────────────────────────────────────────────────────
# CLOUD SECRETS AUTH
# ─────────────────────────────────────────────────────────────
def get_kite_from_secrets():
    """Get pre-configured Kite from Streamlit secrets"""
    try:
        api_key = st.secrets["KITE_API_KEY"]
        access_token = st.secrets["KITE_ACCESS_TOKEN"]
        
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        profile = kite.profile()
        st.sidebar.success(f"✅ Connected: {profile['user_name']}")
        return kite
        
    except Exception as e:
        st.sidebar.error(f"❌ Auth failed: {str(e)[:50]}")
        return None

def get_kite():
    if "kite" in st.session_state:
        return st.session_state.kite
    return None

# ─────────────────────────────────────────────────────────────
# INSTRUMENT MAP
# ─────────────────────────────────────────────────────────────
@st.cache_resource(ttl=86400 * 7)
def load_kite_instrument_map(_kite):
    """Load instrument tokens with separate maps for equities and indices"""
    instruments = _kite.instruments("NSE")
    df = pd.DataFrame(instruments)

    eq = df[(df["segment"] == "NSE") & (df["exchange"] == "NSE")]
    idx = df[df["segment"] == "INDICES"]

    eq_map = dict(zip(eq["tradingsymbol"], eq["instrument_token"]))
    idx_map = dict(zip(idx["tradingsymbol"], idx["instrument_token"]))

    return {"equities": eq_map, "indices": idx_map}

# ─────────────────────────────────────────────────────────────
# PRE-FILTER USING QUOTE API
# ─────────────────────────────────────────────────────────────
def prefilter_universe(kite, symbols, instrument_map, min_liq_cr=5):
    """Pre-filter universe using quote API before historical fetch"""
    equity_map = instrument_map["equities"]
    tokens = []
    symbol_map = {}
    
    for sym in symbols:
        clean = sym.replace(".NS", "")
        token = equity_map.get(clean)
        if token:
            key = f"NSE:{clean}"
            tokens.append(key)
            symbol_map[key] = sym
    
    if not tokens:
        return symbols
    
    passed = []
    
    for batch_idx in range(0, len(tokens), 500):
        batch = tokens[batch_idx:batch_idx+500]
        
        try:
            quotes = kite.quote(batch)
            
            for instrument, data in quotes.items():
                ltp = data.get('last_price', 0)
                volume = data.get('volume', 0)
                
                if ltp > 0 and volume > 0:
                    daily_liq = (ltp * volume) / 1e7
                    if daily_liq >= min_liq_cr * 0.3:
                        passed.append(symbol_map[instrument])
            
            time.sleep(0.35)
                
        except Exception:
            for inst in batch:
                if inst in symbol_map:
                    passed.append(symbol_map[inst])
    
    return passed if len(passed) > 0 else symbols

# ─────────────────────────────────────────────────────────────
# HISTORICAL DATA
# ─────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError))
)
def fetch_kite_historical_core(kite, symbol, days=450, instrument_map=None):
    """Core fetch function"""
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date   = datetime.now().strftime("%Y-%m-%d")

    is_index = symbol in BENCHMARK_CANDIDATES.values()
    key = symbol if is_index else symbol.replace(".NS", "")

    token = instrument_map["indices"].get(key) if is_index else instrument_map["equities"].get(key)

    if token is None:
        return pd.DataFrame()

    try:
        token = int(token)
    except (ValueError, TypeError):
        return pd.DataFrame()

    try:
        data = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval="day",
            continuous=False,
            oi=False
        )

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df

    except Exception:
        return pd.DataFrame()

rate_limiter = RateLimiter(max_calls=3, period=1.0)

@rate_limiter
def fetch_kite_historical(kite, symbol, days=450, instrument_map=None):
    """Rate-limited wrapper for historical data fetch"""
    return fetch_kite_historical_core(kite, symbol, days, instrument_map)

# ─────────────────────────────────────────────────────────────
# PARALLEL BATCH FETCH
# ─────────────────────────────────────────────────────────────
def fetch_in_batches_parallel(kite, symbols, instrument_map, max_workers=5):
    """Parallel fetch with rate limiting"""
    results = {}
    total = len(symbols)
    
    progress_container = st.sidebar.empty()
    status_container = st.sidebar.empty()
    
    processed = [0]
    successful = [0]
    lock = threading.Lock()
    
    def fetch_symbol(sym):
        try:
            df = fetch_kite_historical(kite, sym, days=450, instrument_map=instrument_map)
            
            with lock:
                if not df.empty:
                    successful[0] += 1
                processed[0] += 1
                return sym, df
            
        except Exception:
            with lock:
                processed[0] += 1
            return sym, pd.DataFrame()
    
    status_container.info(f"🚀 Fetching {total} symbols...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_symbol, sym): sym for sym in symbols}
        
        for future in as_completed(futures):
            try:
                sym, df = future.result()
                results[sym] = df
                
                progress = processed[0] / total
                progress_container.progress(progress)
                
                if processed[0] % 10 == 0:
                    status_container.info(f"⏳ {processed[0]}/{total} | {successful[0]} valid")
                
            except Exception:
                pass
    
    progress_container.progress(1.0)
    status_container.success(f"✅ Fetched {successful[0]}/{total} symbols")
    
    return results

# ─────────────────────────────────────────────────────────────
# SEQUENTIAL FETCH FOR BENCHMARKS
# ─────────────────────────────────────────────────────────────
def fetch_benchmarks_sequential(kite, instrument_map):
    """Fetch benchmarks sequentially"""
    bm_data = {}
    
    progress = st.sidebar.empty()
    
    for idx, (name, sym) in enumerate(BENCHMARK_CANDIDATES.items(), 1):
        progress.info(f"📊 Fetching benchmarks... ({idx}/{len(BENCHMARK_CANDIDATES)})")
        df = fetch_kite_historical(kite, sym, days=450, instrument_map=instrument_map)
        
        if not df.empty:
            bm_data[sym] = df
        
        time.sleep(0.35)
    
    progress.success(f"✅ Loaded {len(bm_data)} benchmarks")
    
    return bm_data

# ─────────────────────────────────────────────────────────────
# UNIVERSE LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(ttl=86400)
def load_nse_universe():
    """Load full NSE equity universe"""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.upper().str.strip()

    df = df[df["SERIES"] == "EQ"]
    df["Symbol"] = df["SYMBOL"] + ".NS"
    name_map = dict(zip(df["SYMBOL"], df["NAME OF COMPANY"]))
    return df["Symbol"].tolist(), name_map

@st.cache_resource(ttl=86400)
def load_nifty50_symbols():
    """Load Nifty 50 constituents"""
    url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive"
    }

    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            return [s + ".NS" for s in df["Symbol"]]
        except Exception:
            time.sleep(2)

    return []

# ─────────────────────────────────────────────────────────────
# INDICATORS - FIXED RSI CALCULATION
# ─────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    """Compute RSI with edge case handling"""
    # Ensure series is not empty and has enough data
    if series is None or len(series) < period + 1:
        return None
    
    # Remove any NaN values
    series = series.dropna()
    
    if len(series) < period + 1:
        return None
    
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    
    # Get the last valid value
    gain_val = gain.iloc[-1]
    loss_val = loss.iloc[-1]
    
    if pd.isna(gain_val) or pd.isna(loss_val):
        return None
    if loss_val == 0:
        return 100.0
    if gain_val == 0:
        return 0.0
    
    rs = gain_val / loss_val
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 1)

def resample_to_weekly(close_series):
    """Resample daily close to weekly close (Friday)"""
    if close_series is None or len(close_series) < 20:
        return None
    
    # Resample to weekly, taking last value of each week
    weekly = close_series.resample('W-FRI').last()
    # Drop NaN values
    weekly = weekly.dropna()
    
    return weekly if len(weekly) >= 20 else None

def resample_to_monthly(close_series):
    """Resample daily close to monthly close (month end)"""
    if close_series is None or len(close_series) < 60:
        return None
    
    # Resample to month end, taking last value of each month
    monthly = close_series.resample('M').last()
    # Drop NaN values
    monthly = monthly.dropna()
    
    return monthly if len(monthly) >= 15 else None

def log_rs(p, p0, b, b0):
    """Log-based relative strength calculation"""
    return np.log(p / p0) - np.log(b / b0)

# ─────────────────────────────────────────────────────────────
# RS SCAN
# ─────────────────────────────────────────────────────────────
def rs_scan(kite, symbols, name_map, min_rs, min_liq, benchmark_mode):
    """Main RS scanning logic"""
    
    # Load instrument maps
    instrument_map = load_kite_instrument_map(kite)
    st.session_state.instrument_map = instrument_map
    
    # Fetch benchmarks
    with st.spinner("📊 Fetching benchmark indices..."):
        bm_data = fetch_benchmarks_sequential(kite, instrument_map)
    
    if not bm_data:
        st.error("❌ Failed to fetch benchmark data.")
        st.stop()
    
    # Pre-filter
    with st.spinner("🔍 Pre-filtering universe..."):
        filtered_symbols = prefilter_universe(kite, symbols, instrument_map, min_liq)
    
    reduction_pct = (1 - len(filtered_symbols) / len(symbols)) * 100 if len(symbols) > 0 else 0
    st.info(f"📊 Pre-filter: {len(symbols)} → {len(filtered_symbols)} stocks ({reduction_pct:.1f}% reduction)")
    
    if len(filtered_symbols) == 0:
        st.error("❌ No stocks passed pre-filtering.")
        return pd.DataFrame(), None, pd.DataFrame()

    # Parallel fetch
    with st.spinner(f"📥 Fetching historical data..."):
        stock_data = fetch_in_batches_parallel(
            kite, 
            filtered_symbols, 
            instrument_map,
            max_workers=5
        )

    # Select best benchmark
    best_ret = -1e9
    selected_df = None
    selected_benchmark = None
    benchmark_rows = []
    min_required_days = 250

    for name, sym in BENCHMARK_CANDIDATES.items():
        df = bm_data.get(sym)
        if df is None or df.empty or len(df) < min_required_days:
            continue
        
        if len(df) < RS_LOOKBACK_6M:
            continue

        try:
            ret = df["Close"].iloc[-1] / df["Close"].iloc[-RS_LOOKBACK_6M] - 1
            benchmark_rows.append({
                "Benchmark": name,
                "Return_6M": round(ret * 100, 2),
                "Days": len(df),
                "Status": "✅" if ret > best_ret else ""
            })

            if ret > best_ret:
                best_ret = ret
                selected_df = df
                selected_benchmark = name
        except Exception:
            continue

    if selected_df is None or selected_df.empty:
        st.error("❌ No valid benchmark found.")
        return pd.DataFrame(), None, pd.DataFrame()

    # Scan stocks
    results = []
    filter_stats = {
        "total": len(stock_data),
        "short_history": 0,
        "below_dma200": 0,
        "low_liquidity": 0,
        "passed": 0
    }

    with st.spinner(f"🔍 Analyzing stocks vs {selected_benchmark}..."):
        for sym, df in stock_data.items():
            if df.empty or len(df) < min_required_days:
                filter_stats["short_history"] += 1
                continue

            close = df["Close"]
            
            if len(close) < max(200, RS_LOOKBACK_6M):
                filter_stats["short_history"] += 1
                continue
            
            price = close.iloc[-1]
            dma200 = close.rolling(200).mean().iloc[-1]
            liq = (close * df["Volume"]).tail(30).mean() / 1e7

            if price < dma200 * 0.95:
                filter_stats["below_dma200"] += 1
                continue
                
            if liq < min_liq:
                filter_stats["low_liquidity"] += 1
                continue

            try:
                rs6 = log_rs(price, close.iloc[-RS_LOOKBACK_6M],
                             selected_df["Close"].iloc[-1],
                             selected_df["Close"].iloc[-RS_LOOKBACK_6M])

                rs3 = log_rs(price, close.iloc[-RS_LOOKBACK_3M],
                             selected_df["Close"].iloc[-1],
                             selected_df["Close"].iloc[-RS_LOOKBACK_3M])

                rs_delta = rs3 - rs6
            except Exception:
                continue
            
            # Calculate RSI at multiple timeframes - FIXED
            rsi_d = compute_rsi(close)
            
            # Weekly RSI
            weekly_close = resample_to_weekly(close)
            rsi_w = compute_rsi(weekly_close) if weekly_close is not None else None
            
            # Monthly RSI
            monthly_close = resample_to_monthly(close)
            rsi_m = compute_rsi(monthly_close) if monthly_close is not None else None

            clean = sym.replace(".NS", "")
            tv = f"https://tradingview.com/chart/?symbol=NSE%3A{clean}"

            results.append({
                "Symbol": clean,
                "Name": name_map.get(clean, ""),
                "Price": round(price, 2),
                "RS": round(rs6, 3),
                "RS_3M": round(rs3, 3),
                "RS_6M": round(rs6, 3),
                "RS_Delta": round(rs_delta, 3),
                "LiquidityCr": round(liq, 1),
                "RSI_D": rsi_d,
                "RSI_W": rsi_w,
                "RSI_M": rsi_m,
                "Chart": tv
            })
            filter_stats["passed"] += 1

    df = pd.DataFrame(results)
    
    if len(df) == 0:
        st.warning("⚠️ No stocks passed all filters.")
        with st.expander("🔍 Filter Statistics"):
            for key, val in filter_stats.items():
                st.write(f"**{key}:** {val}")
        return df, selected_benchmark, pd.DataFrame(benchmark_rows) if benchmark_rows else pd.DataFrame()
    
    df["RS_Rank"] = df["RS_6M"].rank(pct=True) * 100
    df["Momentum"] = np.where(
        df["RS_Delta"] > 0, "🚀 Accelerating", 
        np.where(df["RS_Delta"] < 0, "📉 Decelerating", "➡️ Stable")
    )
    
    bm_table = pd.DataFrame(benchmark_rows).sort_values("Return_6M", ascending=False) if benchmark_rows else pd.DataFrame()

    # Summary stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Scan Results")
    st.sidebar.metric("Total Scanned", filter_stats["total"])
    st.sidebar.metric("Passed All Filters", filter_stats["passed"])
    st.sidebar.metric(f"RS Rank ≥ {min_rs}%", len(df[df["RS_Rank"] >= min_rs]))

    return (
        df[df["RS_Rank"] >= min_rs].sort_values("RS_Rank", ascending=False),
        selected_benchmark,
        bm_table
    )

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div style='background: linear-gradient(135deg,#667eea,#764ba2,#f093fb);
                padding:1.2rem;border-radius:14px;color:white;text-align:center'>
        <h2 style='margin:0'>🏆 NSE RS Leaders Scanner</h2>
        <p style='margin:0;font-size:0.9rem'>Relative Strength • Momentum Analysis • Multi-Timeframe RSI</p>
    </div>
    """, unsafe_allow_html=True)

    kite = get_kite_from_secrets()
    if not kite:
        st.error("⚠️ Update KITE_ACCESS_TOKEN in Streamlit Cloud settings")
        st.stop()

    st.sidebar.markdown("### ⚙️ Configuration")
    
    universe = st.sidebar.radio(
        "Universe",
        ["Nifty 50", "Full NSE"],
        help="Nifty 50: ~1 min | Full NSE: ~5 min"
    )
    
    min_rs = st.sidebar.slider(
        "Min RS Rank %",
        60, 95, MIN_RS_RANK,
        help="Minimum RS percentile rank"
    )
    
    min_liq = st.sidebar.slider(
        "Min Liquidity ₹Cr",
        1, 20, MIN_LIQUIDITY_CR,
        help="Minimum 30-day avg daily liquidity"
    )

    if st.sidebar.button("🚀 RUN SCAN", use_container_width=True, type="primary"):
        syms, name_map = load_nse_universe()
        symbols = load_nifty50_symbols() if universe == "Nifty 50" else syms

        if not symbols:
            st.error("❌ Failed to load stock universe")
            st.stop()

        scan_start = time.time()
        df, selected_benchmark, bm_table = rs_scan(kite, symbols, name_map, min_rs, min_liq, "Auto")
        scan_duration = time.time() - scan_start

        if len(df) > 0:
            st.success(f"✅ Found **{len(df)} stocks** in {scan_duration:.0f}s")
            
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### 📊 Benchmark: **{selected_benchmark}**")
            
            with col2:
                if not bm_table.empty:
                    best_perf = bm_table.iloc[0]["Return_6M"]
                    st.metric("6M Return", f"{best_perf}%")

            if not bm_table.empty:
                with st.expander("📈 All Benchmark Returns"):
                    st.dataframe(bm_table, hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown(f"### 🎯 Top RS Leaders (≥ {min_rs}%)")

            def rsi_color(v):
                if pd.isna(v): return ""
                if v >= 60: return "background-color:#d4edda;color:#155724"
                if v <= 40: return "background-color:#f8d7da;color:#721c24"
                return ""

            styled = df.style.format({
                "Price": "₹{:.2f}",
                "RS": "{:.3f}",
                "RS_6M": "{:.3f}",
                "RS_3M": "{:.3f}",
                "RS_Delta": "{:+.3f}",
                "LiquidityCr": "₹{:.1f}Cr",
                "RSI_D": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
                "RSI_W": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
                "RSI_M": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
                "RS_Rank": "{:.1f}%"
            }).background_gradient(
                subset=["RS_Rank"], 
                cmap="RdYlGn",
                vmin=min_rs,
                vmax=100
            ).map(rsi_color, subset=["RSI_D", "RSI_W", "RSI_M"])

            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Chart": st.column_config.LinkColumn("Chart", display_text="📈 View"),
                    "Momentum": st.column_config.TextColumn("Momentum", help="RS 3M vs 6M")
                },
                height=600
            )

            csv = df.to_csv(index=False).encode("utf-8")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "📥 Download CSV",
                csv,
                f"RS_Leaders_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.markdown("---")
            st.markdown("### 💡 Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg RS Rank", f"{df['RS_Rank'].mean():.1f}%")
            
            with col2:
                improving = len(df[df["RS_Delta"] > 0])
                st.metric("Accelerating", f"{improving}/{len(df)}")
            
            with col3:
                overbought = len(df[df["RSI_D"] > 70])
                st.metric("Overbought (D)", f"{overbought}")
            
            with col4:
                avg_liq = df["LiquidityCr"].mean()
                st.metric("Avg Liquidity", f"₹{avg_liq:.1f}Cr")

        else:
            st.warning("⚠️ No stocks passed filters. Try relaxing criteria.")

    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        **Strategy:**
        1. Selects best-performing benchmark (6M return)
        2. Calculates log-based Relative Strength vs benchmark
        3. Filters: Above 200 DMA, Min liquidity ₹5Cr
        4. Ranks by RS percentile (0-100%)
        5. Shows momentum trend (3M vs 6M RS)
        
        **RSI Levels:**
        - 🟢 Green (≥60): Overbought zone
        - 🔴 Red (≤40): Oversold zone
        - ⚪ Neutral (40-60): Normal range
        
        **Best Practices:**
        - Focus on RS Rank > 85% + accelerating momentum
        - Avoid RSI > 70 across all timeframes (overextended)
        - Confirm with chart patterns before entry
        """)

if __name__ == "__main__":
    main()