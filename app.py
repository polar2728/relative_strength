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
    "Nifty Midcap 150": "NIFTY MIDCAP 150"
}

# ─────────────────────────────────────────────────────────────
# KITE CONNECT OAUTH AUTHENTICATION
# ─────────────────────────────────────────────────────────────

def get_kite_instance():
    """
    Initialize Kite Connect with OAuth login flow
    Only API Key needs to be in secrets - no access token needed!
    """
    
    # Check if already authenticated in session
    if 'kite' in st.session_state and 'access_token' in st.session_state:
        return st.session_state.kite
    
    # Get API credentials from secrets (only API key and secret needed)
    try:
        api_key = st.secrets["KITE_API_KEY"]
        api_secret = st.secrets["KITE_API_SECRET"]  # Add this to secrets
    except Exception as e:
        st.sidebar.error("❌ Add KITE_API_KEY and KITE_API_SECRET to Streamlit secrets")
        st.stop()
        return None
    
    # Initialize Kite Connect
    kite = KiteConnect(api_key=api_key)
    
    # Check if we have a request token from callback URL
    query_params = st.query_params
    
    if 'request_token' in query_params:
        # Step 3: Exchange request token for access token
        request_token = query_params['request_token']
        
        try:
            # Generate session
            data = kite.generate_session(request_token, api_secret=api_secret)
            access_token = data["access_token"]
            
            # Store in session state
            st.session_state.access_token = access_token
            kite.set_access_token(access_token)
            st.session_state.kite = kite
            
            # Get user profile
            profile = kite.profile()
            st.session_state.user_name = profile.get('user_name', 'User')
            st.session_state.user_id = profile.get('user_id', '')
            
            # Clear query params
            st.query_params.clear()
            
            st.sidebar.success(f"✅ Logged in as {st.session_state.user_name}")
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"❌ Login failed: {str(e)[:100]}")
            st.query_params.clear()
            return None
    
    elif 'access_token' not in st.session_state:
        # Step 1: Show login button
        st.sidebar.markdown("### 🔐 Kite Connect Login")
        st.sidebar.info("Click below to login with your Zerodha account")
        
        # Generate login URL
        login_url = kite.login_url()
        
        # Show login button
        st.sidebar.markdown(
            f'<a href="{login_url}" target="_self">'
            f'<button style="background:#387ed1;color:white;padding:0.5rem 1rem;'
            f'border:none;border-radius:4px;cursor:pointer;width:100%;font-size:1rem;">'
            f'🚀 Login with Kite</button></a>',
            unsafe_allow_html=True
        )
        
        st.info("👈 Please login with Kite Connect in the sidebar to continue")
        st.stop()
        return None
    
    return st.session_state.kite

def logout_kite():
    """Logout and clear session"""
    if 'access_token' in st.session_state:
        del st.session_state.access_token
    if 'kite' in st.session_state:
        del st.session_state.kite
    if 'user_name' in st.session_state:
        del st.session_state.user_name
    if 'user_id' in st.session_state:
        del st.session_state.user_id
    st.rerun()

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
# INDICATORS - WITH VOLUME ANALYSIS
# ─────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    """Compute RSI with edge case handling"""
    if series is None or len(series) < period + 1:
        return None
    
    series = series.dropna()
    
    if len(series) < period + 1:
        return None
    
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    
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
    
    weekly = close_series.resample('W-FRI').last()
    weekly = weekly.dropna()
    
    return weekly if len(weekly) >= 20 else None

def resample_to_monthly(close_series):
    """Resample daily close to monthly close (month end)"""
    if close_series is None or len(close_series) < 60:
        return None
    
    monthly = close_series.resample('M').last()
    monthly = monthly.dropna()
    
    return monthly if len(monthly) >= 15 else None

def log_rs(p, p0, b, b0):
    """Log-based relative strength calculation"""
    return np.log(p / p0) - np.log(b / b0)

def calculate_volume_metrics(df):
    """Calculate volume shock metrics"""
    if df.empty or len(df) < 20:
        return {
            'vol_ratio': None,
            'vol_spike': None,
            'vol_trend': None,
            'vol_breakout': False
        }
    
    volume = df["Volume"]
    
    vol_20d_avg = volume.rolling(20).mean().iloc[-1]
    vol_today = volume.iloc[-1]
    vol_ratio = round(vol_today / vol_20d_avg, 2) if vol_20d_avg > 0 else None
    
    vol_recent_5d = volume.tail(5).mean()
    vol_prev_20d = volume.iloc[-25:-5].mean() if len(volume) >= 25 else vol_20d_avg
    vol_spike = round(vol_recent_5d / vol_prev_20d, 2) if vol_prev_20d > 0 else None
    
    if len(volume) >= 40:
        vol_ma = volume.rolling(20).mean()
        vol_ma_recent = vol_ma.tail(20)
        vol_trend = "📈" if vol_ma_recent.iloc[-1] > vol_ma_recent.iloc[0] else "📉"
    else:
        vol_trend = "➡️"
    
    close = df["Close"]
    high_52w = close.tail(252).max() if len(close) >= 252 else close.max()
    price_pct_from_high = (close.iloc[-1] / high_52w - 1) * 100
    
    vol_breakout = (vol_ratio and vol_ratio >= 2.0) and (price_pct_from_high >= -5)
    
    return {
        'vol_ratio': vol_ratio,
        'vol_spike': vol_spike,
        'vol_trend': vol_trend,
        'vol_breakout': vol_breakout
    }

# ─────────────────────────────────────────────────────────────
# RS SCAN - WITH TRADING STYLE SUPPORT
# ─────────────────────────────────────────────────────────────
def rs_scan(kite, symbols, name_map, min_rs, min_liq, benchmark_mode, trading_style):
    """Main RS scanning logic with trading style support"""
    
    instrument_map = load_kite_instrument_map(kite)
    st.session_state.instrument_map = instrument_map
    
    if trading_style == "Swing (3M Focus)":
        benchmark_lookback = RS_LOOKBACK_3M
        rs_column = "RS_3M"
        lookback_label = "3M"
    else:
        benchmark_lookback = RS_LOOKBACK_6M
        rs_column = "RS_6M"
        lookback_label = "6M"
    
    with st.spinner("📊 Fetching benchmark indices..."):
        bm_data = fetch_benchmarks_sequential(kite, instrument_map)
    
    if not bm_data:
        st.error("❌ Failed to fetch benchmark data.")
        st.stop()
    
    with st.spinner("🔍 Pre-filtering universe..."):
        filtered_symbols = prefilter_universe(kite, symbols, instrument_map, min_liq)
    
    reduction_pct = (1 - len(filtered_symbols) / len(symbols)) * 100 if len(symbols) > 0 else 0
    st.info(f"📊 Pre-filter: {len(symbols)} → {len(filtered_symbols)} stocks ({reduction_pct:.1f}% reduction)")
    
    if len(filtered_symbols) == 0:
        st.error("❌ No stocks passed pre-filtering.")
        return pd.DataFrame(), None, pd.DataFrame()

    with st.spinner(f"📥 Fetching historical data..."):
        stock_data = fetch_in_batches_parallel(
            kite, 
            filtered_symbols, 
            instrument_map,
            max_workers=5
        )

    best_ret = -1e9
    selected_df = None
    selected_benchmark = None
    benchmark_rows = []
    min_required_days = 250

    for name, sym in BENCHMARK_CANDIDATES.items():
        df = bm_data.get(sym)
        if df is None or df.empty or len(df) < min_required_days:
            continue
        
        if len(df) < benchmark_lookback:
            continue

        try:
            ret = df["Close"].iloc[-1] / df["Close"].iloc[-benchmark_lookback] - 1
            
            benchmark_rows.append({
                "Benchmark": name,
                f"Return_{lookback_label}": round(ret * 100, 2),
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

    results = []
    filter_stats = {
        "total": len(stock_data),
        "short_history": 0,
        "below_dma200": 0,
        "low_liquidity": 0,
        "passed": 0
    }

    with st.spinner(f"🔍 Analyzing stocks vs {selected_benchmark} ({lookback_label})..."):
        for sym, df in stock_data.items():
            if df.empty or len(df) < min_required_days:
                filter_stats["short_history"] += 1
                continue

            close = df["Close"]
            
            if len(close) < max(200, benchmark_lookback):
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
            
            rsi_d = compute_rsi(close)
            weekly_close = resample_to_weekly(close)
            rsi_w = compute_rsi(weekly_close) if weekly_close is not None else None
            monthly_close = resample_to_monthly(close)
            rsi_m = compute_rsi(monthly_close) if monthly_close is not None else None
            
            vol_metrics = calculate_volume_metrics(df)

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
                "Vol_Ratio": vol_metrics['vol_ratio'],
                "Vol_Spike": vol_metrics['vol_spike'],
                "Vol_Trend": vol_metrics['vol_trend'],
                "Vol_Breakout": "🔥" if vol_metrics['vol_breakout'] else "",
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
    
    df["RS_Rank"] = df[rs_column].rank(pct=True) * 100
    
    df["Momentum"] = np.where(
        df["RS_Delta"] > 0, "🚀 Accelerating", 
        np.where(df["RS_Delta"] < 0, "📉 Decelerating", "➡️ Stable")
    )
    
    bm_table = pd.DataFrame(benchmark_rows).sort_values(f"Return_{lookback_label}", ascending=False) if benchmark_rows else pd.DataFrame()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Scan Results")
    st.sidebar.metric("Trading Style", lookback_label)
    st.sidebar.metric("Total Scanned", filter_stats["total"])
    st.sidebar.metric("Passed All Filters", filter_stats["passed"])
    st.sidebar.metric(f"RS Rank ≥ {min_rs}%", len(df[df["RS_Rank"] >= min_rs]))
    
    vol_breakouts = len(df[df["Vol_Breakout"] == "🔥"])
    if vol_breakouts > 0:
        st.sidebar.metric("🔥 Volume Breakouts", vol_breakouts)

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
        <h2 style='margin:0'>🏆 NSE RS Leaders Scanner PRO</h2>
        <p style='margin:0;font-size:0.9rem'>Relative Strength • Volume Analysis • Multi-Timeframe RSI</p>
    </div>
    """, unsafe_allow_html=True)

    # Kite OAuth Login
    kite = get_kite_instance()
    if not kite:
        return

    # Show user info and logout button in sidebar
    if 'user_name' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**👤 {st.session_state.user_name}**")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_kite()

    st.sidebar.markdown("### ⚙️ Configuration")
    
    trading_style = st.sidebar.radio(
        "Trading Style",
        ["Hybrid (6M)", "Swing (3M Focus)", "Position (6M Focus)"],
        help="Swing: 3M lookback | Position: 6M lookback"
    )
    
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
        df, selected_benchmark, bm_table = rs_scan(kite, symbols, name_map, min_rs, min_liq, "Auto", trading_style)
        scan_duration = time.time() - scan_start

        if len(df) > 0:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### 📊 Benchmark: **{selected_benchmark}**")
            
            with col2:
                if not bm_table.empty:
                    if trading_style == "Swing (3M Focus)":
                        perf_col = "Return_3M"
                        timeframe = "3M"
                    else:
                        perf_col = "Return_6M"
                        timeframe = "6M"
                    
                    best_perf = bm_table.iloc[0][perf_col]
                    st.metric(f"{timeframe} Return", f"{best_perf}%")

            if not bm_table.empty:
                with st.expander("📈 All Benchmark Returns"):
                    st.dataframe(bm_table, hide_index=True, use_container_width=True)

            st.markdown(f"### 🎯 Top RS Leaders (≥ {min_rs}%) :green[Found **{len(df)} stocks** in {scan_duration:.0f}s]")

            def rsi_color(v):
                if pd.isna(v): return ""
                if v >= 60: return "background-color:#d4edda;color:#155724"
                if v <= 40: return "background-color:#f8d7da;color:#721c24"
                return ""
            
            def vol_ratio_color(v):
                if pd.isna(v): return ""
                if v >= 2.5: return "background-color:#fff3cd;color:#856404;font-weight:bold"
                if v >= 1.5: return "background-color:#d1ecf1;color:#0c5460"
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
                "Vol_Ratio": lambda x: f"{x:.2f}x" if pd.notna(x) else "-",
                "Vol_Spike": lambda x: f"{x:.2f}x" if pd.notna(x) else "-",
                "RS_Rank": "{:.1f}%"
            }).background_gradient(
                subset=["RS_Rank"], 
                cmap="RdYlGn",
                vmin=min_rs,
                vmax=100
            ).map(rsi_color, subset=["RSI_D", "RSI_W", "RSI_M"]
            ).map(vol_ratio_color, subset=["Vol_Ratio"])

            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Chart": st.column_config.LinkColumn("Chart", display_text="📈 View"),
                    "Momentum": st.column_config.TextColumn("Momentum", help="RS 3M vs 6M"),
                    "Vol_Ratio": st.column_config.TextColumn("Vol Ratio", help="Today vs 20D avg"),
                    "Vol_Spike": st.column_config.TextColumn("Vol Spike", help="Recent 5D vs prev 20D"),
                    "Vol_Trend": st.column_config.TextColumn("Vol Trend", help="20D volume MA trend"),
                    "Vol_Breakout": st.column_config.TextColumn("🔥", help="Vol breakout signal")
                },
                height=420
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

            st.markdown("### 💡 Key Metrics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Avg RS Rank", f"{df['RS_Rank'].mean():.1f}%")
            
            with col2:
                improving = len(df[df["RS_Delta"] > 0])
                st.metric("Accelerating", f"{improving}/{len(df)}")
            
            with col3:
                overbought = len(df[df["RSI_D"] > 70])
                st.metric("Overbought", f"{overbought}")
            
            with col4:
                vol_breakouts = len(df[df["Vol_Breakout"] == "🔥"])
                st.metric("🔥 Vol Breakout", f"{vol_breakouts}")
            
            with col5:
                high_vol = len(df[df["Vol_Ratio"] >= 2.0])
                st.metric("Vol >2x", f"{high_vol}")

        else:
            st.warning("⚠️ No stocks passed filters. Try relaxing criteria.")

    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        **Trading Style:**
        - **Swing (3M Focus)**: More responsive, 3-month lookback
        - **Position (6M Focus)**: More stable, 6-month lookback
        - **Hybrid (6M)**: Default 6-month approach
        
        **Volume Analysis:**
        - **Vol Ratio**: Today vs 20-day average (>2x = significant)
        - **🔥 Vol Breakout**: High volume + price near 52-week high
        
        **RSI Levels:**
        - 🟢 Green (≥60): Overbought | 🔴 Red (≤40): Oversold
        """)

if __name__ == "__main__":
    main()