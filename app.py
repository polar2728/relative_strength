import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import warnings
import time
import urllib.parse
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

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
# CLOUD SECRETS AUTH - NO USER INPUT REQUIRED
# ─────────────────────────────────────────────────────────────
def get_kite_from_secrets():
    """Get pre-configured Kite from Streamlit secrets"""
    try:
        # Access secrets set in Streamlit Cloud dashboard
        api_key = st.secrets["KITE_API_KEY"]
        access_token = st.secrets["KITE_ACCESS_TOKEN"]  # Your 24h token
        
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Quick validation
        profile = kite.profile()
        
        st.sidebar.success(f"✅ Connected: {profile['user_name']}")
        return kite
        
    except Exception as e:
        st.sidebar.error(f"❌ Auth failed: Update secrets → {str(e)[:50]}")
        return None

# ─────────────────────────────────────────────────────────────
# KITE AUTH (FALLBACK OPTION)
# ─────────────────────────────────────────────────────────────
def kite_auth_section():
    st.sidebar.markdown("### 🔌 Kite Connect")

    api_key = st.sidebar.text_input(
        "API Key", type="password",
        value=st.session_state.get("api_key", "")
    )
    access_token = st.sidebar.text_input(
        "Access Token", type="password",
        value=st.session_state.get("access_token", "")
    )

    if st.sidebar.button("TEST CONNECTION", use_container_width=True):
        if not api_key or not access_token:
            st.sidebar.error("Enter API key + token")
            return None

        try:
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            profile = kite.profile()

            st.session_state.api_key = api_key
            st.session_state.access_token = access_token
            st.session_state.kite = kite

            st.sidebar.success(f"Connected: {profile['user_name']}")
        except Exception as e:
            st.sidebar.error(str(e))

    kite = st.session_state.get("kite")
    if kite:
        st.sidebar.success("Kite ready")
    else:
        st.sidebar.warning("Not connected")

    return kite

def get_kite():
    if "kite" in st.session_state:
        return st.session_state.kite

    return None

# ─────────────────────────────────────────────────────────────
# INSTRUMENT MAP - FIXED: SEPARATE EQUITY & INDEX MAPS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(ttl=86400 * 7)
def load_kite_instrument_map(_kite):
    """Load instrument tokens with separate maps for equities and indices to prevent collisions"""
    instruments = _kite.instruments("NSE")
    df = pd.DataFrame(instruments)

    eq = df[(df["segment"] == "NSE") & (df["exchange"] == "NSE")]
    idx = df[df["segment"] == "INDICES"]

    eq_map = dict(zip(eq["tradingsymbol"], eq["instrument_token"]))
    idx_map = dict(zip(idx["tradingsymbol"], idx["instrument_token"]))

    st.sidebar.success(f"📊 {len(eq_map)} equities, {len(idx_map)} indices loaded")
    
    # Return separate maps to avoid symbol collisions
    return {"equities": eq_map, "indices": idx_map}

# ─────────────────────────────────────────────────────────────
# HISTORICAL DATA - FIXED: PROPER MAP LOOKUP
# ─────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError))
)
@st.cache_data(ttl=86400)
def fetch_kite_historical(_kite, symbol, days=365*2):
    """Fetch historical data with proper equity/index map lookup"""
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date   = datetime.now().strftime("%Y-%m-%d")

    is_index = symbol in BENCHMARK_CANDIDATES.values()
    key = symbol if is_index else symbol.replace(".NS", "")

    # Use appropriate map based on symbol type
    instrument_maps = st.session_state.instrument_map
    token = instrument_maps["indices"].get(key) if is_index else instrument_maps["equities"].get(key)

    if token is None:
        print(f"→ Skipping {symbol} | No instrument_token found for key '{key}'")
        return pd.DataFrame()

    try:
        token = int(token)
    except (ValueError, TypeError):
        print(f"→ Invalid token type for {symbol}: {token} (type={type(token)})")
        return pd.DataFrame()

    print(f"→ Fetching {symbol} | token={token} | {from_date} → {to_date}")

    try:
        data = _kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval="day",
            continuous=False,
            oi=False
        )

        if not data:
            print(f"→ No data returned for {symbol} (token {token})")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        print(f"← {symbol}: {len(df)} rows")
        return df

    except Exception as e:
        print(f"!!! FAILED {symbol} (token {token}): {type(e).__name__} → {str(e)}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────
# BATCH FETCH - ENHANCED: BETTER PROGRESS UI
# ─────────────────────────────────────────────────────────────
def fetch_in_batches(kite, symbols, batch_size=50):
    """Fetch data in batches with enhanced progress tracking"""
    all_data = {}
    total = len(symbols)
    prog = st.sidebar.progress(0.0)
    status = st.sidebar.empty()

    processed = 0
    successful = 0
    last_update = time.time()

    for i in range(0, total, batch_size):
        batch = symbols[i:i+batch_size]
        batch_num = i//batch_size + 1
        status.info(f"⏳ Batch {batch_num} | {processed}/{total} processed | {successful} successful")

        for sym in batch:
            df = fetch_kite_historical(kite, sym)
            all_data[sym] = df
            if not df.empty:
                successful += 1
            processed += 1
            time.sleep(0.34)

            # Force UI refresh periodically
            now = time.time()
            if now - last_update > 12:
                prog.progress(processed / total)
                status.info(f"⏳ Working... {processed}/{total} | {successful} valid")
                last_update = now
                time.sleep(0.1)

        prog.progress(processed / total)

    status.success(f"✅ Complete: {successful}/{total} symbols fetched successfully")
    prog.progress(1.0)
    return all_data

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
    """Load Nifty 50 constituents with retry logic"""
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

    st.error("❌ Failed to load Nifty 50 list from NSE")
    return []

# ─────────────────────────────────────────────────────────────
# INDICATORS - FIXED: RSI EDGE CASE
# ─────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    """Compute RSI with edge case handling for flat prices"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    
    # Handle edge case: no losses (flat or only gains)
    if loss.iloc[-1] == 0:
        return 100.0
    
    # Handle edge case: no gains (only losses)
    if gain.iloc[-1] == 0:
        return 0.0
    
    rs = gain.iloc[-1] / loss.iloc[-1]
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 1)

def log_rs(p, p0, b, b0):
    """Log-based relative strength calculation"""
    return np.log(p / p0) - np.log(b / b0)

# ─────────────────────────────────────────────────────────────
# RS SCAN - ENHANCED: BETTER ERROR HANDLING & DIAGNOSTICS
# ─────────────────────────────────────────────────────────────
def rs_scan(kite, symbols, name_map, min_rs, min_liq, benchmark_mode):
    """Main RS scanning logic with enhanced diagnostics"""
    
    # Load instrument maps
    st.session_state.instrument_map = load_kite_instrument_map(kite)

    # Fetch stock data
    with st.spinner("📥 Fetching stock data..."):
        stock_data = fetch_in_batches(kite, symbols)
    
    # Fetch benchmark data
    with st.spinner("📊 Fetching benchmark indices..."):
        bm_data = fetch_in_batches(kite, list(BENCHMARK_CANDIDATES.values()))

    # Select best performing benchmark
    best_ret = -1e9
    selected_df = None
    selected_benchmark = None
    benchmark_rows = []
    min_required_days = max(RS_LOOKBACK_6M, 200) + 30

    for name, sym in BENCHMARK_CANDIDATES.items():
        df = bm_data.get(sym)
        if df.empty or len(df) < min_required_days:
            continue

        ret = df["Close"].iloc[-1] / df["Close"].iloc[-RS_LOOKBACK_6M] - 1
        benchmark_rows.append({
            "Benchmark": name,
            "Return_6M": round(ret * 100, 2),
            "Status": "✅ Selected" if ret > best_ret else ""
        })

        if ret > best_ret:
            best_ret = ret
            selected_df = df
            selected_benchmark = name

    # Validate benchmark selection
    if selected_df is None or selected_df.empty:
        st.error("❌ No valid benchmark found with sufficient history. Check your data connection.")
        st.stop()

    # Scan stocks
    results = []
    filter_stats = {
        "total": len(stock_data),
        "short_history": 0,
        "below_dma200": 0,
        "low_liquidity": 0,
        "passed": 0
    }

    with st.spinner(f"🔍 Scanning {len(stock_data)} stocks against {selected_benchmark}..."):
        for sym, df in stock_data.items():
            if df.empty or len(df) < 250:
                filter_stats["short_history"] += 1
                continue

            close = df["Close"]
            price = close.iloc[-1]
            dma200 = close.rolling(200).mean().iloc[-1]
            liq = (close * df["Volume"]).tail(30).mean() / 1e7

            # Apply filters
            if price < dma200 * 0.95:
                filter_stats["below_dma200"] += 1
                continue
                
            if liq < min_liq:
                filter_stats["low_liquidity"] += 1
                continue

            # Calculate RS metrics
            rs6 = log_rs(price, close.iloc[-RS_LOOKBACK_6M],
                         selected_df["Close"].iloc[-1],
                         selected_df["Close"].iloc[-RS_LOOKBACK_6M])

            rs3 = log_rs(price, close.iloc[-RS_LOOKBACK_3M],
                         selected_df["Close"].iloc[-1],
                         selected_df["Close"].iloc[-RS_LOOKBACK_3M])

            rs_delta = rs3 - rs6
            
            # Calculate RSI at multiple timeframes
            rsi_d = compute_rsi(close)
            rsi_w = compute_rsi(close.resample("W-FRI").last()) if len(close) > 100 else None
            rsi_m = compute_rsi(close.resample("ME").last()) if len(close) > 400 else None

            # Generate TradingView link
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

    # Create results DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        st.warning("⚠️ No stocks passed the filters. Try relaxing the criteria.")
        with st.expander("🔍 Filter Statistics"):
            st.write(f"**Total symbols fetched:** {filter_stats['total']}")
            st.write(f"**Filtered out (insufficient history):** {filter_stats['short_history']}")
            st.write(f"**Filtered out (below 200 DMA):** {filter_stats['below_dma200']}")
            st.write(f"**Filtered out (low liquidity):** {filter_stats['low_liquidity']}")
            st.write(f"**Passed all filters:** {filter_stats['passed']}")
    
    # Calculate RS rank
    df["RS_Rank"] = df["RS_6M"].rank(pct=True) * 100
    df["Momentum"] = np.where(
        df["RS_Delta"] > 0, "🚀 Improving", 
        np.where(df["RS_Delta"] < 0, "📉 Slowing", "➡️ Stable")
    )
    
    # Create benchmark table
    bm_table = pd.DataFrame(benchmark_rows).sort_values("Return_6M", ascending=False)

    # Display filter statistics in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Scan Summary")
    st.sidebar.metric("Stocks Scanned", filter_stats["total"])
    st.sidebar.metric("Passed Filters", filter_stats["passed"])
    st.sidebar.metric("Final Results", len(df[df["RS_Rank"] >= min_rs]))

    return (
        df[df["RS_Rank"] >= min_rs].sort_values("RS_Rank", ascending=False),
        selected_benchmark,
        bm_table
    )

# ─────────────────────────────────────────────────────────────
# MAIN - ENHANCED UX
# ─────────────────────────────────────────────────────────────
def main():
    # Header with gradient
    st.markdown("""
    <div style='background: linear-gradient(135deg,#667eea,#764ba2,#f093fb);
                padding:1.2rem;border-radius:14px;color:white;text-align:center'>
        <h2 style='margin:0'>🏆 NSE RS Leaders PRO</h2>
        <p style='margin:0;font-size:0.9rem'>Dynamic benchmark • RS acceleration • Multi-timeframe RSI</p>
    </div>
    """, unsafe_allow_html=True)

    # Authentication
    kite = get_kite_from_secrets()
    if not kite:
        st.error("⚠️ **Admin:** Update KITE_ACCESS_TOKEN in Streamlit Cloud settings")
        st.info("💡 **Tip:** Generate a new access token and update it in Settings → Secrets")
        st.stop()

    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Scan Configuration")
    
    universe = st.sidebar.radio(
        "Universe",
        ["Nifty 50", "Full NSE"],
        help="Nifty 50 = faster scan, Full NSE = comprehensive but slower"
    )
    
    benchmark_mode = st.sidebar.radio(
        "Benchmark Selection", 
        ["Auto"],
        help="Automatically selects the best performing benchmark index"
    )
    
    min_rs = st.sidebar.slider(
        "Min RS Rank %",
        60, 95, MIN_RS_RANK,
        help="Only show stocks with RS rank above this percentile"
    )
    
    min_liq = st.sidebar.slider(
        "Min Liquidity ₹Cr",
        1, 20, MIN_LIQUIDITY_CR,
        help="Minimum average daily liquidity in crores"
    )

    # Scan button
    if st.sidebar.button("🚀 RUN SCAN", use_container_width=True, type="primary"):
        # Load universe
        syms, name_map = load_nse_universe()
        symbols = load_nifty50_symbols() if universe == "Nifty 50" else syms

        if not symbols:
            st.error("❌ Failed to load stock universe")
            st.stop()

        # Run scan
        scan_start = time.time()
        df, selected_benchmark, bm_table = rs_scan(kite, symbols, name_map, min_rs, min_liq, benchmark_mode)
        scan_duration = time.time() - scan_start

        # Display results
        if len(df) > 0:
            st.success(f"✅ Found **{len(df)} stocks** in {scan_duration:.1f}s")
            
            # Benchmark info
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### 📊 Benchmark: **{selected_benchmark}**")
                st.caption("All RS calculations are relative to this index")
            
            with col2:
                best_perf = bm_table.iloc[0]["Return_6M"]
                st.metric("6M Return", f"{best_perf}%")

            # Benchmark comparison table
            with st.expander("📈 View All Benchmark Returns"):
                st.dataframe(
                    bm_table,
                    hide_index=True,
                    use_container_width=True
                )

            st.markdown("---")
            st.markdown(f"### 🎯 Top RS Leaders (Rank ≥ {min_rs}%)")

            # Style the dataframe
            def rsi_color(v):
                if pd.isna(v): return ""
                if v >= 60: return "background-color:#d4edda;color:#155724"  # Green
                if v <= 40: return "background-color:#f8d7da;color:#721c24"  # Red
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

            # Display dataframe
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Chart": st.column_config.LinkColumn(
                        "Chart",
                        display_text="📈 View"
                    ),
                    "Momentum": st.column_config.TextColumn(
                        "Momentum",
                        help="RS 3M vs 6M trend"
                    )
                },
                height=600
            )

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "📥 Download Results as CSV",
                csv,
                f"RS_Leaders_{selected_benchmark.replace(' ', '_')}_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )

            # Key insights
            st.markdown("---")
            st.markdown("### 💡 Key Insights")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg RS Rank", f"{df['RS_Rank'].mean():.1f}%")
            
            with col2:
                improving = len(df[df["RS_Delta"] > 0])
                st.metric("Improving Momentum", f"{improving} stocks")
            
            with col3:
                overbought = len(df[df["RSI_D"] > 70])
                st.metric("RSI > 70 (Daily)", f"{overbought} stocks")
            
            with col4:
                avg_liq = df["LiquidityCr"].mean()
                st.metric("Avg Liquidity", f"₹{avg_liq:.1f}Cr")

        else:
            st.warning("⚠️ No stocks passed the filters. Try adjusting the criteria.")

    # Info section
    with st.expander("ℹ️ How This Works"):
        st.markdown("""
        **Relative Strength (RS) Strategy:**
        1. Automatically selects the best performing benchmark index (6M return)
        2. Calculates log-based RS for each stock vs. benchmark
        3. Filters stocks trading above 200-day moving average
        4. Applies liquidity filters (min ₹5Cr avg daily volume)
        5. Ranks stocks by RS percentile
        6. Shows multi-timeframe RSI for confluence
        
        **RS Delta:** Difference between 3M and 6M RS (positive = accelerating)
        
        **RSI Color Code:**
        - 🟢 Green (≥60): Potentially overbought
        - 🔴 Red (≤40): Potentially oversold
        - ⚪ Neutral: Between 40-60
        
        **Best Practices:**
        - Focus on RS Rank > 80% with improving momentum
        - Avoid stocks with RSI > 70 on all timeframes (overextended)
        - Check TradingView charts before taking positions
        """)

if __name__ == "__main__":
    main()