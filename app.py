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


def kite_login_ui():
    kite = KiteConnect(api_key=st.secrets["KITE_API_KEY"])

    st.link_button(
        "🔐 Login with Zerodha Kite",
        kite.login_url()
    )



def handle_kite_callback():
    if "kite" in st.session_state:
        return
    params = st.query_params
    request_token = params.get("request_token")
    if request_token and request_token != "None":  # Handle empty strings
        try:
            kite = KiteConnect(api_key=st.secrets["KITE_API_KEY"])
            data = kite.generate_session(request_token, st.secrets["KITE_API_SECRET"])
            kite.set_access_token(data["access_token"])
            st.session_state.kite = kite
            st.session_state.access_token = data["access_token"]
        except Exception as e:
            st.error(f"Auth failed: {e}")
    st.query_params.clear()  # Always clear to prevent param loops


# ─────────────────────────────────────────────────────────────
# KITE AUTH
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
# INSTRUMENT MAP (UNCHANGED)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(ttl=86400 * 7)
def load_kite_instrument_map(_kite):
    instruments = _kite.instruments("NSE")
    df = pd.DataFrame(instruments)

    eq = df[(df["segment"] == "NSE") & (df["exchange"] == "NSE")]
    idx = df[df["segment"] == "INDICES"]

    eq_map = dict(zip(eq["tradingsymbol"], eq["instrument_token"]))
    idx_map = dict(zip(idx["tradingsymbol"], idx["instrument_token"]))

    st.sidebar.success(f"{len(eq_map)} equities, {len(idx_map)} indices loaded")
    return {**eq_map, **idx_map}

# ─────────────────────────────────────────────────────────────
# HISTORICAL DATA (UNCHANGED)
# ─────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),  # Wait 2 seconds between retries
    retry=retry_if_exception_type((requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError))
)
@st.cache_data(ttl=86400)
def fetch_kite_historical(_kite, symbol, days=365*3):
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    is_index = symbol in BENCHMARK_CANDIDATES.values()
    token = st.session_state.instrument_map.get(
        symbol if is_index else symbol.replace(".NS", "")
    )

    if not token:
        return pd.DataFrame()

    data = _kite.historical_data(
        token, from_date, to_date, "day"
    )

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df

# ─────────────────────────────────────────────────────────────
# BATCH FETCH (PROGRESS → SIDEBAR)
# ─────────────────────────────────────────────────────────────
def fetch_in_batches(kite, symbols, batch_size=100):
    all_data = {}
    total = len(symbols)

    prog = st.sidebar.progress(0.0)
    status = st.sidebar.empty()

    for i in range(0, total, batch_size):
        batch = symbols[i:i+batch_size]
        status.info(f"Fetching {i+len(batch)}/{total}")

        for sym in batch:
            all_data[sym] = fetch_kite_historical(kite, sym)
            time.sleep(0.25)

        prog.progress((i + len(batch)) / total)

    status.success("Fetch complete")
    return all_data

# ─────────────────────────────────────────────────────────────
# UNIVERSE LOADERS (UNCHANGED)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(ttl=86400)
def load_nse_universe():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.upper().str.strip()

    df = df[df["SERIES"] == "EQ"]
    df["Symbol"] = df["SYMBOL"] + ".NS"
    name_map = dict(zip(df["SYMBOL"], df["NAME OF COMPANY"]))
    return df["Symbol"].tolist(), name_map

# @st.cache_resource(ttl=86400)
# def load_nifty50_symbols():
#     url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
#     df = pd.read_csv(url)
#     return [s + ".NS" for s in df["Symbol"]]

@st.cache_resource(ttl=86400)
def load_nifty50_symbols():
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

    st.error("Failed to load Nifty 50 list from NSE")
    return []

# ─────────────────────────────────────────────────────────────
# INDICATORS (UNCHANGED)
# ─────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean()
    return round((100 - (100 / (1 + rs))).iloc[-1], 1)

def log_rs(p, p0, b, b0):
    return np.log(p / p0) - np.log(b / b0)

# ─────────────────────────────────────────────────────────────
# RS SCAN (LOGIC UNCHANGED)
# ─────────────────────────────────────────────────────────────
def rs_scan(kite, symbols, name_map, min_rs, min_liq, benchmark_mode):
    st.session_state.instrument_map = load_kite_instrument_map(kite)

    stock_data = fetch_in_batches(kite, symbols)
    bm_data = fetch_in_batches(kite, list(BENCHMARK_CANDIDATES.values()))

    # best_ret = -1e9
    # selected_df = None

    # for name, sym in BENCHMARK_CANDIDATES.items():
    #     df = bm_data.get(sym)
    #     if df is None or len(df) < RS_LOOKBACK_6M:
    #         continue
    #     ret = df["Close"].iloc[-1] / df["Close"].iloc[-RS_LOOKBACK_6M] - 1
    #     if ret > best_ret:
    #         best_ret = ret
    #         selected_df = df

    best_ret = -1e9
    selected_df = None
    selected_benchmark = None
    benchmark_rows = []

    for name, sym in BENCHMARK_CANDIDATES.items():
        df = bm_data.get(sym)
        if df is None or len(df) < RS_LOOKBACK_6M:
            continue

        ret = df["Close"].iloc[-1] / df["Close"].iloc[-RS_LOOKBACK_6M] - 1
        benchmark_rows.append({
            "Benchmark": name,
            "Return_6M": round(ret * 100, 2)
        })

        if ret > best_ret:
            best_ret = ret
            selected_df = df
            selected_benchmark = name

    results = []
    for sym, df in stock_data.items():
        if df.empty or len(df) < 250:
            continue

        close = df["Close"]
        price = close.iloc[-1]
        dma200 = close.rolling(200).mean().iloc[-1]
        liq = (close * df["Volume"]).tail(30).mean() / 1e7

        if price < dma200 * 0.95 or liq < min_liq:
            continue

        rs6 = log_rs(price, close.iloc[-RS_LOOKBACK_6M],
                     selected_df["Close"].iloc[-1],
                     selected_df["Close"].iloc[-RS_LOOKBACK_6M])

        rs3 = log_rs(price, close.iloc[-RS_LOOKBACK_3M],
                     selected_df["Close"].iloc[-1],
                     selected_df["Close"].iloc[-RS_LOOKBACK_3M])

        rs_delta = rs3 - rs6
        rsi_d = compute_rsi(close) 
        rsi_w = compute_rsi(close.resample("W-FRI").last()) if len(close) > 100 else None 
        rsi_m = compute_rsi(close.resample("ME").last()) if len(close) > 400 else None

        clean = sym.replace(".NS", "")
        tv = f"https://in.tradingview.com/chart/?symbol=NSE%3A{clean}"

        results.append({
            "Symbol": clean,  # fixed typo: sym_clean → clean
            "Name": name_map.get(clean, ""),
            "Price": round(price, 2),
            "RS": round(rs6, 3),          # using 6M as main RS (adjust if needed)
            "RS_3M": round(rs3, 3),
            "RS_6M": round(rs6, 3),
            "RS_Delta": round(rs_delta, 3),
            "LiquidityCr": round(liq, 1),
            "RSI_D": rsi_d,
            "RSI_W": rsi_w,
            "RSI_M": rsi_m,
            "Chart": tv
        })

    df = pd.DataFrame(results)
    df["RS_Rank"] = df["RS_6M"].rank(pct=True) * 100
    df["Momentum"] = np.where(df["RS_Delta"] > 0, "🚀 Improving", np.where(df["RS_Delta"] < 0, "📉 Slowing", "➡️ Stable"))
    
    # return df[df["RS_Rank"] >= min_rs].sort_values("RS_Rank", ascending=False)
    bm_table = pd.DataFrame(benchmark_rows).sort_values("Return_6M", ascending=False)

    return (
        df[df["RS_Rank"] >= min_rs].sort_values("RS_Rank", ascending=False),
        selected_benchmark,
        bm_table
    )

def consume_code_once():
    params = st.experimental_get_query_params()

    if "code" not in params or "kite" in st.session_state:
        return

    code = params["code"][0]

    # Fetch token from callback app
    import requests
    resp = requests.get(
        f"https://kc-rs-scanner.streamlit.app/get_token?code={code}",
        timeout=5
    )

    access_token = resp.text.strip()

    kite = KiteConnect(api_key=st.secrets["KITE_API_KEY"])
    kite.set_access_token(access_token)
    st.session_state.kite = kite



# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div style='background: linear-gradient(135deg,#667eea,#764ba2,#f093fb);
                padding:1.2rem;border-radius:14px;color:white;text-align:center'>
        <h2 style='margin:0'>🏆 NSE RS Leaders PRO</h2>
        <p style='margin:0;font-size:0.9rem'>Dynamic benchmark • RS acceleration</p>
    </div>
    """, unsafe_allow_html=True)

    # kite = kite_auth_section()
    # 1️⃣ Consume token ONCE (already discussed)
    consume_kite_token_once()

    # 2️⃣ If Kite not ready → show login UI and STOP
    if "kite" not in st.session_state:
        kite_login_ui()
        st.stop()

    # 3️⃣ Kite is ready → proceed with scanner
    kite = st.session_state.kite

    # kite = st.session_state.get("kite")
    # if not kite:
    #     st.stop()

    st.sidebar.markdown("### ⚙️ Scan Config")
    universe = st.sidebar.radio("Universe", ["Nifty 50", "Full NSE"])
    benchmark_mode = st.sidebar.radio("Benchmark", ["Auto"])
    min_rs = st.sidebar.slider("Min RS Rank", 60, 95, MIN_RS_RANK)
    min_liq = st.sidebar.slider("Min Liquidity ₹Cr", 1, 20, MIN_LIQUIDITY_CR)

    if st.sidebar.button("🚀 Run Scan", use_container_width=True):
        syms, name_map = load_nse_universe()
        symbols = load_nifty50_symbols() if universe == "Nifty 50" else syms

        # df = rs_scan(kite, symbols, name_map, min_rs, min_liq, benchmark_mode)

        df, selected_benchmark, bm_table = rs_scan(kite, symbols, name_map, min_rs, min_liq, benchmark_mode)


        # styled = df.style.background_gradient(
        #     subset=["RS_Rank"], cmap="Greens"
        # ).applymap(
        #     lambda v: "color:green" if v >= 60 else "color:red" if v <= 40 else "",
        #     subset=["RSI_D", "RSI_W", "RSI_M"]
        # )

        # ── Create styled DataFrame FIRST ──
        def rsi_color(v):
            if pd.isna(v): return ""
            if v >= 60: return "background-color:#ccffcc"
            if v <= 40: return "background-color:#ffcccc"
            return ""

        styled = df.style.format({
            "Price": "{:.2f}",
            "RS": "{:.3f}",
            "RS_6M": "{:.3f}",
            "RS_3M": "{:.3f}",
            "RS_Delta": "{:.3f}",
            "LiquidityCr": "{:.2f}",
            "RSI_D"     : "{:.1f}",
            "RSI_W"     : "{:.1f}",
            "RSI_M"     : "{:.1f}",
            "RS_Rank": "{:.1f}%"
        }).background_gradient(subset=["RS_Rank"], cmap="YlGn") \
          .map(rsi_color, subset=["RSI_D", "RSI_W", "RSI_M"])

        # ── Now display it ──
        st.dataframe(
            styled,
            width="stretch",
            hide_index=True,
            column_config={
                "Chart": st.column_config.LinkColumn(
                    "Chart", display_text="📈 View"
                )
            }
        )

        # CSV download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            f"RS_LEADERS_PRO_{datetime.now():%Y%m%d}.csv",
            mime="text/csv",
            width="stretch"
        )

        st.markdown("---")

        st.markdown(
            f"### 📊 Benchmark Used for RS: **{selected_benchmark}**"
        )

        st.dataframe(
            bm_table,
            hide_index=True,
            width="stretch"
        )

if __name__ == "__main__":
    main()
