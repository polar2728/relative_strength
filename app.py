# app.py
# NSE Relative Strength Leaders Scanner – PRO Version
# Ram – Hyderabad – Jan 2026
# Fixed: NSE SERIES issue + True Log RS + RS Acceleration + Robust Benchmark Logic

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import warnings
from datetime import datetime

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
    "Nifty 50": "^NSEI",
    "Nifty Next 50": "^NSMIDCP",
    "Nifty 100": "^CNX100",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
    "Nifty Midcap 150": "NIFTYMIDCAP150.NS",
    "Nifty Total Market": "NIFTY_TOTAL_MKT.NS",
}

# ─────────────────────────────────────────────────────────────
# LOAD NSE UNIVERSE (ROBUST)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(ttl=86400)
def load_nse_universe():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()

        df = pd.read_csv(io.StringIO(r.text), encoding="utf-8-sig")

        # 🔑 Critical fix
        df.columns = df.columns.str.strip().str.upper()

        if "SERIES" not in df.columns:
            st.error("NSE file format changed: 'SERIES' column missing")
            st.write("Columns found:", list(df.columns))
            return [], {}

        df = df[df["SERIES"] == "EQ"]

        if "SYMBOL" not in df.columns:
            st.error("NSE file missing SYMBOL column")
            return [], {}

        df["Symbol"] = df["SYMBOL"].str.strip() + ".NS"

        name_map = {}
        if "NAME OF COMPANY" in df.columns:
            name_map = dict(
                zip(df["SYMBOL"].str.strip(), df["NAME OF COMPANY"].str.strip())
            )

        return df["Symbol"].tolist(), name_map

    except Exception as e:
        st.error(f"Universe load failed: {e}")
        return [], {}

# ─────────────────────────────────────────────────────────────
# LOAD NIFTY 50
# ─────────────────────────────────────────────────────────────
@st.cache_resource(ttl=86400)
def load_nifty50_symbols():
    url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return [s.strip() + ".NS" for s in df["Symbol"].dropna()]
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return round((100 - (100 / (1 + rs))).iloc[-1], 1)

def log_rs(price_now, price_then, bm_now, bm_then):
    return np.log(price_now / price_then) - np.log(bm_now / bm_then)

# ─────────────────────────────────────────────────────────────
# RS SCAN LOGIC
# ─────────────────────────────────────────────────────────────
def rs_scan(symbols, name_map, min_rs_rank, min_liq, benchmark_mode):

    status = st.empty()
    progress = st.progress(0)

    status.info("📥 Downloading stock data…")
    progress.progress(10)
    
    # with st.spinner("Downloading market data…"):
    stock_data = yf.download(
        symbols, period="3y", auto_adjust=True,
        group_by="ticker", threads=True, progress=False
    )
    status.info("📊 Evaluating benchmarks…")
    progress.progress(25)

    bm_data = yf.download(
        list(BENCHMARK_CANDIDATES.values()),
        period="3y", auto_adjust=True,
        group_by="ticker", progress=False
    )

# ───────── Benchmark selection ─────────
    bm_rows = []
    selected_df = None
    selected_name = None
    best_ret = -1e9

    for name, ticker in BENCHMARK_CANDIDATES.items():
        if ticker not in bm_data.columns.levels[0]:
            continue

        df = bm_data[ticker].dropna()
        if len(df) < RS_LOOKBACK_6M + 5:
            continue

        close_bm = df["Close"]  # keep Series
        ret6m = (close_bm.iloc[-1] / close_bm.iloc[-RS_LOOKBACK_6M] - 1) * 100
        bm_rows.append((name, ticker, round(ret6m, 2)))

        if benchmark_mode == "Auto" and ret6m > best_ret:
            best_ret = ret6m
            selected_name = name
            selected_df = df

    if benchmark_mode == "Fixed Nifty 500":
        selected_name = "Nifty 500"
        selected_df = bm_data["^CRSLDX"].dropna()

    st.subheader("Benchmark Comparison (6M)")
    bm_df = pd.DataFrame(bm_rows, columns=["Benchmark", "Ticker", "6M Return %"]) \
             .sort_values("6M Return %", ascending=False)
    st.dataframe(bm_df, hide_index=True)
    st.success(f"Selected Benchmark: **{selected_name}**")

    status.success(f"✅ Selected Benchmark: {selected_name}")
    progress.progress(35)

    # ───────── Stock loop ─────────
    results = []
    total = len(symbols)

    status.info("🔎 Scanning stocks…")

    for i, sym in enumerate(symbols):
        if sym not in stock_data.columns.levels[0]:
            continue

        if i % 25 == 0:
            progress.progress(35 + int(55 * i / total))
            status.info(f"🔎 Scanning… {i}/{total}")

        df = stock_data[sym].dropna()
        if len(df) < 250:
            continue

        close = df["Close"]           # Series – never overwrite with scalar
        price = close.iloc[-1]

        dma200 = close.rolling(200).mean().iloc[-1]
        liq = (close * df["Volume"]).tail(30).mean() / 1e7

        if price < dma200 or liq < min_liq:
            continue

        # Benchmark Series – keep separate
        bm_close = selected_df["Close"]

        # Compute returns without overwriting Series
        bm_ret_3m = (bm_close.iloc[-1] / bm_close.iloc[-RS_LOOKBACK_3M] - 1)
        bm_ret_6m = (bm_close.iloc[-1] / bm_close.iloc[-RS_LOOKBACK_6M] - 1)

        # Stock returns
        stock_ret_6m = price / close.iloc[-RS_LOOKBACK_6M] - 1
        stock_ret_3m = price / close.iloc[-RS_LOOKBACK_3M] - 1

        # Log RS
        rs6 = log_rs(price, close.iloc[-RS_LOOKBACK_6M],
                     bm_close.iloc[-1], bm_close.iloc[-RS_LOOKBACK_6M])

        rs3 = log_rs(price, close.iloc[-RS_LOOKBACK_3M],
                     bm_close.iloc[-1], bm_close.iloc[-RS_LOOKBACK_3M])

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

    progress.progress(95)
    status.info("📐 Ranking…")

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["RS_Rank"] = df["RS_6M"].rank(pct=True) * 100
    df["Momentum"] = np.where(df["RS_Delta"] > 0, "Improving",
                       np.where(df["RS_Delta"] < 0, "Decelerating", "Stable"))

    df = df[df["RS_Rank"] >= min_rs_rank]
    df = df.sort_values("RS_Rank", ascending=False)

    progress.progress(100)
    status.success("✅ Scan complete")


    return df.reset_index(drop=True)

# ───────── UI ─────────
def color_rsi(val):
    if pd.isna(val):
        return ""
    if val >= 60:
        return "background-color: #ffcccc"
    if val <= 40:
        return "background-color: #ccffcc"
    return ""

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    st.title("NSE Relative Strength Leaders Scanner – PRO")

    with st.sidebar:
        universe = st.radio("Universe", ["Full NSE", "Nifty 50"])
        benchmark_mode = st.radio("Benchmark Mode", ["Auto", "Fixed Nifty 500"])
        min_rs = st.slider("Min RS Rank", 60, 95, MIN_RS_RANK)
        min_liq = st.slider("Min Liquidity (Cr)", 1, 20, MIN_LIQUIDITY_CR)
        run = st.button("Run Scan", type="primary")

    full_syms, name_map = load_nse_universe()
    if not full_syms:
        st.stop()

    nifty50 = load_nifty50_symbols()
    symbols = full_syms if universe == "Full NSE" else nifty50

    if run:
        df = rs_scan(symbols, name_map, min_rs, min_liq, benchmark_mode)

        if df.empty:
            st.warning("No stocks passed filters")
            return

        # ── Create styled DataFrame FIRST ──
        def rsi_color(v):
            if pd.isna(v): return ""
            if v >= 60: return "background-color:#ccffcc"
            if v <= 40: return "background-color:#ffcccc"
            return ""

        styled = df.style.format({
            "Price"     : "{:,.2f}",           # 2 decimals + thousand separator
            "RS_6M"     : "{:.3f}",
            "RS_3M"     : "{:.3f}",
            "RS_Delta"  : "{:.3f}",
            "RS_Rank"   : "{:.1f}%",
            "LiquidityCr": "{:.1f}",
            "RSI_D"     : "{:.1f}",
            "RSI_W"     : "{:.1f}",
            "RSI_M"     : "{:.1f}",
        }).background_gradient(subset=["RS_Rank"], cmap="YlGn")
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

if __name__ == "__main__":
    main()
