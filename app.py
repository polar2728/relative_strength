# app.py
# NSE Relative Strength Leaders Scanner – Streamlit Version (TRADINGVIEW LINKS PERFECT)
# Ram – Hyderabad – Jan 2026

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import time
import random
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

st.set_page_config(page_title="NSE RS Leaders Scanner", layout="wide")

# CONFIG ──────────────────────────────────────────────────────────────────────
BENCHMARK          = "^NSEI"
RS_LOOKBACK        = 126
MIN_RS_RANK        = 80
MIN_LIQUIDITY_CR   = 5
DEFAULT_MIN_MCAP   = 2000
MAX_WORKERS        = 6

# ─────────────────────────────────────────────────────────────────────────────
# NSE UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(ttl=86400)
def load_nse_universe():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), header=0, encoding='utf-8-sig')
        df.columns = df.columns.str.strip().str.upper()
        if 'SERIES' not in df.columns:
            st.error("No 'SERIES' column")
            return [], {}
        df = df[df['SERIES'] == 'EQ']
        df['Symbol'] = df['SYMBOL'].str.strip() + '.NS'
        name_map = dict(zip(df['SYMBOL'], df['NAME OF COMPANY'].str.strip()))
        return df['Symbol'].tolist(), name_map
    except Exception as e:
        st.error(f"Universe load failed: {str(e)[:80]}")
        return [], {}

# ─────────────────────────────────────────────────────────────────────────────
# RS Filter WITH TradingView LINKS
# ─────────────────────────────────────────────────────────────────────────────

def rs_filter(symbols, symbol_to_name, rs_lookback, min_rs_rank, min_liq_cr):
    with st.spinner("Computing Relative Strength..."):
        data = yf.download(
            symbols,
            period="2y",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False
        )

        nifty = yf.download(
            BENCHMARK,
            period="2y",
            auto_adjust=True,
            progress=False
        )

        lookback = min(rs_lookback, len(nifty) - 30)
        nifty_ret = float(nifty["Close"].iloc[-1] / nifty["Close"].iloc[-lookback] - 1)

        results = []

        for sym in symbols:
            if sym not in data.columns.levels[0]:
                continue

            try:
                ticker_data = data[sym]
                if ticker_data is None:  # Skip failed downloads
                    continue
                    
                df = ticker_data.dropna()
                if len(df) < lookback + 30:
                    continue

                close = float(df["Close"].iloc[-1])
                ret = close / float(df["Close"].iloc[-lookback]) - 1
                rs = ret / nifty_ret if nifty_ret != 0 else 0

                dma200 = float(df["Close"].rolling(200).mean().iloc[-1])
                liq_cr = float((df["Close"] * df["Volume"]).tail(30).mean() / 1e7)

                if close > dma200 and liq_cr >= min_liq_cr:
                    clean_sym = sym.replace(".NS", "")
                    
                    # 🔥 TRADINGVIEW URL (for CSV)
                    tv_url = f"https://in.tradingview.com/chart/?symbol=NSE%3A{clean_sym}"
                    
                    results.append({
                        "Symbol": clean_sym,
                        "Name": symbol_to_name.get(clean_sym, ""),
                        "Price": round(close, 2),
                        "RS": round(rs, 3),
                        "LiquidityCr": round(liq_cr, 1),
                        "TradingView": tv_url  # Full URL in CSV
                    })

            except Exception:
                continue

        df1 = pd.DataFrame(results)
        if df1.empty:
            return df1
            
        df1["RS_Rank"] = df1["RS"].rank(pct=True) * 100
        df1 = df1[df1["RS_Rank"] >= min_rs_rank]
        df1 = df1.sort_values("RS_Rank", ascending=False).reset_index(drop=True)

        return df1

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP - PERFECT TradingView LINKS
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("NSE Relative Strength Leaders Scanner")

    with st.sidebar:
        rs_lookback = st.slider("RS Lookback (days)", 63, 252, RS_LOOKBACK, step=21)
        min_rs_rank = st.slider("Min RS Rank %", 60, 95, MIN_RS_RANK, step=5)
        min_liq_cr  = st.slider("Min Liquidity (Cr)", 1, 20, MIN_LIQUIDITY_CR, step=1)
        run_now     = st.button("Run Scan", type="primary", use_container_width=True)

    symbols, name_map = load_nse_universe()
    if not symbols:
        st.stop()

    st.caption(f"Universe: {len(symbols):,} stocks")

    if not run_now:
        st.info("Click Run Scan to start")
        return

    # RS Filters
    results = rs_filter(symbols, name_map, rs_lookback, min_rs_rank, min_liq_cr)

    if results.empty:
        st.warning("No stocks found")
        return

    st.subheader("Relative Strength Leaders")
    
    # 🔥 CREATE CLICKABLE SYMBOL COLUMN (WORKS PERFECTLY)
    display_df = results.copy()
    # display_df['Symbol'] = display_df['Symbol'].apply(
    #     lambda x: f'[📊 {x}](https://in.tradingview.com/chart/?symbol=NSE%3A{x})'
    # )
    
    # Show table WITHOUT TradingView column (cleaner display)
    cols_to_show = ['Symbol', 'Name', 'Price', 'RS', 'LiquidityCr', 'RS_Rank']
    st.dataframe(
        display_df[cols_to_show].style.format({
            'Price': '{:.2f}',
            'RS': '{:.3f}', 
            'LiquidityCr': '{:.1f}',
            'RS_Rank': '{:.1f}%'
        }).background_gradient(subset=['RS_Rank'], cmap='YlGn'),
        use_container_width=True,
        hide_index=True
    )

    # 🔥 CSV DOWNLOAD WITH TradingView URLs (complete data)
    csv_stage1 = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV (TradingView URLs included)",
        data=csv_stage1,
        file_name=f"RS_LEADERS_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Show sample TradingView link
    if not results.empty:
        st.caption(f"**Sample**: [📊 {results.iloc[0]['Symbol']} Chart](https://in.tradingview.com/chart/?symbol=NSE%3A{results.iloc[0]['Symbol']})")

if __name__ == "__main__":
    main()
