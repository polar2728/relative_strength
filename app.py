# app.py
# NSE Relative Strength Leaders Scanner – Streamlit Version (TRADINGVIEW LINKS PERFECT)
# Ram – Hyderabad – Jan 2026
# Updated: Ability to switch between Full NSE and Nifty 50 only

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

st.set_page_config(page_title="NSE RS Leaders Scanner", layout="wide")

# CONFIG ──────────────────────────────────────────────────────────────────────
BENCHMARK          = "^NSEI"
RS_LOOKBACK        = 126
MIN_RS_RANK        = 80
MIN_LIQUIDITY_CR   = 5
MAX_WORKERS        = 6  # not actively used in current version but kept for future

# ─────────────────────────────────────────────────────────────────────────────
# NSE FULL UNIVERSE
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
# NIFTY 50 UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(ttl=86400)
def load_nifty50_symbols():
    url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].dropna().str.strip().tolist()
            symbols = [s for s in symbols if s.isalpha() and len(s) >= 3]
            return [s + '.NS' for s in symbols]
        else:
            st.warning("Nifty 50 CSV format issue - no 'Symbol' column")
            return []
    except Exception as e:
        st.error(f"Failed to load Nifty 50 list: {str(e)[:80]}")
        return []

# ─────────────────────────────────────────────────────────────────────────────
# RS + RSI Filter
# ─────────────────────────────────────────────────────────────────────────────

def rs_filter(symbols, symbol_to_name, rs_lookback, min_rs_rank, min_liq_cr):
    with st.spinner("Computing Relative Strength + RSI..."):
        data = yf.download(
            symbols,
            period="3y",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False
        )

        nifty = yf.download(
            BENCHMARK,
            period="3y",
            auto_adjust=True,
            progress=False
        )

        lookback = min(rs_lookback, len(nifty) - 30)
        nifty_ret = float(nifty["Close"].iloc[-1] / nifty["Close"].iloc[-lookback] - 1) if len(nifty) > lookback else 0

        results = []

        for sym in symbols:
            if sym not in data.columns.levels[0]:
                continue

            try:
                ticker_data = data[sym]
                if ticker_data is None or ticker_data.empty:
                    continue
                    
                df = ticker_data.dropna(subset=['Close'])
                if len(df) < max(lookback + 30, 60):
                    continue

                close = df["Close"]

                # Relative Strength
                current_price = float(close.iloc[-1])
                ret = current_price / float(close.iloc[-lookback]) - 1
                rs = ret / nifty_ret if nifty_ret != 0 else 0

                dma200 = float(close.rolling(200).mean().iloc[-1])
                liq_cr = float((close * df["Volume"]).tail(30).mean() / 1e7)

                if current_price <= dma200 or liq_cr < min_liq_cr:
                    continue

                # RSI calculations
                def compute_rsi(series, period=14):
                    if len(series) < period + 1:
                        return None
                    delta = series.diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=period, min_periods=period).mean()
                    avg_loss = loss.rolling(window=period, min_periods=period).mean()
                    rs_val = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs_val))
                    return round(rsi.iloc[-1], 1)

                rsi_daily = compute_rsi(close, 14)

                weekly_close = close.resample('W-FRI').last().dropna()
                rsi_weekly = compute_rsi(weekly_close, 14) if len(weekly_close) >= 20 else None

                monthly_close = close.resample('ME').last().dropna()
                rsi_monthly = compute_rsi(monthly_close, 14) if len(monthly_close) >= 18 else None

                # Build result
                clean_sym = sym.replace(".NS", "")
                tv_url = f"https://in.tradingview.com/chart/?symbol=NSE%3A{clean_sym}"

                results.append({
                    "Symbol": clean_sym,
                    "Name": symbol_to_name.get(clean_sym, ""),
                    "Price": round(current_price, 2),
                    "RS": round(rs, 3),
                    "LiquidityCr": round(liq_cr, 1),
                    "RSI_Daily": rsi_daily,
                    "RSI_Weekly": rsi_weekly,
                    "RSI_Monthly": rsi_monthly,
                    "TradingView": tv_url
                })

            except Exception:
                continue

        df_results = pd.DataFrame(results)
        if df_results.empty:
            return df_results

        df_results["RS_Rank"] = df_results["RS"].rank(pct=True) * 100
        df_results = df_results[df_results["RS_Rank"] >= min_rs_rank]
        df_results = df_results.sort_values("RS_Rank", ascending=False).reset_index(drop=True)

        return df_results

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("NSE Relative Strength Leaders Scanner")

    with st.sidebar:
        universe_mode = st.radio(
            "Scan Universe",
            options=["Full NSE (~2200+ stocks)", "Nifty 50 only"],
            index=0,
            help="Nifty 50 = faster scans, large-cap focus"
        )

        rs_lookback = st.slider("RS Lookback (days)", 63, 252, RS_LOOKBACK, step=21)
        min_rs_rank = st.slider("Min RS Rank %", 60, 95, MIN_RS_RANK, step=5)
        min_liq_cr  = st.slider("Min Liquidity (Cr)", 1, 20, MIN_LIQUIDITY_CR, step=1)
        run_now     = st.button("Run Scan", type="primary", use_container_width=True)

    # Load universes
    full_symbols, full_name_map = load_nse_universe()
    nifty50_symbols_raw = load_nifty50_symbols()

    if not full_symbols:
        st.stop()

    # Select active universe
    if universe_mode == "Nifty 50 only":
        if not nifty50_symbols_raw:
            st.warning("Nifty 50 list unavailable → using full universe")
            symbols = full_symbols
            name_map_active = full_name_map
            universe_text = "Full NSE (fallback)"
        else:
            symbols = nifty50_symbols_raw
            # Create name map only for Nifty 50 symbols
            nifty_clean = [s.replace('.NS', '') for s in symbols]
            name_map_active = {k: full_name_map.get(k, '') for k in nifty_clean}
            universe_text = f"Nifty 50 ({len(symbols)} stocks)"
    else:
        symbols = full_symbols
        name_map_active = full_name_map
        universe_text = f"Full NSE ({len(symbols):,} stocks)"

    st.caption(f"Universe: {universe_text}")

    if not run_now:
        st.info("Click Run Scan to start")
        return

    # Run the scan
    results = rs_filter(symbols, name_map_active, rs_lookback, min_rs_rank, min_liq_cr)

    if results.empty:
        st.warning(f"No stocks found in current universe")
        return

    st.subheader("Relative Strength Leaders + RSI")

    cols_to_show = ['Symbol', 'Name', 'Price', 'RS', 'LiquidityCr', 'RS_Rank',
                    'RSI_Daily', 'RSI_Weekly', 'RSI_Monthly']

    def color_rsi(val):
        if not pd.api.types.is_number(val) or pd.isna(val):
            return ''
        val = float(val)
        if val >= 60:
            return 'background-color: #ffcccc'
        if val <= 40:
            return 'background-color: #ccffcc'
        return ''

    styled = results[cols_to_show].style.format({
        'Price': lambda x: f"{x:.2f}" if pd.notna(x) else "–",
        'RS': lambda x: f"{x:.3f}" if pd.notna(x) else "–",
        'LiquidityCr': lambda x: f"{x:.1f}" if pd.notna(x) else "–",
        'RS_Rank': lambda x: f"{x:.1f}%" if pd.notna(x) else "–",
        'RSI_Daily': lambda x: f"{x:.1f}" if pd.notna(x) else "–",
        'RSI_Weekly': lambda x: f"{x:.1f}" if pd.notna(x) else "–",
        'RSI_Monthly': lambda x: f"{x:.1f}" if pd.notna(x) else "–",
    }).background_gradient(subset=['RS_Rank'], cmap='YlGn')\
      .map(color_rsi, subset=['RSI_Daily', 'RSI_Weekly', 'RSI_Monthly'])

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # CSV Download
    csv_data = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV (with RSI columns)",
        data=csv_data,
        file_name=f"RS_LEADERS_RSI_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
        use_container_width=True
    )

if __name__ == "__main__":
    main()