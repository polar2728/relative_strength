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
# INDICATORS - ENHANCED FOR SWING TRADING
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
    
    # Volume ratio: Today's volume vs 20-day average
    vol_20d_avg = volume.rolling(20).mean().iloc[-1]
    vol_today = volume.iloc[-1]
    vol_ratio = round(vol_today / vol_20d_avg, 2) if vol_20d_avg > 0 else None
    
    # Volume spike: Compare last 5 days avg vs previous 20 days avg
    vol_recent_5d = volume.tail(5).mean()
    vol_prev_20d = volume.iloc[-25:-5].mean() if len(volume) >= 25 else vol_20d_avg
    vol_spike = round(vol_recent_5d / vol_prev_20d, 2) if vol_prev_20d > 0 else None
    
    # Volume trend
    if len(volume) >= 40:
        vol_ma = volume.rolling(20).mean()
        vol_ma_recent = vol_ma.tail(20)
        vol_trend = "📈" if vol_ma_recent.iloc[-1] > vol_ma_recent.iloc[0] else "📉"
    else:
        vol_trend = "➡️"
    
    # Volume breakout
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
# NEW: SWING TRADING SPECIFIC INDICATORS
# ─────────────────────────────────────────────────────────────
def calculate_stage(df):
    """Weinstein Stage Analysis - CRITICAL for swing trading"""
    if df.empty or len(df) < 200:
        return "Unknown"
    
    close = df["Close"]
    price = close.iloc[-1]
    
    # Calculate moving averages
    dma30 = close.rolling(30).mean().iloc[-1]
    dma150 = close.rolling(150).mean().iloc[-1]
    dma200 = close.rolling(200).mean().iloc[-1]
    
    # Check if 150 MA is rising
    dma150_20d_ago = close.rolling(150).mean().iloc[-20] if len(close) >= 170 else dma150
    ma_rising = dma150 > dma150_20d_ago
    
    # Stage 2: Advancing (BEST for swing trading)
    if price > dma30 > dma150 > dma200 and ma_rising:
        return "Stage 2 🚀"
    
    # Stage 1: Basing (potential setup)
    if price > dma200:
        ma_range = abs(dma150 - dma200) / dma200
        if ma_range < 0.03:  # MAs converging
            return "Stage 1 📊"
    
    # Stage 4: Declining (AVOID)
    if price < dma200:
        return "Stage 4 ⚠️"
    
    # Stage 3: Topping
    return "Stage 3 ⚠️"

def calculate_atr_stops(df):
    """ATR-based stop loss and targets"""
    if df.empty or len(df) < 14:
        return {
            'atr': None,
            'stop': None,
            'target': None,
            'risk_pct': None
        }
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    
    # True Range
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    atr_14 = tr.rolling(14).mean().iloc[-1]
    price = close.iloc[-1]
    
    # Stop at 1.5 ATR
    stop = round(price - (1.5 * atr_14), 2)
    
    # Target at 3 ATR (2:1 R/R)
    target = round(price + (3 * atr_14), 2)
    
    # Risk percentage
    risk_pct = round((atr_14 * 1.5 / price) * 100, 2)
    
    return {
        'atr': round(atr_14, 2),
        'stop': stop,
        'target': target,
        'risk_pct': risk_pct
    }

def calculate_pullback_metrics(df):
    """Pullback quality for swing entry timing"""
    if df.empty or len(df) < 20:
        return {
            'pct_from_high': None,
            'pullback_quality': None
        }
    
    close = df["Close"]
    price = close.iloc[-1]
    
    # Recent high (20-day)
    recent_high = close.tail(20).max()
    pct_from_high = round(((price / recent_high) - 1) * 100, 2)
    
    # Pullback quality score
    if -2 >= pct_from_high >= -7:
        quality = "🟢 Ideal"  # Sweet spot
    elif -8 >= pct_from_high >= -12:
        quality = "🟡 Deep"  # Acceptable
    elif pct_from_high > -2:
        quality = "🔴 Extended"  # Too high, wait
    else:
        quality = "⚪ Extreme"  # Too deep
    
    return {
        'pct_from_high': pct_from_high,
        'pullback_quality': quality
    }

def calculate_entry_score(df, rsi_d, vol_ratio, stage, pullback_pct):
    """
    Entry timing score: 0-100
    Higher = better entry setup RIGHT NOW
    """
    score = 0
    
    # 1. Stage (30 points) - Most important
    if stage == "Stage 2 🚀":
        score += 30
    elif stage == "Stage 1 📊":
        score += 15
    
    # 2. RSI sweet spot (25 points)
    if rsi_d:
        if 40 <= rsi_d <= 60:
            score += 25
        elif 35 <= rsi_d < 40 or 60 < rsi_d <= 65:
            score += 15
    
    # 3. Pullback depth (25 points)
    if pullback_pct:
        if -7 <= pullback_pct <= -2:
            score += 25
        elif -12 <= pullback_pct < -7:
            score += 15
    
    # 4. Volume confirmation (20 points)
    if vol_ratio:
        if vol_ratio >= 1.5:
            score += 20
        elif vol_ratio >= 1.2:
            score += 10
    
    return min(score, 100)  # Cap at 100

# ─────────────────────────────────────────────────────────────
# RS SCAN - ENHANCED FOR SWING TRADING
# ─────────────────────────────────────────────────────────────
def rs_scan(kite, symbols, name_map, min_rs, min_liq, benchmark_mode, trading_style):
    """Main RS scanning logic with swing trading enhancements"""
    
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

    # Determine primary RS metric based on trading style
    if trading_style == "Swing (3M Focus)":
        primary_rs = "RS_3M"
        rs_column = "RS_3M"
    else:  # Position or Hybrid
        primary_rs = "RS_6M"
        rs_column = "RS_6M"

    # Scan stocks
    results = []
    filter_stats = {
        "total": len(stock_data),
        "short_history": 0,
        "below_dma200": 0,
        "low_liquidity": 0,
        "stage_4": 0,
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

            # Stage analysis - skip Stage 4
            stage = calculate_stage(df)
            if stage == "Stage 4 ⚠️":
                filter_stats["stage_4"] += 1
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
            
            # Calculate all indicators
            rsi_d = compute_rsi(close)
            weekly_close = resample_to_weekly(close)
            rsi_w = compute_rsi(weekly_close) if weekly_close is not None else None
            monthly_close = resample_to_monthly(close)
            rsi_m = compute_rsi(monthly_close) if monthly_close is not None else None
            
            vol_metrics = calculate_volume_metrics(df)
            atr_metrics = calculate_atr_stops(df)
            pullback_metrics = calculate_pullback_metrics(df)
            
            # Entry score
            entry_score = calculate_entry_score(
                df, 
                rsi_d, 
                vol_metrics['vol_ratio'],
                stage,
                pullback_metrics['pct_from_high']
            )

            clean = sym.replace(".NS", "")
            tv = f"https://tradingview.com/chart/?symbol=NSE%3A{clean}"

            results.append({
                "Symbol": clean,
                "Name": name_map.get(clean, ""),
                "Price": round(price, 2),
                "RS_3M": round(rs3, 3),
                "RS_6M": round(rs6, 3),
                "RS_Delta": round(rs_delta, 3),
                "Stage": stage,
                "Entry_Score": entry_score,
                "RSI_D": rsi_d,
                "RSI_W": rsi_w,
                "RSI_M": rsi_m,
                "Pullback": pullback_metrics['pullback_quality'],
                "Pullback_%": pullback_metrics['pct_from_high'],
                "Vol_Ratio": vol_metrics['vol_ratio'],
                "Vol_Breakout": "🔥" if vol_metrics['vol_breakout'] else "",
                "ATR": atr_metrics['atr'],
                "Stop": atr_metrics['stop'],
                "Target": atr_metrics['target'],
                "Risk_%": atr_metrics['risk_pct'],
                "LiquidityCr": round(liq, 1),
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
    
    # Calculate RS rank based on trading style
    df["RS_Rank"] = df[rs_column].rank(pct=True) * 100
    
    df["Momentum"] = np.where(
        df["RS_Delta"] > 0, "🚀 Accelerating", 
        np.where(df["RS_Delta"] < 0, "📉 Decelerating", "➡️ Stable")
    )
    
    bm_table = pd.DataFrame(benchmark_rows).sort_values("Return_6M", ascending=False) if benchmark_rows else pd.DataFrame()

    # Summary stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Scan Results")
    st.sidebar.metric("Total Scanned", filter_stats["total"])
    st.sidebar.metric("Stage 4 Filtered", filter_stats["stage_4"])
    st.sidebar.metric("Passed All Filters", filter_stats["passed"])
    st.sidebar.metric(f"RS Rank ≥ {min_rs}%", len(df[df["RS_Rank"] >= min_rs]))
    
    # Trading opportunity metrics
    stage2_count = len(df[df["Stage"] == "Stage 2 🚀"])
    high_entry_score = len(df[df["Entry_Score"] >= 70])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Trading Setups")
    st.sidebar.metric("Stage 2 Stocks", stage2_count)
    st.sidebar.metric("Entry Score ≥70", high_entry_score)

    return (
        df[df["RS_Rank"] >= min_rs].sort_values("Entry_Score", ascending=False),
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
        <p style='margin:0;font-size:0.9rem'>Swing Trading Edition • Stage Analysis • Entry Timing</p>
    </div>
    """, unsafe_allow_html=True)

    kite = get_kite_from_secrets()
    if not kite:
        st.error("⚠️ Update KITE_ACCESS_TOKEN in Streamlit Cloud settings")
        st.stop()

    st.sidebar.markdown("### ⚙️ Configuration")
    
    # Trading style selector
    trading_style = st.sidebar.radio(
        "Trading Style",
        ["Hybrid (Recommended)", "Swing (3M Focus)", "Position (6M Focus)"],
        help="Hybrid: Best balance | Swing: More responsive | Position: More stable"
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
            
            # Filter tabs
            tab1, tab2, tab3 = st.tabs(["🎯 Best Setups", "📊 All Results", "🔥 Volume Breakouts"])
            
            with tab1:
                st.markdown("### 🎯 Prime Entry Setups (Entry Score ≥ 70)")
                best_setups = df[df["Entry_Score"] >= 70].sort_values("Entry_Score", ascending=False)
                
                if len(best_setups) > 0:
                    display_results_table(best_setups, min_rs)
                else:
                    st.info("No stocks with Entry Score ≥ 70. Check 'All Results' tab.")
            
            with tab2:
                st.markdown(f"### 📊 All Stocks (RS Rank ≥ {min_rs}%)")
                display_results_table(df, min_rs)
            
            with tab3:
                st.markdown("### 🔥 Volume Breakout Candidates")
                vol_breakouts = df[df["Vol_Breakout"] == "🔥"].sort_values("Entry_Score", ascending=False)
                
                if len(vol_breakouts) > 0:
                    display_results_table(vol_breakouts, min_rs)
                else:
                    st.info("No volume breakouts detected.")

            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "📥 Download Full Results CSV",
                csv,
                f"RS_Leaders_Swing_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Key metrics
            st.markdown("---")
            st.markdown("### 💡 Market Overview")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                stage2 = len(df[df["Stage"] == "Stage 2 🚀"])
                st.metric("Stage 2 Stocks", f"{stage2}/{len(df)}")
            
            with col2:
                accelerating = len(df[df["RS_Delta"] > 0])
                st.metric("Accelerating", f"{accelerating}/{len(df)}")
            
            with col3:
                ideal_pullback = len(df[df["Pullback"] == "🟢 Ideal"])
                st.metric("Ideal Pullback", f"{ideal_pullback}")
            
            with col4:
                high_entry = len(df[df["Entry_Score"] >= 70])
                st.metric("Entry Score ≥70", f"{high_entry}")
            
            with col5:
                avg_risk = df["Risk_%"].mean()
                st.metric("Avg Risk", f"{avg_risk:.1f}%")

        else:
            st.warning("⚠️ No stocks passed filters. Try relaxing criteria.")

    with st.expander("ℹ️ How to Use This Scanner"):
        st.markdown("""
        ## 🎯 Swing Trading Workflow
        
        **1. Filter for Best Setups:**
        - Focus on "Best Setups" tab (Entry Score ≥ 70)
        - Look for Stage 2 🚀 stocks only
        - Prefer 🟢 Ideal or 🟡 Deep pullbacks
        
        **2. Entry Criteria:**
        - Entry Score ≥ 70 (higher = better timing)
        - Stage 2 🚀 only
        - RS Delta > 0 (accelerating momentum)
        - RSI 40-65 range
        - Pullback -2% to -7% from recent high
        
        **3. Risk Management:**
        - Use provided ATR Stop level
        - Position size based on Risk_%
        - Target is 2:1 reward/risk minimum
        
        **4. Indicators Explained:**
        
        **Stage Analysis:**
        - Stage 2 🚀: BEST - uptrend established
        - Stage 1 📊: Basing - potential setup
        - Stage 3/4 ⚠️: AVOID - topping/declining
        
        **Entry Score (0-100):**
        - 80-100: Excellent setup
        - 70-79: Good setup
        - 60-69: Average
        - <60: Wait for better setup
        
        **Pullback Quality:**
        - 🟢 Ideal: -2% to -7% (sweet spot)
        - 🟡 Deep: -8% to -12% (acceptable)
        - 🔴 Extended: Too close to high
        - ⚪ Extreme: Too deep
        
        **Volume:**
        - Vol Ratio >1.5x = Strong interest
        - 🔥 = Breakout signal (2x volume + near high)
        
        **RS Metrics:**
        - RS Rank: 85-100 = Top performers
        - RS Delta > 0: Momentum accelerating
        - RS Delta < 0: Momentum slowing
        
        ## 🎓 Best Practices
        
        1. **Don't chase**: Wait for pullbacks (🟢 or 🟡)
        2. **Confirm stage**: Only trade Stage 2 🚀
        3. **Use stops**: Always use ATR-based stops
        4. **Size properly**: Risk 1-2% per trade based on Risk_%
        5. **Check chart**: Use TradingView link for final confirmation
        
        ## ⚠️ What to Avoid
        
        - Stage 4 stocks (filtered automatically)
        - Extended pullbacks (🔴)
        - RSI > 70 on all timeframes
        - Low Entry Score (<60)
        - Negative RS Delta (decelerating)
        """)

def display_results_table(df, min_rs):
    """Display results table with proper formatting"""
    
    def rsi_color(v):
        if pd.isna(v): return ""
        if v >= 60: return "background-color:#d4edda;color:#155724"
        if v <= 40: return "background-color:#f8d7da;color:#721c24"
        return ""
    
    def entry_score_color(v):
        if pd.isna(v): return ""
        if v >= 80: return "background-color:#d4edda;color:#155724;font-weight:bold"
        if v >= 70: return "background-color:#d1ecf1;color:#0c5460"
        if v >= 60: return "background-color:#fff3cd;color:#856404"
        return ""
    
    def stage_color(v):
        if "Stage 2" in str(v): return "background-color:#d4edda;color:#155724;font-weight:bold"
        if "Stage 1" in str(v): return "background-color:#d1ecf1;color:#0c5460"
        if "Stage 4" in str(v) or "Stage 3" in str(v): return "background-color:#f8d7da;color:#721c24"
        return ""

    styled = df.style.format({
        "Price": "₹{:.2f}",
        "RS_3M": "{:.3f}",
        "RS_6M": "{:.3f}",
        "RS_Delta": "{:+.3f}",
        "Entry_Score": "{:.0f}",
        "RSI_D": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
        "RSI_W": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
        "RSI_M": lambda x: f"{x:.1f}" if pd.notna(x) else "-",
        "Pullback_%": lambda x: f"{x:+.1f}%" if pd.notna(x) else "-",
        "Vol_Ratio": lambda x: f"{x:.2f}x" if pd.notna(x) else "-",
        "ATR": lambda x: f"₹{x:.2f}" if pd.notna(x) else "-",
        "Stop": lambda x: f"₹{x:.2f}" if pd.notna(x) else "-",
        "Target": lambda x: f"₹{x:.2f}" if pd.notna(x) else "-",
        "Risk_%": lambda x: f"{x:.2f}%" if pd.notna(x) else "-",
        "LiquidityCr": "₹{:.1f}Cr",
        "RS_Rank": "{:.1f}%"
    }).background_gradient(
        subset=["RS_Rank"], 
        cmap="RdYlGn",
        vmin=min_rs,
        vmax=100
    ).map(rsi_color, subset=["RSI_D", "RSI_W", "RSI_M"]
    ).map(entry_score_color, subset=["Entry_Score"]
    ).map(stage_color, subset=["Stage"])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Chart": st.column_config.LinkColumn("Chart", display_text="📈 View"),
            "Momentum": st.column_config.TextColumn("Momentum", help="RS 3M vs 6M"),
            "Stage": st.column_config.TextColumn("Stage", help="Weinstein stage"),
            "Entry_Score": st.column_config.NumberColumn("Entry", help="Entry timing score 0-100"),
            "Pullback": st.column_config.TextColumn("Pullback", help="Pullback quality"),
            "Vol_Breakout": st.column_config.TextColumn("🔥", help="Volume breakout"),
            "Stop": st.column_config.NumberColumn("Stop", help="1.5 ATR stop"),
            "Target": st.column_config.NumberColumn("Target", help="3 ATR target"),
        },
        height=600
    )

if __name__ == "__main__":
    main()