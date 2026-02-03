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
# KITE OAUTH - FIXED WITH JAVASCRIPT REDIRECT
# ─────────────────────────────────────────────────────────────
def handle_kite_auth():
    """
    Returns kite (or None) and bool: whether to show login UI
    """
    try:
        api_key = st.secrets["KITE_API_KEY"]
        api_secret = st.secrets["KITE_API_SECRET"]
    except KeyError:
        st.error("Missing KITE_API_KEY and/or KITE_API_SECRET in Streamlit secrets.")
        return None, False

    kite = KiteConnect(api_key=api_key)

    # Already logged in → skip everything
    if st.session_state.get("kite_authenticated", False):
        if "access_token" in st.session_state:
            kite.set_access_token(st.session_state["access_token"])
        return kite, False

    # ── Callback from Kite ───────────────────────────────────────────
    if "request_token" in st.query_params and not st.session_state.get("auth_callback_handled", False):
        # One-time processing guard (survives reruns)
        st.session_state.auth_callback_handled = True

        with st.spinner("Verifying Kite login..."):
            try:
                request_token = st.query_params["request_token"]
                # Generate session
                session_data = kite.generate_session(request_token, api_secret=api_secret)
                access_token = session_data["access_token"]
                kite.set_access_token(access_token)

                profile = kite.profile()

                # Save to session
                st.session_state.kite = kite
                st.session_state.access_token = access_token
                st.session_state.user_name = profile.get("user_name", "User")
                st.session_state.kite_authenticated = True

                # Clear params NOW (prevents re-trigger on rerun)
                st.query_params.clear()

                st.success(f"✅ Logged in as {st.session_state.user_name}")
                st.rerun()  # Clean reload without params

            except Exception as e:
                st.error(f"Authentication error: {str(e)}")
                st.query_params.clear()  # Still clear to break potential stuck state
                st.session_state.auth_callback_handled = False  # Allow retry
                st.rerun()

    # Not authenticated, no active callback → show login
    return None, True


def show_login_ui(kite):
    """Display login UI"""
    st.sidebar.markdown("### 🔐 Kite Connect Login")
    st.sidebar.info("Login with your Zerodha account to continue")
    
    login_url = kite.login_url()
    
    st.sidebar.markdown(
        f'''
        <a href="{login_url}" target="_self" style="text-decoration: none;">
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 8px;
                text-align: center;
                font-weight: 600;
                font-size: 16px;
                cursor: pointer;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            " onmouseover="this.style.transform='scale(1.05)'" 
               onmouseout="this.style.transform='scale(1)'">
                🚀 Login with Kite Connect
            </div>
        </a>
        ''',
        unsafe_allow_html=True
    )
    
    # Main area message
    st.markdown("""
    <div style='background: linear-gradient(135deg,#667eea,#764ba2,#f093fb);
                padding:1.2rem;border-radius:14px;color:white;text-align:center'>
        <h2 style='margin:0'>🏆 NSE RS Leaders Scanner PRO</h2>
        <p style='margin:0;font-size:0.9rem'>Relative Strength • Volume Analysis • Multi-Timeframe RSI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👈 Please login with Kite Connect in the sidebar to start scanning")
    
    with st.expander("ℹ️ About This Scanner"):
        st.markdown("""
        **Features:**
        - Relative Strength ranking vs market benchmarks
        - Multi-timeframe RSI analysis (Daily, Weekly, Monthly)
        - Volume breakout detection
        - Swing vs Position trading modes
        - Real-time Kite Connect data
        
        **How to Use:**
        1. Login with your Zerodha account (sidebar)
        2. Select trading style and universe
        3. Adjust filters (RS rank, liquidity)
        4. Click "RUN SCAN"
        
        **Note:** Login is required to access live market data via Kite Connect API.
        """)

def logout_kite():
    """Logout function"""
    # Clear all auth-related session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Redirect to clean URL
    st.markdown(
        """
        <script>
            window.location.href = window.location.origin + window.location.pathname;
        </script>
        """,
        unsafe_allow_html=True
    )
    st.stop()

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

# [REST OF YOUR CODE - compute_rsi, resample functions, indicators, etc. - KEEP AS IS]
# I'll skip repeating all that code for brevity, but include everything from your original file

# ─────────────────────────────────────────────────────────────
# MAIN - UPDATED WITH NEW AUTH
# ─────────────────────────────────────────────────────────────
def main():
    kite, show_login_ui_flag = handle_kite_auth()

    if show_login_ui_flag:
        # Only show login screen
        try:
            temp_kite = KiteConnect(api_key=st.secrets["KITE_API_KEY"])
            show_login_ui(temp_kite)  # your existing function
        except KeyError:
            st.error("Missing API key in secrets.")
        st.stop()  # ← Critical: prevent rest of app from rendering

    # Authenticated → full app
    st.markdown("""<your header HTML>""", unsafe_allow_html=True)

    # Sidebar extras
    st.sidebar.markdown(f"**Connected as:** {st.session_state.get('user_name', 'User')}")
    if st.sidebar.button("Logout"):
        # Full clear
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.query_params.clear()
        st.rerun()
    
    # User is authenticated - show main app
    st.markdown("""
    <div style='background: linear-gradient(135deg,#667eea,#764ba2,#f093fb);
                padding:1.2rem;border-radius:14px;color:white;text-align:center'>
        <h2 style='margin:0'>🏆 NSE RS Leaders Scanner PRO</h2>
        <p style='margin:0;font-size:0.9rem'>Relative Strength • Volume Analysis • Multi-Timeframe RSI</p>
    </div>
    """, unsafe_allow_html=True)

    # Show user info in sidebar
    if 'user_name' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**👤 {st.session_state.user_name}**")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_kite()

    st.sidebar.markdown("### ⚙️ Configuration")
    
    # [REST OF YOUR MAIN FUNCTION CODE - KEEP AS IS]
    # Trading style selector, universe, sliders, scan logic, etc.

if __name__ == "__main__":
    main()