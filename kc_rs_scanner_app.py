import streamlit as st
from kiteconnect import KiteConnect
import uuid

st.set_page_config(page_title="Kite Callback")

params = st.experimental_get_query_params()

if "request_token" not in params:
    st.error("No request token")
    st.stop()

kite = KiteConnect(api_key=st.secrets["KITE_API_KEY"])
data = kite.generate_session(
    params["request_token"][0],
    api_secret=st.secrets["KITE_API_SECRET"]
)

access_token = data["access_token"]

# 🔑 Store token server-side (temporary)
if "TOKEN_STORE" not in st.session_state:
    st.session_state.TOKEN_STORE = {}

code = str(uuid.uuid4())
st.session_state.TOKEN_STORE[code] = access_token

# Redirect WITHOUT token
scanner_url = f"https://rs-scanner.streamlit.app/?code={code}"

st.markdown(
    f'<meta http-equiv="refresh" content="0; url={scanner_url}">',
    unsafe_allow_html=True
)
