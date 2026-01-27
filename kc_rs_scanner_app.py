import streamlit as st
from kiteconnect import KiteConnect

st.set_page_config(page_title="Kite Callback")

params = st.experimental_get_query_params()

if "request_token" not in params:
    st.error("No request token")
    st.stop()

request_token = params["request_token"][0]

kite = KiteConnect(api_key=st.secrets["KITE_API_KEY"])
data = kite.generate_session(
    request_token,
    api_secret=st.secrets["KITE_API_SECRET"]
)

access_token = data["access_token"]

# Redirect BACK to scanner app
scanner_url = (
    "https://rs-scanner.streamlit.app"
    f"?access_token={access_token}"
)

st.markdown(
    f'<meta http-equiv="refresh" content="0; url={scanner_url}">',
    unsafe_allow_html=True
)
