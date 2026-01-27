import streamlit as st
from kiteconnect import KiteConnect

st.set_page_config(page_title="Kite Callback", layout="centered")

params = st.experimental_get_query_params()
request_token = params.get("request_token", [None])[0]

if not request_token:
    st.error("No request token")
    st.stop()

kite = KiteConnect(api_key=st.secrets["KITE_API_KEY"])
data = kite.generate_session(
    request_token,
    st.secrets["KITE_API_SECRET"]
)

access_token = data["access_token"]

st.markdown(
    f"""
    <script>
        window.location.replace(
            "https://rs-scanner.streamlit.app/?kite_token={access_token}"
        );
    </script>
    """,
    unsafe_allow_html=True
)
