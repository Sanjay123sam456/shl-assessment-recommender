"""
SHL Assessment Recommendation System
Streamlit frontend + FastAPI backend running together

Run: streamlit run app.py
(FastAPI starts automatically on port 8000 in background)
"""

import os
import threading
import time

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="??",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def api_is_live() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def start_api() -> None:
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="error")


if "api_started" not in st.session_state:
    if not api_is_live():
        t = threading.Thread(target=start_api, daemon=True)
        t.start()
    st.session_state.api_started = True

    # Wait briefly for API startup to avoid first-click failures.
    for _ in range(20):
        if api_is_live():
            break
        time.sleep(0.25)


st.markdown(
    """
<style>
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    [data-testid="stAppViewContainer"] { padding: 0 !important; }
    [data-testid="stVerticalBlock"] { gap: 0 !important; padding: 0 !important; }
    section[data-testid="stSidebar"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

frontend_path = os.path.join(os.path.dirname(__file__), "frontend.html")

if os.path.exists(frontend_path):
    with open(frontend_path, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace(
        "</head>",
        f"<script>window.API_BASE = '{API_URL}';</script></head>",
        1,
    )

    components.html(html, height=5000, scrolling=True)
else:
    st.error("frontend.html not found!")
    st.info("Make sure frontend.html is in the same folder as app.py")
