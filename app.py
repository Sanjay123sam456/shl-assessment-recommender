"""
SHL Assessment Recommendation System
Streamlit frontend + FastAPI backend running together

Run: streamlit run app.py
(FastAPI starts automatically on port 8000 in background)
"""

import streamlit as st
import streamlit.components.v1 as components
import threading
import os
import time


# ═══════════════════════════════════════════════
#  PAGE CONFIG  (must be FIRST Streamlit call)
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide"
)


# ═══════════════════════════════════════════════
#  START FASTAPI IN BACKGROUND THREAD
# ═══════════════════════════════════════════════

def start_api():
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="error")

# Only start once — not on every Streamlit rerun
if "api_started" not in st.session_state:
    t = threading.Thread(target=start_api, daemon=True)
    t.start()
    st.session_state.api_started = True
    time.sleep(1.5)  # Give FastAPI time to initialize


# ═══════════════════════════════════════════════
#  HIDE STREAMLIT CHROME
# ═══════════════════════════════════════════════

st.markdown("""
<style>
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    [data-testid="stAppViewContainer"] { padding: 0 !important; }
    [data-testid="stVerticalBlock"] { gap: 0 !important; padding: 0 !important; }
    section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  LOAD & RENDER SHL HTML FRONTEND
# ═══════════════════════════════════════════════

frontend_path = os.path.join(os.path.dirname(__file__), "frontend.html")

if os.path.exists(frontend_path):
    with open(frontend_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Inject API base URL before </head>
    api_url = os.getenv("API_URL", "http://localhost:8000")
    html = html.replace(
        "</head>",
        f"<script>window.API_BASE = '{api_url}';</script></head>",
        1
    )

    components.html(html, height=5000, scrolling=True)

else:
    st.error("❌ frontend.html not found!")
    st.info("Make sure frontend.html is in the same folder as app.py")
