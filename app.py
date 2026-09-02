"""
FireGPT — Streamlit Web App (Phase 6: Deployment)
--------------------------------------------------
This is the public-facing UI. It wraps the FireGPTEngine
(retrieval + Gemini advisory) from firegpt_engine_gemini.py.

DEPLOY NOTE:
  On Streamlit Community Cloud, set your Gemini key under
  "App settings -> Secrets" like this:

      GEMINI_API_KEY = "your-new-key-here"

  Never put the key directly in this file or commit it to GitHub.
"""

import time
from datetime import datetime, timedelta

import streamlit as st

from firegpt_engine_gemini import FireGPTEngine, GEMINI_MODEL

# ============================================================
# CONFIG
# ============================================================
MAX_REQUESTS_PER_SESSION = 8          # cap per browser session
MIN_SECONDS_BETWEEN_REQUESTS = 15     # basic cooldown to avoid spamming Gemini

st.set_page_config(page_title="FireGPT — Fire Safety Advisory", page_icon="🔥", layout="centered")

# ============================================================
# LOAD ENGINE (cached so it only loads once, not per request)
# ============================================================
@st.cache_resource
def load_engine():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "GEMINI_API_KEY not found in Streamlit Secrets. "
            "Add it under App settings → Secrets before using this app."
        )
        st.stop()
    return FireGPTEngine(api_key=api_key)


engine = load_engine()

# ============================================================
# SESSION STATE (per-visitor rate limiting)
# ============================================================
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = None

# ============================================================
# UI — HEADER
# ============================================================
st.title("🔥 FireGPT")
st.caption(
    "Fire safety decision-support tool for incident commanders. "
    "This is a **beta/testing build** — outputs are reference suggestions only, "
    "not a substitute for on-ground judgment."
)

st.divider()

# ============================================================
# UI — INPUT
# ============================================================
st.subheader("Describe the live incident")
live_incident_text = st.text_area(
    "Include location/establishment type, cause (if known), what's burning, "
    "current spread, and any trapped occupants or hazards.",
    height=140,
    placeholder=(
        "e.g. Fire has broken out on the third floor of a hospital in Nagpur. "
        "Cause suspected to be electrical short circuit in the ICU. "
        "Smoke spreading through corridors, patients on ventilators on the same floor."
    ),
)

top_k = st.slider("Number of similar past incidents to reference", min_value=1, max_value=5, value=3)

submit = st.button("Get Advisory", type="primary")

# ============================================================
# RATE LIMITING CHECK
# ============================================================
def rate_limit_ok():
    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.warning(
            f"You've reached the {MAX_REQUESTS_PER_SESSION}-request limit for this session. "
            "Please refresh the page later or come back another time — this keeps the "
            "free-tier API usable for everyone testing it."
        )
        return False

    if st.session_state.last_request_time is not None:
        elapsed = (datetime.now() - st.session_state.last_request_time).total_seconds()
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            wait = int(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
            st.info(f"Please wait {wait}s before submitting another request.")
            return False

    return True


# ============================================================
# HANDLE SUBMISSION
# ============================================================
if submit:
    if not live_incident_text.strip():
        st.error("Please describe the incident before submitting.")
    elif rate_limit_ok():
        st.session_state.request_count += 1
        st.session_state.last_request_time = datetime.now()

        with st.spinner("Retrieving similar past incidents and generating advisory..."):
            retrieved = engine.retrieve(live_incident_text, top_k=top_k)
            prompt = engine.build_prompt(live_incident_text, retrieved)
            response = engine.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            advisory_text = response.text

        st.subheader("Reference incidents used")
        for rank, r in enumerate(retrieved, start=1):
            inc = r["incident"]
            with st.expander(f"[{rank}] {inc['title']} — similarity {r['score']:.2f}"):
                st.write(f"**Location:** {inc.get('location', 'N/A')}  |  **Date:** {inc.get('date', 'N/A')}")
                st.write(f"**Establishment:** {inc.get('establishment_type', 'N/A')}")
                st.write(f"**Cause:** {inc.get('cause', 'N/A')}")
                st.write(f"**Control & Extinguishment:** {inc.get('control_extinguishment', 'N/A')}")
                st.write(f"**Suppression & Mitigation:** {inc.get('suppression_mitigation', 'N/A')}")
                st.write(f"**Evacuation & Rescue:** {inc.get('evacuation_rescue', 'N/A')}")

        st.subheader("FireGPT Advisory")
        st.markdown(advisory_text)

st.divider()

# ============================================================
# FEEDBACK BOX (for testers)
# ============================================================
st.subheader("Feedback")
st.caption("Testing this with friends/colleagues? Flag anything that seemed off, missing, or wrong.")
feedback = st.text_area("Your feedback", height=80, key="feedback_box")
if st.button("Submit feedback"):
    if feedback.strip():
        # For now this just logs to the Streamlit Cloud app logs.
        # Once you're ready, swap this for writing to a Google Sheet,
        # Airtable, or a simple database so feedback isn't lost on restart.
        print(f"[FEEDBACK] {datetime.now().isoformat()} :: {feedback}")
        st.success("Thanks! Your feedback has been recorded.")
    else:
        st.warning("Feedback box is empty.")
