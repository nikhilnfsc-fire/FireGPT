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
import gspread
from google.oauth2.service_account import Credentials
from groq import Groq

from firegpt_engine_gemini import FireGPTEngine, GEMINI_MODEL

GROQ_MODEL = "openai/gpt-oss-20b"

@st.cache_resource
def get_groq_client():
    groq_key = st.secrets.get("GROQ_API_KEY")
    if not groq_key:
        return None
    return Groq(api_key=groq_key)


groq_client = get_groq_client()


def generate_advisory(prompt, max_gemini_attempts=2):
    """
    Try Gemini first (with fast retries). If it fails, fall back to Groq.
    Returns (advisory_text, model_used, last_error) — last_error is None on success.
    """
    last_error = None

    # --- Try Gemini ---
    for attempt in range(1, max_gemini_attempts + 1):
        try:
            response = engine.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            if response.text and response.text.strip():
                return response.text, "Gemini", None
            else:
                last_error = f"[Gemini attempt {attempt}] Empty response received"
        except Exception as e:
            last_error = f"[Gemini attempt {attempt}] {type(e).__name__}: {e}"
            print(last_error)  # goes to Streamlit Cloud logs
            if attempt < max_gemini_attempts:
                time.sleep(1.5)

    # --- Gemini failed after retries, try Groq ---
    if groq_client is not None:
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            msg = completion.choices[0].message
            content = msg.content if msg.content and msg.content.strip() else getattr(msg, 'reasoning', None)
            if content and content.strip():
                return content, "Groq (fallback)", None
            else:
                last_error = f"[Groq] Empty content returned from {GROQ_MODEL}"
        except Exception as e:
            last_error = f"[Groq] {type(e).__name__}: {e}"
            print(last_error)
    else:
        last_error = (last_error or "") + " | [Groq] client not configured (missing GROQ_API_KEY in secrets)"
        print(last_error)

    return None, None, last_error


# ============================================================
# GOOGLE SHEETS SETUP (for feedback storage)
# ============================================================
FEEDBACK_SHEET_NAME = "FireGPT Feedback"   # must match your Google Sheet's name exactly

@st.cache_resource
def get_feedback_sheet():
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        client = gspread.authorize(creds)
        sheet = client.open(FEEDBACK_SHEET_NAME).sheet1
        return sheet
    except Exception as e:
        st.warning(f"Feedback sheet not connected: {e}")
        return None


feedback_sheet = get_feedback_sheet()

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
# ============================================================
# UI — INPUT & QUICK PRESETS
# ============================================================
st.subheader("Describe the live incident")

SCENARIO_PRESETS = {
    "Custom (Type your own)": "",
    "🏥 Hospital ICU Fire (Oxygen & Life Support Hazard)": (
        "Fire has broken out on the 3rd floor ICU of a 5-storey hospital in Nagpur. "
        "Dense smoke filling corridors; power tripped but backup DG is supplying ventilators. "
        "12 ICU patients on life support; medical oxygen pipeline runs along corridor ceiling. "
        "Single main staircase and an external fire escape available."
    ),
    "🏨 High-Rise Hotel Fire (Trapped Occupants)": (
        "Fire reported on 4th floor banquet hall of a 10-storey hotel in Mumbai at 11:30 PM. "
        "Heavy decorative false ceiling and acoustic panels burning rapidly. "
        "Guests trapped on floors 5 through 8; central AC duct spreading smoke to upper floors. "
        "Two stairwells, one heavily smoke-logged."
    ),
    "🏭 Chemical Godown / Solvent Warehouse Fire": (
        "Blaze in a chemical warehouse in an industrial zone. "
        "Storing mixed solvents, paint drums, and plastic raw materials. "
        "Multiple drum BLEVE / explosion sounds heard; adjacent textile godowns 10 meters away. "
        "Dense toxic fumes blowing towards nearby residential colony."
    ),
    "🏬 Congested Commercial Market Fire": (
        "Major fire on ground floor cloth market in a congested bazaar. "
        "Very narrow congested lanes (width < 3 meters), overhead dangling electrical wires. "
        "Fire spreading vertically to residential floors above shopfronts. "
        "Hydrant pressure low, water tenders facing restricted access."
    ),
}

selected_preset = st.selectbox(
    "Scenario template (select to pre-fill or choose Custom)",
    options=list(SCENARIO_PRESETS.keys()),
    index=0,
)

default_text = SCENARIO_PRESETS[selected_preset] if selected_preset != "Custom (Type your own)" else ""

live_incident_text = st.text_area(
    "Include location/establishment type, cause (if known), what's burning, "
    "current spread, and any trapped occupants or hazards.",
    value=default_text,
    height=140,
    placeholder=(
        "e.g. Fire has broken out on the third floor of a hospital in Nagpur. "
        "Cause suspected to be electrical short circuit in the ICU. "
        "Smoke spreading through corridors, patients on ventilators on the same floor."
    ),
)

# ============================================================
# UI — OPERATIONAL FIELD PARAMETERS (INCIDENT COMMAND INPUTS)
# ============================================================
with st.expander("🚒 Operational & Field Dispatch Parameters (Incident Command Inputs)", expanded=True):
    st.caption("Provide real-time dispatch and on-scene constraints to calculate fire growth dynamics and logistic feasibility.")
    col1, col2 = st.columns(2)
    with col1:
        arrival_time_val = st.selectbox(
            "⏱️ Fire Dept Arrival / Elapsed Burn Time",
            options=[
                "< 5 mins (Incipient / Early Fire Growth)",
                "5–15 mins (Rapid Growth / High Flashover Risk)",
                "15–30 mins (Fully Developed / Multi-Compartment Flashover)",
                "> 30 mins (Extended Structural Conflagration / Facade Spread)",
            ],
            index=1,
            help="SFAC benchmark targets 3-5 mins, but Indian urban traffic frequently extends response to 15-25 mins. In t-squared fire dynamics (Q=α*t^2), modern synthetic furnishings reach flashover within 4-8 minutes.",
        )
        road_condition_val = st.selectbox(
            "🛣️ Road & Access Route Condition",
            options=[
                "Standard Motorable Municipal Street (3-6m, single tender approach)",
                "Narrow Congested Gully / Bazaar (< 3m, tender entry blocked; QRV/relay required)",
                "Wide Paved Arterial Road (>= 6m per NBC 2016, dual tender & aerial platform)",
                "Potholed / Under Construction / Blocked by parked two-wheelers & overhead wires",
            ],
            index=0,
            help="NBC 2016 Part 4 specifies min 6m motorable access around buildings >15m. Narrow lanes mandate long hose line relays (>150m).",
        )
        traffic_condition_val = st.selectbox(
            "🚦 Traffic Congestion Level En Route",
            options=[
                "Moderate Urban Traffic (Minor localized choke points)",
                "Severe Gridlock / Peak-Hour Traffic (Mutual aid & water bowsers delayed)",
                "Clear / Fast Corridor (Green corridor coordinated by traffic police / night transit)",
            ],
            index=0,
        )

    with col2:
        weather_condition_val = st.selectbox(
            "🌤️ Weather & Environmental Conditions",
            options=[
                "Normal / Mild Temperature & Calm Winds",
                "High Summer Heat (>38°C) + Dry Air (Accelerates pyrolysis, extreme crew exhaustion)",
                "Strong Gusty Winds (>20 km/h) (Rapid horizontal propagation & ember/brand lofting)",
                "Monsoon Rain / High Humidity (Heavy smoke logging, poor thermal venting)",
            ],
            index=0,
            help="High ambient heat accelerates ignition while gusty winds push horizontal flame fronts.",
        )
        water_supply_val = st.selectbox(
            "💧 On-Scene Water Supply & Static Capacity",
            options=[
                "Tender Onboard Water Only (3000-4500L, lasts ~5-7 min under continuous jetting)",
                "Dedicated On-site Static Fire Tank / Wet Riser Functioning",
                "Municipal Pressurized Street Hydrants Available",
                "Relay Pumping Required from Distant Open Water Source / Canal (>200m)",
            ],
            index=0,
            help="Water supply continuity is critical before interior attack teams are committed.",
        )
        structure_type_val = st.selectbox(
            "🏢 Structure & Occupancy Classification",
            options=[
                "Low-Rise Commercial / Residential (< 15 meters)",
                "High-Rise Building (> 15 meters per NBC 2016 Part 4)",
                "Hospital / Healthcare Facility (ICU & Bedridden Occupants)",
                "Industrial Godown / Warehouse / Chemical Shed (High Fuel Load)",
                "Mixed-Use (Commercial Ground Floor + Residential Upper Floors)",
                "Basement / Underground Enclosure (Severe Heat & Zero Visibility)",
            ],
            index=0,
        )

top_k = st.slider("Number of similar past incidents to reference", min_value=1, max_value=5, value=3)

submit = st.button("Generate Tactical Checkpoints", type="primary")

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

        ops_data = {
            "arrival_time": arrival_time_val,
            "road_condition": road_condition_val,
            "traffic_condition": traffic_condition_val,
            "weather_condition": weather_condition_val,
            "water_supply": water_supply_val,
            "structure_type": structure_type_val,
        }

        # Build comprehensive retrieval query incorporating field reality constraints
        retrieval_query = (
            f"{live_incident_text}. "
            f"Establishment: {structure_type_val}. "
            f"Road access: {road_condition_val}. "
            f"Traffic: {traffic_condition_val}. "
            f"Weather: {weather_condition_val}. "
            f"Water supply: {water_supply_val}."
        )

        with st.spinner("Analyzing incident parameters, matching historical precedents, and synthesizing checkpoints..."):
            retrieved = engine.retrieve(retrieval_query, top_k=top_k)
            prompt = engine.build_prompt(live_incident_text, retrieved, ops_data=ops_data)

            advisory_text, model_used, last_error = generate_advisory(prompt)

            if advisory_text is None:
                st.error(
                    "Both the primary and backup AI models are currently unavailable. "
                    "Please wait a minute and click 'Generate Tactical Checkpoints' again."
                )
                with st.expander("Technical details (for debugging)"):
                    st.code(last_error or "No error details captured.")
                st.stop()

        st.subheader("Reference incidents used")
        relevant_found = any(r.get("is_relevant", True) for r in retrieved)
        if not relevant_found:
            st.info(
                "ℹ️ **No direct high-confidence historical match found in database for this specific scenario.** "
                "The advisory below is synthesized using **National Building Code of India (NBC 2016 Part 4: Fire & Life Safety)** "
                "protocols, t-squared fire growth dynamics, and standard tactical firefighting doctrine."
            )

        for rank, r in enumerate(retrieved, start=1):
            inc = r["incident"]
            conf = r.get("confidence", "Moderate")
            score = r.get("score", 0.0)
            badge = "🟢 High Match" if conf == "High" else ("🟡 Moderate Match" if conf == "Moderate" else "⚪ Low Match")
            with st.expander(f"[{rank}] {inc.get('title', 'Historical Record')} — {badge} (score: {score:.2f})"):
                st.write(f"**Location:** {inc.get('location', 'N/A')}  |  **Date:** {inc.get('date', 'N/A')}")
                st.write(f"**Establishment:** {inc.get('establishment_type', 'N/A')}")
                st.write(f"**Field Realities:** Arrival: ~{inc.get('arrival_time_min', 'N/A')} min | Road: {inc.get('road_condition', 'N/A')}")
                st.write(f"**Traffic & Weather:** Traffic: {inc.get('traffic_condition', 'N/A')} | Weather: {inc.get('weather_condition', 'N/A')}")
                st.write(f"**Water Supply on Arrival:** {inc.get('water_supply_status', 'N/A')}")
                st.write(f"**Control & Extinguishment:** {inc.get('control_extinguishment', 'N/A')}")
                st.write(f"**Suppression & Mitigation:** {inc.get('suppression_mitigation', 'N/A')}")
                st.write(f"**Evacuation & Rescue:** {inc.get('evacuation_rescue', 'N/A')}")

        st.subheader("📋 Incident Command Tactical Checkpoints")
        if model_used == "Groq (fallback)":
            st.caption("⚡ Generated using backup model (Gemini was unavailable)")
        
        st.markdown(advisory_text)

        st.download_button(
            label="📥 Export Tactical Checkpoints (.txt)",
            data=advisory_text,
            file_name=f"FireGPT_Checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )



st.divider()

# ============================================================
# FEEDBACK BOX (for testers)
# ============================================================
st.subheader("Feedback")
st.caption("Testing this with friends/colleagues? Flag anything that seemed off, missing, or wrong.")
tester_name = st.text_input("Your name", key="tester_name")
feedback = st.text_area("Your feedback", height=80, key="feedback_box")
if st.button("Submit feedback"):
    if not tester_name.strip():
        st.warning("Please enter your name before submitting.")
    elif feedback.strip():
        timestamp = datetime.now().isoformat()
        # Always log locally too, as a backup
        print(f"[FEEDBACK] {timestamp} :: {tester_name} :: {feedback}")

        if feedback_sheet is not None:
            try:
                feedback_sheet.append_row([timestamp, tester_name, feedback, live_incident_text])
                st.success("Thanks! Your feedback has been recorded.")
            except Exception as e:
                st.error(f"Could not save feedback to sheet: {e}")
        else:
            st.warning("Feedback sheet isn't connected — feedback only logged locally for now.")
    else:
        st.warning("Feedback box is empty.")
