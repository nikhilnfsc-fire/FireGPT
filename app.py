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
import importlib
from google.oauth2.service_account import Credentials
from groq import Groq

import firegpt_engine_gemini
importlib.reload(firegpt_engine_gemini)
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

    # --- Try Gemini across candidate models ---
    try:
        from firegpt_engine_gemini import DEFAULT_GEMINI_MODELS
        gemini_candidates = DEFAULT_GEMINI_MODELS
    except ImportError:
        gemini_candidates = [
            getattr(firegpt_engine_gemini, "GEMINI_MODEL", "gemini-2.5-flash"),
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]
    for gem_model in gemini_candidates:
        for attempt in range(1, max_gemini_attempts + 1):
            try:
                response = engine.client.models.generate_content(
                    model=gem_model,
                    contents=prompt,
                )
                if response.text and response.text.strip():
                    return response.text, f"Gemini ({gem_model})", None
                else:
                    last_error = f"[{gem_model} attempt {attempt}] Empty response received"
            except Exception as e:
                last_error = f"[{gem_model} attempt {attempt}] {type(e).__name__}: {e}"
                print(last_error)  # goes to Streamlit Cloud logs
                if attempt < max_gemini_attempts:
                    time.sleep(1.0)
                else:
                    # Model failed; advance to next fallback model
                    break

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
MAX_REQUESTS_PER_SESSION = 15         # cap per browser session
MIN_SECONDS_BETWEEN_REQUESTS = 10     # basic cooldown to avoid spamming Gemini

st.set_page_config(
    page_title="FireGPT — Fire Safety Advisory",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Emergency Command Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Command Center Glassmorphism Containers */
    .command-header-card {
        padding: 1.4rem 1.8rem;
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.14) 0%, rgba(17, 24, 39, 0.85) 100%);
        border: 1px solid rgba(255, 75, 75, 0.3);
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }

    .tactical-threat-panel {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .threat-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .threat-red {
        background: rgba(239, 68, 68, 0.25);
        color: #fca5a5;
        border: 1px solid #ef4444;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);
    }
    .threat-orange {
        background: rgba(249, 115, 22, 0.25);
        color: #fdba74;
        border: 1px solid #f97316;
        box-shadow: 0 0 12px rgba(249, 115, 22, 0.4);
    }
    .threat-yellow {
        background: rgba(234, 179, 8, 0.25);
        color: #fde047;
        border: 1px solid #eab308;
        box-shadow: 0 0 12px rgba(234, 179, 8, 0.4);
    }
    .threat-blue {
        background: rgba(59, 130, 246, 0.25);
        color: #93c5fd;
        border: 1px solid #3b82f6;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.4);
    }
    
    .checklist-category-header {
        background: linear-gradient(90deg, rgba(255, 75, 75, 0.2) 0%, rgba(255, 75, 75, 0.02) 100%);
        border-left: 4px solid #ff4b4b;
        padding: 9px 16px;
        margin-top: 1.4rem;
        margin-bottom: 0.75rem;
        font-weight: 700;
        font-size: 1.05rem;
        color: #ffffff;
        border-radius: 0 8px 8px 0;
        letter-spacing: 0.02em;
    }

    .stCheckbox > label {
        font-size: 0.96rem !important;
        line-height: 1.5 !important;
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }

    .quick-stat-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ENGINE (cached so it only loads once, not per request)
# ============================================================
ENGINE_VERSION = "2.3.0"

@st.cache_resource
def load_engine(version=ENGINE_VERSION):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "GEMINI_API_KEY not found in Streamlit Secrets. "
            "Add it under App settings → Secrets before using this app."
        )
        st.stop()
    return firegpt_engine_gemini.FireGPTEngine(api_key=api_key)


engine = load_engine(ENGINE_VERSION)


# ============================================================
# SESSION STATE (per-visitor rate limiting and live states)
# ============================================================
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = None
if "current_advisory" not in st.session_state:
    st.session_state.current_advisory = None
if "current_retrieved" not in st.session_state:
    st.session_state.current_retrieved = []
if "current_model_used" not in st.session_state:
    st.session_state.current_model_used = None
if "current_threat" not in st.session_state:
    st.session_state.current_threat = None
if "incident_text_input" not in st.session_state:
    st.session_state.incident_text_input = ""

# ============================================================
# UI — HEADER
# ============================================================
st.markdown("""
<div style="padding: 1.2rem 1.5rem; background: linear-gradient(135deg, rgba(255, 75, 75, 0.12) 0%, rgba(25, 28, 36, 0.8) 100%); border: 1px solid rgba(255, 75, 75, 0.25); border-radius: 12px; margin-bottom: 1.5rem;">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
        <div>
            <h1 style="margin:0; font-size: 2rem; font-weight:800; color:#ffffff; display:flex; align-items:center; gap:10px;">
                🔥 FireGPT <span style="font-size:0.75rem; padding:3px 8px; background:rgba(255,75,75,0.3); border:1px solid #ff4b4b; border-radius:4px; font-weight:700;">COMMAND V2.2</span>
            </h1>
            <p style="margin:4px 0 0 0; color:#cbd5e1; font-size:0.92rem;">
                Operational fire safety decision-support synthesizing <b>NBC 2016 Part 4</b>, field dispatch constraints, and historical precedent analysis.
            </p>
        </div>
        <div>
            <span style="font-size:0.8rem; background:rgba(16, 185, 129, 0.15); color:#34d399; padding:4px 10px; border-radius:20px; border:1px solid #059669; font-weight:600;">
                ● AI Engines Active (Gemini + Groq Fallback)
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# UI — INPUT & QUICK PRESETS
# ============================================================
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

def on_preset_change():
    chosen = st.session_state.get("selected_preset_key")
    if chosen and chosen != "Custom (Type your own)":
        st.session_state.incident_text_input = SCENARIO_PRESETS[chosen]
    elif chosen == "Custom (Type your own)":
        st.session_state.incident_text_input = ""

input_col, dispatch_col = st.columns([1.1, 0.9], gap="large")

with input_col:
    st.subheader("1. Incident Report")
    
    st.selectbox(
        "⚡ Scenario Preset Template",
        options=list(SCENARIO_PRESETS.keys()),
        index=0,
        key="selected_preset_key",
        on_change=on_preset_change,
        help="Quickly pre-fill realistic high-risk fire scenarios to test tactical checklists."
    )

    live_incident_text = st.text_area(
        "Describe on-scene status, location, burning materials, hazards, and trapped victims:",
        key="incident_text_input",
        height=160,
        placeholder=(
            "e.g. Fire has broken out on the 3rd floor of a hospital in Nagpur. "
            "Suspected electrical short circuit in ICU. Smoke spreading in corridors, "
            "patients on ventilators, oxygen line in ceiling."
        ),
    )
    
    top_k = st.slider("Historical reference cases to correlate", min_value=1, max_value=5, value=3)

with dispatch_col:
    st.subheader("2. Operational Field Realities")
    st.caption("Provide real-time dispatch and on-scene constraints to calculate flashover risk and logistics.")
    
    col1, col2 = st.columns(2)
    with col1:
        arrival_time_val = st.selectbox(
            "⏱️ Elapsed Burn / Arrival Time",
            options=[
                "< 5 mins (Incipient / Early Fire Growth)",
                "5–15 mins (Rapid Growth / High Flashover Risk)",
                "15–30 mins (Fully Developed / Multi-Compartment Flashover)",
                "> 30 mins (Extended Structural Conflagration / Facade Spread)",
            ],
            index=1,
            help="In t-squared fire dynamics, synthetic fuels reach flashover in 4-8 minutes.",
        )
        road_condition_val = st.selectbox(
            "🛣️ Road & Access Route",
            options=[
                "Standard Motorable Municipal Street (3-6m, single tender approach)",
                "Narrow Congested Gully / Bazaar (< 3m, tender entry blocked; QRV/relay required)",
                "Wide Paved Arterial Road (>= 6m per NBC 2016, dual tender & aerial platform)",
                "Potholed / Under Construction / Blocked by parked vehicles & overhead wires",
            ],
            index=0,
        )
        traffic_condition_val = st.selectbox(
            "🚦 Traffic Congestion En Route",
            options=[
                "Moderate Urban Traffic (Minor localized choke points)",
                "Severe Gridlock / Peak-Hour Traffic (Mutual aid & water bowsers delayed)",
                "Clear / Fast Corridor (Green corridor coordinated by traffic police / night transit)",
            ],
            index=0,
        )

    with col2:
        weather_condition_val = st.selectbox(
            "🌤️ Weather & Environmental",
            options=[
                "Normal / Mild Temperature & Calm Winds",
                "High Summer Heat (>38°C) + Dry Air (Accelerates pyrolysis, crew exhaustion)",
                "Strong Gusty Winds (>20 km/h) (Rapid horizontal propagation & ember lofting)",
                "Monsoon Rain / High Humidity (Heavy smoke logging, poor thermal venting)",
            ],
            index=0,
        )
        water_supply_val = st.selectbox(
            "💧 On-Scene Water Supply",
            options=[
                "Tender Onboard Water Only (3000-4500L, lasts ~5-7 min under continuous jetting)",
                "Dedicated On-site Static Fire Tank / Wet Riser Functioning",
                "Municipal Pressurized Street Hydrants Available",
                "Relay Pumping Required from Distant Open Water Source (>200m)",
            ],
            index=0,
        )
        structure_type_val = st.selectbox(
            "🏢 Structure Classification",
            options=[
                "Hospital / Healthcare Facility (ICU & Bedridden Occupants)",
                "High-Rise Building (> 15 meters per NBC 2016 Part 4)",
                "Low-Rise Commercial / Residential (< 15 meters)",
                "Industrial Godown / Warehouse / Chemical Shed (High Fuel Load)",
                "Mixed-Use (Commercial Ground Floor + Residential Upper Floors)",
                "Basement / Underground Enclosure (Severe Heat & Zero Visibility)",
            ],
            index=0,
        )

# Tactical threat level calculation
def calculate_threat_level(elapsed, structure, road, water):
    score = 0
    if "15–30 mins" in elapsed: score += 2
    elif "> 30 mins" in elapsed: score += 3
    elif "5–15 mins" in elapsed: score += 1
    
    if "Hospital" in structure or "Chemical" in structure or "Basement" in structure: score += 2
    if "Narrow Congested" in road or "Blocked" in road: score += 2
    if "Tender Onboard Water Only" in water or "Relay Pumping" in water: score += 2
    
    if score >= 6:
        return "CRITICAL (CODE RED)", "threat-red", "High flashover/collapse probability with acute life-hazard constraints. Prioritize progressive compartment evacuation and immediate uninterrupted water relay."
    elif score >= 4:
        return "HIGH (CODE ORANGE)", "threat-orange", "Elevated structural risk or access choke points. Mandatory charged line backup before any vent actions."
    elif score >= 2:
        return "MODERATE (CODE YELLOW)", "threat-yellow", "Standard high-alert offensive fire attack. Establish 360-degree size-up and monitor utility isolations."
    else:
        return "CONTAINED (CODE BLUE)", "threat-blue", "Incipient stage fire. Standard offensive knock-down with rapid primary search."

submit = st.button("🚨 Generate Incident Command Checkpoints", type="primary", use_container_width=True)

# ============================================================
# RATE LIMITING CHECK
# ============================================================
def rate_limit_ok():
    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.warning(
            f"You've reached the {MAX_REQUESTS_PER_SESSION}-request limit for this session. "
            "Please refresh the page later to continue testing."
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
    current_input = st.session_state.get("incident_text_input", "").strip()
    if not current_input:
        st.error("Please provide details of the incident before submitting.")
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

        # Calculate threat assessment
        threat_label, threat_class, threat_desc = calculate_threat_level(
            arrival_time_val, structure_type_val, road_condition_val, water_supply_val
        )
        st.session_state.current_threat = {
            "label": threat_label,
            "class": threat_class,
            "desc": threat_desc
        }

        # Clean normalized query for TF-IDF to avoid noise
        clean_road = "narrow road" if "< 3m" in road_condition_val else "arterial road"
        clean_water = "onboard tender water" if "Tender Onboard" in water_supply_val else "static water supply"
        retrieval_query = f"{current_input}. {structure_type_val}. {clean_road}. {clean_water}."

        with st.spinner("Analyzing tactical parameters, matching historical precedents, and synthesizing checkpoints..."):
            retrieved = engine.retrieve(retrieval_query, top_k=top_k)
            prompt = engine.build_prompt(current_input, retrieved, ops_data=ops_data)
            advisory_text, model_used, last_error = generate_advisory(prompt)

            if advisory_text is None:
                st.error("Both the primary and backup AI models are currently unavailable. Please retry shortly.")
                with st.expander("Technical details"):
                    st.code(last_error or "No error details captured.")
            else:
                st.session_state.current_advisory = advisory_text
                st.session_state.current_retrieved = retrieved
                st.session_state.current_model_used = model_used


# ============================================================
# DISPLAY ADVISORY & INTERACTIVE CHECKLISTS
# ============================================================
if st.session_state.current_advisory:
    st.divider()
    
    threat = st.session_state.current_threat
    if threat:
        st.markdown(f"""
        <div style="background:rgba(25, 28, 36, 0.95); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:16px; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:10px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:0.85rem; color:#94a3b8; font-weight:600; text-transform:uppercase;">Tactical Threat Assessment:</span>
                    <span class="threat-badge {threat['class']}">{threat['label']}</span>
                </div>
                <div style="color:#94a3b8; font-size:0.85rem;">
                    Engine: <b style="color:#f1f5f9;">{st.session_state.current_model_used}</b>
                </div>
            </div>
            <p style="margin:0; font-size:0.95rem; color:#e2e8f0; line-height:1.5;">{threat['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Reference incidents exp/tabs
    retrieved = st.session_state.current_retrieved
    with st.expander(f"📚 Correlated Historical Precedents & Verified Sources ({len(retrieved)} Records Evaluated)", expanded=False):
        for rank, r in enumerate(retrieved, start=1):
            inc = r["incident"]
            conf = r.get("confidence", "Moderate")
            score = r.get("score", 0.0)
            badge = "🟢 High Match" if conf == "High" else ("🟡 Moderate Match" if conf == "Moderate" else "⚪ Low Match")
            st.markdown(f"**[{rank}] {inc.get('title', 'Historical Incident')}** — {badge} `(Score: {score:.2f})`")
            st.caption(f"📍 {inc.get('location', 'N/A')} | 📅 {inc.get('date', 'N/A')} | 🏢 {inc.get('establishment_type', 'N/A')}")
            st.markdown(f"- **Traffic & Road Access:** {inc.get('traffic_condition', 'N/A')} | {inc.get('road_condition', 'N/A')} (~{inc.get('arrival_time_min', 'N/A')} min arrival)")
            st.markdown(f"- **Weather Factor:** {inc.get('weather_condition', 'N/A')}")
            st.markdown(f"- **Key Suppression Action:** {inc.get('control_extinguishment', 'N/A')}")
            st.markdown(f"- **Rescue & Evacuation:** {inc.get('evacuation_rescue', 'N/A')}")
            st.markdown(f"- **Verified Source / Investigation:** `{inc.get('primary_source', 'Official Inquiry Log')}`")
            st.markdown("---")

    # Render Interactive Checklist with Task Completion Counter
    st.subheader("📋 Operational Incident Command Checklist")
    st.caption("Check off items as they are executed and verified on the ground by dispatch teams.")

    lines = st.session_state.current_advisory.split("\n")
    checklist_items = [l.strip()[5:].strip() for l in lines if l.strip().startswith("- [ ]") or l.strip().startswith("- [x]")]
    total_items = len(checklist_items)
    
    # Count checked items
    completed_items = sum(1 for i in range(total_items) if st.session_state.get(f"tactical_task_{i}", False))
    progress_val = (completed_items / total_items) if total_items > 0 else 0.0

    col_action, col_meta = st.columns([2.5, 1], gap="large")

    with col_action:
        # Progress indicator bar
        st.progress(progress_val, text=f"Field Task Execution: {completed_items} of {total_items} Checkpoints Verified ({int(progress_val*100)}%)")
        
        checkbox_idx = 0
        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith("### "):
                section_title = trimmed.replace("### ", "")
                st.markdown(f'<div class="checklist-category-header">{section_title}</div>', unsafe_allow_html=True)
            elif trimmed.startswith("- [ ]") or trimmed.startswith("- [x]"):
                item_text = trimmed[5:].strip()
                st.checkbox(item_text, key=f"tactical_task_{checkbox_idx}")
                checkbox_idx += 1
            elif trimmed.startswith("⚠️"):
                st.warning(trimmed)
            elif trimmed:
                st.markdown(trimmed)

    with col_meta:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.09); border-radius:12px; padding:18px; margin-bottom:16px;">
            <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Incident Operational Status</div>
            <div style="font-size:1.2rem; font-weight:800; color:{'#34d399' if progress_val == 1.0 else '#f59e0b'}; margin-bottom:8px;">
                {'✅ ALL TASKS COMPLETE' if progress_val == 1.0 else f'{completed_items} / {total_items} Executed'}
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px;">
                <div class="quick-stat-box">
                    <div style="font-size:0.7rem; color:#94a3b8;">Burn Time</div>
                    <div style="font-size:0.85rem; font-weight:700; color:#f8fafc;">{arrival_time_val.split(' ')[0]}</div>
                </div>
                <div class="quick-stat-box">
                    <div style="font-size:0.7rem; color:#94a3b8;">Water Mode</div>
                    <div style="font-size:0.85rem; font-weight:700; color:#f8fafc;">{'Relay' if 'Relay' in water_supply_val else 'Tank/Line'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Export Checklist (.txt)",
            data=st.session_state.current_advisory,
            file_name=f"FireGPT_Checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

        if st.button("🔄 Reset Checklist State", use_container_width=True):
            for i in range(total_items):
                st.session_state[f"tactical_task_{i}"] = False
            st.rerun()

st.divider()

# ============================================================
# FEEDBACK BOX
# ============================================================
st.subheader("Field Feedback & Accuracy Flagging")
st.caption("Testing this with officers or colleagues? Flag anything that seemed inaccurate, missing, or impractical.")

fb_col1, fb_col2 = st.columns([1, 2])
with fb_col1:
    tester_name = st.text_input("Officer / Responder Name", key="tester_name")
with fb_col2:
    feedback = st.text_area("Observations, corrections, or field feasibility remarks", height=70, key="feedback_box")

if st.button("Submit Field Feedback"):
    if not tester_name.strip():
        st.warning("Please enter your name before submitting feedback.")
    elif feedback.strip():
        timestamp = datetime.now().isoformat()
        print(f"[FEEDBACK] {timestamp} :: {tester_name} :: {feedback}")

        if feedback_sheet is not None:
            try:
                feedback_sheet.append_row([timestamp, tester_name, feedback, st.session_state.get("incident_text_input", "")])
                st.success("Feedback logged to central dispatch sheet.")
            except Exception as e:
                st.error(f"Could not save feedback to sheet: {e}")
        else:
            st.info("Feedback recorded locally.")
    else:
        st.warning("Feedback field is empty.")

