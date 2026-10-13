"""
FireGPT — Phase 3: Retrieval + LLM Advisory Generation
--------------------------------------------------------
What changed from Phase 1:
  - Retrieval (TF-IDF over 79 incidents) stays exactly the same.
  - NEW: retrieved incidents are now handed to Gemini (Google's LLM)
    as context, and Gemini writes a proper, coherent advisory in
    plain language — instead of just dumping raw bullet points.

SETUP (one-time):
  1. pip install google-genai scikit-learn
  2. Get a free key from https://aistudio.google.com -> Get API key
  3. Set it as an environment variable (NEVER paste it into this file):
       export GEMINI_API_KEY="your-key-here"     (Mac/Linux)
       setx GEMINI_API_KEY "your-key-here"        (Windows, new terminal after)

  IMPORTANT: Treat the key like a password. If it's ever pasted into
  code, a screenshot, or a chat, revoke it in AI Studio and generate
  a new one immediately.

Run:
    python firegpt_engine_gemini.py
"""

import os
import sys
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# Configure UTF-8 for console output on Windows (handles formulas like CO₂ and emojis)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# 1. CONFIGURATION
# ============================================================
# The key is NEVER hardcoded here. It's read from the environment
# (locally) or from Streamlit's st.secrets (once deployed) — whichever
# is passed in by the caller. See get_api_key() below.
DATASET_PATH = "firegpt_incidents_dataset.json"
TOP_K = 3
GEMINI_MODEL = "gemini-3.8-flash"   # Fast, up-to-date and reliable
SIMILARITY_THRESHOLD_MODERATE = 0.12  # Minimum score to consider an incident moderately relevant
SIMILARITY_THRESHOLD_HIGH = 0.22      # High relevance threshold


def get_api_key():
    """
    Resolve the Gemini API key without ever writing it into source code.
    - Locally: set an environment variable before running, e.g.
        export GEMINI_API_KEY="your-new-key-here"      (Mac/Linux)
        setx GEMINI_API_KEY "your-new-key-here"         (Windows)
    - On Streamlit Cloud: app.py will pass st.secrets["GEMINI_API_KEY"]
      in directly via the api_key= argument to FireGPTEngine(), so this
      function is only the local-development fallback.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Set it as an environment variable "
            "before running locally, or pass it in via api_key= when "
            "creating FireGPTEngine (Streamlit app does this using st.secrets)."
        )
    return key


# ============================================================
# 2. LOAD DATA
# ============================================================
def load_incidents(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_corpus_text(incident):
    return (
        f"Establishment: {incident.get('establishment_type', '')}. "
        f"Cause: {incident.get('cause', '')}. "
        f"Spread pattern: {incident.get('description', '')}. "
        f"Arrival time / Response: {incident.get('arrival_time_min', '')} minutes. "
        f"Road access: {incident.get('road_condition', '')}. "
        f"Traffic: {incident.get('traffic_condition', '')}. "
        f"Weather conditions: {incident.get('weather_condition', '')}. "
        f"Water supply status: {incident.get('water_supply_status', '')}. "
        f"Fire growth stage: {incident.get('fire_growth_stage', '')}. "
        f"Control actions taken: {incident.get('control_extinguishment', '')} "
        f"Mitigation actions taken: {incident.get('suppression_mitigation', '')} "
        f"Evacuation actions taken: {incident.get('evacuation_rescue', '')}"
    )



# ============================================================
# 3. THE ENGINE
# ============================================================
class FireGPTEngine:
    def __init__(self, dataset_path=DATASET_PATH, api_key=None):
        if api_key is None:
            api_key = get_api_key()
        print(f"[FireGPT] Loading dataset from {dataset_path} ...")
        self.incidents = load_incidents(dataset_path)

        print("[FireGPT] Building TF-IDF vectors for all incidents ...")
        self.corpus_texts = [build_corpus_text(inc) for inc in self.incidents]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=5000
        )
        self.doc_vectors = self.vectorizer.fit_transform(self.corpus_texts)
        print(f"[FireGPT] Index ready: {len(self.incidents)} incidents.\n")

        print("[FireGPT] Connecting to Gemini ...")
        self.client = genai.Client(api_key=api_key)
        print("[FireGPT] Ready.\n")

    def retrieve(self, query_text, top_k=TOP_K):
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.doc_vectors)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        
        results = []
        for idx in ranked_idx:
            score = float(sims[idx])
            if score >= SIMILARITY_THRESHOLD_HIGH:
                confidence = "High"
            elif score >= SIMILARITY_THRESHOLD_MODERATE:
                confidence = "Moderate"
            else:
                confidence = "Low"
            
            results.append({
                "score": score,
                "confidence": confidence,
                "is_relevant": score >= SIMILARITY_THRESHOLD_MODERATE,
                "incident": self.incidents[idx],
            })
        return results

    def build_prompt(self, live_incident_text, retrieved, ops_data=None, *args, **kwargs):
        relevant_incidents = [r for r in retrieved if r.get("is_relevant", True)]

        if relevant_incidents:
            context_blocks = []
            for r in relevant_incidents:
                inc = r["incident"]
                context_blocks.append(
                    f"""
--- Past Incident: {inc.get('title', 'Historical Record')} ({inc.get('date', 'N/A')}, {inc.get('location', 'N/A')}) [Match: {r['confidence']}, Score: {r['score']:.2f}] ---
Establishment: {inc.get('establishment_type', 'N/A')}
Field Realities: Arrival ~{inc.get('arrival_time_min', 'N/A')} min | Road: {inc.get('road_condition', 'N/A')} | Weather: {inc.get('weather_condition', 'N/A')} | Water: {inc.get('water_supply_status', 'N/A')}
Cause & Spread: {inc.get('cause', 'N/A')} | {inc.get('description', 'N/A')}
Control & Suppression: {inc.get('control_extinguishment', 'N/A')} | {inc.get('suppression_mitigation', 'N/A')}
Evacuation: {inc.get('evacuation_rescue', 'N/A')}
"""
                )
            context = "\n".join(context_blocks)
            precedent_instructions = f"""Below are RELEVANT PAST INCIDENTS retrieved from historical Indian fire records:
{context}

Draw tactical insights from these matching incidents and cite which past incident informs each recommendation where relevant. However, adapt all actions strictly to the live scenario."""
        else:
            precedent_instructions = """NOTE ON HISTORICAL PRECEDENTS:
No historical incidents in the database met the relevance threshold for this specific scenario.
DO NOT hallucinate or force-fit false historical precedents. Formulate the advisory based strictly on standard firefighting doctrine, the National Building Code of India (NBC 2016 Part 4: Fire & Life Safety), and general incident command procedures."""

        ops_section = ""
        if ops_data:
            ops_section = f"""
============================================================
LIVE OPERATIONAL & ON-GROUND DISPATCH PARAMETERS:
- Time of Fire Dept Arrival (Elapsed Burn Time): {ops_data.get('arrival_time', 'Not specified')}
- Road & Approach Conditions: {ops_data.get('road_condition', 'Not specified')}
- Traffic Gridlock Status: {ops_data.get('traffic_condition', 'Not specified')}
- Weather & Environmental Factor: {ops_data.get('weather_condition', 'Not specified')}
- On-Scene Water Supply / Static Capacity: {ops_data.get('water_supply', 'Not specified')}
- Structure / Occupancy Type: {ops_data.get('structure_type', 'Not specified')}

TACTICAL FACTOR ASSESSMENT:
* If arrival time > 15 mins: Fire has reached or passed compartment flashover (t-squared fire growth model Q = alpha*t^2); anticipate extensive structural thermal penetration and zero visibility in corridors.
* If road access is narrow (< 3m): Standard large fire tenders cannot enter; mandate Quick Response Vehicles (QRV), portable pumps, or high-pressure hose relays (> 150m) from the main road.
* If weather is high ambient heat (>38°C) or windy (>20 km/h): Factor in accelerated horizontal flame leaps, brand lofting, and extreme crew heat stress.
* If water supply is limited to onboard tender tanks (3000-4500L): Continuous jetting lasts only 5-7 minutes. Immediately order continuous relay pumping / water bowsers before committing interior lines.
============================================================
"""

        prompt = f"""You are FireGPT, an operational fire safety decision-support assistant for fire officers and Incident Commanders (ICs) in India.
You provide rapid, high-priority, point-by-point tactical action checklists. You do NOT replace the Incident Commander's on-ground judgment.

LIVE / ONGOING INCIDENT REPORT:
\"\"\"{live_incident_text}\"\"\"

{ops_section}

{precedent_instructions}

============================================================
CRITICAL TACTICAL SAFETY RULES & GUARDRAILS:
1. ELECTRICAL HAZARDS (Class C / Energized):
   - Always confirm electrical isolation before applying water or conductive extinguishing agents.
   - For solar PV arrays or UPS/battery rooms, treat as continuously energized even if mains are tripped.
2. VENTILATION & FLASH OVER / BACKDRAFT:
   - NEVER open doors, windows, or roofs without charged hose lines in place at the entry point.
   - Coordinate horizontal/vertical ventilation with fire attack crews.
3. HAZARDOUS MATERIALS & UTILITIES:
   - Immediately locate and isolate domestic/commercial piped gas lines (PNG), LPG cylinders, diesel storage, and medical oxygen manifolds (especially in hospitals).
4. FIREFIGHTER SAFETY & CREW ACCOUNTABILITY:
   - Mandate Personal Protective Equipment (SCBA + turnout gear). Maintain primary and secondary escape paths.
   - Monitor structural integrity (cracked RCC, sagging steel, false ceilings, falling ACP facade panels).
5. NBC 2016 & LIFE SAFETY:
   - Adhere to the principles of National Building Code of India (NBC 2016 Part 4) regarding compartmentation, stairwell smoke control/pressurization, and horizontal evacuation in institutional/hospital occupancies.
============================================================

CRITICAL FORMAT REQUIREMENT — ACTIONABLE CHECKPOINTS ONLY:
Do NOT output lengthy text paragraphs, academic narratives, or essay-style prose. An Incident Commander under extreme time pressure needs rapid, point-by-point tactical checkpoints (- [ ]) that can be executed and verified sequentially by the command team.

Structure the response into EXACTLY THREE CHECKLIST SECTIONS:

### 1. SUPPRESSION TACTICS (Tactical Checkpoints)
- [ ] **[0-3 MINS / ATTACK ROUTE]** Specify primary attack entry point (e.g. external fire escape vs. pressurized stairwell) and hose deployment.
- [ ] **[EXTINGUISHING AGENT & METHOD]** Exact agent specification (e.g. CO2, DCP powder, high-pressure fog stream, AFFF foam) matched to fuel/electrical hazards.
- [ ] **[WATER CONTINUITY & RELAY]** Specific action for tender turnaround, relay pumping, or static tank utilization given the reported road & water constraints.
- [ ] **[PRECEDENT LESSON]** Concrete tactical action drawn from matching precedent(s) or NBC doctrine.

### 2. MITIGATION & EXPOSURE PROTECTION (Tactical Checkpoints)
- [ ] **[UTILITY ISOLATION]** Exact sequence for electrical breakers, piped gas (PNG), LPG banks, medical O2 zone valves, or rooftop solar arrays.
- [ ] **[SMOKE & VENTILATION CONTROL]** Critical ventilation rule: caution against premature window breaching; coordinate venting with charged lines.
- [ ] **[EXPOSURE PROTECTION]** Actions to stop upward/horizontal spread (cooling facade, boundary water curtains, closing fire doors).
- [ ] **[STRUCTURAL SAFETY WATCH]** Specific failure indicators to monitor (false ceilings, duct collapse, sagging trusses, facade panel drop).

### 3. EVACUATION & RESCUE PRIORITIES (Tactical Checkpoints)
- [ ] **[PRIORITY SEQUENCE]** Priority order: immediate fire compartment -> floor directly above -> general building egress.
- [ ] **[RESCUE METHODOLOGY]** Clear instruction: horizontal progressive relocation to safe smoke compartments (NBC 2016) vs. vertical stairwell descent vs. external aerial ladders.
- [ ] **[SPECIALIZED CARE RESCUE]** Action for high-vulnerability occupants (e.g., ventilator/ICU patients via Ambu-bag paired teams, mobility-impaired persons).
- [ ] **[TRIAGE & CORRIDOR COMMAND]** Safe exterior Casualty Collection Point (CCP) setup and traffic police coordination for ambulance green corridor.

End with:
"⚠️ **Operational Notice:** This checklist is an AI-assisted tactical decision-support aid. The Incident Commander's on-scene assessment and real-time ground conditions take absolute precedence."
"""
        return prompt


    def advise(self, live_incident_text, top_k=TOP_K, ops_data=None, *args, **kwargs):
        retrieved = self.retrieve(live_incident_text, top_k=top_k)

        print("=" * 90)
        print(f"LIVE INCIDENT:\n  {live_incident_text}")
        if ops_data:
            print(f"OPERATIONAL DATA:\n  {ops_data}")
        print("=" * 90)
        print(f"\nTop {len(retrieved)} similar past incidents used as reference:")
        for rank, r in enumerate(retrieved, start=1):
            inc = r["incident"]
            print(f"  [{rank}] {inc['title']}  (match: {r['confidence']}, score: {r['score']:.3f})")

        prompt = self.build_prompt(live_incident_text, retrieved, ops_data=ops_data)

        print("\n[FireGPT] Asking Gemini to synthesize advisory ...\n")
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        print("-" * 90)
        print("FIREGPT TACTICAL CHECKLIST")
        try:
            print(response.text)
        except UnicodeEncodeError:
            print(response.text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
        print("=" * 90 + "\n")

        return response.text



# ============================================================
# 4. RUN IT
# ============================================================
if __name__ == "__main__":
    engine = FireGPTEngine()

    test_query = (
        "Fire has broken out on the third and fourth floor in a Taj Hotel , Nepal late at night "
        "Guest are trapped on the fifth floor "
        "Thick smoke is spreading through corridors "
        "single narrow staircase available for evacuation."
    )
    engine.advise(test_query)
