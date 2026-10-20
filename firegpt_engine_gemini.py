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
DEFAULT_GEMINI_MODELS = [
    os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]
GEMINI_MODEL = DEFAULT_GEMINI_MODELS[0]
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
        f"Traffic condition: {incident.get('traffic_condition', '')}. "
        f"Weather factor: {incident.get('weather_condition', '')}. "
        f"Water supply status: {incident.get('water_supply_status', '')}. "
        f"Fire growth stage: {incident.get('fire_growth_stage', '')}. "
        f"Control actions taken: {incident.get('control_extinguishment', '')} "
        f"Mitigation actions taken: {incident.get('suppression_mitigation', '')} "
        f"Evacuation actions taken: {incident.get('evacuation_rescue', '')} "
        f"Verified Source: {incident.get('primary_source', '')} - {incident.get('source_reference', '')}"
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
Field Realities & Weather/Traffic Impact: Arrival ~{inc.get('arrival_time_min', 'N/A')} min | Road: {inc.get('road_condition', 'N/A')} | Traffic: {inc.get('traffic_condition', 'N/A')} | Weather: {inc.get('weather_condition', 'N/A')} | Water: {inc.get('water_supply_status', 'N/A')}
Cause & Spread: {inc.get('cause', 'N/A')} | {inc.get('description', 'N/A')}
Control & Suppression: {inc.get('control_extinguishment', 'N/A')} | {inc.get('suppression_mitigation', 'N/A')}
Evacuation & Rescue: {inc.get('evacuation_rescue', 'N/A')}
Verified Reference / Source: {inc.get('primary_source', 'Official Inquiry Log')} | Authority: {inc.get('investigating_agency', 'State Fire Directorate')} | Reference: {inc.get('source_reference', 'National Safety Records')}
"""
                )
            context = "\n".join(context_blocks)
            precedent_instructions = f"""Below are RELEVANT PAST INCIDENTS retrieved from verified historical fire records:
{context}

Draw tactical lessons from these matching incidents and cite which past incident informs each recommendation in simple terms. However, adapt all actions strictly to the live scenario."""
        else:
            precedent_instructions = """NOTE ON HISTORICAL PRECEDENTS:
No historical incidents in the database met the relevance threshold for this specific scenario.
Formulate the advisory based strictly on standard firefighting doctrine, the National Building Code of India (NBC 2016 Part 4: Fire & Life Safety), and general incident command procedures."""

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

ENVIRONMENTAL & LOGISTICAL IMPACT ON FIREFIGHTING:
* TRAFFIC EFFECT: If traffic is severe or access roads are congested, mutual-aid water bowsers will be delayed. Order continuous water relay pumping immediately.
* WEATHER EFFECT: If hot (>38°C) or windy (>20 km/h), flames spread faster across windows/facades and embers can jump to nearby buildings; rotate firefighting crews early to prevent heat exhaustion. If raining or humid, heavy smoke stays trapped inside corridors—do not break windows without charged water hoses ready.
* ROAD EFFECT: If narrow lane (< 3m), large fire engines cannot enter. Commit Quick Response Vehicles (QRV) and run 150m+ hose relays from the main road.
* WATER EFFECT: If water is tender-only (lasts only 5-7 minutes of continuous spraying), order continuous water refills before sending crews into interior rooms.
============================================================
"""

        prompt = f"""You are FireGPT, an operational fire safety decision-support assistant for fire officers and firefighters in India.
Your mission is to provide clear, practical, easy-to-understand tactical action steps for firefighters on the ground.

CRITICAL TONE & CLARITY RULE:
- KEEP IT SIMPLE AND EASY TO UNDERSTAND: Avoid overly dense, academic, or theoretical jargon. Firefighters are in high-stress, noisy conditions. Use plain, direct, active language (e.g., instead of "pyrolytic thermal penetration and compartmentation breach", write "Fire is intense; room is at risk of sudden flashover — keep charged hose lines ready").
- Focus on practical, life-saving actions that can be read and executed in seconds.

LIVE / ONGOING INCIDENT REPORT:
\"\"\"{live_incident_text}\"\"\"

{ops_section}

{precedent_instructions}

============================================================
CORE FIREFIGHTER SAFETY RULES:
1. ELECTRICAL HAZARDS: Always shut off electricity before spraying water. Treat battery rooms and solar panels as live even after mains power is off.
2. SMOKE & VENTILATION: Never break windows or open doors until charged water hoses are standing right at the door to prevent a backdraft or flashover.
3. GAS & FUEL HAZARDS: Immediately locate and shut off piped cooking gas (PNG), LPG gas cylinders, and generator fuel lines.
4. FIREFIGHTER SAFETY: Wear complete protective gear with breathing apparatus (SCBA). Keep two clear exit routes open. Watch for falling false ceilings, cracked concrete, and dropping facade panels.
5. EVACUATION PRIORITY: Get people out from the fire room first, then the floor directly above, then the rest of the building.
============================================================

STRUCTURE THE ADVISORY INTO EXACTLY FOUR SECTIONS:
Use checklist bullet points (- [ ]) that the commander can check off one by one.

### 1. FIRE ATTACK & SUPPRESSION (Clear, Step-by-Step Actions)
- [ ] **[WHERE TO ENTER]** Safe entry route (e.g. use exterior fire escape or protected stairwell; NEVER use elevators).
- [ ] **[WHAT TO SPRAY]** Simple agent choice (water spray, foam for oils/chemicals, or CO2/powder for electrical).
- [ ] **[WATER RELAY PLAN]** How to keep water flowing continuously given the reported road and traffic constraints.
- [ ] **[PRACTICAL LESSON]** Direct lesson from matching past incident or standard fire rules.

### 2. HAZARD CONTROL & SPREAD PREVENTION (Clear Safety Steps)
- [ ] **[SHUT OFF POWER & GAS]** Simple shutoff instructions for electricity, gas pipelines, and backup generators.
- [ ] **[SAFE SMOKE CLEARING]** Warning: Do not break windows until water lines are charged and ready.
- [ ] **[STOP SPREAD TO NEIGHBORS]** Spray water curtains on outer walls, cool adjacent buildings, and close fire doors.
- [ ] **[WATCH FOR COLLAPSE]** Warning signs to watch (sagging ceilings, burning cables, cracked walls).

### 3. RESCUE & EVACUATION (Clear Life-Saving Priorities)
- [ ] **[WHO TO RESCUE FIRST]** Priority: fire room first, floor directly above second, then other floors.
- [ ] **[HOW TO MOVE PEOPLE]** Move people sideways into safe smoke-free sections or down stairs with breathing support.
- [ ] **[CRITICAL PATIENTS / TRAPPED PEOPLE]** Simple instructions for moving patients on life-support or elderly/injured persons.
- [ ] **[SAFE GATHERING POINT]** Set up a safe triage area outside and coordinate green corridors with traffic police for ambulances.

### 4. VERIFIED REFERENCES & SOURCES (Mandatory at Bottom)
List the verified historical cases and safety references used to generate this plan:
* **Historical Reference(s):** [Incident name, Date, Location] — Source: [Name of investigation report, Fire Department log, or public inquiry]
* **Doctrine & Standard:** National Building Code of India (NBC 2016 Part 4: Fire & Life Safety) and Standard Fire Operational Manual.

End with:
"⚠️ **Operational Notice:** This checklist is an AI-assisted decision-support tool. Real-time conditions on the ground and the Incident Commander's direct observation always take first priority."
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
