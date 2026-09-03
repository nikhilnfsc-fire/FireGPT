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
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# ============================================================
# 1. CONFIGURATION
# ============================================================
# The key is NEVER hardcoded here. It's read from the environment
# (locally) or from Streamlit's st.secrets (once deployed) — whichever
# is passed in by the caller. See get_api_key() below.
DATASET_PATH = "firegpt_incidents_dataset.json"
TOP_K = 3
GEMINI_MODEL = "gemini-2.5-flash"   # fast + free-tier friendly


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
        f"Establishment: {incident['establishment_type']}. "
        f"Cause: {incident['cause']}. "
        f"Spread pattern: {incident['description']}. "
        f"Control actions taken: {incident['control_extinguishment']} "
        f"Mitigation actions taken: {incident['suppression_mitigation']} "
        f"Evacuation actions taken: {incident['evacuation_rescue']}"
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
        return [
            {"score": float(sims[idx]), "incident": self.incidents[idx]}
            for idx in ranked_idx
        ]

    def build_prompt(self, live_incident_text, retrieved):
        context_blocks = []
        for r in retrieved:
            inc = r["incident"]
            context_blocks.append(
                f"""
--- Past Incident: {inc['title']} ({inc['date']}, {inc['location']}) ---
Establishment: {inc['establishment_type']}
Cause: {inc['cause']}
Spread: {inc['description']}
Control & Extinguishment actions: {inc['control_extinguishment']}
Suppression & Mitigation actions: {inc['suppression_mitigation']}
Evacuation & Rescue actions: {inc['evacuation_rescue']}
"""
            )
        context = "\n".join(context_blocks)

        prompt = f"""You are FireGPT, a fire safety decision-support assistant for fire officers and incident commanders in India. You are NOT replacing the incident commander's judgment — you provide reference-based suggestions only.

A LIVE / ONGOING incident has been reported:
\"\"\"{live_incident_text}\"\"\"

Below are the most similar PAST incidents from a historical database, including what actions were actually taken in each:
{context}

Based on patterns from these past incidents, write a structured advisory with exactly three sections:

1. SUPPRESSION TACTICS — concrete firefighting/extinguishment actions relevant to this situation, citing which past incident(s) the idea is drawn from.
2. MITIGATION STEPS — actions to prevent escalation or secondary hazards (e.g. isolating power, cooling nearby stock, isolating gas lines), citing which past incident(s) informed it.
3. EVACUATION PRIORITIES — order of operations and specific tactics for getting people out safely, citing which past incident(s) informed it.

Keep it practical and specific to the live incident described — do not just copy the past incidents verbatim. End with one line reminding the reader this is decision-support only and the incident commander's on-ground judgment takes precedence.
"""
        return prompt

    def advise(self, live_incident_text, top_k=TOP_K):
        retrieved = self.retrieve(live_incident_text, top_k=top_k)

        print("=" * 90)
        print(f"LIVE INCIDENT:\n  {live_incident_text}")
        print("=" * 90)
        print(f"\nTop {len(retrieved)} similar past incidents used as reference:")
        for rank, r in enumerate(retrieved, start=1):
            inc = r["incident"]
            print(f"  [{rank}] {inc['title']}  (similarity: {r['score']:.3f})")

        prompt = self.build_prompt(live_incident_text, retrieved)

        print("\n[FireGPT] Asking Gemini to synthesize advisory ...\n")
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        print("-" * 90)
        print("FIREGPT ADVISORY")
        print("-" * 90)
        print(response.text)
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
