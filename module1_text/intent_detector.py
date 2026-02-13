import torch
from sentence_transformers import util
from module3_semantic.embedder import embed

# ============================================================
# Intent Templates (Semantic Descriptions)
# ============================================================

INTENT_TEMPLATES = {
    "tamil_name": [
        "What is the Tamil name of the plant?",
        "Plant name in Tamil"
    ],
    "telugu_name": [
        "What is the Telugu name of the plant?",
        "Plant name in Telugu"
    ],
    "malayalam_name": [
        "What is the Malayalam name of the plant?",
        "Plant name in Malayalam"
    ],
    "hindi_name": [
        "What is the Hindi name of the plant?",
        "Plant name in Hindi"
    ],
    "sanskrit_name": [
        "What is the Sanskrit name of the plant?",
        "Plant name in Sanskrit"
    ],
    "english_name": [
        "What is the English name of the plant?"
    ],
    "botanical_name": [
        "What is the botanical name of the plant?",
        "Scientific name of the plant"
    ],
    "uses": [
        "What are the uses of the plant?",
        "Medicinal uses of the plant",
        "Health benefits of the plant"
    ],
    "general": [
        "Tell me about the plant",
        "Information about the plant",
        "Describe the plant"
    ],
    "symptom_remedy": [
        "I have a fever give me a remedy",
        "Remedy for fever",
        "How to cure fever naturally",
        "I am suffering from fever",
        "I have fever what should I do",
           "I have a fever give me a remedy",
        "Remedy for cold and cough",
        "Natural treatment for headache",
        "How to reduce body pain",
        "Home remedy for stomach problems"
    ]
}

# ============================================================
# Precompute Intent Embeddings
# ============================================================

INTENT_EMBEDDINGS = {
    intent: embed(sentences).cpu()
    for intent, sentences in INTENT_TEMPLATES.items()
}

CONFIDENCE_THRESHOLD = 0.55


# ============================================================
# Semantic Intent Detection (FINAL)
# ============================================================

def detect_intent_semantic(query: str):
    """
    Detect user intent using:
    1. Symptom priority detection
    2. Name-query normalization
    3. Semantic similarity fallback
    """

    q = query.lower()

    # --------------------------------------------------------
    # 1️⃣ SYMPTOM → REMEDY (HIGHEST PRIORITY)
    # --------------------------------------------------------
    SYMPTOM_KEYWORDS = [
        "fever", "cold", "cough",
        "headache", "pain",
        "infection", "illness",
        "suffering",
        "asthma",
        "diabetes",
        "stomach",
        "ulcer",
        "wound",
        "inflammation"
    ]


    if any(word in q for word in SYMPTOM_KEYWORDS):
        return "symptom_remedy", 0.95

    # --------------------------------------------------------
    # 2️⃣ NAME LOOKUP (LANGUAGE / BOTANICAL)
    # --------------------------------------------------------
    NAME_KEYWORDS = {
        "tamil": "tamil_name",
        "telugu": "telugu_name",
        "malayalam": "malayalam_name",
        "hindi": "hindi_name",
        "sanskrit": "sanskrit_name",
        "botanical": "botanical_name",
        "scientific": "botanical_name",
        "english name": "english_name"
    }

    for key, intent in NAME_KEYWORDS.items():
        if key in q:
            return intent, 0.9

    # --------------------------------------------------------
    # 3️⃣ USES / BENEFITS
    # --------------------------------------------------------
    USES_KEYWORDS = ["use", "uses", "usage", "benefit", "benefits", "purpose"]

    if any(word in q for word in USES_KEYWORDS):
        return "uses", 0.85

    # --------------------------------------------------------
    # 4️⃣ SEMANTIC MATCHING (FALLBACK)
    # --------------------------------------------------------
    query_embedding = embed([query]).cpu()

    best_intent = "unknown"
    best_score = -1.0

    for intent, intent_emb in INTENT_EMBEDDINGS.items():
        score = util.cos_sim(query_embedding, intent_emb).max().item()

        if score > best_score:
            best_score = score
            best_intent = intent

    if best_score < CONFIDENCE_THRESHOLD:
        return "unknown", round(best_score, 3)

    return best_intent, round(best_score, 3)
