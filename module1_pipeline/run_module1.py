import json
import pandas as pd
import torch

# =====================================================
# CUSTOM MODULE IMPORTS
# =====================================================
from module1_text.preprocess import clean_text
from module1_text.intent_detector import detect_intent_semantic
from module1_text.plant_detector import detect_plant_semantic
from module3_semantic.embedder import embed
from module3_semantic.search import semantic_search

# =====================================================
# LOAD DATASET (GLOBAL, CPU ONLY)
# =====================================================
DATASET_PATH = "data/🍀Dataset - FYP.xlsx"

df = pd.read_excel(DATASET_PATH, engine="openpyxl").fillna("")

# =====================================================
# UTILITY
# =====================================================
def remove_empty_fields(data: dict) -> dict:
    """Remove empty / null values from dict"""
    return {
        k: v for k, v in data.items()
        if v not in ("", None) and str(v).strip() != ""
    }

# =====================================================
# MODULE-1 PIPELINE
# =====================================================
def run_module1(
    translated_query: str,
    original_query: str = "",
    detected_lang: str = "en"
) -> dict:
    """
    Optimized Module-1 pipeline.

    🔹 Translation is handled outside (HybridTranslator)
    🔹 No model loading here
    🔹 Safe for CUDA + Streamlit
    """

    # -----------------------------
    # 1️⃣ Preprocess
    # -----------------------------
    cleaned_query = clean_text(translated_query)

    # -----------------------------
    # 2️⃣ Intent Detection
    # -----------------------------
    intent, intent_confidence = detect_intent_semantic(cleaned_query)

    # -----------------------------
    # 3️⃣ Plant Detection
    # -----------------------------
    try:
        # Symptom-based queries
        if intent == "symptom_remedy":
            symptom_query = f"medicinal plant remedy for {cleaned_query}"

            # Embed query (keep device consistent)
            query_embedding = embed([symptom_query])

            # Ensure embedding is on CPU for FAISS / similarity search
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.detach().cpu()

            plant_id = semantic_search(query_embedding)
            plant_name = df.iloc[plant_id]["English Name"]

        # Direct plant queries
        else:
            plant_id, plant_name = detect_plant_semantic(cleaned_query)

    except Exception as e:
        # Fallback safety
        plant_id = 0
        plant_name = df.iloc[0]["English Name"]
        print("⚠️ Plant detection fallback:", e)

    # -----------------------------
    # 4️⃣ Fetch Plant Data
    # -----------------------------
    plant_data = remove_empty_fields(
        df.iloc[int(plant_id)].to_dict()
    )

    # -----------------------------
    # 5️⃣ Build Output JSON
    # -----------------------------
    output = {
        "original_query": original_query,
        "detected_language": detected_lang,
        "translated_query": translated_query,
        "cleaned_query": cleaned_query,
        "intent": intent,
        "intent_confidence": round(float(intent_confidence), 4),
        "plant_name": plant_name,
        "plant_data": plant_data
    }

    # Add poem only for symptom remedies
    if intent == "symptom_remedy":
        output["poem"] = plant_data.get("Tamil poem", "")
        output["poet"] = plant_data.get("Poet", "")

    return output

# =====================================================
# CLI TEST (SAFE)
# =====================================================
if __name__ == "__main__":
    q = input("Enter query (English): ")
    result = run_module1(
        translated_query=q,
        original_query=q,
        detected_lang="en"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
