# =====================================================
# PROMPT BUILDER (STRICT, NO HALLUCINATION)
# =====================================================

def condense_text(text: str, max_sentences=4):
    if not text:
        return ""
    parts = str(text).replace(";", ".").split(".")
    condensed = [p.strip() for p in parts if len(p.strip()) > 15]
    return ". ".join(condensed[:max_sentences])

def extract_keywords(text: str, max_keywords=8):
    if not text:
        return []
    parts = str(text).replace(";", ",").split(",")
    keywords = [p.strip() for p in parts if len(p.strip()) > 3]
    return keywords[:max_keywords]

def build_prompt(intent, plant_name, plant_data):
    uses = condense_text(plant_data.get("Usages", ""))
    actions = ", ".join(extract_keywords(plant_data.get("Actions", "")))

    # IMPORTANT: we intentionally DO NOT include botanical description
    # to prevent geography / species hallucination

    context = (
        f"Plant Name: {plant_name}\n"
        f"Traditional Uses: {uses}\n"
        f"Medicinal Actions: {actions}\n"
    )

    if intent == "symptom_remedy":
        return (
            f"{context}\n"
            f"Instruction:\n"
            f"Rewrite and expand the information above into exactly 4 to 5 sentences "
            f"explaining how {plant_name} is traditionally used to manage fever. "
            f"Use only the words and ideas already present in the context. "
            f"Do NOT mention geography, origin, species, or any information not shown above."
        )

    elif intent == "uses":
        return (
            f"{context}\n"
            f"Instruction:\n"
            f"Rewrite the traditional uses above into exactly 4 to 5 clear sentences. "
            f"Only rephrase the given information without adding new details."
        )

    else:
        return (
            f"{context}\n"
            f"Instruction:\n"
            f"Rewrite the medicinal actions and uses into exactly 4 to 5 factual sentences. "
            f"Do not introduce new botanical or scientific information."
        )
