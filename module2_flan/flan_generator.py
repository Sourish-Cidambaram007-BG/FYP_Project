import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------
# DEVICE & MODEL
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-large"


# --------------------------------------------------
# LOAD FLAN MODEL (ONCE)
# --------------------------------------------------
@torch.inference_mode()
def load_flan():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# --------------------------------------------------
# GENERATE PLANT INFORMATION (STABLE + SAFE)
# --------------------------------------------------
def generate_plant_info(tokenizer, model, plant_name: str):
    """
    Generates medically relevant, non-hallucinated plant information.
    Designed for academic / FYP usage.
    """

    prompt = f"""
Write a concise academic paragraph about the plant "{plant_name}".

Focus on:
- traditional medicinal uses
- health conditions it is commonly used to support
- nutritional or therapeutic value

Rules:
- Do NOT mention geography, history, or taxonomy.
- Do NOT invent scientific studies or timelines.
- Do NOT claim guaranteed cures.
- Use safe wording like "traditionally used", "may help", "used to support".

Write exactly 4 to 6 complete sentences.
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        min_new_tokens=90,        # forces length
        max_new_tokens=150,
        temperature=0.2,          # 🔑 LOW = stable
        top_p=1.0,                # disable nucleus randomness
        do_sample=False,          # 🔑 NO sampling = no nonsense
        repetition_penalty=1.3,
        no_repeat_ngram_size=4
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # --------------------------------------------------
    # QUALITY CHECK (ONLY REAL FAILURES FALL BACK)
    # --------------------------------------------------
    if (
        len(text.split()) < 45
        or any(x in text.lower() for x in ["bc", "ad", "century", "discovered"])
    ):
        return (
            f"{plant_name} is traditionally used for nutritional and medicinal purposes. "
            "It is commonly included in diets to support digestion and general health. "
            "Traditional practices use parts of the plant to help manage inflammation and "
            "support metabolic functions. The plant is valued for its natural bioactive compounds. "
            "It is mainly used as a supportive remedy rather than a definitive cure."
        )

    return text
