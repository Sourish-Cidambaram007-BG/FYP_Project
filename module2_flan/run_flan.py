import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =====================================================
# GLOBALS (LAZY LOAD – LOAD ONCE)
# =====================================================
_flan_model = None
_flan_tokenizer = None

# =====================================================
# LOAD FLAN RESOURCES (FINAL & STABLE)
# =====================================================
def get_flan_resources():
    global _flan_model, _flan_tokenizer

    model_id = "google/flan-t5-large"
    # If VRAM is tight:
    # model_id = "google/flan-t5-base"

    if _flan_model is None:
        print("🚀 Loading FLAN-T5 (FP16, GPU-safe)")

        _flan_tokenizer = AutoTokenizer.from_pretrained(model_id)

        _flan_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        _flan_model.eval()

    return _flan_model, _flan_tokenizer

# =====================================================
# INTERNAL: SENTENCE FILTER (ANTI-HALLUCINATION)
# =====================================================
def filter_sentences(text: str, plant_data: dict, min_sent=4, max_sent=5):
    """
    Keeps only sentences that overlap with dataset vocabulary
    """
    if not text:
        return text

    allowed_text = (
        f"{plant_data.get('Usages','')} "
        f"{plant_data.get('Actions','')}"
    ).lower()

    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]

    grounded = [
        s for s in sentences
        if sum(1 for w in s.lower().split() if w in allowed_text) >= 2
    ]

    final = grounded[:max_sent]

    if len(final) < min_sent:
        final = sentences[:max_sent]

    return ". ".join(final) + "."

# =====================================================
# FLAN INFERENCE
# =====================================================
def run_flan(module1_output: dict):
    intent = module1_output["intent"]
    plant_name = module1_output["plant_name"]
    plant_data = module1_output["plant_data"]

    # -------------------------------------------------
    # DIRECT LOOKUPS (NO LLM)
    # -------------------------------------------------
    lookups = {
        "tamil_name": "Tamil Name",
        "telugu_name": "Telugu Name",
        "malayalam_name": "Malayalam Name",
        "hindi_name": "Hindi Name",
        "sanskrit_name": "Sanskrit Name"
    }

    if intent in lookups:
        value = plant_data.get(lookups[intent], "not available")
        return f"The {intent.replace('_', ' ')} of {plant_name} is {value}."

    if intent == "botanical_name":
        botanical = plant_data.get("Botanical name \n", "not available")
        return f"The botanical name of {plant_name} is {botanical}."

    # -------------------------------------------------
    # AI GENERATION
    # -------------------------------------------------
    from module2_flan.generate_answer import build_prompt

    model, tokenizer = get_flan_resources()
    prompt = build_prompt(intent, plant_name, plant_data)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            min_length=120,
            num_beams=4,
            temperature=0.75,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    raw_text = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )

    # -------------------------------------------------
    # POST-FILTER (GROUNDING + 4–5 SENTENCES)
    # -------------------------------------------------
    final_text = filter_sentences(raw_text, plant_data)

    # -------------------------------------------------
    # SYMPTOM REMEDY RESPONSE
    # -------------------------------------------------
    if intent == "symptom_remedy":
        response = {
            "plant_name": plant_name,
            "remedy": final_text
        }

        if module1_output.get("poem"):
            response["poem"] = module1_output.get("poem")
            response["poet"] = module1_output.get("poet")

        return response

    return final_text
