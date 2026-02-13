SYMPTOMS = {
    "fever": ["fever", "temperature", "body heat"],
    "cough": ["cough", "throat pain"],
    "cold": ["cold", "runny nose"]
}

def detect_symptom(text: str):
    text = text.lower()
    for symptom, keywords in SYMPTOMS.items():
        for k in keywords:
            if k in text:
                return symptom
    return None
