import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True
).to(DEVICE)

model.eval()
ip = IndicProcessor(inference=True)

LANG_MAP = {
    "ta": "tam_Taml",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
    "mr": "mar_Deva"
}

def translate_to_english(text: str, lang_code: str) -> str:
    if lang_code not in LANG_MAP:
        return text

    batch = ip.preprocess_batch(
        [text],
        src_lang=LANG_MAP[lang_code],
        tgt_lang="eng_Latn"
    )

    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5
        )

    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    return ip.postprocess_batch(decoded, lang="eng_Latn")[0]
