import fasttext

model = fasttext.load_model("models/fasttext/lid.176.bin")

def detect_language(text: str) -> str:
    lang = model.predict(text)[0][0]
    return lang.replace("__label__", "")
