import os
import uuid
from gtts import gTTS

# =====================================================
# TEXT → SPEECH (LANGUAGE AWARE)
# =====================================================
def text_to_speech(text: str, lang: str = "en"):
    """
    Convert text to speech and return audio file path.

    lang examples:
    - en → English
    - ta → Tamil
    - hi → Hindi
    """

    if not text:
        return None

    # gTTS language mapping
    lang_map = {
        "en": "en",
        "ta": "ta",
        "hi": "hi",
        "te": "te",
        "ml": "ml"
    }

    tts_lang = lang_map.get(lang, "en")

    # output folder
    os.makedirs("outputs/audio", exist_ok=True)

    filename = f"outputs/audio/{uuid.uuid4().hex}.mp3"

    tts = gTTS(text=text, lang=tts_lang)
    tts.save(filename)

    return filename
