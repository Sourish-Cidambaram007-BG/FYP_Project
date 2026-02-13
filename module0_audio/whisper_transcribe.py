import whisper

# =====================================================
# FORCE WHISPER TO CPU (VERY IMPORTANT)
# =====================================================
DEVICE = "cpu"

# Medium is PERFECT for FYP (accurate + fast)
model = whisper.load_model("medium", device=DEVICE)


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio to text using Whisper (CPU).
    """
    result = model.transcribe(audio_path)
    return result["text"]
