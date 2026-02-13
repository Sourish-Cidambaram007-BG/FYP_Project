import whisper

model = whisper.load_model("large-v3")

def audio_to_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"], result["language"]
