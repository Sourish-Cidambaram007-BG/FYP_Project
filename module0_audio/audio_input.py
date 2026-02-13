import uuid
import os

AUDIO_DIR = "temp_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


def save_uploaded_audio(uploaded_file):
    """
    Save uploaded audio file to disk.
    """
    file_path = os.path.join(
        AUDIO_DIR,
        f"{uuid.uuid4()}_{uploaded_file.name}"
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    return file_path
