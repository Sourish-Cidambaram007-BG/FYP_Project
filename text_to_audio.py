import pyttsx3

def text_to_wav(text, output_path="output.wav"):
    engine = pyttsx3.init()

    # Optional voice settings
    engine.setProperty("rate", 150)   # speed
    engine.setProperty("volume", 1.0) # volume

    engine.save_to_file(text, output_path)
    engine.runAndWait()

    print(f"✅ Audio saved as {output_path}")


# ===============================
# TEST
# ===============================
if __name__ == "__main__":
    text = "I have fever give me remedy"
    text_to_wav(text)
