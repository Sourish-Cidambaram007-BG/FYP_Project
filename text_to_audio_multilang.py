from gtts import gTTS

def text_to_wav(text, lang="en", output="output.wav"):
    tts = gTTS(text=text, lang=lang)
    tts.save(output)
    print(f"✅ Audio saved as {output}")

# ======================
# TEST CASES
# ======================
if __name__ == "__main__":

    # Tamil
    tamil_text = "எனக்கு காய்ச்சல் உள்ளது, மருந்து சொல்லுங்கள்"
    text_to_wav(tamil_text, lang="ta", output="fever_tamil.wav")

    # Hindi
    hindi_text = "मुझे बुखार है, कृपया उपाय बताइए"
    text_to_wav(hindi_text, lang="hi", output="fever_hindi.wav")

    # English
    english_text = "I have fever give me remedy"
    text_to_wav(english_text, lang="en", output="fever_english.wav")
