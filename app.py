# =====================================================
# 🌿 MEDICINAL PLANT ASSISTANT – FULL MERGED APP
# Image + NLP + Audio | FYP 2026
# =====================================================

import os, sys, json, torch, psutil, warnings
import streamlit as st
import pandas as pd

from PIL import Image
from torchvision import models, transforms
from deep_translator import GoogleTranslator

# =====================================================
# ENVIRONMENT
# =====================================================
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.extend([
    BASE_DIR,
    os.path.join(BASE_DIR, "PlantNet-300K"),
    os.path.join(BASE_DIR, "module2_flan")
])

# =====================================================
# IMPORTS
# =====================================================
from module1_pipeline.run_module1 import run_module1
from module2_flan.run_flan import run_flan
from module0_audio.audio_input import save_uploaded_audio
from module0_audio.whisper_transcribe import transcribe_audio
from module0_audio.text_to_speech import text_to_speech
from module1_text.hybrid_translator import HybridTranslator

from utils import load_model
from module2_flan.flan_generator import load_flan, generate_plant_info

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config("🌿 Medicinal Plant Assistant", "🌿", "centered")

# =====================================================
# GLOBAL LANGUAGE SELECTOR (NEW)
# =====================================================
LANGUAGES = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Marathi": "mr",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Urdu": "ur",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Arabic": "ar",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese": "zh-CN",
    "Indonesian": "id",
    "Thai": "th",
    "Vietnamese": "vi",
    "Turkish": "tr",
    "Dutch": "nl",
    "Polish": "pl",
    "Swedish": "sv",
    "Norwegian": "no",
    "Danish": "da"
}

st.sidebar.header("🌍 Output Language")
selected_lang_name = st.sidebar.selectbox(
    "Choose output language",
    list(LANGUAGES.keys())
)
OUTPUT_LANG = LANGUAGES[selected_lang_name]

# =====================================================
# GOOGLE TRANSLATE
# =====================================================
def google_translate(text, target):
    if not text or target == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target).translate(text)
    except Exception:
        return text

# =====================================================
# LOAD NLP
# =====================================================
@st.cache_resource
def load_nlp():
    return HybridTranslator()

translator = load_nlp()

# =====================================================
# LOAD CNN
# =====================================================
@st.cache_resource
def load_cnn():
    model = models.resnet18(num_classes=1081)
    load_model(
        model,
        os.path.join(
            BASE_DIR,
            "PlantNet-300K/results/fast_resnet18/fast_resnet18_weights_best_acc.tar"
        ),
        use_gpu=torch.cuda.is_available()
    )
    model.eval().to(DEVICE)
    return model

model = load_cnn()

# =====================================================
# LOAD CLASS MAP
# =====================================================
with open(
    os.path.join(BASE_DIR, "PlantNet-300K/class_idx_to_species_name.json"),
    "r", encoding="utf-8"
) as f:
    id_to_name = {int(k): v for k, v in json.load(f).items()}

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =====================================================
# LOAD FLAN
# =====================================================
@st.cache_resource
def load_flan_model():
    return load_flan()

tokenizer, flan_model = load_flan_model()

# =====================================================
# UI HEADER
# =====================================================
st.markdown("""
<div style="background:#2e7d32;padding:20px;border-radius:16px;color:white;text-align:center">
<h1>🌿 Medicinal Plant Assistant</h1>
<p>Hybrid Vision–Language AI System</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

tab_img, tab_nlp = st.tabs(["📷 Plant Identification", "💬 Medicinal Q&A"])

# =====================================================
# 📷 IMAGE TAB (TRANSLATED OUTPUT)
# =====================================================
with tab_img:
    img = st.file_uploader("Upload plant image", ["jpg","jpeg","png"])

    if img:
        image = Image.open(img).convert("RGB")
        st.image(image, width=300)

        x = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            top5 = torch.topk(probs, 5)

        st.subheader("🌿 Top-5 Predictions")
        top1 = None

        for i in range(5):
            idx = top5.indices[0][i].item()
            conf = top5.values[0][i].item()*100
            plant = id_to_name.get(idx,"Unknown")
            if i==0: top1 = plant
            st.write(f"{i+1}. {plant.replace('_',' ')} — {conf:.2f}%")

        if top1:
            info = generate_plant_info(
                tokenizer, flan_model, top1.replace("_"," ")
            )
            translated_info = google_translate(info, OUTPUT_LANG)
            st.subheader("📘 Plant Information")
            st.write(translated_info)

# =====================================================
# 💬 NLP TAB
# =====================================================
with tab_nlp:
    text = st.text_input("Ask your question")
    if st.button("Get Answer"):
        detected = translator.detect_language(text)
        base = translator.translate_to_english(text) if detected!="en" else text

        out1 = run_module1(base, text, detected)
        answer = run_flan(out1)
        final_en = answer.get("remedy","")
        final = google_translate(final_en, OUTPUT_LANG)

        st.subheader("📌 Expert Answer")
        st.write(final)
        st.audio(text_to_speech(final, OUTPUT_LANG))

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("🎓 FYP 2026 | Multilingual Hybrid Medicinal Plant AI System")
