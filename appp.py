import sys
import os
import json
import torch
import streamlit as st
from torchvision import models, transforms
from PIL import Image

# ==================================================
# PATH FIXES (IMPORTANT FOR STREAMLIT)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "PlantNet-300K"))
sys.path.append(os.path.join(BASE_DIR, "module2_flan"))

from utils import load_model
from module2_flan.flan_generator import load_flan, generate_plant_info

# ==================================================
# DEVICE
# ==================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================
# STREAMLIT PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Hybrid Plant Identification System",
    page_icon="🌿",
    layout="centered"
)

# ==================================================
# HEADER UI
# ==================================================
st.markdown(
    """
    <div style="
        background-color:#e8f5e9;
        padding:18px;
        border-radius:14px;
        text-align:center;
    ">
        <h2>🌿 Hybrid Vision–Language Plant Identification System</h2>
        <p style="font-size:15px;">
            Deep Learning–based plant identification with
            AI-generated botanical & medicinal insights
        </p>
        <p style="font-size:13px; color:#2e7d32;">
            Architecture: CNN for visual recognition + Transformer for knowledge generation
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==================================================
# LOAD CLASS → SPECIES NAME MAP
# ==================================================
CLASS_MAP_PATH = os.path.join(
    BASE_DIR,
    "PlantNet-300K",
    "class_idx_to_species_name.json"
)

with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    id_to_name = json.load(f)

id_to_name = {int(k): v for k, v in id_to_name.items()}

# ==================================================
# LOAD CNN MODEL (RESNET-18 BACKEND, HIDDEN)
# ==================================================
@st.cache_resource(show_spinner=True)
def load_cnn_model():
    model = models.resnet18(num_classes=1081)
    load_model(
        model,
        os.path.join(
            BASE_DIR,
            "PlantNet-300K",
            "results",
            "fast_resnet18",
            "fast_resnet18_weights_best_acc.tar"
        ),
        use_gpu=torch.cuda.is_available()
    )
    model.eval().to(DEVICE)
    return model

model = load_cnn_model()

# ==================================================
# LOAD FLAN-T5-LARGE (CACHED)
# ==================================================
@st.cache_resource(show_spinner=True)
def load_flan_cached():
    return load_flan()

tokenizer, flan_model = load_flan_cached()

# ==================================================
# IMAGE TRANSFORM
# ==================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================================================
# IMAGE UPLOAD
# ==================================================
st.subheader("📷 Upload Plant Image")

uploaded_file = st.file_uploader(
    "Supported formats: JPG, JPEG, PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=360)

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top5 = torch.topk(probs, k=5)

    st.markdown("---")
    st.subheader("🌿 Top-5 Predicted Species")

    top1_name = None

    for i in range(5):
        class_id = top5.indices[0][i].item()
        confidence = top5.values[0][i].item() * 100
        plant = id_to_name.get(class_id, "Unknown species")

        if i == 0:
            top1_name = plant

        st.write(f"**{i+1}. {plant.replace('_',' ')}** — {confidence:.2f}%")

    # ==================================================
    # FLAN-T5 PLANT INFORMATION
    # ==================================================
    if top1_name and top1_name != "Unknown species":
        st.markdown("---")
        with st.spinner("🧠 Generating botanical & medicinal information..."):
            info = generate_plant_info(
                tokenizer,
                flan_model,
                top1_name.replace("_", " ")
            )

        st.markdown(
            """
            <div style="
                background-color:#f9fbe7;
                padding:16px;
                border-radius:12px;
            ">
                <h4>📘 Plant Information (AI-Generated)</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(info)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:12px;'>"
    "Academic Prototype • Hybrid Vision–Language AI System"
    "</p>",
    unsafe_allow_html=True
)
