import tensorflow as tf
import numpy as np
import json
import os

# 1. PATHS - Using the FIXED labels file
MODEL_PATH = r'C:\Users\FYP_2026_06\Desktop\FYP\plantnet_warmup.keras'
LABELS_PATH = r'C:\Users\FYP_2026_06\Desktop\FYP\class_labels_fixed.json'
NAMES_JSON = r'C:\Users\FYP_2026_06\Desktop\FYP\data\plantnet_300K\plantnet_300K\plantnet300K_species_names.json'

print("🔄 Running Prediction with Synced Mapping...")

# 2. LOAD DATA
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    class_indices = json.load(f)

with open(NAMES_JSON, 'r') as f:
    id_to_name = json.load(f)

def identify(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Error: {img_path} not found.")
        return

    # Preprocessing
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)

    # Prediction
    preds = model.predict(img_array, verbose=0)
    top_3_idx = np.argsort(preds[0])[-3:][::-1]
    
    print(f"\n🌿 Module 3: Medicinal Plant Identification")
    print("="*45)
    
    for i, idx in enumerate(top_3_idx):
        folder_id = str(class_indices[idx])
        # Lookup name from the dictionary
        name = id_to_name.get(folder_id, f"Unknown ID: {folder_id}")
        confidence = preds[0][idx] * 100
        
        star = "⭐" if i == 0 else "  "
        print(f"{star} {name}")
        print(f"   Confidence Score: {confidence:.2f}%")

# --- EXECUTE ---
identify(r'C:\Users\FYP_2026_06\Desktop\FYP\neem.jpg')