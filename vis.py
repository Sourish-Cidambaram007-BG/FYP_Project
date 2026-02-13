import tensorflow as tf
import numpy as np
import json
import os
import random

# 1. Paths
MODEL_PATH = r'C:\Users\FYP_2026_06\Desktop\FYP\plantnet_warmup.keras'
TEST_DIR = r'C:\Users\FYP_2026_06\Desktop\FYP\data\plantnet_300K\plantnet_300K\images_test'
NAMES_JSON = r'C:\Users\FYP_2026_06\Desktop\FYP\data\plantnet_300K\plantnet_300K\plantnet300K_species_names.json'
LABELS_PATH = r'C:\Users\FYP_2026_06\Desktop\FYP\class_labels.json'

print("🔄 Loading Model and Database...")

# 2. Load Metadata
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    class_indices = json.load(f)

with open(NAMES_JSON, 'r') as f:
    id_to_name = json.load(f) # Directly loading as dictionary

print("\n🔍 Visual Validation (Testing 5 Random Species)")
print("="*60)

# Get list of folders in the test directory
test_folders = [f for f in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, f))]
samples = random.sample(test_folders, 5)

matches = 0

for folder in samples:
    # Pick a random image from that folder
    folder_path = os.path.join(TEST_DIR, folder)
    img_list = [i for i in os.listdir(folder_path) if i.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not img_list: continue
    
    img_name = random.choice(img_list)
    img_path = os.path.join(folder_path, img_name)
    
    # 3. Predict
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)
    
    preds = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(preds[0])
    pred_folder_id = str(class_indices[pred_idx]) # Ensure it's a string for dict lookup
    
    # 4. Lookup Names
    actual_name = id_to_name.get(str(folder), "Unknown Species")
    predicted_name = id_to_name.get(pred_folder_id, "Unknown Species")
    confidence = np.max(preds[0]) * 100
    
    print(f"📂 Actual Folder: {folder} ({actual_name})")
    print(f"🤖 AI Prediction: {predicted_name} ({confidence:.2f}%)")
    
    if str(folder) == pred_folder_id:
        print("✅ SUCCESS: Match Found!")
        matches += 1
    else:
        print("❌ MISMATCH")
    print("-" * 40)

print(f"\n🎯 Validation Score: {matches}/5 Correct")
print(f"Estimated Sample Accuracy: {(matches/5)*100}%")