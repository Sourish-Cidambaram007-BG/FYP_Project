import tensorflow as tf
import json
import os
import numpy as np

# 1. Paths
MODEL_PATH = r'C:\Users\FYP_2026_06\Desktop\FYP\plantnet_warmup.keras'
TEST_DIR = r'C:\Users\FYP_2026_06\Desktop\FYP\data\plantnet_300K\plantnet_300K\images_test'
LABELS_PATH = r'C:\Users\FYP_2026_06\Desktop\FYP\class_labels.json'

print("🔄 Loading Model for Accuracy Evaluation...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    class_names = json.load(f)

print("🧪 Evaluating Accuracy on Test Dataset (Sampling)...")

# We use a smaller validation subset for a quick 2 PM result
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='int',
    class_names=class_names, # Ensures mapping matches your model
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=42
)

# Normalize data
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

# Take a sample of 10 batches (~320 images) to get a quick score
sample_ds = test_ds.take(10)

results = model.evaluate(sample_ds)
accuracy = results[1] * 100

print("\n" + "="*30)
print(f"✅ FINAL ACCURACY SCORE: {accuracy:.2f}%")
print("="*30)
print(f"Interpretation: The model correctly identifies the plant in {accuracy:.1f} out of 100 cases.")