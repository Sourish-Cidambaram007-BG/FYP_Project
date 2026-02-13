import tensorflow as tf
import numpy as np

# 1. Path to your saved model
model_path = r'C:\Users\FYP_2026_06\Desktop\FYP\plantnet_warmup.keras'

print("🔄 Loading Model... Please wait.")
try:
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # 2. Check the model's 'memory'
    # This prints how many plant classes it knows
    output_shape = model.output_shape[-1]
    print(f"🌿 This model is trained to recognize {output_shape} different plant species.")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")