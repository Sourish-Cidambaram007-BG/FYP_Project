import os
# IMPORTANT: This must be set BEFORE importing keras
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
from PIL import Image

def run_standalone_test(img_path, model_path='plant_efficientnetb0.keras'):
    # 1. Load the model
    print("--- Loading Keras 3 Model ---")
    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Define Class Labels
    # Update this list to match the EXACT order used during training
    class_names = [
        "Aloe Vera", "Amla", "Ashwagandha", "Brahmi", "Ginger", 
        "Hibiscus", "Mint", "Neem", "Tulsi", "Turmeric"
    ]

    # 3. Preprocess the Image
    print(f"--- Processing Image: {img_path} ---")
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))  # EfficientNetB0 standard size
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # 4. Prediction
    predictions = model.predict(img_array)
    
    # Get index of highest probability
    predicted_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    print("\n" + "="*40)
    print(f"IDENTIFIED PLANT: {class_names[predicted_idx]}")
    print(f"CONFIDENCE LEVEL: {confidence * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # Ensure you have an image file named 'test_image.jpg' in the folder
    target_image = "test_image.jpg"
    
    if os.path.exists(target_image):
        run_standalone_test(target_image)
    else:
        print(f"File Not Found: {target_image}. Please add an image to test.")