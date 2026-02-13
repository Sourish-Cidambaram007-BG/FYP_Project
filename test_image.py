import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
from PIL import Image

def get_labels(data_dir):
    if not os.path.exists(data_dir):
        return []
    # Sorts folders alphabetically to match Keras training order
    return sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

def run_standalone_test():
    # --- ABSOLUTE PATHS ---
    model_path = r"C:\Users\FYP_2026_06\Desktop\FYP\ImageTest\plant_efficientnetb0.keras"
    data_path  = r"C:\Users\FYP_2026_06\Desktop\FYP\data\plant_images"
    
    possible_images = [
        r"C:\Users\FYP_2026_06\Desktop\FYP\test_leaf.jpg",
        r"C:\Users\FYP_2026_06\Desktop\FYP\test_leaf.jpeg"
    ]
    
    image_path = next((path for path in possible_images if os.path.exists(path)), None)

    print("--- ⏳ System Check ---")
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at: {model_path}")
        return
    if not image_path:
        print(f"❌ ERROR: Image not found. Place a 'test_leaf.jpg' in the FYP folder.")
        return

    try:
        print(f"✅ Using image: {os.path.basename(image_path)}")
        print("⏳ Loading Keras 3 Model...")
        model = keras.models.load_model(model_path)
        
        class_names = get_labels(data_path)
        print(f"✅ Found {len(class_names)} plant classes.")

        # --- Preprocessing ---
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Common normalization
        img_array = np.expand_dims(img_array, axis=0)

        # --- Prediction ---
        print("🧠 Running Prediction...")
        predictions = model.predict(img_array)
        
        # Get index and confidence
        idx = np.argmax(predictions[0])
        conf = np.max(predictions[0])

        # --- Threshold Logic ---
        # If confidence is below 40%, we consider it an "Unknown" or "Incorrect" plant
        THRESHOLD = 0.40 

        print("\n" + "="*40)
        if conf < THRESHOLD:
            print("⚠️ RESULT: UNCERTAIN")
            print(f"The model is not confident enough (Only {conf*100:.2f}%).")
            print(f"Closest match was: {class_names[idx]}")
            print("\nSuggestion: Try a clearer photo or a plant from the dataset.")
        else:
            print(f"🌿 IDENTIFIED PLANT: {class_names[idx]}")
            print(f"🎯 CONFIDENCE LEVEL: {conf * 100:.2f}%")
        print("="*40)
        
    except Exception as e:
        print(f"❌ Execution Error: {e}")

if __name__ == "__main__":
    run_standalone_test()