import os
import json

# The directory where your training folders are located
train_dir = r'C:\Users\FYP_2026_06\Desktop\FYP\data\plantnet_300K\plantnet_300K\images_train'

try:
    # Keras/TensorFlow sorts folders alphabetically by default
    # This list MUST match the order used during model.fit()
    folders = sorted([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])

    # We save this as the "Source of Truth"
    mapping_filename = 'class_labels_fixed.json'
    
    with open(mapping_filename, 'w') as f:
        json.dump(folders, f)

    print(f"✅ SUCCESS!")
    print(f"File Created: {mapping_filename}")
    print(f"Total Classes: {len(folders)}")
    print(f"First 3 Folders: {folders[:3]}")

except Exception as e:
    print(f"❌ Error: {e}")