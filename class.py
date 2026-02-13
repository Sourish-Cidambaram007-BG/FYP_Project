import os
import json

train_dir = r'C:\Users\FYP_2026_06\Desktop\FYP\data\plantnet_300K\plantnet_300K\images_train'

try:
    # Filter out .DS_Store and only keep directories
    class_names = [f for f in os.listdir(train_dir) 
                   if os.path.isdir(os.path.join(train_dir, f)) and not f.startswith('.')]
    
    # Sort them to match how Keras sorts them
    class_names.sort()
    
    with open('class_labels.json', 'w') as f:
        json.dump(class_names, f)

    print(f"✅ FIXED! Created class_labels.json with {len(class_names)} species.")
    print(f"Index 0 is now correctly: '{class_names[0]}'")

except Exception as e:
    print(f"❌ Error: {e}")