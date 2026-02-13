import os
import json

# Your actual training folder path
train_dir = r'C:\Users\FYP_2026_06\Desktop\FYP\data\plantnet_300K\plantnet_300K\images_train'

# Keras uses ALPHABETICAL order. We must match that exactly.
correct_labels = sorted([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])

with open('class_labels_fixed.json', 'w') as f:
    json.dump(correct_labels, f)

print(f"✅ Created class_labels_fixed.json")
print(f"Total Folders Found: {len(correct_labels)}")
print(f"Index 0 is: {correct_labels[0]}")