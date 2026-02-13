import tensorflow as tf
import os

print(f"📦 TensorFlow Version: {tf.__version__}")

# Check GPU - Method 1
gpus = tf.config.list_physical_devices('GPU')
print(f"✅ GPU List (New Method): {gpus}")

# Check GPU - Method 2 (Legacy Backup)
print(f"✅ GPU Available (Legacy): {tf.test.is_gpu_available()}")

# Check Path
path = r"C:\Users\FYP_2026_06\Desktop\FYP\data\plantnet_300K\images_train"
print(f"📂 Path Exists: {os.path.exists(path)}")