import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import matplotlib.pyplot as plt
import os
import datetime
import time

# 1. 🔥 GPU & PERFORMANCE CONFIG
mixed_precision.set_global_policy('mixed_float16')

# 2. PATHS & CONFIG
DATA_DIR = r"C:\Users\FYP_2026_06\Desktop\FYP\data\plantnet_300K\plantnet_300K"
IMG_SIZE = 224
BATCH_SIZE = 32  
NUM_CLASSES = 1081

# --- TIME TRACKING CALLBACK ---
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.total_start_time = time.time()
        print(f"\n⏱️  Training Phase Started at: {datetime.datetime.now().strftime('%H:%M:%S')}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start_time
        total_elapsed = time.time() - self.total_start_time
        remaining_epochs = self.params['epochs'] - (epoch + 1)
        eta_seconds = remaining_epochs * duration
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        print(f"\n⏱️  Epoch {epoch+1} took: {duration:.2f}s | Total Elapsed: {total_elapsed/60:.2f} min | Est. Time Left: {eta_str}")

# 3. DATA PIPELINE
def load_dataset(subset):
    path = os.path.join(DATA_DIR, subset)
    print(f"📂 Loading from: {path}")
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int'
    ).prefetch(tf.data.AUTOTUNE)

train_ds = load_dataset('images_train')
val_ds = load_dataset('images_val')

# 4. HYBRID MODEL ARCHITECTURE
def build_hybrid_model():
    input_layer = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="main_input")
    
    # Data Augmentation
    aug = layers.RandomFlip("horizontal_and_vertical")(input_layer)
    aug = layers.RandomRotation(0.15)(aug)
    aug = layers.RandomZoom(0.1)(aug)
    aug = layers.RandomContrast(0.1)(aug)

    # Branch 1: EfficientNetV2B0 (V2 fixes the TF 2.10 serialization bug)
    eff_base = tf.keras.applications.EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=aug)
    eff_base.trainable = False 

    # Branch 2: ResNet50V2
    res_scale = layers.Rescaling(1./127.5, offset=-1)(aug) 
    res_base = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_tensor=res_scale)
    res_base.trainable = False 

    # Fusion
    eff_feat = layers.GlobalAveragePooling2D()(eff_base.output)
    res_feat = layers.GlobalAveragePooling2D()(res_base.output)
    combined = layers.Concatenate()([eff_feat, res_feat])
    
    x = layers.BatchNormalization()(combined)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='relu')(x)
    output = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    
    return models.Model(inputs=input_layer, outputs=output), eff_base, res_base

model, eff_base, res_base = build_hybrid_model()

# 5. COMPILE (PHASE 1)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5')]
)

# 6. CALLBACKS - UPDATED TO PREVENT CRASH
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def get_callbacks(model_name):
    return [
        tf.keras.callbacks.ModelCheckpoint(f"{model_name}.keras", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2),
        # write_graph=False is essential to avoid the EagerTensor serialization error
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False),
        TimeHistory()
    ]

# 7. PHASE 1: WARM-UP
print("🚀 PHASE 1: Training the Head...")
history_warmup = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=get_callbacks("plantnet_warmup"))

# 8. PHASE 2: FINE-TUNING
print("\n🔬 PHASE 2: Unfreezing for Fine-Tuning...")
eff_base.trainable = True
res_base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy', tf.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5')]
)

history_finetune = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=get_callbacks("plantnet_final_hybrid"))

# 9. PLOT RESULTS
def plot_results(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    plt.figure(figsize=(10, 6))
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.axvline(x=len(h1.history['accuracy'])-1, color='r', linestyle='--', label='Fine-Tuning')
    plt.title('PlantNet Hybrid Performance')
    plt.legend()
    plt.savefig('Final_Training_Report.png')
    plt.show()

plot_results(history_warmup, history_finetune)
print("✅ Done! Final model: plantnet_final_hybrid.keras")