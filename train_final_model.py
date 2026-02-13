import tensorflow as tf
import os

# 1. SETTINGS
DATASET_PATH = r'C:\Users\FYP_2026_06\Desktop\FYP\data\plantnet_300K\plantnet_300K\images_train'
MODEL_SAVE_PATH = r'C:\Users\FYP_2026_06\Desktop\FYP\plantnet_final_medicinal.keras'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("🚀 Initializing STABLE Deep Brain Training on RTX A4000...")

# 2. LOAD DATASET
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Rescaling and Optimization
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(buffer_size=tf.data.AUTOTUNE)

# 3. LOAD WARMUP MODEL (The last stable version)
model = tf.keras.models.load_model(r'C:\Users\FYP_2026_06\Desktop\FYP\plantnet_warmup.keras')

# 4. UNFREEZE THE BRAIN
model.trainable = True

# 5. COMPILE WITH SURVIVAL SETTINGS (The Magic Fix)
# We add 'clipnorm' to prevent the loss from exploding to 168 again
# We add 'top_k' to track if the right plant is in the AI's top 5 guesses
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6, clipnorm=1.0), 
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy', 
        tf.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    ]
)

# 6. CALLBACKS
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

# 7. THE RESCUE RUN
# We reduce to 3 epochs because fine-tuning doesn't need 10 if the math is stable.
print("\n🛰️ Training is starting. Expected duration: ~6-8 hours.")
print("The 'clipnorm' will keep the math stable. Top-5 accuracy should grow nicely.")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,  
    callbacks=[checkpoint]
)

print(f"✅ Rescue Mission Complete! Model saved to {MODEL_SAVE_PATH}")