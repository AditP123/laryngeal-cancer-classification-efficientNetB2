import tensorflow as tf
import os
import time
import numpy as np
from prepare_data import load_data_for_fold, DATASET_PATH

# Optional: enable mixed precision if you have a recent NVIDIA GPU
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Using mixed precision (fp16).")
except Exception as e:
    print("Mixed precision not enabled:", e)

# --- Configuration ---
WARMUP_EPOCHS = 8          # train top classifier with base frozen
FINETUNE_EPOCHS = 12       # fine-tune upper layers
TOTAL_EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS

LR_WARMUP = 1e-3
LR_FINETUNE = 1e-5
WEIGHT_DECAY = 1e-6        # mild L2 via AdamW if available

# --- 1. Load and Combine Data from All Folds ---
print("Loading data from all folds...")
train_ds_1, val_ds_1, num_classes = load_data_for_fold(os.path.join(DATASET_PATH, 'FOLD 1'))
train_ds_2, val_ds_2, _ = load_data_for_fold(os.path.join(DATASET_PATH, 'FOLD 2'))
train_ds_3, val_ds_3, _ = load_data_for_fold(os.path.join(DATASET_PATH, 'FOLD 3'))

full_train_ds = train_ds_1.concatenate(train_ds_2).concatenate(train_ds_3)
full_val_ds = val_ds_1.concatenate(val_ds_2).concatenate(val_ds_3)

# Shuffle combined datasets
full_train_ds = full_train_ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)
full_val_ds = full_val_ds.shuffle(buffer_size=2048, reshuffle_each_iteration=True)

for images, labels in full_train_ds.take(1):
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("-" * 30)

print(f"\n✅ All data loaded and combined. Total number of classes: {num_classes}")

# --- 2. Build the Model using Transfer Learning (EfficientNetB2) ---
def build_model(num_classes):
    # Base model with imagenet weights; no top classifier
    base_model = tf.keras.applications.EfficientNetB2(
        input_shape=(260, 260, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # freeze for warmup

    inputs = tf.keras.Input(shape=(260, 260, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    dtype = tf.float32  # ensure final Dense runs in fp32 even if mixed precision is on
    x = tf.keras.layers.Dense(256, activation='relu', dtype=dtype)(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype=dtype)(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNetB2_laryngeal")
    return model

print("Building the centralized model (EfficientNetB2)...")
model = build_model(num_classes)

# Optimizer (AdamW if available, else Adam)
try:
    optimizer_warmup = tf.keras.optimizers.AdamW(learning_rate=LR_WARMUP, weight_decay=WEIGHT_DECAY)
    optimizer_finetune = tf.keras.optimizers.AdamW(learning_rate=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
except Exception:
    optimizer_warmup = tf.keras.optimizers.Adam(learning_rate=LR_WARMUP)
    optimizer_finetune = tf.keras.optimizers.Adam(learning_rate=LR_FINETUNE)

loss = 'categorical_crossentropy'
metrics = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

# --- Callbacks ---
ckpt_path = "checkpoints/b2_best.keras"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', mode='max', patience=6, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
]

# --- 3a. Warmup: train classifier head with base frozen ---
print("\nStage 1/2: Warmup training (frozen base)...")
model.compile(optimizer=optimizer_warmup, loss=loss, metrics=metrics)
model.summary()

start_time = time.time()
history_warmup = model.fit(
    full_train_ds,
    epochs=WARMUP_EPOCHS,
    validation_data=full_val_ds,
    callbacks=callbacks,
    verbose=1
)

# --- 3b. Fine-tuning: unfreeze top layers of the base model ---
print("\nStage 2/2: Fine-tuning upper layers...")
base_model = model.get_layer(index=1)  # EfficientNetB2 base should be at index 1
base_model.trainable = True

# Unfreeze last ~30% of layers for stable fine-tuning
fine_tune_at = int(len(base_model.layers) * 0.7)
for i, layer in enumerate(base_model.layers[:fine_tune_at]):
    layer.trainable = False

model.compile(optimizer=optimizer_finetune, loss=loss, metrics=metrics)

history_finetune = model.fit(
    full_train_ds,
    initial_epoch=WARMUP_EPOCHS,
    epochs=TOTAL_EPOCHS,
    validation_data=full_val_ds,
    callbacks=callbacks,
    verbose=1
)
end_time = time.time()
centralized_training_time = end_time - start_time

# --- 4. Evaluate the Final Model ---
print("\nEvaluating the best checkpoint on validation data...")
best_model = tf.keras.models.load_model(ckpt_path)

results = best_model.evaluate(full_val_ds, verbose=1)
final_loss = results[0]
final_accuracy = results[1]
final_precision = results[2]
final_recall = results[3]
final_auc = results[4]

# F1 from precision & recall
final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

# Model size (float32 params; mixed precision stores weights in fp32)
model_size_bytes = np.sum([np.prod(v.shape) for v in best_model.get_weights()]) * 4
model_size_mb = model_size_bytes / (1024 * 1024)

print(f"\n--- EfficientNetB2 Centralized Benchmark (Final Validation) ---")
print(f"  Loss: {final_loss:.4f}")
print(f"  Accuracy: {final_accuracy * 100:.2f}%")
print(f"  Precision: {final_precision:.4f}")
print(f"  Recall: {final_recall:.4f}")
print(f"  F1-Score: {final_f1:.4f}")
print(f"  AUC-ROC: {final_auc:.4f}")
print(f"  Training Time: {centralized_training_time:.2f} seconds")
print(f"  Model Size: {model_size_mb:.2f} MB")
