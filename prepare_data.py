import tensorflow as tf
import numpy as np
import os

# --- Configuration (tuned for EfficientNetB2) ---
IMG_SIZE = (260, 260)  # Input size for EfficientNetB2
BATCH_SIZE = 16
DATASET_PATH = './laryngeal_dataset/'  # Path to your main dataset folder
SEED = 1337

def load_data_for_fold(fold_path):
    """
    Loads images from a given fold path, splits them into training and validation sets,
    and preprocesses them for the model.

    Args:
        fold_path (str): The path to the fold directory (e.g., './laryngeal_dataset/FOLD 1').

    Returns:
        tuple: A tuple containing (train_ds, val_ds, num_classes).
    """
    # Load the dataset from the directory
    full_dataset = tf.keras.utils.image_dataset_from_directory(
        fold_path,
        labels='inferred',
        label_mode='categorical',   # one-hot
        image_size=IMG_SIZE,
        interpolation='bicubic',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        color_mode='rgb'  
    )

    num_classes = len(full_dataset.class_names)
    print(f"Found {num_classes} classes in {fold_path}: {full_dataset.class_names}")

    # 80/20 split using dataset cardinality
    dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
    train_size = int(dataset_size * 0.8)

    train_ds = full_dataset.take(train_size)
    val_ds = full_dataset.skip(train_size)

    # --- Data augmentation & performance ---
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.05),
    ])

    # Note: tf.keras.applications.EfficientNet* include internal rescaling.
    # We only add augmentation on-the-fly to training.
    def prepare(ds, augment=False):
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(tf.cast(x, tf.float32), training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    train_ds = prepare(train_ds, augment=True)
    val_ds = prepare(val_ds, augment=False)

    return train_ds, val_ds, num_classes

if __name__ == '__main__':
    print("Loading data for FOLD 1...")
    fold1_path = os.path.join(DATASET_PATH, 'FOLD 1')
    train_ds_1, val_ds_1, num_classes_1 = load_data_for_fold(fold1_path)

    for images, labels in train_ds_1.take(1):
        print("Images shape:", images.shape)
        print("Labels shape:", labels.shape)
        print("-" * 30)

    print("\nLoading data for FOLD 2...")
    fold2_path = os.path.join(DATASET_PATH, 'FOLD 2')
    train_ds_2, val_ds_2, _ = load_data_for_fold(fold2_path)

    print("\nLoading data for FOLD 3...")
    fold3_path = os.path.join(DATASET_PATH, 'FOLD 3')
    train_ds_3, val_ds_3, _ = load_data_for_fold(fold3_path)

    print("\n✅ Data loading script finished successfully!")
