import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
IMG_SIZE = (260, 260) # Input size for EfficientNetB0
BATCH_SIZE = 32
DATASET_PATH = './laryngeal_dataset/' # Path to your main dataset folder

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
        label_mode='categorical', # Use 'categorical' for one-hot encoding
        image_size=IMG_SIZE,
        interpolation='bicubic',
        batch_size=BATCH_SIZE,
        shuffle=True,
        color_mode='rgb'
    )
    
    num_classes = len(full_dataset.class_names)
    print(f"Found {num_classes} classes in {fold_path}: {full_dataset.class_names}")

    # Create a 80/20 split for training and validation
    dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
    train_size = int(dataset_size * 0.8)
    
    train_ds = full_dataset.take(train_size)
    val_ds = full_dataset.skip(train_size)

    # --- Preprocessing and Performance Optimization ---
    
    # Create a sequential model for data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # EfficientNet models have their own preprocessing, but we apply augmentation
    # and use prefetch for performance.
    def prepare(ds, augment=False):
        # Apply augmentation only to the training set
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(lambda x, y: (x, y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
        
        ds = ds.shuffle(buffer_size=100)   # smaller buffer
        ds = ds.prefetch(buffer_size=1)    # fixed prefetch
        return ds

    train_ds = prepare(train_ds, augment=True)
    val_ds = prepare(val_ds, augment=False)
    
    # Preserve class names on the returned datasets so callers can inspect mapping
    train_ds.class_names = full_dataset.class_names
    val_ds.class_names = full_dataset.class_names
    
    return train_ds, val_ds, num_classes

if __name__ == '__main__':
    # --- Example of how to use the function ---
    print("Loading data for FOLD 1...")
    fold1_path = os.path.join(DATASET_PATH, 'FOLD 1')
    train_ds_1, val_ds_1, num_classes_1 = load_data_for_fold(fold1_path)

    # You can inspect a batch of data to check
    for images, labels in train_ds_1.take(1):
        print("Images shape:", images.shape) # (Batch Size, Height, Width, Channels)
        print("Labels shape:", labels.shape) # (Batch Size, Num Classes)
        print("-" * 30)

    print("\nLoading data for FOLD 2...")
    fold2_path = os.path.join(DATASET_PATH, 'FOLD 2')
    train_ds_2, val_ds_2, _ = load_data_for_fold(fold2_path)
    
    print("\nLoading data for FOLD 3...")
    fold3_path = os.path.join(DATASET_PATH, 'FOLD 3')
    train_ds_3, val_ds_3, _ = load_data_for_fold(fold3_path)
    
    print("\n✅ Data loading script finished successfully!")