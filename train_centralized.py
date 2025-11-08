import os
import time
import numpy as np
import tensorflow as tf
from prepare_data import load_data_for_fold, DATASET_PATH # Import from our previous script

# imports for metrics & plotting
import matplotlib
matplotlib.use("Agg")  # safe backend for scripts
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)

def build_model(num_classes):
    # Load the pre-trained EfficientNetB2 model, without its top classification layer
    base_model = tf.keras.applications.EfficientNetB2(
        input_shape=(260, 260, 3),
        include_top=False, # We'll add our own top layer
        weights='imagenet' # Use weights pre-trained on ImageNet
    )
    
    # Freeze the base model layers to prevent them from being re-trained
    base_model.trainable = False
    
    # Create the new model on top
    inputs = tf.keras.Input(shape=(260, 260, 3))
    # We apply the data augmentation we defined earlier
    x = base_model(inputs, training=False) # Set training=False for the frozen base
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x) # Regularization
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    # --- Configuration ---
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # lr=0.001, dropout=0.2 (default)
    # lr=0.000787, dropout=0.41 (optimized)

    # --- 1. Load and Combine Data from All Folds ---
    print("Loading data from all folds...")
    # Load data from each fold
    train_ds_1, val_ds_1, num_classes = load_data_for_fold(os.path.join(DATASET_PATH, 'FOLD 1'))
    train_ds_2, val_ds_2, _ = load_data_for_fold(os.path.join(DATASET_PATH, 'FOLD 2'))
    train_ds_3, val_ds_3, _ = load_data_for_fold(os.path.join(DATASET_PATH, 'FOLD 3'))
    
    # preserve class names in a variable for later use in reports/plots
    CLASS_NAMES = train_ds_1.class_names
    
    # Concatenate them into a single, large dataset
    full_train_ds = train_ds_1.concatenate(train_ds_2).concatenate(train_ds_3)
    full_val_ds = val_ds_1.concatenate(val_ds_2).concatenate(val_ds_3)

    # Shuffle the combined datasets for good measure
    full_train_ds = full_train_ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    full_val_ds = full_val_ds.shuffle(buffer_size=1024, reshuffle_each_iteration=True)

    print(f"\n✅ All data loaded and combined. Total number of classes: {num_classes}")

    # --- 2. Build the Model using Transfer Learning (EfficientNetB2) ---
    print("Building the centralized model...")
    model = build_model(num_classes)

    # --- 3. Compile the Model ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )

    print("Model built and compiled successfully.")
    model.summary()

    # --- 4. Train the Model ---
    print("\nStarting centralized training...")
    start_time = time.time()
    history = model.fit(
        full_train_ds,
        epochs=NUM_EPOCHS,
        validation_data=full_val_ds
    )
    end_time = time.time()
    centralized_training_time = end_time - start_time

    # --- 5. Evaluate the Final Model ---
    print("\nCentralized training finished.")
    results = model.evaluate(full_val_ds)
    final_loss = results[0]
    final_accuracy = results[1]
    final_precision = results[2]
    final_recall = results[3]
    final_auc = results[4]

    # Calculate F1-score from precision and recall
    final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

    # Calculate Model Size ---
    model_size_bytes = np.sum([np.prod(v.shape) for v in model.get_weights()]) * 4
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Compute predictions on the full validation set ---
    # Robust: collect predictions and true labels in a single pass to preserve order
    y_scores_list = []
    y_true_onehot_list = []
    for x_batch, y_batch in full_val_ds:                  # single iteration, same order
        preds = model.predict_on_batch(x_batch)           # (batch_size, num_classes)
        y_scores_list.append(preds)
        y_true_onehot_list.append(y_batch.numpy())

    y_scores = np.concatenate(y_scores_list, axis=0)
    y_true_onehot = np.concatenate(y_true_onehot_list, axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_scores, axis=1)

    # Sanity check
    assert y_true.shape[0] == y_pred.shape[0], "Mismatch between true labels and predictions"

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Classification report (precision, recall, f1 per class)
    target_names = CLASS_NAMES
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=target_names))

    # --- Precision-Recall curves & Average Precision per class ---
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_true_onehot[:, i], y_scores[:, i])

    # Micro-average PR
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_onehot.ravel(), y_scores.ravel())
    average_precision["micro"] = average_precision_score(y_true_onehot, y_scores, average="micro")

    # Print Average Precision scores
    print("\nAverage Precision (AP) per class:")
    for i in range(num_classes):
        print(f"  class_{i}: AP = {average_precision[i]:.4f}")
    print(f"  micro-average AP = {average_precision['micro']:.4f}")

    # --- Save confusion matrix figure ---
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
    # annotate cells with counts
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
            ha="center", va="center", color="black")
            
    plt.tight_layout()
    cm_path = os.path.join(os.getcwd(), "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    # --- Save PR curve figure (micro + per-class) ---
    plt.figure(figsize=(8, 6))
    plt.step(recall["micro"], precision["micro"], where='post', label=f"micro-average (AP={average_precision['micro']:.2f})", color='k')
    for i in range(num_classes):
        plt.step(recall[i], precision[i], where='post', label=f"{CLASS_NAMES[i]} (AP={average_precision[i]:.2f})", alpha=0.6)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend(loc="lower left")
    plt.grid(True)
    pr_path = os.path.join(os.getcwd(), "pr_curves.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"Saved precision-recall curves to: {pr_path}")
    
    print(f"\n--- Benchmark Model Final Validation Metrics ---")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_accuracy * 100:.2f}%")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall: {final_recall:.4f}")
    print(f"  F1-Score: {final_f1:.4f}")
    print(f"  AUC-ROC: {final_auc:.4f}")
    print(f"  Training Time: {centralized_training_time:.2f} seconds")
    print(f"  Model Size: {model_size_mb:.2f} MB")