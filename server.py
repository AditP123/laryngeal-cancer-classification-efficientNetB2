import os
import flwr as fl
import tensorflow as tf
from typing import List, Tuple, Dict, Union
from flwr.common import Metrics, parameters_to_ndarrays
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
# Import helpers to build model and load data
from train_centralized import build_model
from prepare_data import load_data_for_fold, DATASET_PATH

# --- Configuration ---
NUM_ROUNDS = 10
WEIGHTS_SAVE_DIR = os.path.join(os.getcwd(), "federated_weights")
os.makedirs(WEIGHTS_SAVE_DIR, exist_ok=True)

# Global variables to track metrics across rounds
total_communication_bytes = 0
total_training_time_accumulated = 0

# --- Custom FedAvg that saves aggregated weights each round ---
class SaveWeightsFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            # Save as .npz
            np.savez(os.path.join(WEIGHTS_SAVE_DIR, f"global_weights_round_{rnd}.npz"), *ndarrays)
            print(f"Saved aggregated global weights for round {rnd} -> {WEIGHTS_SAVE_DIR}")
        return aggregated_parameters, metrics

# --- Define a function to aggregate the evaluation metrics ---
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Union[float, int]]:
    """An aggregation function that calculates the weighted average of all client metrics."""
    # Create a dictionary to store the aggregated metrics
    aggregated_metrics = {}
    
    # Get a list of all metric keys from the first client (e.g., 'accuracy', 'precision')
    if not metrics:
        return {}
    metric_keys = metrics[0][1].keys()

    total_examples = sum([num_examples for num_examples, _ in metrics])

    for key in metric_keys:
        # Calculate the weighted sum of the metric
        weighted_sum = sum([num_examples * m[key] for num_examples, m in metrics])
        # Calculate the weighted average
        aggregated_metrics[key] = weighted_sum / total_examples
        
    return aggregated_metrics

# --- New: Define a function to aggregate fit metrics ---
def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Union[float, int]]:
    """Aggregate fit metrics using weighted average for accuracy and summing for time and bytes."""
    global total_communication_bytes, total_training_time_accumulated
    
    if not metrics:
        return {}
    
    # Debug: Print received metrics
    print(f"Aggregating fit metrics from {len(metrics)} clients:")
    for i, (num_examples, client_metrics) in enumerate(metrics):
        print(f"  Client {i+1}: {num_examples} examples, metrics: {client_metrics}")
    
    # Aggregate accuracy using weighted average
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_accuracy = sum(accuracies) / sum(examples)
    
    # Sum training time and bytes sent for this round
    print(metrics)
    round_training_time = max(m["training_time"] for _, m in metrics)
    round_bytes_sent = max(m["bytes_sent"] for _, m in metrics)
    
    # Accumulate totals across all rounds
    total_training_time_accumulated += round_training_time
    total_communication_bytes += round_bytes_sent

    aggregated = {
        "accuracy": aggregated_accuracy,
        "training_time": round_training_time,
        "bytes_sent": round_bytes_sent,
    }
    
    print(f"  Aggregated for this round: {aggregated}")
    print(f"  Total accumulated - Training time: {total_training_time_accumulated:.2f}s, Bytes: {total_communication_bytes/1024/1024:.2f} MB")
    return aggregated

# Define the strategy, now including our custom metric aggregation function
strategy = SaveWeightsFedAvg(
    min_available_clients=3,
    min_fit_clients=3,
    min_evaluate_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average, # This ensures accuracy is aggregated
    fit_metrics_aggregation_fn=fit_metrics_aggregation # Custom function for fit metrics
)

# Start the Flower server
print(f"Starting Federated Learning server for {NUM_ROUNDS} rounds...")
history = fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    # Add this line to increase the message size limit (1GB)
    grpc_max_message_length=1024 * 1024 * 1024,
)

print("Server finished.")

print(f"\nTotal Training Time (accumulated): {total_training_time_accumulated:.2f} seconds")
print(f"Total Communication Overhead (accumulated): {total_communication_bytes/(1024*1024):.2f} MB")

# -------------------------
# Post-training: centralized evaluation using final global weights
# -------------------------
# Find the last saved weights file
saved_files = sorted([f for f in os.listdir(WEIGHTS_SAVE_DIR) if f.endswith(".npz")])
if not saved_files:
    print("No saved federated weights found. Skipping centralized evaluation.")
else:
    final_weights_file = os.path.join(WEIGHTS_SAVE_DIR, saved_files[-1])
    print(f"Loading final weights from: {final_weights_file}")
    data = np.load(final_weights_file)
    ndarrays = [data[f"arr_{i}"] for i in range(len(data.files))]
    
    # Build model (ensure num_classes consistent with training)
    _, val_ds_sample, num_classes = load_data_for_fold(os.path.join(DATASET_PATH, "FOLD 1"))
    model = build_model(num_classes)
    model.set_weights(ndarrays)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), 
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    
    # Prepare combined validation dataset (same as centralized eval)
    val_ds1 = load_data_for_fold(os.path.join(DATASET_PATH, "FOLD 1"))[1]
    val_ds2 = load_data_for_fold(os.path.join(DATASET_PATH, "FOLD 2"))[1]
    val_ds3 = load_data_for_fold(os.path.join(DATASET_PATH, "FOLD 3"))[1]
    full_val_ds = val_ds1.concatenate(val_ds2).concatenate(val_ds3)
    
    # Single-pass prediction to preserve alignment
    y_scores_list = []
    y_true_onehot_list = []
    for x_b, y_b in full_val_ds:
        preds = model.predict_on_batch(x_b)
        y_scores_list.append(preds)
        y_true_onehot_list.append(y_b.numpy())
    y_scores = np.concatenate(y_scores_list, axis=0)
    y_true_onehot = np.concatenate(y_true_onehot_list, axis=0)
    y_true = y_true_onehot.argmax(axis=1)
    y_pred = y_scores.argmax(axis=1)
    
    # Sanity check
    assert y_true.shape[0] == y_pred.shape[0], "Mismatch between true labels and predictions"
    
    # Compute metrics
    cm = confusion_matrix(y_true, y_pred)
    print("\nFederated centralized evaluation - Confusion Matrix:\n", cm)
    
    # --- Scalar metrics (accuracy, precision, recall, f1, AUC) ---
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    acc = accuracy_score(y_true, y_pred)
    prec_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    rec_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    try:
        auc_roc = roc_auc_score(y_true_onehot, y_scores, average="micro", multi_class="ovr")
    except Exception:
        auc_roc = float("nan")

    print(f"\nSummary metrics:")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision : {prec_micro:.4f}")
    print(f"  Recall   : {rec_micro:.4f}")
    print(f"  F1-score : {f1_micro:.4f}")
    print(f"  AUC-ROC  : {auc_roc:.4f}")
    
    CLASS_NAMES = full_val_ds.class_names if hasattr(full_val_ds, "class_names") else [f"class_{i}" for i in range(num_classes)]
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # PR curves & AP
    precision = {}
    recall = {}
    average_precision = {}
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_true_onehot[:, i], y_scores[:, i])
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_onehot.ravel(), y_scores.ravel())
    average_precision["micro"] = average_precision_score(y_true_onehot, y_scores, average="micro")
    
    print("\nAverage Precision (AP) per class:")
    for i in range(num_classes):
        print(f"  {CLASS_NAMES[i]}: AP = {average_precision[i]:.4f}")
    print(f"  micro-average AP = {average_precision['micro']:.4f}")
    
    # Save confusion matrix figure (annotated)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Federated - Final Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    fmt = 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            plt.text(j, i, format(val, fmt), ha="center", va="center",
                     color="white" if val > thresh else "black")
    plt.tight_layout()
    cm_path = os.path.join(os.getcwd(), "federated_confusion_matrix_final.png")
    plt.savefig(cm_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved federated confusion matrix: {cm_path}")
    
    # Save PR curve figure (micro + per-class)
    plt.figure(figsize=(8, 6))
    plt.step(recall["micro"], precision["micro"], where='post', label=f"micro-average (AP={average_precision['micro']:.2f})", color='k')
    for i in range(num_classes):
        plt.step(recall[i], precision[i], where='post', label=f"{CLASS_NAMES[i]} (AP={average_precision[i]:.2f})", alpha=0.6)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Federated - Precision-Recall curves")
    plt.legend(loc="lower left"); plt.grid(True)
    pr_path = os.path.join(os.getcwd(), "federated_pr_curves_final.png")
    plt.savefig(pr_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved federated PR curves: {pr_path}")