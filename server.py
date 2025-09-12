import flwr as fl
from typing import List, Tuple, Dict, Union
from flwr.common import Metrics
import numpy as np

# --- Configuration ---
NUM_ROUNDS = 10

# Global variables to track metrics across rounds
total_communication_bytes = 0
total_training_time_accumulated = 0

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
    round_bytes_sent = sum(m["bytes_sent"] for _, m in metrics)
    
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
strategy = fl.server.strategy.FedAvg(
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

# Debug: Print available metrics in history
print(f"Debug - Available metrics_centralized keys: {list(history.metrics_centralized.keys()) if hasattr(history, 'metrics_centralized') and history.metrics_centralized else 'None'}")
print(f"Debug - Available metrics_distributed keys: {list(history.metrics_distributed.keys()) if hasattr(history, 'metrics_distributed') and history.metrics_distributed else 'None'}")

# Use the global accumulated values instead of trying to extract from history
total_bytes_mb = total_communication_bytes / (1024*1024)

# The history object will now contain the aggregated metrics for each round
print("\n--- Federated Learning Final Aggregated Metrics ---")
if history.metrics_distributed:
    for metric_name, values in history.metrics_distributed.items():
        # values is a list of tuples (round, value)
        if values:
            final_value = values[-1][1]
            if metric_name == 'accuracy':
                print(f"  {metric_name.capitalize()}: {final_value * 100:.2f}%")
            else:
                # Format metric names like f1_score to F1-score
                formatted_name = metric_name.replace('_', '-').capitalize()
                print(f"  {formatted_name}: {final_value:.4f}")
    
    print(f"  Total Training Time: {total_training_time_accumulated:.2f} seconds")
    print(f"  Total Communication Overhead: {total_bytes_mb:.2f} MB")

else:
    print("No metrics were distributed.")