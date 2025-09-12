import flwr as fl
import tensorflow as tf
import os
import argparse
import time
import numpy as np
from prepare_data import load_data_for_fold, DATASET_PATH
from train_centralized import build_model

LEARNING_RATE = 0.001

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Flower client")
parser.add_argument(
    "--fold", type=int, choices=[1, 2, 3], required=True, help="Specify the fold number (1, 2, or 3)"
)
args = parser.parse_args()

# --- 1. Load Local Data and Build Model ---
print(f"Client starting for FOLD {args.fold}...")
fold_path = os.path.join(DATASET_PATH, f'FOLD {args.fold}')
train_ds, val_ds, num_classes = load_data_for_fold(fold_path)

model = build_model(num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC()
    ]
)

# --- 2. Define Flower Client ---
class LaryngealClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        
        start_time = time.time()
        history = model.fit(train_ds, epochs=1, validation_data=val_ds, verbose=0)
        end_time = time.time()
        
        training_time = end_time - start_time
        
        updated_weights = model.get_weights()
        
        # Calculate communication overhead more accurately
        # Account for both incoming and outgoing model weights
        incoming_bytes = sum(param.nbytes for param in parameters)
        outgoing_bytes = sum(weight.nbytes for weight in updated_weights)
        total_bytes_sent = incoming_bytes + outgoing_bytes
        
        print(f"Client FOLD {args.fold} - Training time: {training_time:.2f}s, Bytes sent: {total_bytes_sent/1024/1024:.2f} MB")
        
        return model.get_weights(), len(list(train_ds)), {
            "accuracy": history.history["accuracy"][0],
            "training_time": training_time,
            "bytes_sent": total_bytes_sent
        }

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy, precision, recall, auc = model.evaluate(val_ds, verbose=0)
        num_examples = len(list(val_ds))
        
        # Calculate F1-score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return loss, num_examples, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "f1_score": f1_score
        }

# --- 3. Start the Client ---
fl.client.start_numpy_client(
    server_address="127.0.0.1:8081",
    client=LaryngealClient(),
    # Add this line to increase the message size limit (1GB)
    grpc_max_message_length=1024 * 1024 * 1024,
)