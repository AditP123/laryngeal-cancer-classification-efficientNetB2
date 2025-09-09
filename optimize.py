"""
Template for applying Nature-Inspired Optimization Algorithms (NIOs)
to tune hyperparameters for EfficientNetB2 training.

Supported optimizers: DBRO, AWO, POA, CBO, AO, BWO
(implementations need to be imported or added manually).
"""

import numpy as np
import time
from train_centralized import build_model   # reuse your model builder
from prepare_data import load_data_for_fold, DATASET_PATH
import tensorflow as tf
import os
from mealpy import FloatVar, AO

# --- Load Data Once (shared across runs) ---
train_ds1, val_ds1, num_classes = load_data_for_fold(os.path.join(DATASET_PATH, "FOLD 1"))
train_ds2, val_ds2, _ = load_data_for_fold(os.path.join(DATASET_PATH, "FOLD 2"))
train_ds3, val_ds3, _ = load_data_for_fold(os.path.join(DATASET_PATH, "FOLD 3"))

train_ds = train_ds1.concatenate(train_ds2).concatenate(train_ds3)
val_ds = val_ds1.concatenate(val_ds2).concatenate(val_ds3)

print(f"\n✅ Dataset ready for optimization. Num classes = {num_classes}")

# --- Training Wrapper ---
def train_with_hyperparams(lr, dropout, epochs=5):
    """
    Build and train EfficientNetB2 with given hyperparameters.
    Returns validation accuracy after training.
    """
    tf.keras.backend.clear_session()  # Clear previous models from memory
    model = build_model(num_classes)

    # Modify dropout layer dynamically (optional: adjust dropout in build_model instead)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = dropout

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=0
    )

    val_acc = history.history["val_accuracy"][-1]
    return val_acc

# --- Objective Function for Optimizers ---
def objective(params):
    """Params is a list/array: [learning_rate, dropout]."""
    lr, dropout = params
    start = time.time()
    acc = train_with_hyperparams(lr, dropout)
    end = time.time()
    print(f"Tested lr={lr:.6f}, dropout={dropout:.2f} → val_acc={acc:.4f}, time={end-start:.1f}s")
    return 1 - acc   # many optimizers minimize, so invert accuracy

# 4. Run Aquila Optimizer
def run_aquila(bounds, pop_size=10, max_iter=20):
    problem_dict = {
        "bounds": FloatVar(lb=[bounds[0][0], bounds[1][0]],
                           ub=[bounds[0][1], bounds[1][1]],
                           name="hyperparams"),
        "obj_func": objective,
        "minmax": "min",
    }

    model = AO.OriginalAO(epoch=max_iter, pop_size=pop_size)
    g_best = model.solve(problem_dict)
    return g_best.solution, g_best.target.fitness

if __name__ == "__main__":
    # Define bounds for hyperparameters
    bounds = [(1e-5, 1e-3), (0.1, 0.5)]  # (lr range, dropout range)

    # Run the AO optimizer to find best hyperparameters
    best_params, best_score = run_aquila(bounds, pop_size=8, max_iter=3)

    print("\n--- Optimization Finished ---")
    print(f"Best Params: lr={best_params[0]:.6f}, dropout={best_params[1]:.2f}")
    print(f"Best Validation Accuracy: {1 - best_score:.4f}")