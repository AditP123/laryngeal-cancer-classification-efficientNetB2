# Laryngeal Image Classification

This project demonstrates system for classifying laryngeal images, the model architecture used is EfficientNetB2 with transfer learning.

The system now includes comprehensive metrics tracking including communication overhead analysis, training time comparison, and enhanced visualization of both performance and operational efficiency metrics.

---

## Directory Structure

The project should be organized as follows. The `laryngeal_dataset` is not included in this repository and should be placed manually.

```
.
├── laryngeal_dataset/
│   ├── FOLD 1/
│   │   ├── Hbv/
│   │   ├── He/
│   │   ├── IPCL/
│   │   └── Le/
│   ├── FOLD 2/
│   │   └── ...
│   └── FOLD 3/
│       └── ...
├── prepare_data.py
├── train_centralized.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/AditP123/laryngeal-cancer-classification-efficientNetB2
    cd LARYNGEAL-CANCER-CLASSIFICATION-EFFICIENTNETB2
    ```

2.  **Activate Virtual Environment (Recommended)**
    ```bash
    conda create -n myenv python=3.11.13
    conda activate myenv 
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add the Dataset**
    - Download and place the `laryngeal_dataset` directory into the root of the project folder as shown in the directory structure above.

---

## How to Run the Experiments

### Step 1: Run the Aquila Optimizer for optimal parameters

This script runs the aquila optimizer imported from the mealpy library to determine the optimal values for learning rate and dropout rate.

```bash
python optimizepy
```

### Step 2: Run the Centralized Benchmark

This script trains the model on the entire dataset at once to establish the best-case performance benchmark.

```bash
python train_centralized.py
```
The script will print the final validation accuracy, model size, and training time metrics, which serve as our benchmark.

### Step 3: Run the Federated Learning Simulation

This requires **four separate terminal windows** (with the virtual environment activated in each).

1.  **In Terminal 1, start the server:**
    ```bash
    python server.py
    ```
    The server will start and wait for 3 clients to connect. It will track and aggregate communication overhead and training time metrics across all rounds.

2.  **In Terminals 2, 3, and 4, start one client each:**
    ```bash
    # In Terminal 2
    python client.py --fold 1

    # In Terminal 3
    python client.py --fold 2

    # In Terminal 4
    python client.py --fold 3
    ```
    Once all clients connect, the federated training will begin. Each client will report its training time and communication overhead per round. The server will print the final federated accuracy, total training time, and total communication overhead upon completion.