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

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add the Dataset**
    - Download and place the `laryngeal_dataset` directory into the root of the project folder as shown in the directory structure above.

---

## How to Run the Experiments

### Step 1: Run the Centralized Benchmark

This script trains the model on the entire dataset at once to establish the best-case performance benchmark.

```bash
python train_centralized.py
```
The script will print the final validation accuracy, model size, and training time metrics, which serve as our benchmark.