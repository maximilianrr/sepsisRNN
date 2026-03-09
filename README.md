# Assignment 3: Time Series Prediction of Sepsis

A from-scratch NumPy RNN implementation for early sepsis detection, predicting onset 2, 4, and 6 hours in advance from ICU time-series data.

## Project Overview

This project implements a Recurrent Neural Network (RNN) without high-level ML frameworks (e.g., TensorFlow or Keras) to classify whether a patient will develop sepsis within a given time horizon. The full workflow covers:

1. **Exploratory Data Analysis** – class balance, sequence length distribution, missing value checks, correlation analysis, normality tests (Shapiro-Wilk), and outlier detection (IQR method).
2. **Data Preprocessing** – outlier capping (Winsorizer), robust scaling (median / IQR), and sequence padding with masking.
3. **Cross-validation** – 4-fold cross-validation using four data partitions (A–D); scaling statistics are fitted exclusively on training folds to prevent data leakage.
4. **Model Architecture** – single hidden-layer RNN with `tanh` activation, sigmoid output, and masked binary cross-entropy loss.
5. **Hyperparameter Tuning** – grid search over learning rate, hidden size, batch size, and epochs for each prediction horizon (2 h / 4 h / 6 h).
6. **Evaluation** – accuracy, precision, recall, F1-score, and ROC-AUC reported per fold and aggregated across folds.

## Repository Structure

```
Assignment3/
├── Maximilian_Rauer_assignment3.ipynb  # Main notebook
├── requirements.txt                    # Python dependencies
├── models/                             # Saved best-model weights (.npy)
│   ├── best_rnn_2hrs_shift.npy
│   ├── best_rnn_4hrs_shift.npy
│   └── best_rnn_6hrs_shift.npy
└── raw_data/                           # Raw TSV data partitions (A–D)
    ├── sepsisexp_timeseries_partition-A.tsv
    ├── sepsisexp_timeseries_partition-B.tsv
    ├── sepsisexp_timeseries_partition-C.tsv
    └── sepsisexp_timeseries_partition-D.tsv
```

## Dataset

The project uses the **Sepsis Experiment Time-Series** dataset split into four partitions (A–D).  
Each partition contains ~245 non-sepsis and ~74 sepsis patients with 30-minute interval measurements.

> **Download the dataset and place the `.tsv` files in the `raw_data/` directory before running the notebook.**
>
> Download the files from the following link: https://www.cl.uni-heidelberg.de/statnlpgroup/sepsisexp/#data
> It is important to create a `raw_data/` folder under: `Assignment3/raw_data/`

## Prerequisites

- Python 3.8+
- [Conda](https://docs.conda.io/) (recommended) or any Python virtual environment manager

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd Assignment3
```

### 2. Create and activate a virtual environment

**With Conda:**
```bash
conda create -n sepsis-rnn python=3.10
conda activate sepsis-rnn
```

**With venv:**
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Notebook

### In VS Code

1. Open `Maximilian_Rauer_assignment3.ipynb` in VS Code.
2. Select the correct Python kernel (the environment created above).
3. Run all cells via **Run All** (`Ctrl+F9` / `Cmd+F9`).

### In JupyterLab / Jupyter Notebook

```bash
jupyter lab
# or
jupyter notebook
```

Then open `Maximilian_Rauer_assignment3.ipynb` and run all cells sequentially.

> **Important:** Run cells in order from top to bottom. Later cells depend on variables defined in earlier cells.

## Pre-trained Models

The `models/` directory contains pre-trained weight files (`.npy`) for each prediction horizon.  
You can use these for evaluation. However, the code in the notebook does not consider these and only saves them in the directory.

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | RNN implementation and array operations |
| `pandas` | Data loading and manipulation |
| `matplotlib` | Plotting |
| `seaborn` | Correlation and distribution visualizations |
| `scikit-learn` | Train/test splitting and evaluation metrics |
| `scipy` | Shapiro-Wilk normality test |
| `feature-engine` | Winsorizer for outlier capping |
| `ipykernel` | Jupyter kernel support |

## Author

Maximilian Rauer
