# E.SUN AI Cup 2025 — Account-Level Fraud Detection

A cleaned portfolio version of an E.SUN AI Cup 2025 fraud-detection pipeline.

The project transforms bidirectional transaction records into account-level training data, engineers **27 behavioral features**, and trains a **Positive-Unlabeled (PU) learning + LightGBM** ensemble with 5-fold cross-validation.

## Pipeline

```text
raw transactions
      |
      v
account-centric long format
      |
      v
leakage-safe reference cutoff
      |
      v
27 account-level behavioral features
      |
      v
iterative reliable-negative mining
      |
      v
confidence-weighted LightGBM
      |
      v
5-fold ensemble + OOF threshold tuning
```

### 1. Data preparation

`notebooks/01_data_preparation.ipynb`

- Converts each transaction into inbound and outbound account perspectives.
- Normalizes transaction amounts to TWD.
- Builds the positive, unlabeled, and prediction account sets.
- Removes transactions after the alert reference date for known positive accounts.

### 2. Feature engineering

`notebooks/02_feature_engineering.ipynb`

The final CV pipeline uses **27 features** covering:

- transaction volume and amount
- inbound / outbound behavior
- self-transaction behavior
- temporal activity
- directional amount statistics
- net fund flow
- counterparty-network behavior
- transaction velocity
- account-type distribution

### 3. PU learning and LightGBM

`notebooks/03_pu_lightgbm_cv.ipynb`

The modeling pipeline:

1. Splits accounts into 5 folds with `StratifiedGroupKFold`.
2. Holds out a small fraction of known positives as **spies**.
3. Trains a lightweight PU classifier on core positives vs. unlabeled + spy samples.
4. Iteratively selects low-scoring **reliable negatives (RN)**.
5. Assigns larger weight to positives and confidence-based weights to RN samples.
6. Trains the final LightGBM model with early stopping.
7. Tunes the classification threshold on concatenated out-of-fold predictions.
8. Averages probabilities from the five fold models for final predictions.

## Repository structure

```text
.
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_pu_lightgbm_cv.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```
