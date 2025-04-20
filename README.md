# Credit Card Fraud Detection

This repository contains a **Machine Learning pipeline** for detecting fraudulent credit card transactions using classical models. The project handles highly imbalanced data, applies custom feature engineering, interprets the model with SHAP, and evaluates its robustness on a holdout test set.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Installation](#installation)
- [Saved Artifacts](#saved-artifacts)
- [Model Performance](#model-performance)
- [References](#references)
- [Author](#author)

## Project Overview

This project tackles the challenge of fraud detection using a real-world anonymized dataset from credit card transactions. The data is extremely imbalanced, with frauds representing only **0.172%** of all observations. The final model was designed to prioritize **recall** while maintaining **interpretability** and good overall performance.

Key objectives include:

- Handle extreme class imbalance
- Engineer relevant features to improve fraud signal
- Train and tune multiple classical ML models (Logistic Regression, Random Forest, XGBoost)
- Interpret decisions using SHAP
- Simulate predictions in a real-time pipeline
- Evaluate final performance on a separate holdout test set

## Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Frauds**: 492 (0.173%)
- **Features**:
  - `V1` to `V28`: Anonymized PCA-transformed features
  - `Time`: Seconds since first transaction
  - `Amount`: Transaction value
  - `Class`: Target (0 = legitimate, 1 = fraud)

## Project Structure

```
credit-card-fraud-detection/
├── data/
│   |── processed/
│       ├── train/
│       ├── val/
│       └── test/
|   └── raw/
├── models/
│   ├── xgboost_final_model.joblib
│   └── amount_scaler.joblib
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── results/
│   └── test_results.json
├── src/
│   ├── data_loader.py
│   ├── model_utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Jupyter Notebooks

| Step                            | Notebook                                                         | Description                                                                   |
| ------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **1. Exploratory Analysis**     | [01_eda.ipynb](notebooks/01_eda.ipynb)                           | Visualizes fraud distribution, amount, time, PCA components, and correlations. |
| **2. Preprocessing & Features** | [02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb)       | Applies scaling, creates new features, and splits data.                        |
| **3. Model Training**           | [03_model_training.ipynb](notebooks/03_model_training.ipynb)     | Trains and tunes Logistic Regression, Random Forest and XGBoost.               |
| **4. Evaluation & Simulation**  | [04_model_evaluation.ipynb](notebooks/04_model_evaluation.ipynb) | Uses SHAP for interpretation, tests real-time prediction and test performance. |

## Installation

To set up the project locally:

```bash
git clone https://github.com/your-user/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

## Saved Artifacts

| File Path                           | Description                           |
| ----------------------------------- | ------------------------------------- |
| `models/xgboost_final_model.joblib` | Trained final model (XGBoost)         |
| `models/amount_scaler.joblib`       | Scaler used to normalize `Amount`     |
| `results/test_results.json`         | Final performance metrics on test set |

## Model Performance

- **Selected model**: XGBoost
- **Threshold used**: `0.8` (to improve precision-recall trade-off)
- **Holdout test set results**:
  - **Recall**: ~0.86
  - **Precision**: ~0.55
  - **AUC-ROC**: ~1.00
- **Top engineered features**:
  - `Amount_to_mean_ratio`
  - `Amount_to_std_ratio`
  - `Hour` (derived from `Time`)

## References

- [Credit card fraud detection on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Author

Developed by Gustavo Gomes

- [LinkedIn](https://www.linkedin.com/in/gustavo-gomes-581975333/)

