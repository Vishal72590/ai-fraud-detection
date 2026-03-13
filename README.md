# AI Credit Card Fraud Detection System

Production-grade fraud detection using ensemble of 5 ML models.

## Project Overview
| | |
|---|---|
| Dataset | Kaggle Credit Card Fraud - 284,807 transactions |
| Fraud Rate | 0.17% (highly imbalanced) |
| Models | Logistic Regression, Random Forest, XGBoost, Isolation Forest, Autoencoder |
| API | FastAPI REST endpoint - real time prediction |
| Database | MySQL - stores all predictions and alerts |
| GPU Required | No - runs fully on CPU |

## Project Structure
```
ai-fraud-detection/
notebooks/
    NB1_data_loading_eda.ipynb
    NB2_feature_engineering.ipynb
    NB3_model_training.ipynb
    NB4_stacking_evaluation.ipynb
    NB5_deployment.ipynb
api/
    main.py
models/
data/
requirements.txt
README.md
```

## Quick Start
pip install -r requirements.txt
Run notebooks NB1 to NB5 in order
uvicorn api.main:app --reload --port 8000

## API Usage
POST http://localhost:8000/predict
Input: {"features": [...31 values...]}
Output: {"fraud_probability": 1.0, "decision": "BLOCK"}

## Decision Logic
| Score | Decision | Action |
|---|---|---|
| Less than 0.30 | APPROVE | Transaction allowed |
| 0.30 to 0.70 | REVIEW | Manual check |
| More than 0.70 | BLOCK | Transaction blocked |

## Models Used
| Model | Type |
|---|---|
| Logistic Regression | Supervised |
| Random Forest | Supervised |
| XGBoost | Supervised - Best performer |
| Isolation Forest | Unsupervised |
| Autoencoder | Deep Learning |
| Meta Learner | Stacking all 5 |

## MySQL Tables
| Table | Description |
|---|---|
| raw_transactions | Original CSV data |
| processed_transactions | Scaled features |
| model_predictions | All predictions |
| fraud_alerts | BLOCK and REVIEW cases |
| model_metrics | Evaluation scores |

## Tech Stack
Python, XGBoost, scikit-learn, PyTorch, FastAPI, MySQL, SHAP

## Dataset
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
