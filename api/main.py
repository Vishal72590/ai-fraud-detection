
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import mysql.connector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List

class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

print("Loading models...")
scaler     = joblib.load("models/scaler.pkl")
lr_model   = joblib.load("models/logistic_regression.pkl")
rf_model   = joblib.load("models/random_forest.pkl")
xgb_model  = joblib.load("models/xgboost.pkl")
iso_model  = joblib.load("models/isolation_forest.pkl")
meta_model = joblib.load("models/meta_learner.pkl")
config     = joblib.load("models/prediction_config.pkl")
ae_cfg     = joblib.load("models/autoencoder_config.pkl")

ae_model = FraudAutoencoder(ae_cfg["input_dim"])
ae_model.load_state_dict(
    torch.load("models/autoencoder.pt", map_location="cpu"))
ae_model.eval()

THRESHOLD     = config["threshold"]
FEATURE_NAMES = config["feature_names"]
print("All models loaded!")

def get_db_connection():
    return mysql.connector.connect(
        host     = "localhost",
        user     = "root",
        password = "Rathod@9vishal",
        database = "fraud_detection_db"
    )

app = FastAPI(
    title       = "AI Fraud Detection API",
    description = "Credit card fraud detection using 5 ML models.",
    version     = "1.0.0"
)

class TransactionInput(BaseModel):
    features: List[float]

    @validator("features")
    def check_length(cls, v):
        if len(v) != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} features, got {len(v)}")
        return v

class PredictionOutput(BaseModel):
    fraud_probability : float
    decision          : str
    risk_level        : str
    model_scores      : dict
    saved_to_mysql    : bool

def get_ae_score(X_df):
    with torch.no_grad():
        t   = torch.FloatTensor(X_df.values)
        err = ((t - ae_model(t)) ** 2).mean(dim=1).numpy()
    return err

def save_to_mysql(scores, final_prob, decision, risk):
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_predictions
            (lr_score, rf_score, xgb_score, iso_score, ae_score,
             final_probability, decision, risk_level)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            scores["logistic_regression"],
            scores["random_forest"],
            scores["xgboost"],
            scores["isolation_forest"],
            scores["autoencoder"],
            final_prob, decision, risk
        ))
        pred_id = cursor.lastrowid
        if decision in ("REVIEW", "BLOCK"):
            cursor.execute("""
                INSERT INTO fraud_alerts
                (prediction_id, fraud_probability, decision)
                VALUES (%s,%s,%s)
            """, (pred_id, final_prob, decision))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"MySQL error: {e}")
        return False

@app.get("/")
def root():
    return {"status": "running",
            "message": "Fraud Detection API is live!"}

@app.get("/health")
def health():
    try:
        conn = get_db_connection()
        conn.close()
        mysql_status = "connected"
    except:
        mysql_status = "disconnected"
    return {"api_status": "healthy",
            "mysql_status": mysql_status}

@app.get("/stats")
def stats():
    try:
        conn   = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT COUNT(*) as total FROM model_predictions")
        total = cursor.fetchone()["total"]
        cursor.execute("""
            SELECT decision, COUNT(*) as count
            FROM model_predictions GROUP BY decision""")
        decisions = cursor.fetchall()
        cursor.close()
        conn.close()
        return {"total_predictions": total,
                "decisions": decisions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    try:
        X = pd.DataFrame(
            [transaction.features], columns=FEATURE_NAMES)
        cols = ["Amount", "amount_log", "hour_of_day"]
        existing = [c for c in cols if c in X.columns]
        if existing:
            X[existing] = scaler.transform(X[existing])
        lr_score  = float(lr_model.predict_proba(X)[:, 1][0])
        rf_score  = float(rf_model.predict_proba(X)[:, 1][0])
        xgb_score = float(xgb_model.predict_proba(X)[:, 1][0])
        iso_score = float(-iso_model.score_samples(X)[0])
        ae_score  = float(get_ae_score(X)[0])
        stack      = np.array([[lr_score, rf_score,
                                 xgb_score, iso_score, ae_score]])
        final_prob = float(
            meta_model.predict_proba(stack)[:, 1][0])
        if final_prob < 0.30:
            decision, risk = "APPROVE", "LOW"
        elif final_prob < 0.70:
            decision, risk = "REVIEW", "MEDIUM"
        else:
            decision, risk = "BLOCK", "HIGH"
        scores = {
            "logistic_regression": round(lr_score, 4),
            "random_forest"      : round(rf_score, 4),
            "xgboost"            : round(xgb_score, 4),
            "isolation_forest"   : round(iso_score, 4),
            "autoencoder"        : round(ae_score, 4),
        }
        saved = save_to_mysql(
            scores, round(final_prob, 4), decision, risk)
        return PredictionOutput(
            fraud_probability = round(final_prob, 4),
            decision          = decision,
            risk_level        = risk,
            model_scores      = scores,
            saved_to_mysql    = saved)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
