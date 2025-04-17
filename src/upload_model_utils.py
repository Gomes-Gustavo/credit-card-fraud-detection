import os
import joblib

def save_model(model, model_path="../models/xgboost_final_model.joblib"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


def save_scaler(scaler, scaler_path="../models/amount_scaler.joblib"):
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
