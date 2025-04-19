import os
import joblib


def save_model(model, model_path="../models/xgboost_final_model.joblib"):
    """
    Saves the trained model to the specified path.

    Parameters:
        model: Trained model object (e.g., XGBoost classifier)
        model_path: Path to save the model file
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


def save_scaler(scaler, scaler_path="../models/amount_scaler.joblib"):
    """
    Saves the fitted scaler to the specified path.

    Parameters:
        scaler: Fitted StandardScaler or similar object
        scaler_path: Path to save the scaler file
    """
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)


def load_model(model_path="../models/xgboost_final_model.joblib"):
    """
    Loads a previously saved model from the specified path.

    Parameters:
        model_path: Path to the saved model file

    Returns:
        Loaded model object
    """
    return joblib.load(model_path)


def load_scaler(scaler_path="../models/amount_scaler.joblib"):
    """
    Loads a previously saved scaler from the specified path.

    Parameters:
        scaler_path: Path to the saved scaler file

    Returns:
        Loaded scaler object
    """
    return joblib.load(scaler_path)
