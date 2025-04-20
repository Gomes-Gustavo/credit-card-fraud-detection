import os
import joblib
from src.data_loader import _get_project_root

def save_model(model, model_path="../models/xgboost_final_model.joblib"):
    """
    Saves the given model to the specified path.

    Parameters:
        model: Trained model object to be saved.
        model_path (str): Relative or absolute path where the model will be saved.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

def save_scaler(scaler, scaler_path="../models/amount_scaler.joblib"):
    """
    Saves the given scaler to the specified path.

    Parameters:
        scaler: Fitted scaler object to be saved.
        scaler_path (str): Relative or absolute path where the scaler will be saved.
    """
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

def load_model(model_name="xgboost_final_model.joblib"):
    """
    Loads a previously saved model from the models directory.

    Parameters:
        model_name (str): Filename of the saved model.

    Returns:
        Loaded model object.
    """
    root = _get_project_root()
    path = os.path.join(root, "models", model_name)
    return joblib.load(path)

def load_scaler(scaler_name="amount_scaler.joblib"):
    """
    Loads a previously saved scaler from the models directory.

    Parameters:
        scaler_name (str): Filename of the saved scaler.

    Returns:
        Loaded scaler object.
    """
    root = _get_project_root()
    path = os.path.join(root, "models", scaler_name)
    return joblib.load(path)
