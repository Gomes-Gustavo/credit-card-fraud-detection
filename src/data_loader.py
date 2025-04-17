import pandas as pd
import os

def _get_project_root():
    """Returns the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_raw_data(path=None):
    """Loads the raw dataset from the data/raw folder."""
    if path is None:
        root = _get_project_root()
        path = os.path.join(root, "data", "raw", "creditcard.csv")
    return pd.read_csv(path)

def load_processed_data(split="train"):
    """Loads a processed dataset split: 'train', 'val' or 'test'."""
    root = _get_project_root()
    file_path = os.path.join(root, "data", "processed", split, "data.csv")
    return pd.read_csv(file_path)

def save_processed_data(df, split="train"):
    """Saves the processed dataset split to the corresponding folder."""
    root = _get_project_root()
    dir_path = os.path.join(root, "data", "processed", split)
    os.makedirs(dir_path, exist_ok=True)
    df.to_csv(os.path.join(dir_path, "data.csv"), index=False)