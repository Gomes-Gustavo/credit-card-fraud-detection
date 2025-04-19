import pandas as pd
import os


def _get_project_root():
    """
    Returns the absolute path to the root directory of the project.

    This is used to construct file paths relative to the project structure.
    
    Returns:
        str: Absolute path to the project root
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_raw_data(path=None):
    """
    Loads the raw dataset from the data/raw directory.

    Parameters:
        path (str, optional): Custom path to the dataset. If None, uses default project path.

    Returns:
        pd.DataFrame: Raw dataset as a pandas DataFrame
    """
    if path is None:
        root = _get_project_root()
        path = os.path.join(root, "data", "raw", "creditcard.csv")
    return pd.read_csv(path)


def load_processed_data(split="train"):
    """
    Loads a processed dataset split (train, val, or test) from the corresponding folder.

    Parameters:
        split (str): One of 'train', 'val', or 'test'

    Returns:
        pd.DataFrame: Processed dataset split
    """
    root = _get_project_root()
    file_path = os.path.join(root, "data", "processed", split, "data.csv")
    return pd.read_csv(file_path)


def save_processed_data(df, split="train"):
    """
    Saves a processed dataset split (train, val, or test) to the corresponding folder.

    Parameters:
        df (pd.DataFrame): The processed DataFrame to save
        split (str): One of 'train', 'val', or 'test'
    """
    root = _get_project_root()
    dir_path = os.path.join(root, "data", "processed", split)
    os.makedirs(dir_path, exist_ok=True)
    df.to_csv(os.path.join(dir_path, "data.csv"), index=False)
