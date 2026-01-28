import pandas as pd
import sys
import os

def load_raw_data(path):
    """
    Load raw fraud detection dataset.

    Parameters:
        path (str): Path to raw CSV file

    Returns:
        pd.DataFrame
    """
    return pd.read_csv(path)
sys.path.append(os.path.abspath(".."))