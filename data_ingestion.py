"""Load the supplied tab-separated customer data."""
import pandas as pd
from .config import DATA_PATH

def load_data():
    """Read the original data and report a clear error if it is missing."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Put customer_performance.csv in data/raw/.")
    # The supplied file uses tabs rather than commas.
    return pd.read_csv(DATA_PATH, sep="\t")
