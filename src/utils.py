"""Utility functions for fraud detection."""

import pandas as pd
import numpy as np


def check_python_version():
    """Check if Python version is 3.13+."""
    import sys

    return sys.version_info >= (3, 13)


def load_csv_safe(filepath):
    """Load CSV file with Python 3.13 compatibility."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def get_memory_usage(df):
    """Get memory usage of DataFrame."""
    if isinstance(df, pd.DataFrame):
        return df.memory_usage(deep=True).sum() / 1024**2  # MB
    return 0
