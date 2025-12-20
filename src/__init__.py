"""Fraud Detection Package."""

__version__ = "1.0.0"
__author__ = "Tsegay"
__python_version__ = "3.13"

import sys

# Check Python version
if sys.version_info < (3, 13):
    print(
        f"Warning: Python {sys.version_info.major}.{sys.version_info.minor} detected."
    )
    print("This package is optimized for Python 3.13+")
