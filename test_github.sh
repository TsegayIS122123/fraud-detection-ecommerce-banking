#!/bin/bash
echo "Simulating GitHub Actions Test..."

# Create virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install minimal packages
pip install pandas numpy scikit-learn pytest

# Run tests
python -m pytest tests/test_basic.py -v

# Cleanup
deactivate
rm -rf test_env

echo "If all tests pass, GitHub Actions should work!"
