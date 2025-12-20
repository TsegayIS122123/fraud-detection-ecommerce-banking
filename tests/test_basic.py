"""Basic tests for the fraud detection project."""

import sys
import os
import importlib.metadata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_imports():
    """Test that main packages can be imported."""
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost
        import matplotlib.pyplot as plt

        # Get versions
        pd_version = pd.__version__
        np_version = np.__version__
        sklearn_version = sklearn.__version__
        xgboost_version = xgboost.__version__

        print(f"pandas: {pd_version}")
        print(f"numpy: {np_version}")
        print(f"scikit-learn: {sklearn_version}")
        print(f"xgboost: {xgboost_version}")

        assert pd_version is not None
        assert np_version is not None
        assert sklearn_version is not None
        assert xgboost_version is not None
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False


def test_data_structure():
    """Test basic data structure."""
    try:
        import pandas as pd

        # Create test data
        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )

        assert data.shape == (3, 3)
        assert "target" in data.columns
        print("Data structure test passed")
        return True
    except Exception as e:
        print(f"Data structure test failed: {e}")
        return False


def test_project_structure():
    """Test that project structure exists."""
    try:
        import os

        required_dirs = ["data/raw", "data/processed", "notebooks", "src", "tests"]
        required_files = ["requirements.txt", "README.md", "pyproject.toml"]

        for directory in required_dirs:
            if not os.path.exists(directory):
                print(f"Missing directory: {directory}")
                return False

        for file in required_files:
            if not os.path.exists(file):
                print(f"Missing file: {file}")
                return False

        print("Project structure test passed")
        return True
    except Exception as e:
        print(f"Project structure test failed: {e}")
        return False


def test_python_version():
    """Test Python version compatibility."""
    import sys

    python_version = sys.version_info
    print(
        f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    # Check if it's Python 3.13 or higher
    if python_version.major == 3 and python_version.minor >= 13:
        print("Python 3.13+ detected - compatible")
        return True
    else:
        print(
            f"Python {python_version.major}.{python_version.minor} detected - may have compatibility issues"
        )
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running Fraud Detection Project Tests")
    print("=" * 60)

    tests = [
        test_python_version,
        test_imports,
        test_data_structure,
        test_project_structure,
    ]

    results = []
    for test_func in tests:
        test_name = test_func.__name__
        print(f"\nRunning {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} - {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ FAILED - {test_name} with error: {e}")

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed successfully!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    # Run tests directly when script is executed
    success = run_all_tests()
    sys.exit(0 if success else 1)
