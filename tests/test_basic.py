def test_imports():
    """Test that main packages can be imported."""
    import pandas as pd
    import numpy as np
    import sklearn
    import xgboost
    
    assert pd.__version__ is not None
    assert np.__version__ is not None
    assert sklearn.__version__ is not None
    assert xgboost.__version__ is not None

def test_data_structure():
    """Test basic data structure."""
    import pandas as pd
    
    # Create test data
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    
    assert data.shape == (3, 3)
    assert 'target' in data.columns

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])