import numpy as np
from unittest.mock import patch
from counterpart.frankmocap import fit_expressions

def test_fit_expressions_mean():
    # Mock predictor to return different vectors for each image
    dummy1 = {
        'pose': np.zeros(72, dtype=np.float32),
        'betas': np.zeros(10, dtype=np.float32),
        'exp': np.ones(10, dtype=np.float32)
    }
    dummy2 = {
        'pose': np.ones(72, dtype=np.float32),
        'betas': np.ones(10, dtype=np.float32),
        'exp': np.zeros(10, dtype=np.float32)
    }
    dummy3 = {
        'pose': np.full(72, 2.0, dtype=np.float32),
        'betas': np.full(10, 2.0, dtype=np.float32),
        'exp': np.full(10, 2.0, dtype=np.float32)
    }
    
    with patch("counterpart.frankmocap.FrankMocapPredictor.predict", 
               side_effect=[dummy1, dummy2, dummy3]):
        result = fit_expressions(["img1.jpg", "img2.jpg", "img3.jpg"], "weights.pkl")
        
        # Check shapes
        assert result['pose'].shape == (72,)
        assert result['betas'].shape == (10,)
        assert result['exp'].shape == (10,)
        
        # Check means
        # pose: (0 + 1 + 2) / 3 = 1.0
        assert np.allclose(result['pose'], 1.0)
        # betas: (0 + 1 + 2) / 3 = 1.0
        assert np.allclose(result['betas'], 1.0)
        # exp: (1 + 0 + 2) / 3 = 1.0
        assert np.allclose(result['exp'], 1.0) 