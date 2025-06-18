import numpy as np
from unittest.mock import patch
from counterpart.ringnet import fit_flame_shape

def test_fit_flame_shape_mean():
    # Mock predictor to return different vectors for each image
    dummy1 = np.ones(300, dtype=np.float32)
    dummy2 = np.zeros(300, dtype=np.float32)
    dummy3 = np.full(300, 2.0, dtype=np.float32)
    with patch("counterpart.ringnet.RingNetPredictor.predict", side_effect=[dummy1, dummy2, dummy3]):
        result = fit_flame_shape(["img1.jpg", "img2.jpg", "img3.jpg"], "weights.pkl")
        assert result.shape == (300,)
        # The mean should be (1 + 0 + 2) / 3 = 1.0 for all elements
        assert np.allclose(result, 1.0) 