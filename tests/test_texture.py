import numpy as np
from unittest.mock import patch, MagicMock
from counterpart.texture import TextureBaker, load_image, save_texture

def test_texture_baking():
    # Create dummy FLAME parameters
    flame_params = {
        'shape': np.zeros(300, dtype=np.float32),
        'exp': np.zeros(50, dtype=np.float32),
        'pose': np.zeros(6, dtype=np.float32)
    }
    
    # Create dummy UV coordinates
    uv_coords = np.random.rand(1000, 2).astype(np.float32)
    
    # Create dummy images
    images = ["img1.jpg", "img2.jpg", "img3.jpg"]
    
    # Mock image loading
    with patch("counterpart.texture.load_image") as mock_load:
        mock_load.return_value = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Create baker and bake texture
        baker = TextureBaker(device="cpu")
        texture = baker.bake_texture(images, flame_params, uv_coords)
        
        # Check texture shape
        assert texture.shape == (1024, 1024, 3)
        assert texture.dtype == np.uint8
        
        # Verify image loading was called for each image
        assert mock_load.call_count == len(images)
        for img in images:
            mock_load.assert_any_call(img)

def test_save_texture():
    # Create dummy texture
    texture = np.zeros((1024, 1024, 3), dtype=np.uint8)
    
    # Mock PIL Image save
    with patch("PIL.Image.Image.save") as mock_save:
        save_texture(texture, "texture.png")
        mock_save.assert_called_once_with("texture.png") 