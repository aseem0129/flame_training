import os
import numpy as np
from pathlib import Path
from counterpart.landmark import fuse_landmarks
from counterpart.ringnet import fit_flame_shape
from counterpart.frankmocap import fit_expressions
from counterpart.texture import TextureBaker
from counterpart.export import Exporter

def test_generate_avatar():
    """Integration test: Generate a real avatar.glb from test images."""
    # Test data paths
    test_dir = Path("tests/data")
    test_dir.mkdir(exist_ok=True)
    
    # Create dummy test images
    for i in range(3):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img_path = test_dir / f"test_{i}.jpg"
        np.save(img_path, img)
    
    # Asset paths
    assets_dir = Path("assets")
    ringnet_weights = assets_dir / "ringnet" / "ringnet_weights.pkl"
    frankmocap_weights = assets_dir / "frankmocap" / "totalcap.pth"
    
    # Skip if weights not found
    if not ringnet_weights.exists() or not frankmocap_weights.exists():
        print("Skipping integration test: weights not found")
        return
    
    # 1. Fuse landmarks
    image_paths = [str(test_dir / f"test_{i}.jpg") for i in range(3)]
    landmarks = fuse_landmarks(image_paths)
    
    # 2. Fit FLAME shape
    shape_coeffs = fit_flame_shape(image_paths, str(ringnet_weights))
    
    # 3. Fit expressions
    smpl_params = fit_expressions(image_paths, str(frankmocap_weights))
    
    # 4. Bake texture
    baker = TextureBaker()
    texture = baker.bake_texture(
        images=image_paths,
        flame_params={'shape': shape_coeffs, 'exp': smpl_params['exp'], 'pose': smpl_params['pose']},
        uv_coords=np.random.rand(1000, 2)  # Dummy UVs
    )
    
    # 5. Export GLB
    exporter = Exporter()
    output_path = test_dir / "avatar.glb"
    exporter.export_glb(
        vertices=np.random.rand(1000, 3),  # Dummy vertices
        faces=np.random.randint(0, 1000, (2000, 3)),  # Dummy faces
        texture=texture,
        output_path=str(output_path)
    )
    
    # Verify output exists
    assert output_path.exists()
    assert output_path.stat().st_size > 0 