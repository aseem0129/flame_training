"""
Counterpart CLI for generating avatars from images.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from .landmark import fuse_landmarks
from .ringnet import fit_flame_shape
from .frankmocap import fit_expressions
from .texture import TextureBaker
from .export import Exporter

def main():
    parser = argparse.ArgumentParser(description="Generate a 3D avatar from face images")
    parser.add_argument("images", nargs="+", help="Input image paths")
    parser.add_argument("--output", "-o", default="avatar.glb", help="Output GLB file path")
    parser.add_argument("--ringnet-weights", default="assets/ringnet/ringnet_weights.pkl",
                      help="Path to RingNet weights")
    parser.add_argument("--frankmocap-weights", default="assets/frankmocap/totalcap.pth",
                      help="Path to FrankMocap weights")
    args = parser.parse_args()

    # Verify input images exist
    for img_path in args.images:
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

    print("1. Fusing landmarks...")
    landmarks = fuse_landmarks(args.images)
    
    print("2. Fitting FLAME shape...")
    shape_coeffs = fit_flame_shape(args.images, args.ringnet_weights)
    
    print("3. Fitting expressions...")
    smpl_params = fit_expressions(args.images, args.frankmocap_weights)
    
    print("4. Baking texture...")
    baker = TextureBaker()
    texture = baker.bake_texture(
        images=args.images,
        flame_params={'shape': shape_coeffs, 'exp': smpl_params['exp'], 'pose': smpl_params['pose']},
        uv_coords=np.random.rand(1000, 2)  # Dummy UVs for now
    )
    # Save the baked texture as a PNG
    Image.fromarray(texture).save("baked_texture.png")
    print("Texture image saved as baked_texture.png")
    
    print("5. Exporting GLB...")
    exporter = Exporter()
    exporter.export_glb(
        vertices=np.random.rand(1000, 3),  # Dummy vertices for now
        faces=np.random.randint(0, 1000, (2000, 3)),  # Dummy faces for now
        texture=texture,
        output_path=args.output
    )
    
    print(f"✅ Avatar generated: {args.output}")
    print("You can view it by opening viewer/index.html in your browser and dragging the GLB file onto it.")

if __name__ == "__main__":
    main() 