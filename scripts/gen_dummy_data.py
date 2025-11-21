#!/usr/bin/env python3
"""
Generate synthetic dummy data for testing the medical imaging pipeline.
Creates random noise images and a transforms.json file with dummy camera parameters.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image


def generate_dummy_data(output_dir: Path, num_images: int = 5, image_size: tuple = (512, 512)):
    """
    Generate dummy medical imaging data for testing.
    
    Args:
        output_dir: Directory to save the generated data
        num_images: Number of images to generate
        image_size: Size of each image (height, width)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} dummy images in {output_dir}...")
    
    # Generate random noise images
    for i in range(1, num_images + 1):
        # Create random noise image (grayscale converted to RGB)
        noise = np.random.randint(0, 256, size=(*image_size, 3), dtype=np.uint8)
        img = Image.fromarray(noise, mode='RGB')
        
        # Save with zero-padded filename
        filename = f"{i:03d}.png"
        img.save(output_dir / filename)
        print(f"  Created {filename}")
    
    # Create transforms.json with dummy camera parameters
    transforms = {
        "frames": []
    }
    
    # Identity camera intrinsics (simplified)
    focal_length = 500.0
    cx, cy = image_size[1] / 2, image_size[0] / 2
    
    for i in range(1, num_images + 1):
        filename = f"{i:03d}.png"
        
        # Create a simple rotation around the object
        angle = (i - 1) * (2 * np.pi / num_images)
        radius = 2.0
        
        # Camera position in a circle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.5
        
        # Simple look-at transformation (camera looking at origin)
        # For simplicity, using identity rotation with translation
        transform_matrix = [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0]
        ]
        
        frame = {
            "file_path": filename,
            "transform_matrix": transform_matrix,
            "intrinsics": {
                "fx": focal_length,
                "fy": focal_length,
                "cx": cx,
                "cy": cy,
                "width": image_size[1],
                "height": image_size[0]
            }
        }
        
        transforms["frames"].append(frame)
    
    # Save transforms.json
    transforms_path = output_dir / "transforms.json"
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"\nCreated {transforms_path}")
    print(f"✓ Dummy data generation complete!")
    print(f"  - {num_images} images")
    print(f"  - 1 transforms.json file")
    print(f"  - Total size: ~{num_images * image_size[0] * image_size[1] * 3 / 1024:.1f} KB")


if __name__ == "__main__":
    # Generate dummy data in data/dummy_medical
    output_path = Path(__file__).parent.parent / "data" / "dummy_medical"
    generate_dummy_data(output_path, num_images=5, image_size=(512, 512))
