#!/usr/bin/env python3
import imageio.v2 as imageio
import argparse
from pathlib import Path
import glob
import os

def create_gif(input_pattern, output_file, fps=5):
    images = []
    file_list = sorted(glob.glob(input_pattern))
    
    if not file_list:
        print(f"No files found matching: {input_pattern}")
        return
    
    print(f"Found {len(file_list)} images.")
    for filename in file_list:
        images.append(imageio.imread(filename))
    
    imageio.mimsave(output_file, images, fps=fps, loop=0)
    print(f"Saved GIF to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing images")
    parser.add_argument("output_file", help="Output GIF filename")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    args = parser.parse_args()
    
    input_pattern = os.path.join(args.input_dir, "*.png")
    create_gif(input_pattern, args.output_file, args.fps)
