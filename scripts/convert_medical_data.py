#!/usr/bin/env python3
"""
Convert medical imaging data (DICOM or NIfTI) to the format required by mvsplat.

This script:
1. Loads a 3D medical volume from DICOM series or NIfTI file
2. Applies intensity normalization (min-max or CT windowing)
3. Extracts axial slices (along Z-axis)
4. Saves each slice as a PNG image
5. Generates a transforms.json file with camera poses

The camera model assumes a virtual camera moving linearly along the Z-axis,
with each slice representing a different camera position.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    print("Error: pydicom not installed. Run: pip install pydicom")
    sys.exit(1)

try:
    import nibabel as nib
except ImportError:
    print("Error: nibabel not installed. Run: pip install nibabel")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not installed. Run: pip install Pillow")
    sys.exit(1)


def load_dicom_series(dicom_dir: Path) -> Tuple[np.ndarray, float]:
    """
    Load a DICOM series from a directory.
    
    Args:
        dicom_dir: Directory containing DICOM files
        
    Returns:
        volume: 3D numpy array (H, W, D)
        slice_thickness: Physical spacing between slices in mm
    """
    print(f"Loading DICOM series from {dicom_dir}...")
    
    # Find all DICOM files
    dicom_files = []
    for file_path in dicom_dir.iterdir():
        if file_path.is_file():
            try:
                pydicom.dcmread(str(file_path), stop_before_pixels=True)
                dicom_files.append(file_path)
            except (InvalidDicomError, Exception):
                continue
    
    if not dicom_files:
        raise ValueError(f"No valid DICOM files found in {dicom_dir}")
    
    print(f"  Found {len(dicom_files)} DICOM files")
    
    # Read all slices and sort by ImagePositionPatient (Z coordinate)
    slices = []
    for file_path in dicom_files:
        ds = pydicom.dcmread(str(file_path))
        slices.append(ds)
    
    # Sort by Instance Number or ImagePositionPatient Z coordinate
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        print("  Sorted by ImagePositionPatient (Z coordinate)")
    except (AttributeError, KeyError):
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
            print("  Sorted by InstanceNumber")
        except (AttributeError, KeyError):
            print("  Warning: Could not sort slices. Using file order.")
    
    # Stack into 3D volume
    volume = np.stack([s.pixel_array for s in slices], axis=-1)
    
    # Get slice thickness
    try:
        slice_thickness = float(slices[0].SliceThickness)
    except (AttributeError, KeyError):
        # Calculate from ImagePositionPatient if available
        if len(slices) > 1:
            try:
                z1 = float(slices[0].ImagePositionPatient[2])
                z2 = float(slices[1].ImagePositionPatient[2])
                slice_thickness = abs(z2 - z1)
                print(f"  Calculated slice thickness from positions: {slice_thickness:.2f} mm")
            except (AttributeError, KeyError):
                slice_thickness = 1.0
                print("  Warning: Could not determine slice thickness. Using 1.0 mm")
        else:
            slice_thickness = 1.0
            print("  Warning: Could not determine slice thickness. Using 1.0 mm")
    
    print(f"  Volume shape: {volume.shape}")
    print(f"  Slice thickness: {slice_thickness:.2f} mm")
    
    return volume, slice_thickness


def load_nifti(nifti_path: Path) -> Tuple[np.ndarray, float]:
    """
    Load a NIfTI volume.
    
    Args:
        nifti_path: Path to .nii or .nii.gz file
        
    Returns:
        volume: 3D numpy array (H, W, D)
        slice_thickness: Physical spacing between slices in mm
    """
    print(f"Loading NIfTI volume from {nifti_path}...")
    
    img = nib.load(str(nifti_path))
    volume = img.get_fdata()
    
    # Get voxel spacing (slice thickness is the Z dimension)
    voxel_spacing = img.header.get_zooms()
    slice_thickness = float(voxel_spacing[2]) if len(voxel_spacing) > 2 else 1.0
    
    print(f"  Volume shape: {volume.shape}")
    print(f"  Voxel spacing: {voxel_spacing}")
    print(f"  Slice thickness (Z): {slice_thickness:.2f} mm")
    
    return volume, slice_thickness


def normalize_intensity(
    volume: np.ndarray,
    method: str = "minmax",
    window_center: Optional[float] = None,
    window_width: Optional[float] = None
) -> np.ndarray:
    """
    Normalize volume intensity to 0-255 range.
    
    Args:
        volume: Input 3D volume
        method: "minmax" or "window"
        window_center: Center of CT window (for method="window")
        window_width: Width of CT window (for method="window")
        
    Returns:
        Normalized volume as uint8
    """
    print(f"\nNormalizing intensity using method: {method}")
    
    if method == "minmax":
        # Simple min-max normalization
        vmin, vmax = volume.min(), volume.max()
        print(f"  Original range: [{vmin:.2f}, {vmax:.2f}]")
        normalized = ((volume - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        
    elif method == "window":
        # CT windowing (Hounsfield units)
        if window_center is None or window_width is None:
            raise ValueError("window_center and window_width required for windowing")
        
        print(f"  Window center: {window_center}, Window width: {window_width}")
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        
        # Clip and normalize
        windowed = np.clip(volume, lower, upper)
        normalized = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    print(f"  Normalized range: [0, 255]")
    return normalized


def save_slices_as_png(
    volume: np.ndarray,
    output_dir: Path,
    axis: int = 2,
    resize: Optional[Tuple[int, int]] = None
) -> int:
    """
    Save volume slices as PNG images.
    
    Args:
        volume: 3D volume (H, W, D)
        output_dir: Output directory
        axis: Axis to slice along (0=sagittal, 1=coronal, 2=axial)
        resize: Optional (width, height) to resize images to
        
    Returns:
        Number of slices saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_slices = volume.shape[axis]
    print(f"\nSaving {num_slices} slices to {output_dir}...")
    
    for i in range(num_slices):
        # Extract slice
        if axis == 0:
            slice_data = volume[i, :, :]
        elif axis == 1:
            slice_data = volume[:, i, :]
        else:  # axis == 2
            slice_data = volume[:, :, i]
        
        # Convert to RGB (grayscale replicated across channels)
        rgb_slice = np.stack([slice_data] * 3, axis=-1)
        
        # Create image
        img = Image.fromarray(rgb_slice, mode='RGB')
        
        # Resize if requested
        if resize is not None:
            img = img.resize(resize, Image.Resampling.LANCZOS)
        
        # Save as PNG
        filename = f"{i+1:03d}.png"
        img.save(output_dir / filename)
        
        if (i + 1) % 10 == 0 or i == num_slices - 1:
            print(f"  Saved {i+1}/{num_slices} slices")
    
    return num_slices


def generate_transforms_json(
    num_slices: int,
    slice_thickness: float,
    image_shape: Tuple[int, int],
    output_path: Path,
    focal_length: Optional[float] = None
):
    """
    Generate transforms.json with camera poses.
    
    Assumes a virtual camera moving linearly along the Z-axis.
    
    Args:
        num_slices: Number of slices
        slice_thickness: Physical spacing between slices (mm)
        image_shape: (height, width) of images
        output_path: Path to save transforms.json
        focal_length: Camera focal length (defaults to image width)
    """
    print(f"\nGenerating transforms.json...")
    
    height, width = image_shape
    
    # Default focal length (simple pinhole camera)
    if focal_length is None:
        focal_length = float(width)
    
    # Principal point at image center
    cx, cy = width / 2.0, height / 2.0
    
    # Normalize slice thickness to a reasonable camera step
    # (convert mm to normalized units, assuming 1mm = 0.001 units)
    z_step = slice_thickness * 0.001
    
    print(f"  Image shape: {height} x {width}")
    print(f"  Focal length: {focal_length:.2f}")
    print(f"  Principal point: ({cx:.2f}, {cy:.2f})")
    print(f"  Slice thickness: {slice_thickness:.2f} mm")
    print(f"  Z-step (camera translation): {z_step:.6f} units")
    
    transforms = {"frames": []}
    
    for i in range(num_slices):
        filename = f"{i+1:03d}.png"
        
        # Camera translation along Z-axis
        # Start at z=0 and move forward
        z_position = i * z_step
        
        # 4x4 transformation matrix
        # Identity rotation, translation along Z
        transform_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, z_position],
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
                "width": width,
                "height": height
            }
        }
        
        transforms["frames"].append(frame)
    
    # Save transforms.json
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"  Saved transforms.json to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert medical imaging data to mvsplat format"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to DICOM directory or NIfTI file (.nii/.nii.gz)"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--normalization",
        type=str,
        choices=["minmax", "window"],
        default="minmax",
        help="Intensity normalization method"
    )
    parser.add_argument(
        "--window-center",
        type=float,
        default=None,
        help="CT window center (Hounsfield units, for --normalization=window)"
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=None,
        help="CT window width (Hounsfield units, for --normalization=window)"
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Camera focal length (defaults to image width)"
    )
    parser.add_argument(
        "--axis",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Slicing axis: 0=sagittal, 1=coronal, 2=axial (default: 2)"
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        help="Resize images to specific width and height (e.g., 512 512)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    
    # Validate input
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Load volume
    if input_path.is_dir():
        # DICOM series
        volume, slice_thickness = load_dicom_series(input_path)
    elif input_path.suffix in ['.nii', '.gz']:
        # NIfTI file
        volume, slice_thickness = load_nifti(input_path)
    else:
        print(f"Error: Unsupported file format. Expected DICOM directory or .nii/.nii.gz file")
        sys.exit(1)
    
    # Normalize intensity
    normalized_volume = normalize_intensity(
        volume,
        method=args.normalization,
        window_center=args.window_center,
        window_width=args.window_width
    )
    
    # Save slices as PNG
    resize_dims = tuple(args.resize) if args.resize else None
    num_slices = save_slices_as_png(
        normalized_volume, 
        output_dir, 
        axis=args.axis,
        resize=resize_dims
    )
    
    # Get image shape
    if resize_dims:
        # PIL resize is (width, height), so shape is (height, width)
        image_shape = (resize_dims[1], resize_dims[0])
    else:
        if args.axis == 0:
            image_shape = (volume.shape[1], volume.shape[2])
        elif args.axis == 1:
            image_shape = (volume.shape[0], volume.shape[2])
        else:  # axis == 2
            image_shape = (volume.shape[0], volume.shape[1])
    
    # Generate transforms.json
    transforms_path = output_dir / "transforms.json"
    generate_transforms_json(
        num_slices,
        slice_thickness,
        image_shape,
        transforms_path,
        focal_length=args.focal_length
    )
    
    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Number of slices: {num_slices}")
    print(f"  Image shape: {image_shape[0]} x {image_shape[1]}")
    print(f"\nNext steps:")
    print(f"  1. Update data_root in your training config to: {output_dir.absolute()}")
    print(f"  2. Update image_shape to: [{image_shape[0]}, {image_shape[1]}]")
    print(f"  3. Run training with: python src/train_real.py")


if __name__ == "__main__":
    main()
