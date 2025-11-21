# Walkthrough: Refactoring mvsplat for Medical Imaging

I have successfully refactored the `mvsplat` repository to be a static-image-only solver tailored for medical imaging.

## Changes
- **Cleanup**: Removed all video processing code, `webdataset` support, and unnecessary scripts.
- **Dependencies**: Streamlined `requirements.txt` and removed `moviepy`, `lpips`, `sk-video`.
- **Dataset**: Implemented `MedicalDataset` in `src/dataset/dataset_medical.py` to load static images and camera poses from `transforms.json`.
- **Configuration**: Created `config/medical.yaml` and updated `src/main.py` to use it as the default.
- **Imports**: Enforced explicit imports by clearing `__init__.py` files and updating source code.
- **Output**: Simplified output to save RGB images, Depth maps, and Point Clouds (`.ply`), removing video generation.

## Verification
- **Environment**: Created a new conda environment `mvsplat_medical` with PyTorch (CUDA 11.8) and all dependencies.
- **Execution**: Verified that `python -m src.main --help` runs successfully, loading the new `medical.yaml` configuration.

## How to Run
1. Activate the environment:
   ```bash
   conda activate mvsplat_medical
   ```
2. Run training (requires a valid dataset path in `config/medical.yaml`):
   ```bash
   python -m src.main dataset_path=/path/to/dataset
   ```
