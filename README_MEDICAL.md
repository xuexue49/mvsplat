# Medical Imaging with MVSplat

This guide explains how to use MVSplat with real medical imaging data (DICOM or NIfTI).

## Overview

The workflow consists of three main steps:
1. **Install dependencies** for medical imaging
2. **Convert your data** from DICOM/NIfTI to the required format
3. **Run production training** on the converted data

---

## 1. Installation

### Install Medical Imaging Dependencies

The medical data conversion script requires additional libraries:

```bash
pip install pydicom nibabel Pillow
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import pydicom, nibabel; print('✓ Medical imaging libraries installed')"
```

---

## 2. Data Conversion

The `convert_medical_data.py` script converts 3D medical volumes into the format required by MVSplat:
- **Input:** DICOM series (directory) or NIfTI file (`.nii` / `.nii.gz`)
- **Output:** PNG slices + `transforms.json` with camera poses

### Basic Usage

#### For DICOM Series

```bash
python scripts/convert_medical_data.py \
    /path/to/dicom/directory \
    data/real_patient_001
```

#### For NIfTI Volume

```bash
python scripts/convert_medical_data.py \
    /path/to/volume.nii.gz \
    data/real_patient_001
```

### Advanced Options

#### CT Windowing (Hounsfield Units)

For CT scans, you can apply windowing to enhance specific tissue types:

```bash
# Soft tissue window (center=40, width=400)
python scripts/convert_medical_data.py \
    /path/to/ct_scan.nii.gz \
    data/ct_soft_tissue \
    --normalization window \
    --window-center 40 \
    --window-width 400

# Bone window (center=300, width=1500)
python scripts/convert_medical_data.py \
    /path/to/ct_scan.nii.gz \
    data/ct_bone \
    --normalization window \
    --window-center 300 \
    --window-width 1500

# Lung window (center=-600, width=1500)
python scripts/convert_medical_data.py \
    /path/to/ct_scan.nii.gz \
    data/ct_lung \
    --normalization window \
    --window-center -600 \
    --window-width 1500
```

#### Custom Slicing Axis

By default, slices are extracted along the axial plane (Z-axis). You can change this:

```bash
# Sagittal slices (X-axis)
python scripts/convert_medical_data.py \
    /path/to/volume.nii.gz \
    data/sagittal_slices \
    --axis 0

# Coronal slices (Y-axis)
python scripts/convert_medical_data.py \
    /path/to/volume.nii.gz \
    data/coronal_slices \
    --axis 1

# Axial slices (Z-axis, default)
python scripts/convert_medical_data.py \
    /path/to/volume.nii.gz \
    data/axial_slices \
    --axis 2
```

#### Custom Focal Length

```bash
python scripts/convert_medical_data.py \
    /path/to/volume.nii.gz \
    data/custom_focal \
    --focal-length 800.0
```

### Understanding the Output

After conversion, your output directory will contain:

```
data/real_patient_001/
├── 001.png          # First slice
├── 002.png          # Second slice
├── ...
├── 128.png          # Last slice
└── transforms.json  # Camera poses and intrinsics
```

The script will print important information:

```
Loading NIfTI volume from /path/to/volume.nii.gz...
  Volume shape: (512, 512, 128)
  Voxel spacing: (0.5, 0.5, 1.0)
  Slice thickness (Z): 1.00 mm

Normalizing intensity using method: minmax
  Original range: [-1024.00, 3071.00]
  Normalized range: [0, 255]

Saving 128 slices to data/real_patient_001...
  Saved 128/128 slices

Generating transforms.json...
  Image shape: 512 x 512
  Focal length: 512.00
  Principal point: (256.00, 256.00)
  Slice thickness: 1.00 mm
  Z-step (camera translation): 0.001000 units
  Saved transforms.json to data/real_patient_001/transforms.json

✓ Conversion complete!
  Output directory: data/real_patient_001
  Number of slices: 128
  Image shape: 512 x 512

Next steps:
  1. Update data_root in your training config to: /absolute/path/to/data/real_patient_001
  2. Update image_shape to: [512, 512]
  3. Run training with: python src/train_real.py
```

**Important values to note:**
- **Image shape:** Update `image_shape` in `src/train_real.py`
- **Number of slices:** Determines training duration
- **Z-step:** Physical spacing between camera positions

---

## 3. Production Training

### Configure Training

Edit `src/train_real.py` to match your dataset:

```python
@dataclass
class ProductionConfig:
    # Update these values!
    dataset_path: str = "data/real_patient_001"  # Your converted data
    image_shape: list[int] = field(default_factory=lambda: [512, 512])  # From conversion output
    
    # Training parameters
    max_steps: int = 30000
    val_check_interval: int = 500
    batch_size_train: int = 1  # Typically 1 for high-res medical images
    
    # WandB logging
    use_wandb: bool = True
    wandb_project: str = "mvsplat-medical"
    wandb_entity: str = "your-username"  # Optional
```

### Run Training

```bash
python src/train_real.py
```

### Monitor Training

If WandB is enabled, you can monitor training at:
```
https://wandb.ai/<your-entity>/mvsplat-medical
```

Otherwise, check the local logs in:
```
outputs/production/runs/run_medical-production/
```

### Checkpoints

Checkpoints are saved to:
```
outputs/production/runs/run_medical-production/checkpoints/
```

The script keeps the top 3 checkpoints based on training steps.

### Resume Training

To resume from a checkpoint:

```python
# In src/train_real.py
load_checkpoint: str = "outputs/production/runs/run_medical-production/checkpoints/step_010000.ckpt"
resume_training: bool = True
```

---

## 4. Common Workflows

### Multiple Patients

Process multiple patients by creating separate directories:

```bash
# Patient 1
python scripts/convert_medical_data.py \
    /data/patient_001/dicom \
    data/patient_001

# Patient 2
python scripts/convert_medical_data.py \
    /data/patient_002/dicom \
    data/patient_002

# Train on patient 1
# (Update dataset_path in src/train_real.py to "data/patient_001")
python src/train_real.py
```

### Different Tissue Types (CT)

Create multiple datasets with different windowing:

```bash
# Soft tissue
python scripts/convert_medical_data.py \
    ct_scan.nii.gz \
    data/ct_soft_tissue \
    --normalization window \
    --window-center 40 \
    --window-width 400

# Bone
python scripts/convert_medical_data.py \
    ct_scan.nii.gz \
    data/ct_bone \
    --normalization window \
    --window-center 300 \
    --window-width 1500
```

### Quick Test Run

Before running a full training, test with a small subset:

```bash
# Convert only first 20 slices (manually copy from full dataset)
mkdir -p data/test_subset
cp data/real_patient_001/{001..020}.png data/test_subset/
cp data/real_patient_001/transforms.json data/test_subset/

# Edit transforms.json to keep only first 20 frames
# Then run a short training
# (Update max_steps to 100 in src/train_real.py)
python src/train_real.py
```

---

## 5. Troubleshooting

### "No valid DICOM files found"

- Ensure the directory contains `.dcm` files
- Check file permissions
- Try a different DICOM directory

### "Could not determine slice thickness"

- The script will default to 1.0 mm
- You can manually verify spacing in the DICOM metadata
- This affects camera pose spacing but shouldn't break training

### Out of Memory (OOM)

If you encounter OOM errors during training:

1. **Reduce image resolution** during conversion:
   ```bash
   # Resize images after conversion
   mogrify -resize 256x256 data/patient_001/*.png
   # Update image_shape to [256, 256] in train_real.py
   ```

2. **Reduce batch size** (already at 1, can't go lower)

3. **Reduce model size** in `src/train_real.py`:
   ```python
   d_feature: int = 32  # Default: 64
   num_depth_candidates: int = 32  # Default: 64
   ```

### Training is too slow

- Ensure `num_workers > 0` (default: 4)
- Use `persistent_workers=True` (default)
- Check GPU utilization with `nvidia-smi`
- Consider reducing `val_check_interval` to save time

---

## 6. Expected Results

After training completes, you should have:

1. **Checkpoints:** `outputs/production/runs/run_medical-production/checkpoints/`
2. **Training logs:** WandB dashboard or local logs
3. **Reconstructed images:** Generated during validation

To visualize results, you can run inference on the test set (see main documentation).

---

## 7. Next Steps

- **Experiment with hyperparameters** (learning rate, depth candidates, etc.)
- **Try different normalization methods** for your specific imaging modality
- **Combine multiple views** by adjusting `num_context_views`
- **Evaluate reconstruction quality** using test metrics

For more advanced usage, refer to the main MVSplat documentation.
