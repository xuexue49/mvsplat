#!/bin/bash
set -e

# Initialize conda
eval "$(conda shell.bash hook)"

# Create environment if it doesn't exist, otherwise just activate
if ! conda info --envs | grep -q "mvsplat_medical"; then
    conda create -n mvsplat_medical python=3.10 -y
fi

# Activate environment
conda activate mvsplat_medical

# Install PyTorch (idempotent)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt

# Install hydra-core explicitly
pip install hydra-core

# Install rasterizer with no build isolation
pip install --no-build-isolation git+https://github.com/dcharatan/diff-gaussian-rasterization-modified

echo "Environment setup complete."
