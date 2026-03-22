#!/bin/bash
# Download SKU-110K dataset
# SKU-110K: 11,762 images, ~1.73M bounding box annotations
# Official split: 8,233 train / 588 val / 2,941 test
set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "=== SKU-110K Dataset Download ==="

# Check if already downloaded
if [ -d "$DATA_DIR/SKU110K_fixed" ]; then
    echo "Dataset already exists at $DATA_DIR/SKU110K_fixed"
    exit 0
fi

echo "Downloading SKU-110K dataset..."
echo "This is a large download (~13.6 GB). Consider using a subset for faster iteration."

# Try primary download
if command -v wget &> /dev/null; then
    wget -c "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz" -P "$DATA_DIR/" || true
elif command -v curl &> /dev/null; then
    curl -L -C - -o "$DATA_DIR/SKU110K_fixed.tar.gz" "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz" || true
fi

# Extract
if [ -f "$DATA_DIR/SKU110K_fixed.tar.gz" ]; then
    echo "Extracting dataset..."
    tar -xzf "$DATA_DIR/SKU110K_fixed.tar.gz" -C "$DATA_DIR/"
    echo "Dataset extracted to $DATA_DIR/SKU110K_fixed/"
else
    echo "ERROR: Download failed. Please download manually from:"
    echo "  http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
    echo "Or use Kaggle/HuggingFace mirrors."
    exit 1
fi

echo "=== Download Complete ==="
