#!/bin/bash
# Quick start script for BGT60TR13C simulation

echo "=========================================="
echo "BGT60TR13C Finger Classification Pipeline"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "rcs_simulator.py" ]; then
    echo "Error: Please run this script from the src-model directory"
    exit 1
fi

echo "Step 1: Testing the simulator..."
python3 rcs_simulator.py
if [ $? -ne 0 ]; then
    echo "Error: Simulator test failed"
    exit 1
fi

echo ""
echo "Step 2: Generating dataset (100 samples per class for quick test)..."
python3 data_generator.py --samples 100 --output data --seed 42

if [ $? -ne 0 ]; then
    echo "Error: Data generation failed"
    exit 1
fi

echo ""
echo "Step 3: Checking generated data..."
python3 << 'EOF'
import numpy as np
data = np.load('data/train_data.npz')
print(f"Training data shape: {data['X'].shape}")
print(f"Training labels shape: {data['y'].shape}")
print(f"Classes: {data['labels']}")
print(f"Unique labels: {np.unique(data['y'])}")
print("✓ Data looks good!")
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. For larger dataset: python3 data_generator.py --samples 2000 --output data"
echo "  2. To train model: python3 train.py --train-data data/train_data.npz --val-data data/val_data.npz"
echo "  3. See README.md for full documentation"
echo "  4. See IMPLEMENTATION_SUMMARY.md for technical details"
echo ""
