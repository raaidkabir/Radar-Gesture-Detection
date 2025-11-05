# BGT60TR13C Training Guide

Quick guide to train the finger classification model on BGT60TR13C radar data.

## Quick Start

### 1. Generate Training Data
```bash
python data_generator.py --samples 500 --output data
```

This creates:
- `data/train_data.npz`: 80% training data (2000 samples)
- `data/val_data.npz`: 20% validation data (500 samples)

**Options:**
- `--samples N`: Samples per class (default 100)
- `--output DIR`: Output directory (default data/)
- `--seed N`: Random seed for reproducibility
- `--chirps N`: Number of chirps per sample (default 128)
- `--samples-per-chirp N`: Samples per chirp (default 256)

### 2. Train the Model
```bash
python train.py --train-data data/train_data.npz --val-data data/val_data.npz
```

**Training Options:**
- `--config config.yaml`: Config file (default config.yaml)
- `--epochs N`: Number of epochs (default 100)
- `--batch-size N`: Batch size (default 32)
- `--device cuda`: Use GPU/MPS/CPU (auto-detects by default)

**Outputs:**
- `models/best_model.pth`: Best model (highest val accuracy)
- `models/checkpoint_epoch_N.pth`: Periodic checkpoints
- `models/final_model.pth`: Last epoch model
- `models/training_history.npz`: Loss/accuracy curves

### 3. Evaluate Results
```bash
python evaluate.py --model models/best_model.pth --test-data data/val_data.npz
```

**Outputs in `results/`:**
- `confusion_matrix.png`: 5×5 confusion matrix
- `per_class_accuracy.png`: Bar chart for each finger
- `training_history.png`: Loss and accuracy curves
- `confidence_analysis.png`: Prediction confidence stats
- `evaluation_results.txt`: Text summary

## Model Architecture

**RangeDopplerCNN** (default):
- Input: (batch, 1, 128, 256) Range-Doppler maps
- 4 convolutional blocks with attention
- Feature maps: 32 → 64 → 128 → 256
- Dual pooling (avg + max)
- 3-layer classifier: 512 → 256 → 128 → 5
- ~1.3M parameters

**LightweightRangeDopplerNet** (faster):
- 3 convolutional blocks
- Feature maps: 16 → 32 → 64
- ~200K parameters
- To use: Set `model.name: lightweight` in config.yaml

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  input_shape: [128, 256]  # Range-Doppler map size
  num_classes: 5           # 5 fingers
  
model:
  name: rangedoppler       # or 'lightweight'
  feature_maps: [32, 64, 128, 256]
  dropout: 0.3
  use_attention: true
  
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: adam          # or 'sgd'
  scheduler: cosine        # or 'step'
```

## Hardware-Specific Parameters

The simulator uses **BGT60TR13C** specs:
- Frequency: 60.75 GHz center (58-63.5 GHz range)
- Bandwidth: 5.5 GHz
- Channels: 1 TX, 3 RX
- ADC: 12-bit @ 4 MSps
- Range resolution: 2.73 cm
- Detection range: 0.15-0.7 m optimal

These are hard-coded in `rcs_simulator.py` and don't need adjustment unless simulating different hardware.

## Data Augmentation

Enabled by default in config.yaml:
- **Gaussian noise**: σ=0.05 (simulates thermal noise)
- **Intensity scaling**: 90-110% (simulates power variations)
- **Range shift**: ±5 samples (simulates position uncertainty)
- **Doppler shift**: ±5 samples (simulates motion variations)

Disable: Set `augmentation.enabled: false` in config.yaml

## Tips for Good Results

1. **More data helps**: Generate 1000+ samples per class
   ```bash
   python data_generator.py --samples 1000 --output data
   ```

2. **Monitor training**: Watch for overfitting
   - Training accuracy >> validation accuracy
   - Solution: More data, higher dropout, early stopping

3. **GPU acceleration**: ~10-20x faster
   ```bash
   python train.py --device cuda
   ```
   
4. **Learning rate tuning**: If loss plateaus early
   - Increase: `learning_rate: 0.005`
   - Decrease: `learning_rate: 0.0001`

5. **Model size vs speed**:
   - RangeDopplerCNN: Best accuracy (~1.3M params)
   - LightweightRangeDopplerNet: Faster inference (~200K params)

## Example Full Workflow

```bash
# 1. Generate 1000 samples per class (5000 total)
python data_generator.py --samples 1000 --output data

# 2. Train for 100 epochs
python train.py --train-data data/train_data.npz --val-data data/val_data.npz --epochs 100

# 3. Evaluate on validation set
python evaluate.py --model models/best_model.pth --test-data data/val_data.npz --output results

# 4. Check results
open results/confusion_matrix.png
open results/training_history.png
cat results/evaluation_results.txt
```

## Expected Performance

With 500+ samples per class and default settings:
- **Training time**: ~5 min/epoch on CPU, ~30s/epoch on GPU
- **Target accuracy**: 85-95% on validation set
- **Per-class**: Middle/index fingers easiest (largest RCS), pinky hardest

**Note:** Current test run (100 samples, 2 epochs) shows 21% accuracy - this is expected with very limited data and training. Real performance requires more samples and full training.

## Troubleshooting

**Low accuracy after training:**
- Generate more data (1000+ samples/class)
- Train for full 100 epochs
- Check augmentation isn't too aggressive
- Try different learning rates

**Out of memory:**
- Reduce batch size: `--batch-size 16`
- Use lightweight model: `model.name: lightweight`

**Training too slow:**
- Use GPU: `--device cuda` or `--device mps` (Mac M1/M2)
- Reduce samples: `--samples 500`
- Use lightweight model

**Model not converging:**
- Lower learning rate: `learning_rate: 0.0001`
- Try different optimizer: `optimizer: sgd`
- Check data quality (visualize samples)
