"""
Simple Pattern-Based Data Generator for Testing
Creates clearly distinguishable patterns to verify the training pipeline works.

Intended for sanity checks only. For realistic data, use data_generator.py
"""

import numpy as np
from pathlib import Path
import argparse


def generate_simple_dataset(
    samples_per_class=1000,
    num_chirps=128,
    num_samples=256,
    train_split=0.8,
    output_path='data',
    seed=42
):
    """
    Generate simple synthetic data with VERY distinct patterns for each finger.
    This is for testing - to verify the model can learn when patterns are clear.
    """
    rng = np.random.RandomState(seed)
    
    finger_labels = ['thumb', 'index', 'middle', 'ring', 'pinky']
    
    # Define VERY distinct patterns for each finger
    # Pattern strategy: Different combinations of range bins and intensities
    patterns = {
        'thumb': {
            'range_bins': [40, 50, 60],  # Close range cluster
            'doppler_offset': 10,  # Approaching motion
            'intensity': 5.0,
            'width': 8
        },
        'index': {
            'range_bins': [80, 90, 100],  # Mid-close range
            'doppler_offset': 5,
            'intensity': 4.0,
            'width': 6
        },
        'middle': {
            'range_bins': [120, 130, 140],  # Mid range
            'doppler_offset': 0,  # Static
            'intensity': 4.5,
            'width': 7
        },
        'ring': {
            'range_bins': [160, 170, 180],  # Mid-far range
            'doppler_offset': -5,
            'intensity': 3.5,
            'width': 6
        },
        'pinky': {
            'range_bins': [200, 210, 220],  # Far range
            'doppler_offset': -10,  # Receding motion
            'intensity': 3.0,
            'width': 5
        }
    }
    
    all_samples = []
    all_labels = []
    
    print(f"Generating SIMPLE test dataset: {samples_per_class} samples per class")
    print("=" * 60)
    
    for label_idx, finger in enumerate(finger_labels):
        print(f"\nGenerating {finger}...")
        pattern = patterns[finger]
        
        for i in range(samples_per_class):
            if i % 200 == 0:
                print(f"  Progress: {i}/{samples_per_class}")
            
            # Create base noise
            rd_map = rng.randn(num_chirps, num_samples) * 0.1
            
            # Add finger-specific pattern
            doppler_center = num_chirps // 2 + pattern['doppler_offset']
            
            for range_bin in pattern['range_bins']:
                # Add random variation to prevent exact memorization
                r_var = rng.randint(-5, 6)
                d_var = rng.randint(-3, 4)
                
                r_center = np.clip(range_bin + r_var, 0, num_samples - 1)
                d_center = np.clip(doppler_center + d_var, 0, num_chirps - 1)
                
                # Add Gaussian peak
                for di in range(-pattern['width'], pattern['width'] + 1):
                    for ri in range(-pattern['width'], pattern['width'] + 1):
                        d_idx = d_center + di
                        r_idx = r_center + ri
                        
                        if 0 <= d_idx < num_chirps and 0 <= r_idx < num_samples:
                            dist = np.sqrt(di**2 + ri**2)
                            if dist < pattern['width']:
                                amplitude = pattern['intensity'] * np.exp(-(dist**2) / (2 * (pattern['width']/2)**2))
                                rd_map[d_idx, r_idx] += amplitude
            
            # Add random noise
            rd_map += rng.randn(num_chirps, num_samples) * 0.2
            
            # Clip to positive values
            rd_map = np.clip(rd_map, 0, None)
            
            all_samples.append(rd_map)
            all_labels.append(label_idx)
    
    # Convert to arrays
    X = np.array(all_samples, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    
    # Shuffle
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split train/val
    n_train = int(len(X) * train_split)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # Save
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save compressed for consistency with main generator
    np.savez_compressed(output_dir / 'train_data.npz', X=X_train, y=y_train)
    np.savez_compressed(output_dir / 'val_data.npz', X=X_val, y=y_val)
    
    print(f"\n{'='*60}")
    print("Dataset Summary:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Data range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  Data mean: {X.mean():.2f}, std: {X.std():.2f}")
    print(f"\nClass distribution (training):")
    for i, finger in enumerate(finger_labels):
        count = np.sum(y_train == i)
        print(f"  {finger}: {count} ({100*count/len(y_train):.1f}%)")
    print(f"\nSaved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Generate simple synthetic dataset for sanity checks')
    parser.add_argument('--samples', type=int, default=1000, help='Samples per class')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--chirps', type=int, default=128, help='Number of chirps (doppler bins)')
    parser.add_argument('--samples-per-chirp', type=int, default=256, help='Number of samples per chirp (range bins)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    generate_simple_dataset(
        samples_per_class=args.samples,
        num_chirps=args.chirps,
        num_samples=args.samples_per_chirp,
        train_split=0.8,
        output_path=args.output,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
