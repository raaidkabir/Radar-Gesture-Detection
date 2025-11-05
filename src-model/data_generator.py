"""
Synthetic Data Generator for BGT60TR13C Radar-based Finger Classification

Generates realistic training data based on BGT60TR13C radar specifications:
- 60 GHz FMCW radar
- Range-Doppler processing
- Realistic SNR levels
- Multi-finger scenarios
- Various hand poses and gestures
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from rcs_simulator import BGT60RCSSimulator, BGT60TR13CConfig, FINGER_GEOMETRIES


class BGT60DataGenerator:
    """Generate synthetic radar data for finger classification training"""
    
    def __init__(
        self,
        simulator: BGT60RCSSimulator = None,
        num_chirps: int = 128,
        num_samples: int = 256,
        seed: int = 42
    ):
        """
        Initialize data generator.
        
        Args:
            simulator: RCS simulator instance
            num_chirps: Number of chirps per frame (Doppler dimension)
            num_samples: Number of ADC samples per chirp (range dimension)
            seed: Random seed for reproducibility
        """
        self.simulator = simulator or BGT60RCSSimulator()
        self.num_chirps = num_chirps
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)
        
        self.finger_labels = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.label_to_idx = {label: idx for idx, label in enumerate(self.finger_labels)}
        
    def generate_single_sample(
        self,
        finger_type: str,
        distance_range: Tuple[float, float] = (0.15, 1.0),
        velocity_range: Tuple[float, float] = (-0.5, 0.5),
        azimuth_range: Tuple[float, float] = (-30, 30),
        elevation_range: Tuple[float, float] = (-20, 20),
        flexion_range: Tuple[float, float] = (0, 60),
        add_noise: bool = True,
        add_clutter: bool = True
    ) -> Tuple[np.ndarray, int, Dict]:
        """
        Generate a single training sample for a finger.
        
        Args:
            finger_type: Type of finger
            distance_range: Min/max distance in meters
            velocity_range: Min/max velocity in m/s
            azimuth_range: Min/max azimuth angle in degrees
            elevation_range: Min/max elevation angle in degrees
            flexion_range: Min/max finger flexion in degrees
            add_noise: Whether to add thermal noise
            add_clutter: Whether to add background clutter
        
        Returns:
            Tuple of (range_doppler_map, label_idx, metadata)
        """
        # Sample random parameters
        distance = self.rng.uniform(*distance_range)
        velocity = self.rng.uniform(*velocity_range)
        azimuth = self.rng.uniform(*azimuth_range)
        elevation = self.rng.uniform(*elevation_range)
        flexion = self.rng.uniform(*flexion_range)
        
        # Generate base Range-Doppler map
        rd_map, metadata = self.simulator.generate_range_doppler_signature(
            finger_type=finger_type,
            distance_m=distance,
            velocity_ms=velocity,
            num_chirps=self.num_chirps,
            num_samples=self.num_samples
        )
        
        # Add realistic clutter (static objects, multipath)
        if add_clutter:
            # Static clutter (walls, desk, etc.)
            num_clutter = self.rng.randint(2, 6)
            for _ in range(num_clutter):
                clutter_range_bin = self.rng.randint(0, self.num_samples)
                clutter_doppler_bin = self.rng.randint(
                    self.num_chirps // 2 - 5,  # Near zero Doppler (static)
                    self.num_chirps // 2 + 5
                )
                clutter_amplitude = self.rng.uniform(0.5, 2.0)
                
                # Add Gaussian clutter blob
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        r_idx = clutter_range_bin + i
                        d_idx = clutter_doppler_bin + j
                        if 0 <= r_idx < self.num_samples and 0 <= d_idx < self.num_chirps:
                            spread = np.exp(-(i**2 + j**2) / 4)
                            rd_map[d_idx, r_idx] += clutter_amplitude * spread
        
        # Add thermal noise (modeled as complex Gaussian)
        if add_noise:
            # Noise level based on SNR from simulator
            noise_power = 10**(-metadata['rcs_data']['snr_db'] / 20)
            noise = self.rng.randn(self.num_chirps, self.num_samples) * noise_power
            rd_map += noise
        
        # Convert to magnitude (typical radar processing)
        rd_magnitude = np.abs(rd_map)
        
        # Normalize to [0, 1] range (simulate ADC quantization effect)
        rd_magnitude = np.clip(rd_magnitude, 0, None)
        max_val = np.percentile(rd_magnitude, 99.5)  # Robust normalization
        if max_val > 0:
            rd_magnitude = rd_magnitude / max_val
        
        # Quantize to 12-bit (simulate BGT60TR13C ADC)
        rd_magnitude = np.round(rd_magnitude * 4095) / 4095
        
        label_idx = self.label_to_idx[finger_type]
        
        metadata['finger_type'] = finger_type
        metadata['label_idx'] = label_idx
        metadata['distance_m'] = distance
        metadata['velocity_ms'] = velocity
        metadata['azimuth_deg'] = azimuth
        metadata['elevation_deg'] = elevation
        metadata['flexion_deg'] = flexion
        
        return rd_magnitude, label_idx, metadata
    
    def generate_dataset(
        self,
        samples_per_class: int = 1000,
        train_split: float = 0.8,
        output_path: str = 'data',
        balanced: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate a complete dataset for finger classification.
        
        Args:
            samples_per_class: Number of samples to generate per finger type
            train_split: Fraction of data for training (rest is validation)
            output_path: Directory to save the dataset
            balanced: Whether to balance classes equally
        
        Returns:
            Dictionary with train/val data and labels
        """
        print(f"Generating dataset with {samples_per_class} samples per class...")
        print(f"Total samples: {samples_per_class * len(self.finger_labels)}")
        
        all_samples = []
        all_labels = []
        all_metadata = []
        
        # Generate samples for each finger type
        for finger_type in self.finger_labels:
            print(f"\nGenerating data for {finger_type}...")
            
            for i in range(samples_per_class):
                # Vary parameters to create diverse dataset
                if i % 100 == 0:
                    print(f"  Progress: {i}/{samples_per_class}")
                
                # Sample from different distance ranges for realism
                if self.rng.rand() < 0.7:  # Most gestures at close range
                    distance_range = (0.15, 0.5)
                else:  # Some at farther range
                    distance_range = (0.5, 1.0)
                
                rd_map, label_idx, metadata = self.generate_single_sample(
                    finger_type=finger_type,
                    distance_range=distance_range,
                    velocity_range=(-0.5, 0.5),
                    azimuth_range=(-30, 30),
                    elevation_range=(-20, 20),
                    flexion_range=(0, 60),
                    add_noise=True,
                    add_clutter=True
                )
                
                all_samples.append(rd_map)
                all_labels.append(label_idx)
                all_metadata.append(metadata)
        
        # Convert to numpy arrays
        X = np.array(all_samples)  # Shape: (N, num_chirps, num_samples)
        y = np.array(all_labels)    # Shape: (N,)
        
        # Shuffle the dataset
        shuffle_idx = self.rng.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        all_metadata = [all_metadata[i] for i in shuffle_idx]
        
        # Split into train/val
        split_idx = int(len(X) * train_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"\nDataset generated:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Data shape: {X_train.shape}")
        print(f"  Classes: {self.finger_labels}")
        
        # Class distribution
        print(f"\nClass distribution (training):")
        for i, label in enumerate(self.finger_labels):
            count = np.sum(y_train == i)
            print(f"  {label}: {count} ({count/len(y_train)*100:.1f}%)")
        
        # Save dataset
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_file = output_dir / 'train_data.npz'
        val_file = output_dir / 'val_data.npz'
        
        np.savez_compressed(
            train_file,
            X=X_train,
            y=y_train,
            labels=self.finger_labels
        )
        
        np.savez_compressed(
            val_file,
            X=X_val,
            y=y_val,
            labels=self.finger_labels
        )
        
        print(f"\nDataset saved:")
        print(f"  Training: {train_file}")
        print(f"  Validation: {val_file}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'labels': self.finger_labels,
            'metadata': all_metadata
        }
    
    def generate_gesture_sequence(
        self,
        gesture_name: str,
        num_frames: int = 50,
        frame_rate: int = 20
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate a temporal sequence of radar frames for a gesture.
        
        Useful for gesture recognition (not just finger classification).
        
        Args:
            gesture_name: Name of gesture (e.g., 'swipe_right', 'tap', 'wave')
            num_frames: Number of frames in sequence
            frame_rate: Frames per second
        
        Returns:
            Tuple of (frame_sequence, frame_metadata_list)
        """
        gesture_patterns = {
            'swipe_right': self._generate_swipe_right,
            'swipe_left': self._generate_swipe_left,
            'tap': self._generate_tap,
            'wave': self._generate_wave,
            'pinch': self._generate_pinch,
        }
        
        if gesture_name not in gesture_patterns:
            raise ValueError(f"Unknown gesture: {gesture_name}")
        
        return gesture_patterns[gesture_name](num_frames, frame_rate)
    
    def _generate_swipe_right(
        self,
        num_frames: int,
        frame_rate: int
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Generate swipe right gesture (index finger moves right)"""
        frames = []
        metadata_list = []
        
        # Swipe parameters
        start_distance = 0.3  # meters
        velocity = 0.5  # m/s lateral
        
        for i in range(num_frames):
            t = i / frame_rate
            # Lateral motion (azimuth changes)
            azimuth = -20 + (40 * i / num_frames)  # -20° to +20°
            
            rd_map, label_idx, metadata = self.generate_single_sample(
                finger_type='index',
                distance_range=(start_distance - 0.05, start_distance + 0.05),
                velocity_range=(velocity - 0.1, velocity + 0.1),
                azimuth_range=(azimuth - 2, azimuth + 2),
                elevation_range=(0, 5),
                flexion_range=(0, 10)
            )
            
            frames.append(rd_map)
            metadata_list.append(metadata)
        
        return np.array(frames), metadata_list
    
    def _generate_swipe_left(self, num_frames: int, frame_rate: int):
        """Generate swipe left gesture"""
        frames, metadata = self._generate_swipe_right(num_frames, frame_rate)
        # Mirror the azimuth angles
        return frames[::-1], metadata[::-1]
    
    def _generate_tap(self, num_frames: int, frame_rate: int):
        """Generate tap gesture (finger approaches and retreats)"""
        frames = []
        metadata_list = []
        
        # Tap has approach and retreat phases
        for i in range(num_frames):
            progress = i / num_frames
            
            if progress < 0.4:  # Approach phase
                distance = 0.5 - 0.3 * (progress / 0.4)
                velocity = -0.6
            elif progress < 0.6:  # Contact phase
                distance = 0.2
                velocity = 0.0
            else:  # Retreat phase
                distance = 0.2 + 0.3 * ((progress - 0.6) / 0.4)
                velocity = 0.6
            
            rd_map, label_idx, metadata = self.generate_single_sample(
                finger_type='index',
                distance_range=(distance - 0.02, distance + 0.02),
                velocity_range=(velocity - 0.1, velocity + 0.1),
                azimuth_range=(-5, 5),
                elevation_range=(-5, 5),
                flexion_range=(0, 5)
            )
            
            frames.append(rd_map)
            metadata_list.append(metadata)
        
        return np.array(frames), metadata_list
    
    def _generate_wave(self, num_frames: int, frame_rate: int):
        """Generate wave gesture (hand moves back and forth)"""
        frames = []
        metadata_list = []
        
        for i in range(num_frames):
            t = i / frame_rate
            # Sinusoidal motion
            azimuth = 15 * np.sin(2 * np.pi * t * 0.5)  # 0.5 Hz wave
            velocity = 15 * np.cos(2 * np.pi * t * 0.5) * 2 * np.pi * 0.5 / 57.3  # derivative
            
            # Multiple fingers visible during wave
            finger_type = self.rng.choice(['index', 'middle', 'ring'])
            
            rd_map, label_idx, metadata = self.generate_single_sample(
                finger_type=finger_type,
                distance_range=(0.3, 0.4),
                velocity_range=(velocity - 0.1, velocity + 0.1),
                azimuth_range=(azimuth - 3, azimuth + 3),
                elevation_range=(-10, 10),
                flexion_range=(10, 30)
            )
            
            frames.append(rd_map)
            metadata_list.append(metadata)
        
        return np.array(frames), metadata_list
    
    def _generate_pinch(self, num_frames: int, frame_rate: int):
        """Generate pinch gesture (thumb and index approach each other)"""
        # This would require multi-finger simulation
        # Simplified version: just thumb with changing distance
        return self._generate_tap(num_frames, frame_rate)


def main():
    """Generate dataset for finger classification"""
    parser = argparse.ArgumentParser(description='Generate BGT60TR13C radar dataset')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Samples per class')
    parser.add_argument('--output', type=str, default='data',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--chirps', type=int, default=128,
                       help='Number of chirps per frame')
    parser.add_argument('--samples-per-chirp', type=int, default=256,
                       help='Number of ADC samples per chirp')
    
    args = parser.parse_args()
    
    print("BGT60TR13C Data Generator")
    print("=" * 50)
    
    # Initialize simulator and generator
    simulator = BGT60RCSSimulator()
    generator = BGT60DataGenerator(
        simulator=simulator,
        num_chirps=args.chirps,
        num_samples=args.samples_per_chirp,
        seed=args.seed
    )
    
    # Generate dataset
    dataset = generator.generate_dataset(
        samples_per_class=args.samples,
        train_split=0.8,
        output_path=args.output
    )
    
    print("\nDataset generation complete!")
    print(f"Training data shape: {dataset['X_train'].shape}")
    print(f"Validation data shape: {dataset['X_val'].shape}")


if __name__ == "__main__":
    main()
