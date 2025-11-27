import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from algorithms.range_doppler import DopplerAlgo
from natsort import natsorted

# ==============================
# Configuration
# ==============================

radar_config = {
    "num_chirps_per_frame": 64,
    "num_samples_per_chirp": 64
}
NUM_ANTENNAS = 1

# ==============================
# Processing Function
# ==============================

def process_sample_folder(folder_path, algo_class, config, max_frames=40):
    algo = algo_class(config, NUM_ANTENNAS, mti_alpha=0.8)
    
    frame_files = natsorted(glob.glob(os.path.join(folder_path, "*.npy")))
    processed_frames = []

    for f_path in frame_files:
        raw_frame = np.squeeze(np.load(f_path))
        if raw_frame.ndim == 2:
            raw_frame = raw_frame[:, :, np.newaxis]
        rdm_complex = algo.compute_doppler_map(raw_frame[:, :, 0], i_ant=0)
        rdm_mag = np.abs(rdm_complex)
        processed_frames.append(rdm_mag)

    if len(processed_frames) == 0:
        return np.zeros((1, max_frames, 64, 64), dtype=np.float32)

    tensor_stack = np.stack(processed_frames, axis=0)
    T, H, W = tensor_stack.shape

    if T < max_frames:
        pad = max_frames - T
        tensor_stack = np.pad(tensor_stack, ((0, pad), (0,0), (0,0)))
    else:
        tensor_stack = tensor_stack[:max_frames]

    tensor_stack = tensor_stack[np.newaxis, :, :, :]
    return tensor_stack.astype(np.float32)

# ==============================
# Dataset
# ==============================

class RadarGestureDataset(Dataset):
    def __init__(self, root_dir, config, algo_class, max_frames=40):
        self.root_dir = root_dir
        self.config = config
        self.algo_class = algo_class
        self.max_frames = max_frames

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls_name in self.classes:
            folders = natsorted(glob.glob(os.path.join(root_dir, cls_name, "sample_*")))
            for f in folders:
                self.samples.append((f, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        data = process_sample_folder(sample_path, self.algo_class, self.config, self.max_frames)
        data = (data - np.mean(data)) / (np.std(data) + 1e-6)
        return torch.from_numpy(data), label

# ==============================
# Model
# ==============================

class Radar3DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,2,2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )

        self.gap = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        features = self.features(x)
        features = self.gap(features)
        features = features.squeeze(-1).squeeze(-1)
        features = features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]
        return self.classifier(lstm_out)

# ==============================
# Load trained model
# ==============================

data_root = "data"
print("Loading dataset...", flush=True)
dataset = RadarGestureDataset(data_root, radar_config, DopplerAlgo, max_frames=40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = Radar3DCNN(num_classes=len(dataset.classes)).to(device)
model.load_state_dict(torch.load("radar_gesture_cnn.pth", map_location=device))
model.eval()

print(f"Loaded model from radar_gesture_cnn.pth")
print(f"Classes: {dataset.classes}\n")

# ==============================
# Test on multiple random samples
# ==============================

num_predictions = 10
correct_predictions = 0

print(f"--- {num_predictions} RANDOM SAMPLE PREDICTIONS ---")
for i in range(num_predictions):
    sample_idx = np.random.randint(0, len(dataset))
    sample_data, sample_label = dataset[sample_idx]
    sample_data = sample_data.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sample_data)
        pred = torch.argmax(output, dim=1).item()
    
    is_correct = "✓" if pred == sample_label else "✗"
    if pred == sample_label:
        correct_predictions += 1
    
    print(f"{i+1}. True: {dataset.classes[sample_label]:12s} | Pred: {dataset.classes[pred]:12s} {is_correct}")

accuracy = correct_predictions / num_predictions
print(f"\nSample Prediction Accuracy: {correct_predictions}/{num_predictions} ({accuracy:.1%})")
