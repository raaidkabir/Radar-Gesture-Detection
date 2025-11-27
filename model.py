import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from algorithms.range_doppler import DopplerAlgo
from natsort import natsorted
from tqdm import tqdm

# ==============================
# 1. Radar / Config
# ==============================

radar_config = {
    "num_chirps_per_frame": 64,
    "num_samples_per_chirp": 64
}
NUM_ANTENNAS = 1


# ==============================
# 2. Processing Function
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

    tensor_stack = tensor_stack[np.newaxis, :, :, :]  # (1, T, H, W)
    return tensor_stack.astype(np.float32)


# ==============================
# 3. Dataset
# ==============================

class RadarGestureDataset(Dataset):
    def __init__(self, root_dir, config, algo_class, max_frames=40):
        self.root_dir = root_dir
        self.config = config
        self.algo_class = algo_class
        self.max_frames = max_frames

        self.classes = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
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

        # new instance every single __getitem__
        algo_class = self.algo_class
        config = self.config
        data = process_sample_folder(sample_path, algo_class, config, self.max_frames)

        data = (data - np.mean(data)) / (np.std(data) + 1e-6)

        return torch.from_numpy(data), label


# ==============================
# 4. Model
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

        # Global average pooling to reduce spatial dimensions
        self.gap = nn.AdaptiveAvgPool3d((None, 1, 1))  # Keep time, pool spatial
        
        # LSTM for temporal modeling (input: 64 features per timestep)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, T, H, W)
        B, C, T, H, W = x.shape
        
        # Extract spatial features for each time step
        features = self.features(x)  # (B, 64, T', H', W')
        
        # Apply global average pooling on spatial dimensions
        features = self.gap(features)  # (B, 64, T', 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 64, T')
        features = features.permute(0, 2, 1)  # (B, T', 64)
        
        # LSTM processes temporal sequence
        lstm_out, _ = self.lstm(features)  # (B, T', 128)
        
        # Use last time step output
        lstm_out = lstm_out[:, -1, :]  # (B, 128)
        
        return self.classifier(lstm_out)


# ==============================
# 5. Training Script
# ==============================

data_root = "data"
print("Loading dataset... (this may take a few minutes)", flush=True)
dataset = RadarGestureDataset(data_root, radar_config, DopplerAlgo, max_frames=40)
print(f"Dataset loaded: {len(dataset)} samples", flush=True)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Radar3DCNN(num_classes=len(dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Classes: {dataset.classes}", flush=True)
print(f"Training on {device}...", flush=True)
print(f"Total samples: {len(dataset)}, Train: {train_size}, Val: {val_size}", flush=True)

num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()

            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total

    print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {val_acc:.3f}", flush=True)

print("Training Complete.")

# ==============================
# 6. Save the Model
# ==============================

torch.save(model.state_dict(), "radar_gesture_cnn.pth")
print("Saved model to radar_gesture_cnn.pth")


# ==============================
# 7. Predict on Multiple Random Samples
# ==============================

model.eval()
num_predictions = 10
correct_predictions = 0

print(f"\n--- {num_predictions} RANDOM SAMPLE PREDICTIONS ---", flush=True)
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
    
    print(f"{i+1}. True: {dataset.classes[sample_label]:12s} | Pred: {dataset.classes[pred]:12s} {is_correct}", flush=True)

accuracy = correct_predictions / num_predictions
print(f"\nSample Prediction Accuracy: {correct_predictions}/{num_predictions} ({accuracy:.1%})", flush=True)
