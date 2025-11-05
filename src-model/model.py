"""
Neural Network Model for BGT60TR13C Range-Doppler Finger Classification

This model is specifically designed for Range-Doppler map input (128×256)
from the BGT60TR13C radar simulator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Spatial attention mechanism for Range-Doppler maps"""
    
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        
    def forward(self, x):
        # Generate attention map
        attention = torch.sigmoid(self.conv(x))
        # Apply attention
        return x * attention


class RangeDopplerCNN(nn.Module):
    """
    CNN for classifying fingers from Range-Doppler maps.
    
    Input: (batch, 1, 128, 256) - [batch, channels, chirps, samples]
    Output: (batch, 5) - 5 finger classes
    """
    
    def __init__(self, num_classes=5, feature_maps=[32, 64, 128, 256], 
                 dropout=0.3, use_attention=True):
        super(RangeDopplerCNN, self).__init__()
        
        self.use_attention = use_attention
        
        # Convolutional blocks for Range-Doppler feature extraction
        self.conv1 = self._make_conv_block(1, feature_maps[0])
        self.conv2 = self._make_conv_block(feature_maps[0], feature_maps[1])
        self.conv3 = self._make_conv_block(feature_maps[1], feature_maps[2])
        self.conv4 = self._make_conv_block(feature_maps[2], feature_maps[3])
        
        # Attention mechanisms (optional)
        if use_attention:
            self.attention1 = AttentionBlock(feature_maps[0])
            self.attention2 = AttentionBlock(feature_maps[1])
            self.attention3 = AttentionBlock(feature_maps[2])
            self.attention4 = AttentionBlock(feature_maps[3])
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(feature_maps[3] * 2, 256)  # *2 for concat of avg+max pool
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Batch normalization for classifier
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block with Conv2D, BatchNorm, ReLU, Pooling"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Input: (batch, 1, 128, 256)
        
        # Conv block 1: (batch, 32, 64, 128)
        x = self.conv1(x)
        if self.use_attention:
            x = self.attention1(x)
        
        # Conv block 2: (batch, 64, 32, 64)
        x = self.conv2(x)
        if self.use_attention:
            x = self.attention2(x)
        
        # Conv block 3: (batch, 128, 16, 32)
        x = self.conv3(x)
        if self.use_attention:
            x = self.attention3(x)
        
        # Conv block 4: (batch, 256, 8, 16)
        x = self.conv4(x)
        if self.use_attention:
            x = self.attention4(x)
        
        # Global pooling (dual pooling: avg + max)
        avg_pool = self.global_avg_pool(x)  # (batch, 256, 1, 1)
        max_pool = self.global_max_pool(x)  # (batch, 256, 1, 1)
        
        # Concatenate and flatten
        x = torch.cat([avg_pool, max_pool], dim=1)  # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 512)
        
        # Fully connected classifier
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


class LightweightRangeDopplerNet(nn.Module):
    """
    Lightweight version for faster training/inference.
    Good for testing or resource-constrained environments.
    """
    
    def __init__(self, num_classes=5, dropout=0.3):
        super(LightweightRangeDopplerNet, self).__init__()
        
        # Simpler architecture
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(config):
    """
    Factory function to create model from config.
    
    Args:
        config: Dictionary with model configuration
        
    Returns:
        PyTorch model
    """
    model_name = config.get('name', 'RangeDopplerCNN')
    num_classes = config.get('num_classes', 5)
    
    if model_name == 'RangeDopplerCNN':
        return RangeDopplerCNN(
            num_classes=num_classes,
            feature_maps=config.get('feature_maps', [32, 64, 128, 256]),
            dropout=config.get('dropout', 0.3),
            use_attention=config.get('use_attention', True)
        )
    elif model_name == 'LightweightRangeDopplerNet':
        return LightweightRangeDopplerNet(
            num_classes=num_classes,
            dropout=config.get('dropout', 0.3)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing RangeDopplerCNN model...\n")
    
    # Create model
    model = RangeDopplerCNN(num_classes=5)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 256)  # BGT60 Range-Doppler format
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output[0]}")
    
    # Test probabilities
    probs = F.softmax(output, dim=1)
    print(f"\nProbabilities (first sample): {probs[0]}")
    print(f"Predicted class: {torch.argmax(probs[0]).item()}")
    
    # Test lightweight model
    print("\n" + "="*60)
    print("Testing LightweightRangeDopplerNet model...\n")
    
    model_light = LightweightRangeDopplerNet(num_classes=5)
    print(f"Model: {model_light.__class__.__name__}")
    print(f"Parameters: {count_parameters(model_light):,}")
    
    with torch.no_grad():
        output_light = model_light(x)
    
    print(f"\nOutput shape: {output_light.shape}")
    print(f"Parameters comparison: {count_parameters(model):,} vs {count_parameters(model_light):,}")
    print(f"Reduction: {(1 - count_parameters(model_light)/count_parameters(model))*100:.1f}%")
