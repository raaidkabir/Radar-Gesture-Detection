"""
Evaluation script for BGT60TR13C finger classification model
"""

import torch
import torch.nn.functional as F
import numpy as np
import yaml
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from model import create_model
from train import RangeDopplerDataset
from torch.utils.data import DataLoader


def evaluate_model(model, data_loader, device, labels):
    """
    Evaluate model on dataset
    
    Returns:
        accuracy, all_preds, all_targets, all_probs
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    accuracy = 100.0 * np.mean(all_preds == all_targets)
    
    return accuracy, all_preds, all_targets, all_probs


def plot_confusion_matrix(cm, labels, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


def plot_per_class_accuracy(cm, labels, output_path):
    """Plot per-class accuracy"""
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, per_class_acc, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.xlabel('Finger', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.ylim([0, 105])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class accuracy: {output_path}")


def plot_training_history(history_path, output_dir):
    """Plot training curves"""
    if not Path(history_path).exists():
        print(f"Warning: Training history not found at {history_path}")
        return
    
    history = np.load(history_path)
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    train_accs = history['train_accs']
    val_accs = history['val_accs']
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training history: {output_path}")


def analyze_predictions(probs, targets, labels, output_dir):
    """Analyze prediction confidence"""
    # Get confidence of predicted class
    max_probs = np.max(probs, axis=1)
    predicted_labels = np.argmax(probs, axis=1)
    correct = predicted_labels == targets
    
    # Separate correct and incorrect predictions
    correct_confidences = max_probs[correct]
    incorrect_confidences = max_probs[~correct]
    
    plt.figure(figsize=(12, 5))
    
    # Histogram of confidences
    plt.subplot(1, 2, 1)
    plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
    plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Confidence by class
    plt.subplot(1, 2, 2)
    class_confidences = [max_probs[targets == i] for i in range(len(labels))]
    bp = plt.boxplot(class_confidences, tick_labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    plt.ylabel('Prediction Confidence', fontsize=12)
    plt.xlabel('True Label', fontsize=12)
    plt.title('Confidence by True Class', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'confidence_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence analysis: {output_path}")
    
    # Print statistics
    print(f"\nConfidence Statistics:")
    print(f"  Correct predictions: mean={np.mean(correct_confidences):.3f}, "
          f"std={np.std(correct_confidences):.3f}")
    print(f"  Incorrect predictions: mean={np.mean(incorrect_confidences):.3f}, "
          f"std={np.std(incorrect_confidences):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate BGT60TR13C finger classifier')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--test-data', type=str, default='data/val_data.npz',
                       help='Path to test data')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda, cpu, or mps')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device_name = config.get('device', 'cuda')
        if device_name == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif device_name == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print("BGT60TR13C Finger Classification Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Test data: {args.test_data}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    model_config = checkpoint['config']['model'].copy()
    model_config['num_classes'] = checkpoint['config']['data']['num_classes']
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"\nModel loaded from epoch {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    # Load test data
    test_dataset = RangeDopplerDataset(args.test_data, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )
    
    # Get labels
    labels = config['data']['labels']
    
    # Evaluate
    print("\n" + "=" * 60)
    accuracy, preds, targets, probs = evaluate_model(model, test_loader, device, labels)
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print("\n" + "=" * 60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=labels))
    
    # Confusion matrix
    cm = confusion_matrix(targets, preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save confusion matrix
    plot_confusion_matrix(cm, labels, output_dir / 'confusion_matrix.png')
    
    # Per-class accuracy
    plot_per_class_accuracy(cm, labels, output_dir / 'per_class_accuracy.png')
    
    # Training history
    history_path = Path(config['paths']['model_dir']) / 'training_history.npz'
    plot_training_history(history_path, output_dir)
    
    # Confidence analysis
    analyze_predictions(probs, targets, labels, output_dir)
    
    # Save results to text file
    results_file = output_dir / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write("BGT60TR13C Finger Classification Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test data: {args.test_data}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(targets, preds, target_names=labels))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - confusion_matrix.png")
    print(f"  - per_class_accuracy.png")
    print(f"  - training_history.png")
    print(f"  - confidence_analysis.png")
    print(f"  - evaluation_results.txt")


if __name__ == "__main__":
    main()
