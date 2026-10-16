"""
Evaluate the trained model on the held-out test set: overall accuracy,
per-class precision/recall/F1, and a confusion matrix heatmap.

Run this after train.py has produced best_model.pth.

Usage:
    py -3.11 -m pip install torch torchvision scikit-learn pandas pillow matplotlib seaborn
    py -3.11 evaluate.py

Output:
    confusion_matrix.png
    classification_report.txt
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dataset import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, num_classes: int) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model, checkpoint


def main():
    _, _, test_loader, label_map = get_dataloaders(batch_size=32, num_workers=0)
    num_classes = len(label_map)
    idx_to_series = {v: k for k, v in label_map.items()}

    model, checkpoint = load_model("best_model.pth", num_classes)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.4f})\n")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    class_names = [idx_to_series[i] for i in range(num_classes)]

    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Wrote classification_report.txt")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted series")
    plt.ylabel("True series")
    plt.title("Confusion matrix - test set")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Wrote confusion_matrix.png")

    overall_acc = (all_preds == all_labels).mean()
    print(f"\nOverall test accuracy: {overall_acc:.4f}")
    print("\nLook at the confusion matrix for which series get confused with")
    print("each other - visually/stylistically adjacent eras (e.g. Sun & Moon")
    print("vs Sword & Shield) are the most likely confusions, and that's a")
    print("genuinely interesting thing to discuss in your writeup either way.")


if __name__ == "__main__":
    main()