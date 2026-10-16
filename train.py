"""
Train the Pokémon TCG series classifier.

Uses a pretrained ResNet18 backbone (frozen early layers, fine-tuned later
layers + new classification head) rather than training a CNN fully from
scratch - with ~21k images across 19 imbalanced classes, transfer learning
from ImageNet features gives a much stronger starting point than random
weights, especially for the small classes (McDonald's Collection: 88 images
is not enough to learn good low-level filters from zero).

Uses weighted CrossEntropyLoss so mistakes on rare classes are penalized
more than mistakes on common ones (Sword & Shield: 3679 images vs
McDonald's Collection: 88 images is a ~42x imbalance).

Run this locally (needs a GPU to be practical; will run on CPU but slowly).

Usage:
    py -3.11 -m pip install torch torchvision scikit-learn pandas pillow matplotlib
    py -3.11 train.py

Output:
    best_model.pth           -- checkpoint with the best validation accuracy
    training_history.csv     -- per-epoch train/val loss and accuracy
    training_curves.png      -- loss/accuracy plot over epochs
"""

import json
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

from dataset import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_WORKERS = 4  # lower to 0 on Windows if you hit multiprocessing errors


def build_model(num_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze the early conv layers (generic edge/texture/color features that
    # transfer well from ImageNet regardless of domain); fine-tune layer3,
    # layer4, and the new head, since those capture higher-level features
    # that need to adapt to what actually distinguishes card series (art
    # style, border design, holo patterns, layout conventions per era).
    for name, param in model.named_parameters():
        if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Replace the final classification head for our 19 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def compute_class_weights(train_csv: str, label_map: dict) -> torch.Tensor:
    """Inverse-frequency weights: rare classes get a higher weight so the
    loss penalizes mistakes on them more heavily."""
    df = pd.read_csv(train_csv)
    counts = df["series"].value_counts()
    num_classes = len(label_map)
    weights = torch.zeros(num_classes)
    total = len(df)
    for series, idx in label_map.items():
        class_count = counts.get(series, 1)  # avoid div-by-zero if a class is somehow absent
        weights[idx] = total / (num_classes * class_count)
    return weights.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over the data. If optimizer is given, trains; otherwise
    just evaluates (used for both val and test)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cpu":
        print("WARNING: no GPU detected. Training on CPU will be slow for")
        print("21k images - consider Google Colab (free GPU) if this environment")
        print("has no CUDA-capable card.\n")

    train_loader, val_loader, test_loader, label_map = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    num_classes = len(label_map)
    print(f"Training on {num_classes} classes.\n")

    model = build_model(num_classes)
    class_weights = compute_class_weights("train.csv", label_map)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    history = []
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        elapsed = time.time() - start
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_map": label_map,
                "val_acc": val_acc,
                "epoch": epoch,
            }, "best_model.pth")
            print(f"  -> New best val_acc ({val_acc:.4f}), saved best_model.pth")

    pd.DataFrame(history).to_csv("training_history.csv", index=False)
    print("\nWrote training_history.csv")

    # Plot loss and accuracy curves - the classic overfitting/generalization
    # check: if train_acc keeps climbing while val_acc plateaus or drops,
    # that's overfitting, worth discussing in your writeup regardless of
    # outcome.
    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="train")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist_df["epoch"], hist_df["train_acc"], label="train")
    axes[1].plot(hist_df["epoch"], hist_df["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Wrote training_curves.png")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print("Run evaluate.py next for test-set accuracy and a confusion matrix.")


if __name__ == "__main__":
    main()