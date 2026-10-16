"""
PyTorch Dataset + DataLoader for Pokémon TCG series classification.

Reads train.csv / val.csv / test.csv (from split_dataset.py) and label_map.json,
loads images on demand from their original locations (no copying), applies
standard augmentation on train only, and exposes ready-to-use DataLoaders.

This file is meant to be imported by train.py, not run directly - but running
it directly does a quick smoke test (loads one batch, prints shapes) so you
can confirm everything wires up before writing the full training loop.

Usage (smoke test):
    py -3.11 -m pip install torch torchvision pillow pandas
    py -3.11 dataset.py
"""

import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

IMAGE_SIZE = 224  # standard input size for common pretrained backbones (ResNet, MobileNet, etc.)

# ImageNet normalization stats - use these if you plan to use a pretrained
# backbone (recommended given our data size); if training fully from scratch
# you could use dataset-specific stats instead, but ImageNet stats are a fine
# default either way.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.RandomHorizontalFlip(p=0.3),  # cards are readable either way visually,
                                     # but keep this mild - card text becomes
                                     # unreadable/mirrored, so don't overdo
                                     # augmentation that destroys real signal
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    T.RandomRotation(degrees=5),   # small only - real scans are near-upright
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

eval_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


class PokemonSeriesDataset(Dataset):
    def __init__(self, csv_path: str, label_map_path: str, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        with open(label_map_path, encoding="utf-8") as f:
            self.label_map = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        series = row["series"]
        label = self.label_map[series]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Surface the bad path clearly rather than crashing the whole
            # training run on one unreadable file deep into an epoch.
            raise RuntimeError(f"Could not open image at {image_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(batch_size: int = 32, num_workers: int = 4):
    train_ds = PokemonSeriesDataset("train.csv", "label_map.json", transform=train_transform)
    val_ds = PokemonSeriesDataset("val.csv", "label_map.json", transform=eval_transform)
    test_ds = PokemonSeriesDataset("test.csv", "label_map.json", transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.label_map


if __name__ == "__main__":
    train_loader, val_loader, test_loader, label_map = get_dataloaders(batch_size=8, num_workers=0)
    print(f"Classes: {len(label_map)}")
    images, labels = next(iter(train_loader))
    print(f"Batch image tensor shape: {images.shape}")  # expect [8, 3, 224, 224]
    print(f"Batch labels: {labels.tolist()}")
    print("\nDataset wiring looks correct if the shape above is [batch, 3, 224, 224]")
    print("and labels are integers in range 0 to", len(label_map) - 1)