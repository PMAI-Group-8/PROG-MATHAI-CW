"""
Clean CIFAR-100 data pipeline with deterministic splits and correct normalization.
Uses local CIFAR-100 dataset files from dataset/cifar100/.
"""

from pathlib import Path
from typing import Callable, Tuple
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

# Normalization stats for CIFAR-100 (channel-wise)
_CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
_CIFAR100_STD = [0.2675, 0.2565, 0.2761]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _unpickle(file_path: Path) -> dict:
    """
    Unpickle a CIFAR-100 data file.
    
    :param file_path: Path to the pickle file.
    :return: Dictionary containing the unpickled data.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


class CIFAR100Dataset(Dataset):
    """
    Custom CIFAR-100 dataset that loads from local pickle files.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.train = train
        self.transform = transform

        # Resolve primary and fallback paths
        split_name = 'train' if train else 'test'
        data_file = self.root / split_name

        if not data_file.exists():
            # Fallback to repo layout: <this_file>/../dataset/cifar100
            script_dir = Path(__file__).resolve().parent
            alt_root = script_dir.parent / "dataset" / "cifar100"
            alt_data_file = alt_root / split_name
            if alt_data_file.exists():
                self.root = alt_root
                data_file = alt_data_file
            else:
                raise FileNotFoundError(f"Expected data file at {data_file}")

        data_dict = _unpickle(data_file)

        # Extract images and labels
        self.data = data_dict[b'data']
        self.targets = data_dict[b'fine_labels']
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

        # Load class names (with the same fallback)
        meta_file = self.root / 'meta'
        if not meta_file.exists():
            script_dir = Path(__file__).resolve().parent
            alt_root = script_dir.parent / "dataset" / "cifar100"
            alt_meta = alt_root / 'meta'
            if alt_meta.exists():
                meta_file = alt_meta
                self.root = alt_root
            else:
                raise FileNotFoundError(f"Expected meta file at {meta_file}")

        meta_dict = _unpickle(meta_file)
        self.classes = [name.decode('utf-8') for name in meta_dict[b'fine_label_names']]
    
    def __len__(self) -> int:
        """
        Return the total number of samples.
        
        :return: Length of the dataset.
        """
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        :param idx: Index of the sample.
        :return: Tuple of (image, label).
        """
        img = self.data[idx]
        target = self.targets[idx]
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


def get_transforms(transform_mode: str = "cifar", image_size: int = 128) -> Tuple[Callable, Callable]:
    """
    Build train and val/test transforms.
    Modes:
      - 'cifar': 32x32, CIFAR-100 normalization
      - 'imagenet': resize to `image_size`, ImageNet normalization (for pretrained backbones)
    """
    if transform_mode == "imagenet_strong":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.6, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
        val_test_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    elif transform_mode == "imagenet":
        train_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        val_test_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    elif transform_mode == "cifar_strong":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
            transforms.RandomErasing(p=0.25),
        ])
        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ])
        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ])
    return train_transform, val_test_transform


def get_datasets(
    data_root: str,
    val_split: int,
    seed: int,
    transform_mode: str = "cifar",
    image_size: int = 128,
) -> Tuple[Dataset, Dataset, Dataset, list[str]]:
    """
    Load CIFAR-100 datasets with deterministic train/val split.
    """
    train_transform, val_test_transform = get_transforms(transform_mode=transform_mode, image_size=image_size)

    # Full training set (with train transforms)
    full_train = CIFAR100Dataset(
        root=data_root,
        train=True,
        transform=train_transform,
    )
    class_names = full_train.classes

    # Deterministic split into train/val
    train_size = len(full_train) - val_split
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset_with_train_tx = random_split(
        full_train,
        [train_size, val_split],
        generator=generator,
    )
    val_indices = val_subset_with_train_tx.indices

    # Validation dataset with val/test transforms
    full_train_val_tx = CIFAR100Dataset(
        root=data_root,
        train=True,
        transform=val_test_transform,
    )
    val_subset = Subset(full_train_val_tx, val_indices)

    # Test dataset
    test_dataset = CIFAR100Dataset(
        root=data_root,
        train=False,
        transform=val_test_transform,
    )

    return train_subset, val_subset, test_dataset, class_names


def get_dataloaders(
    data_root: str = "./dataset/cifar100",
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: int = 5000,
    seed: int = 42,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    transform_mode: str = "cifar",
    image_size: int = 128,
) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Build CIFAR-100 data loaders with deterministic train/val split.
    """
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_dataset, val_dataset, test_dataset, class_names = get_datasets(
        data_root=data_root,
        val_split=val_split,
        seed=seed,
        transform_mode=transform_mode,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, class_names


def save_sample_grid(
    loader: DataLoader,
    class_names: list[str],
    out_path: str = "task2-src/results/cifar100_samples.png",
    n: int = 36,
) -> None:
    """
    Save a grid of sample images from the data loader.

    :param loader: DataLoader to sample from.
    :param class_names: List of class names for labeling.
    :param out_path: Output path for the saved image.
    :param n: Number of samples to display (must be a perfect square).
    """
    # Denormalization stats as tensors for broadcasting
    mean = torch.tensor(_CIFAR100_MEAN).view(3, 1, 1)
    std = torch.tensor(_CIFAR100_STD).view(3, 1, 1)

    images, labels = next(iter(loader))
    images = images[:n]
    labels = labels[:n]

    # Denormalize and clamp to [0, 1]
    images = torch.clamp(images * std + mean, 0, 1)

    grid_size = int(n**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for idx, (img, label) in enumerate(zip(images, labels)):
        ax = axes[idx]
        img_np = img.permute(1, 2, 0).numpy()  # CHW -> HWC
        ax.imshow(img_np)
        ax.set_title(class_names[label.item()], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample grid saved to: {out_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    dataset_path = script_dir.parent / "dataset" / "cifar100"
    
    print("Loading CIFAR-100 datasets...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root=str(dataset_path),
        batch_size=128,
        num_workers=4,
        val_split=5000,
        seed=42,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(class_names)}")

    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: images={images.shape}, labels={labels.shape}")

    print("\nGenerating sample grid...")
    save_sample_grid(
        train_loader,
        class_names,
        out_path="task2-src/results/cifar100_samples.png",
        n=36,
    )
    print("Done!")