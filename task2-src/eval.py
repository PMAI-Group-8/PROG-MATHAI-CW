from pathlib import Path
from typing import Tuple

import argparse
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from cifar100_loader import get_dataloaders


# --- Model (same as in train.py) ---
class CNN(nn.Module):
    def __init__(self, num_classes: int = 100):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        # 32x32 -> conv3 -> 30x30 -> maxpool -> 15x15
        # -> conv3 -> 13x13 -> maxpool -> 6x6
        self.fc1 = nn.Linear(20 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.fc2(x)
        return x


def accuracy_topk(outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> list[float]:
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k * 100.0 / batch_size).item())
    return res


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0
    sum_top1 = 0.0
    sum_top5 = 0.0
    all_preds = []
    all_targets = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        top1, top5 = accuracy_topk(outputs, labels, topk=(1, 5))
        batch_size = labels.size(0)
        total += batch_size
        sum_top1 += top1 * batch_size / 100.0
        sum_top5 += top5 * batch_size / 100.0
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    top1_acc = 100.0 * sum_top1 / total
    top5_acc = 100.0 * sum_top5 / total
    return top1_acc, top5_acc, np.concatenate(all_preds), np.concatenate(all_targets)


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)
    plt.tight_layout()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def save_per_class_accuracy(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_class_acc = (cm.diagonal() / cm.sum(axis=1)).astype(float)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "accuracy"])
        for name, acc in zip(class_names, per_class_acc):
            writer.writerow([name, f"{acc * 100.0:.2f}"])
    print(f"Saved per-class accuracy to {out_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    default_data_root = str((script_dir.parent / "dataset" / "cifar100").resolve())
    default_ckpt = str((script_dir / "results" / "best_model.pt").resolve())

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=default_ckpt)
    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=str((script_dir / "results" / "eval").resolve()))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Resolve dataset path robustly
    data_root_path = Path(args.data_root)
    if not data_root_path.exists():
        alt = (script_dir.parent / "dataset" / "cifar100").resolve()
        print(f"--data-root not found at {data_root_path}. Falling back to {alt}")
        data_root_path = alt
    print(f"Using data root: {data_root_path}")

    # Data
    _, _, test_loader, class_names = get_dataloaders(
        data_root=str(data_root_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=5000,
        seed=args.seed,
        pin_memory=(device.type == "cuda"),
    )

    # Model and checkpoint
    model = CNN(num_classes=len(class_names)).to(device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        alt_ckpt = (script_dir / "results" / "best_model.pt").resolve()
        if alt_ckpt.exists() and alt_ckpt != ckpt_path:
            print(f"Checkpoint not found at {ckpt_path}. Using {alt_ckpt}")
            ckpt_path = alt_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])

    # Evaluate on test set
    print("Evaluating on CIFAR-100 test set...")
    top1, top5, preds, targets = evaluate(model, test_loader, device)
    print(f"Test Top-1: {top1:.2f}% | Test Top-5: {top5:.2f}%")

    # Confusion matrix and per-class accuracy
    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(cm, class_names, save_dir / "confusion_matrix.png")
    save_per_class_accuracy(cm, class_names, save_dir / "per_class_accuracy.csv")

    # Save summary metrics
    with open(save_dir / "test_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["top1", f"{top1:.2f}"])
        writer.writerow(["top5", f"{top5:.2f}"])
    print(f"Saved test metrics to {save_dir / 'test_metrics.csv'}")


if __name__ == "__main__":
    main()