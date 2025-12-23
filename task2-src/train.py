# train.py
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from cifar100_loader import get_dataloaders


# ---------- Model ----------

class SimpleCifar100CNN(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------- Metrics ----------

def accuracy_topk(outputs, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)      # (B, maxk)
    pred = pred.t()                                  # (maxk, B)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k * 100.0 / batch_size).item())
    return res  # list of percentages


# ---------- Train / Val loops ----------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    sum_top1 = 0.0
    sum_top5 = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        top1, top5 = accuracy_topk(outputs, labels, topk=(1, 5))
        sum_top1 += top1 * batch_size / 100.0
        sum_top5 += top5 * batch_size / 100.0

    epoch_loss = running_loss / total
    epoch_top1 = 100.0 * sum_top1 / total
    epoch_top5 = 100.0 * sum_top5 / total
    return epoch_loss, epoch_top1, epoch_top5


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    sum_top1 = 0.0
    sum_top5 = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            top1, top5 = accuracy_topk(outputs, labels, topk=(1, 5))
            sum_top1 += top1 * batch_size / 100.0
            sum_top5 += top5 * batch_size / 100.0

    epoch_loss = running_loss / total
    epoch_top1 = 100.0 * sum_top1 / total
    epoch_top5 = 100.0 * sum_top5 / total
    return epoch_loss, epoch_top1, epoch_top5


# ---------- Plotting (line epoch graph) ----------

def plot_history(history, out_path: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 6))

    # Loss (left y-axis)
    ax1 = plt.gca()
    ax1.plot(epochs, history["train_loss"], label="Train loss", color="tab:blue")
    ax1.plot(epochs, history["val_loss"], label="Val loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # Accuracy (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_top1"], label="Val top-1 (%)", color="tab:green")
    ax2.plot(epochs, history["val_top5"], label="Val top-5 (%)", color="tab:red")
    ax2.set_ylabel("Accuracy (%)")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Loss & Accuracy against Epoch")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------- Main ----------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # build loaders from your helper
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root="dataset/cifar100",
        batch_size=128,
        num_workers=4,
        val_split=5000,
        seed=42,
        pin_memory=(device.type == "cuda"),
    )  # pin_memory wired to device to avoid warnings [file:42]

    model = SimpleCifar100CNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    num_epochs = 20
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": [],
    }

    best_top1 = 0.0
    ckpt_path = Path("task2-src/results/best_model.pt")
    curve_path = Path("task2-src/results/train_curve.png")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_top1, train_top5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_top1, val_top5 = eval_one_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_top1"].append(val_top1)
        history["val_top5"].append(val_top5)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| val_top1={val_top1:.2f}% "
            f"| val_top5={val_top5:.2f}%"
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                ckpt_path,
            )

    plot_history(history, curve_path)
    print(f"Best val top-1: {best_top1:.2f}%")
    print(f"Saved best model to {ckpt_path}")
    print(f"Saved training curve to {curve_path}")


if __name__ == "__main__":
    main()