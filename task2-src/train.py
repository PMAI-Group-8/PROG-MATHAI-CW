# train.py
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

from cifar100_loader import get_dataloaders

class CNN(nn.Module):
    def __init__(self, num_classes: int = 100):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    arch = arch.lower()
    if arch == "cnn":
        return CNN(num_classes=num_classes)
    if arch == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if arch == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if arch == "densenet121":
        w = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.densenet121(weights=w); m.classifier = nn.Linear(m.classifier.in_features, num_classes); return m
    raise ValueError(f"Unknown arch: {arch}")

def accuracy_topk(outputs, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return [(correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size).item() for k in topk]

def main():
    script_dir = Path(__file__).resolve().parent
    default_data_root = str((script_dir.parent / "dataset" / "cifar100").resolve())

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arch", type=str, default="resnet18", choices=["cnn","resnet18","resnet50","densenet121"])
    parser.add_argument("--pretrained", type=int, default=1)  # 1=True, 0=False
    parser.add_argument("--freeze-features", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_root_path = Path(args.data_root)
    if not data_root_path.exists():
        alt = (script_dir.parent / "dataset" / "cifar100").resolve()
        print(f"--data-root not found at {data_root_path}. Falling back to {alt}")
        data_root_path = alt
    print(f"Using data root: {data_root_path}")

    transform_mode = "imagenet" if args.arch != "cnn" else "cifar"
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root=str(data_root_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=5000,
        seed=args.seed,
        pin_memory=(device.type == "cuda"),
        transform_mode=transform_mode,
        image_size=args.image_size,
    )

    model = build_model(args.arch, num_classes=len(class_names), pretrained=bool(args.pretrained)).to(device)

    if args.arch != "cnn" and bool(args.freeze_features):
        for p in model.parameters(): p.requires_grad = False
        if hasattr(model, "fc"):
            for p in model.fc.parameters(): p.requires_grad = True
        elif hasattr(model, "classifier"):
            for p in model.classifier.parameters(): p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    train_losses, valid_losses, val_top1_hist, val_top5_hist = [], [], [], []
    results_dir = script_dir / "results"; results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = results_dir / "best_model.pt"; plot_path = results_dir / "train_curve.png"

    best_val_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Generating epoch {epoch:03d}/{args.epochs:03d}...")
        model.train(); train_loss = 0.0
        for data, target in train_loader:
            data = data.to(device); target = target.to(device)
            optimizer.zero_grad(); output = model(data)
            loss = criterion(output, target); loss.backward(); optimizer.step()
            train_loss += loss.item() * data.size(0)

        model.eval(); valid_loss = 0.0; total = 0; sum_top1 = 0.0; sum_top5 = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device); target = target.to(device)
                output = model(data); loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)
                top1, top5 = accuracy_topk(output, target, topk=(1, 5))
                b = target.size(0); total += b; sum_top1 += top1 * b / 100.0; sum_top5 += top5 * b / 100.0

        scheduler.step()

        train_loss /= len(train_loader.dataset); valid_loss /= len(val_loader.dataset)
        val_top1 = 100.0 * sum_top1 / total; val_top5 = 100.0 * sum_top5 / total
        train_losses.append(train_loss); valid_losses.append(valid_loss)
        val_top1_hist.append(val_top1); val_top5_hist.append(val_top5)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}%")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save({"epoch": epoch, "arch": args.arch, "transform_mode": transform_mode, "model_state": model.state_dict()}, ckpt_path)

    print(f"Best validation top-1: {best_val_top1:.2f}%")
    epochs = range(1, args.epochs + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, train_losses, label="Training loss", color="tab:blue")
    ax1.plot(epochs, valid_losses, label="Validation loss", color="tab:orange")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_top1_hist, label="Val top-1 (%)", color="tab:green")
    ax2.plot(epochs, val_top5_hist, label="Val top-5 (%)", color="tab:red")
    ax2.set_ylabel("Accuracy (%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Loss & Accuracy against Epoch"); plt.tight_layout()
    plt.savefig(plot_path, dpi=150); plt.close()
    print(f"Saved best model to {ckpt_path}"); print(f"Saved training curve to {plot_path}")

if __name__ == "__main__":
    main()