# train.py
from pathlib import Path

import argparse
import math
import random
import numpy as np
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
    if arch == "mobilenet_v3_small":
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        in_features = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_features, num_classes)
        return m
    if arch == "shufflenet_v2_x1_0":
        w = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.shufflenet_v2_x1_0(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
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
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", choices=["cnn","resnet18","resnet50","densenet121","mobilenet_v3_small","shufflenet_v2_x1_0"])
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--freeze-features", type=int, default=1)
    parser.add_argument("--unfreeze-epoch", type=int, default=5)
    parser.add_argument("--lr-head-mult", type=float, default=10.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--augment", type=str, default="light", choices=["light","strong"])
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd","adamw"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-prob", type=float, default=0.0)
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()

    if args.cpu_threads and args.cpu_threads > 0:
        try:
            torch.set_num_threads(args.cpu_threads)
        except Exception:
            pass

    device = torch.device(args.device)
    print("Using device:", device)

    data_root_path = Path(args.data_root)
    if not data_root_path.exists():
        alt = (script_dir.parent / "dataset" / "cifar100").resolve()
        print(f"--data-root not found at {data_root_path}. Falling back to {alt}")
        data_root_path = alt
    print(f"Using data root: {data_root_path}")

    if args.arch != "cnn":
        transform_mode = "imagenet_strong" if args.augment == "strong" else "imagenet"
    else:
        transform_mode = "cifar_strong" if args.augment == "strong" else "cifar"
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

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    # Differential LR: backbone vs head for non-CNN
    if args.arch == "cnn":
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        param_groups = [{"params": params, "lr": args.lr}]
    else:
        head_params = []
        backbone_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if ("fc." in name) or ("classifier." in name):
                head_params.append(p)
            else:
                backbone_params.append(p)
        param_groups = [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.lr_head_mult},
        ]
    if args.optimizer == "sgd":
        optimizer = optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)

    train_losses, valid_losses, val_top1_hist, val_top5_hist = [], [], [], []
    results_dir = script_dir / "results"; results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = results_dir / "best_model.pt"; plot_path = results_dir / "train_curve.png"

    best_val_top1 = 0.0
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed); random.seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        print(f"Generating epoch {epoch:03d}/{args.epochs:03d}...")
        # Optional progressive unfreeze for pretrained backbones
        if args.arch != "cnn" and bool(args.freeze_features) and args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            for p in model.parameters(): p.requires_grad = True
            print("Unfroze backbone parameters.")
        model.train(); train_loss = 0.0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            use_cutmix = (args.cutmix_alpha > 0 and rng.rand() < args.cutmix_prob)
            use_mixup = (args.mixup_alpha > 0 and not use_cutmix)
            lam = 1.0; target_a = target; target_b = target
            if use_mixup:
                lam = rng.beta(args.mixup_alpha, args.mixup_alpha)
                index = torch.randperm(data.size(0), device=data.device)
                data = lam * data + (1 - lam) * data[index]
                target_a, target_b = target, target[index]
            elif use_cutmix:
                lam = rng.beta(args.cutmix_alpha, args.cutmix_alpha)
                W = data.size(-1); H = data.size(-2)
                cut_rat = math.sqrt(1. - lam)
                cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
                cx = rng.randint(W); cy = rng.randint(H)
                bbx1 = max(cx - cut_w // 2, 0); bby1 = max(cy - cut_h // 2, 0)
                bbx2 = min(cx + cut_w // 2, W); bby2 = min(cy + cut_h // 2, H)
                index = torch.randperm(data.size(0), device=data.device)
                data[:, :, bby1:bby2, bbx1:bbx2] = data[index, :, bby1:bby2, bbx1:bbx2]
                target_a, target_b = target, target[index]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            optimizer.zero_grad()
            output = model(data)
            if use_mixup or use_cutmix:
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
                loss = criterion(output, target)
            loss.backward(); optimizer.step()
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
    plt.title("Loss & Accuracy vs Epoch"); plt.tight_layout()
    plt.savefig(plot_path, dpi=150); plt.close()
    print(f"Saved best model to {ckpt_path}"); print(f"Saved training curve to {plot_path}")

if __name__ == "__main__":
    main()