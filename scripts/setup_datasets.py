from pathlib import Path

base = Path("dataset")
paths = [base, base / "cifar10", base / "cifar100"]

all_exist = all(p.exists() for p in paths)
if all_exist:
    print("Dataset directories already exist. No action needed.")
    exit(0)
else:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
    print("Dataset directories created successfully.")


print("Now download the CIFAR-10 and CIFAR-100 datasets from their official website and place them in the respective directories:")
print("Then copy CIFAR-10 files to 'dataset/cifar10/' and CIFAR-100 files to 'dataset/cifar100/'")
