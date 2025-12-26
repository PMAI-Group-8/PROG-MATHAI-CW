from pathlib import Path

base = Path("dataset")
cifar10 = base / "cifar10"
cifar100 = base / "cifar100"

cifar10_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
    "batches.meta",
]

cifar100_files = [
    "train",
    "test",
    "meta",
]

def check_folder(path: Path, required_files):
    if not path.exists():
        print(f"X   | Folder missing: {path}")
        return False

    print(f"TRUE | Folder exists: {path}")
    missing = [f for f in required_files if not (path / f).exists()]
    if missing:
        print(f"X    | Missing files in {path}: {', '.join(missing)}")
        return False
    else:
        print(f"TRUE | All required files present in {path}")
        return True

print("Checking dataset structure...\n")

ok_base = base.exists()
if not ok_base:
    print(f"X    | Folder missing: {base}\nRun setup_datasets.py first.")
else:
    print(f"TRUE | Folder exists: {base}")

ok_c10 = check_folder(cifar10, cifar10_files) if ok_base else False
ok_c100 = check_folder(cifar100, cifar100_files) if ok_base else False

print("\nSummary:")
if ok_base and ok_c10 and ok_c100:
    print("TRUE | CIFAR-10 and CIFAR-100 appear to be correctly downloaded and placed.")
else:
    print("X    | Dataset setup is incomplete. See messages above for what is missing.")