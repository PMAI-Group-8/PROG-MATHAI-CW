import sys
import re
import types
from pathlib import Path
import importlib.util
import builtins

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.py"
LOADER_PATH = ROOT / "cifar100_loader.py"
EVAL_PATH = ROOT / "eval.py"
sys.path.insert(0, str(ROOT))

def load_module_from_path(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

class EpochState:
    current_epoch: int | None = None

epoch_state = EpochState()
_orig_print = builtins.print
def _patched_print(*args, **kwargs):
    try:
        msg = " ".join(str(a) for a in args)
        m = re.search(r"Generating epoch\s+(\d+)", msg)
        if m:
            epoch_state.current_epoch = int(m.group(1))
        else:
            m2 = re.search(r"Epoch\s+(\d+)\s*/\s*\d+", msg)
            if m2:
                epoch_state.current_epoch = int(m2.group(1))
    except Exception:
        pass
    _orig_print(*args, **kwargs)

class TqdmLoader:
    def __init__(self, loader, desc_prefix: str, include_epoch: bool = True):
        self._loader = loader
        self._desc_prefix = desc_prefix
        self._include_epoch = include_epoch

    def __iter__(self):
        if not TQDM_AVAILABLE:
            return iter(self._loader)
        desc = self._desc_prefix
        if self._include_epoch and epoch_state.current_epoch is not None:
            desc = f"{self._desc_prefix} {epoch_state.current_epoch:03d}"
        try:
            total = len(self._loader)
        except Exception:
            total = None
        return iter(tqdm(self._loader, total=total, desc=desc, unit="batch", leave=False))

    def __len__(self):
        return len(self._loader)

    @property
    def dataset(self):
        return self._loader.dataset

    def __getattr__(self, name):
        return getattr(self._loader, name)

def main():
    args = sys.argv[1:]
    run_eval = True
    if "--eval" in args:
        run_eval = True
        args = [a for a in args if a != "--eval"]

    # Set defaults
    def _has_flag(a_list, flag):
        return any(a == flag or a.startswith(flag + "=") for a in a_list)
    
    if not _has_flag(args, "--device"):
        args += ["--device", "cpu"]
    if not _has_flag(args, "--arch"):
        args += ["--arch", "mobilenet_v3_small"]
    if not _has_flag(args, "--pretrained"):
        args += ["--pretrained", "1"]
    if not _has_flag(args, "--augment"):
        args += ["--augment", "light"]
    if not _has_flag(args, "--image-size"):
        args += ["--image-size", "128"]
    if not _has_flag(args, "--epochs"):
        args += ["--epochs", "15"]
    if not _has_flag(args, "--batch-size"):
        args += ["--batch-size", "128"]
    if not _has_flag(args, "--num-workers"):
        args += ["--num-workers", "0"]
    if not _has_flag(args, "--optimizer"):
        args += ["--optimizer", "sgd"]
    if not _has_flag(args, "--lr"):
        args += ["--lr", "0.01"]
    if not _has_flag(args, "--momentum"):
        args += ["--momentum", "0.9"]
    if not _has_flag(args, "--weight-decay"):
        args += ["--weight-decay", "1e-4"]
    if not _has_flag(args, "--label-smoothing"):
        args += ["--label-smoothing", "0.1"]
    if not _has_flag(args, "--freeze-features"):
        args += ["--freeze-features", "1"]
    if not _has_flag(args, "--unfreeze-epoch"):
        args += ["--unfreeze-epoch", "5"]
    if not _has_flag(args, "--lr-head-mult"):
        args += ["--lr-head-mult", "10"]

    # Load and patch modules
    cifar_loader = load_module_from_path("cifar100_loader", LOADER_PATH)
    builtins.print = _patched_print

    original_get_dataloaders = cifar_loader.get_dataloaders

    def patched_get_dataloaders(*p_args, **p_kwargs):
        tl, vl, test_l, class_names = original_get_dataloaders(*p_args, **p_kwargs)
        tl = TqdmLoader(tl, desc_prefix="Train", include_epoch=True)
        vl = TqdmLoader(vl, desc_prefix="Val", include_epoch=True)
        test_l = TqdmLoader(test_l, desc_prefix="Test", include_epoch=False)
        return tl, vl, test_l, class_names

    cifar_loader.get_dataloaders = patched_get_dataloaders
    sys.modules["cifar100_loader"] = cifar_loader

    # Train
    train_mod = load_module_from_path("train", TRAIN_PATH)
    sys.argv = ["train.py"] + args
    train_mod.main()

    # Eval with test progress bar
    if run_eval:
        eval_mod = load_module_from_path("eval", EVAL_PATH)
        image_size_args = []
        for i, a in enumerate(args):
            if a == "--image-size" and i + 1 < len(args):
                image_size_args = ["--image-size", args[i + 1]]
                break
        sys.argv = ["eval.py"] + image_size_args
        eval_mod.main()

if __name__ == "__main__":
    main()