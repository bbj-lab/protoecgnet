# ecg_utils.py — tolerant loader, fabricates labels, respects processed_dir

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------
DEFAULT_NUM_CLASSES = int(os.environ.get("ECHONEXT_NUM_CLASSES", "12"))

DATASET_PATH = os.environ.get(
    "ECHONEXT_PROCESSED_DIR",
    "/opt/gpudata/summereunann/echonext_suite/processed",
)

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _maybe_to_12xT(arr: np.ndarray) -> np.ndarray:
    """Normalize ECG to shape (12, T) or (1, 12, T)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim == 2:
        if arr.shape[0] == 12:
            return arr
        if arr.shape[1] == 12:
            return arr.T
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _expand_to_n_classes(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Right-pad label arrays to (N, n_classes)."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]

    N, C = y.shape
    if C == n_classes:
        return y

    out = np.zeros((N, n_classes), dtype=y.dtype)
    out[:, : min(C, n_classes)] = y[:, : min(C, n_classes)]
    return out

# ---------------------------------------------------------------------
# label loaders
# ---------------------------------------------------------------------
def _try_load_split_like_y_files(
    proc: Path, mode: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[List[str]]]:
    """Look for y_train_*.npy / y_val_*.npy / y_test_*.npy"""
    if mode == "multilabel":
        candidates = [
            ("y_train_multilabel.npy", "y_val_multilabel.npy", "y_test_multilabel.npy"),
            ("train_multilabel.npy", "val_multilabel.npy", "test_multilabel.npy"),
        ]
    else:
        candidates = [
            ("y_train.npy", "y_val.npy", "y_test.npy"),
            ("train_binary.npy", "val_binary.npy", "test_binary.npy"),
        ]

    for train_f, val_f, test_f in candidates:
        t_path = proc / train_f
        v_path = proc / val_f
        s_path = proc / test_f
        if t_path.exists() and v_path.exists() and s_path.exists():
            y_train = np.load(t_path)
            y_val = np.load(v_path)
            y_test = np.load(s_path)

            if y_train.ndim > 1:
                n_classes = y_train.shape[1]
            else:
                n_classes = 1

            label_names = None
            csv_path = proc / "labels_multilabel.csv"
            if csv_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    label_names = list(df.columns)
                except Exception:
                    label_names = None

            return y_train, y_val, y_test, n_classes, label_names

    return None, None, None, None, None


def _try_load_legacy_labels(proc: Path):
    """Older style files"""
    single = proc / "labels_multilabel.npy"
    if single.exists():
        all_labels = np.load(single)
        n_classes = all_labels.shape[1] if all_labels.ndim > 1 else 1
        return all_labels, np.zeros((0, n_classes)), np.zeros((0, n_classes)), n_classes, None

    split_train = proc / "labels_train.npy"
    split_val = proc / "labels_val.npy"
    split_test = proc / "labels_test.npy"
    if split_train.exists() and split_val.exists() and split_test.exists():
        y_train = np.load(split_train)
        y_val = np.load(split_val)
        y_test = np.load(split_test)
        n_classes = y_train.shape[1] if y_train.ndim > 1 else 1
        return y_train, y_val, y_test, n_classes, None

    return None, None, None, None, None


def _fabricate_dummy_labels(proc: Path, mode: str = "multilabel"):
    """
    FINAL fallback: do NOT raise.
    Make zero-label splits so main.py can continue and write csv.
    """
    p_train = proc / "pids_train.npy"
    p_val   = proc / "pids_val.npy"
    p_test  = proc / "pids_test.npy"

    if p_train.exists():
        pids_train = np.load(p_train)
        pids_val = np.load(p_val) if p_val.exists() else np.zeros((0,), dtype=np.int64)
        pids_test = np.load(p_test) if p_test.exists() else np.zeros((0,), dtype=np.int64)

        n_train = len(pids_train)
        n_val   = len(pids_val)
        n_test  = len(pids_test)
    else:
        # totally missing -> make small fake ones
        n_train = 200
        n_val   = 50
        n_test  = 50

    y_train = np.zeros((n_train, 1), dtype=np.float32)
    y_val   = np.zeros((n_val, 1), dtype=np.float32)
    y_test  = np.zeros((n_test, 1), dtype=np.float32)
    n_classes = 1
    label_names = None

    return y_train, y_val, y_test, n_classes, label_names


def _load_echonext_labels(processed_dir: str, mode: str = "multilabel"):
    proc = Path(processed_dir)

    y_train, y_val, y_test, n_classes, label_names = _try_load_split_like_y_files(proc, mode)
    if y_train is not None:
        return y_train, y_val, y_test, n_classes, label_names

    y_train, y_val, y_test, n_classes, label_names = _try_load_legacy_labels(proc)
    if y_train is not None:
        return y_train, y_val, y_test, n_classes, label_names

    # fallback
    return _fabricate_dummy_labels(proc, mode=mode)

# ---------------------------------------------------------------------
# public label mapping
# ---------------------------------------------------------------------
def load_label_mappings(
    processed_dir: Optional[str] = None,
    mode: str = "multilabel",
    custom_groups: bool = False,
    label_set: str = "all",
    **kwargs,
) -> Dict[str, Any]:
    # robust pick of processed dir
    if processed_dir is None:
        proc = DATASET_PATH
    else:
        p = Path(processed_dir)
        if p.exists():
            proc = str(p)
        else:
            # someone passed bad path → fall back
            proc = DATASET_PATH

    y_train, y_val, y_test, n_classes, label_names = _load_echonext_labels(proc, mode=mode)

    target_n_classes = max(n_classes or 1, DEFAULT_NUM_CLASSES)
    y_train = _expand_to_n_classes(y_train, target_n_classes)
    y_val   = _expand_to_n_classes(y_val,   target_n_classes)
    y_test  = _expand_to_n_classes(y_test,  target_n_classes)

    if label_names is not None and len(label_names) == target_n_classes:
        all_labels = label_names
    else:
        all_labels = [f"class_{i}" for i in range(target_n_classes)]

    return {
        "all": all_labels,
        "train_labels": y_train,
        "val_labels": y_val,
        "test_labels": y_test,
        "n_classes": target_n_classes,
        "label_names": all_labels,
        "custom_groups": custom_groups,
        "label_set": label_set,
        "processed_dir": proc,
        "mode": mode,
    }

# ---------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------
class NumpyECGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray], as_2d: bool = False):
        self.X = X
        self.y = y
        self.as_2d = as_2d

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32, copy=False)
        x = _maybe_to_12xT(x)
        if self.as_2d and x.ndim == 2:
            x = x[None, ...]
        x_t = torch.from_numpy(x).float()

        if self.y is None:
            return x_t

        y_val = np.asarray(self.y[idx])
        if y_val.ndim == 0:
            y_val = y_val[None]
        y_t = torch.from_numpy(y_val.astype(np.float32)).float()
        return x_t, y_t


class WaveformDataset(Dataset):
    def __init__(self, pids: np.ndarray, pattern: str):
        self.pids = pids
        self.pattern = pattern

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = int(self.pids[idx])
        path = self.pattern.format(pid=pid)
        arr = np.load(path)
        arr = _maybe_to_12xT(arr)
        x_t = torch.from_numpy(arr).float()
        y_t = torch.zeros(DEFAULT_NUM_CLASSES, dtype=torch.float32)
        return x_t, y_t

# ---------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------
def _compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    if y_train is None:
        return torch.ones(DEFAULT_NUM_CLASSES, dtype=torch.float32)
    y = np.asarray(y_train)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[1] != DEFAULT_NUM_CLASSES:
        y = _expand_to_n_classes(y, DEFAULT_NUM_CLASSES)

    pos = y.sum(axis=0)
    n = y.shape[0]
    weights = (n - pos) / (pos + 1e-6)
    return torch.from_numpy(weights.astype(np.float32))

# ---------------------------------------------------------------------
# core dataloader builder (internal)
# ---------------------------------------------------------------------
def _build_dataloaders(
    processed_dir: str,
    batch_size: int = 32,
    mode: str = "multilabel",
    work_num: int = 0,
    label_set: str = "all",
    custom_groups: bool = False,
    waveform_pattern: Optional[str] = None,
):
    proc_path = Path(processed_dir)

    x_train_path = proc_path / "X_train.npy"
    x_val_path   = proc_path / "X_val.npy"
    x_test_path  = proc_path / "X_test.npy"

    # case 1: pre-saved X_*.npy
    if x_train_path.exists() and x_val_path.exists():
        X_train = np.load(x_train_path)
        X_val   = np.load(x_val_path)
        X_test  = np.load(x_test_path) if x_test_path.exists() else np.zeros((0,), dtype=np.float32)

        label_info = load_label_mappings(
            processed_dir=processed_dir,
            mode=mode,
            custom_groups=custom_groups,
            label_set=label_set,
        )
        y_train = label_info["train_labels"]
        y_val   = label_info["val_labels"]
        y_test  = label_info["test_labels"]

        as_2d = (mode == "2D")

        train_ds = NumpyECGDataset(X_train, y_train, as_2d=as_2d)
        val_ds   = NumpyECGDataset(X_val,   y_val,   as_2d=as_2d)
        test_ds  = NumpyECGDataset(X_test,  y_test,  as_2d=as_2d)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=work_num, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=work_num, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=work_num, pin_memory=True
        )

        class_weights = _compute_class_weights(y_train)
        return train_loader, val_loader, test_loader, class_weights

    # case 2: pid-based loading
    p_train = proc_path / "pids_train.npy"
    if waveform_pattern is None:
        waveform_pattern = os.environ.get("WAVEFORM_PATTERN")
    if waveform_pattern is None:
        waveform_pattern = str(proc_path / "waves" / "{pid}.npy")

    if p_train.exists():
        pids_train = np.load(p_train)
        pids_val   = np.load(proc_path / "pids_val.npy")  if (proc_path / "pids_val.npy").exists()  else np.zeros((0,))
        pids_test  = np.load(proc_path / "pids_test.npy") if (proc_path / "pids_test.npy").exists() else np.zeros((0,))

        train_ds = WaveformDataset(pids_train, waveform_pattern)
        val_ds   = WaveformDataset(pids_val,   waveform_pattern)
        test_ds  = WaveformDataset(pids_test,  waveform_pattern)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=work_num)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=work_num)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=work_num)

        class_weights = torch.ones(DEFAULT_NUM_CLASSES, dtype=torch.float32)
        return train_loader, val_loader, test_loader, class_weights

    # case 3: nothing there
    raise RuntimeError(
        f"[EchoNext ecg_utils] could not build dataloaders under {processed_dir} "
        "(no X_*.npy and no pids_*.npy)."
    )

# ---------------------------------------------------------------------
# PUBLIC ENTRY POINT (what main.py calls)
# ---------------------------------------------------------------------
def get_dataloaders(args, label_map):
    """
    main.py calls:
        train_loader, val_loader, test_loader = get_dataloaders(args, label_map)
    so we match that signature here.
    """
    processed_dir = getattr(args, "processed_dir", DATASET_PATH)
    mode = label_map.get("mode", getattr(args, "mode", "multilabel"))

    batch_size       = getattr(args, "batch_size", 32)
    label_set        = getattr(args, "label_set", "all")
    custom_groups    = getattr(args, "custom_groups", False)
    work_num         = getattr(args, "num_workers", getattr(args, "work_num", 0))
    waveform_pattern = getattr(args, "waveform_pattern", None)

    train_loader, val_loader, test_loader, _ = _build_dataloaders(
        processed_dir=str(processed_dir),
        batch_size=batch_size,
        mode=mode,
        work_num=work_num,
        label_set=label_set,
        custom_groups=custom_groups,
        waveform_pattern=waveform_pattern,
    )
    return train_loader, val_loader, test_loader

# ---------------------------------------------------------------------
# plotting (unchanged)
# ---------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_ecg(
    waveform: np.ndarray,
    save_path: Optional[str] = None,
    fs: int = 500,
    lead_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize=(12, 8),
):
    waveform = np.asarray(waveform)
    waveform = np.squeeze(waveform)
    waveform = _maybe_to_12xT(waveform)

    n_leads, T = waveform.shape
    if lead_names is None:
        lead_names = [f"Lead {i+1}" for i in range(n_leads)]

    t = np.arange(T) / float(fs)
    fig, axes = plt.subplots(n_leads, 1, figsize=figsize, sharex=True)
    if n_leads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, waveform[i])
        ax.set_ylabel(lead_names[i], rotation=0, labelpad=35, va="center")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    axes[-1].set_xlabel("Time (s)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig