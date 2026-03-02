"""
FACED + DETransformer Training Script
======================================

Trains a lightweight temporal transformer (DETransformer) on FACED pre-extracted
Differential Entropy features mapped to Muse 2's 4-channel layout.

Architecture:
  Input : (batch, seq_len=30, input_dim=57) — 30 one-second DE feature windows
  Model : 2-layer TransformerEncoder, d_model=64, 4 heads
  Output: (batch, 3) — positive / neutral / negative

Key advantage over current LightGBM:
  - Captures temporal dynamics within a 30-second clip (habituation, ramp-up)
  - Models how band powers evolve across seconds
  - Can be fine-tuned on personal Muse 2 data after deployment

Activation at inference time:
  Only activates when >= 30 seconds of data are available (7680 samples at 256 Hz).
  Shorter epochs (4-second live windows) fall through to mega-LGBM as before.

Data: FACED DE features — 123 subjects, 32 channels → 4 Muse-equivalent channels
  Channel mapping: T7(23)→TP9, FP1(0)→AF7, FP2(2)→AF8, T8(24)→TP10
  Per clip: 28 videos × 30 seconds × 57 features = temporal sequence of shape (30, 57)

Download:
  Option A (automatic, no account needed):
    pip install torcheeg && python train_reve_faced.py --download-torcheeg

  Option B (manual):
    Download EEG_Features.zip from Synapse syn52368847
    Extract to ml/data/faced/EEG_Features/DE/

Usage:
  cd ml/
  python training/train_reve_faced.py
  python training/train_reve_faced.py --epochs 80 --d-model 128
  python training/train_reve_faced.py --download-torcheeg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent     # ml/training/
_ML_ROOT  = _HERE.parent                         # ml/

# Add ml/ to path so imports from training/processing/models work
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
_MUSE_CH = [23, 0, 2, 24]          # T7(23)→TP9, FP1(0)→AF7, FP2(2)→AF8, T8(24)→TP10
SEQ_LEN  = 30                       # seconds per FACED video clip
INPUT_DIM = 57                       # features per 1-second window (from _de_to_features)
N_CLASSES = 3                        # positive / neutral / negative

DE_DIR        = _ML_ROOT / "data" / "faced" / "EEG_Features" / "DE"
MODEL_DIR     = _ML_ROOT / "models" / "saved"
BENCHMARK_DIR = _ML_ROOT / "benchmarks"

FACED_VIDEO_LABELS_28: List[int] = [
    1, 1, 1,       # disgust   (0–2)
    2, 2, 2,       # fear      (3–5)
    3, 3, 3,       # sadness   (6–8)
    4, 4, 4, 4,    # neutral   (9–12)
    5, 5, 5,       # amusement (13–15)
    7, 7, 7,       # joy       (16–18)
    6, 6, 6,       # inspiration (19–21)
    8, 8, 8,       # tenderness  (22–24)
    0, 0, 0,       # anger     (25–27)
]
assert len(FACED_VIDEO_LABELS_28) == 28

FACED_9_TO_3: Dict[int, int] = {
    5: 0, 6: 0, 7: 0, 8: 0,   # positive
    4: 1,                       # neutral
    0: 2, 1: 2, 2: 2, 3: 2,   # negative
}

_DELTA, _THETA, _ALPHA, _BETA, _GAMMA = 0, 1, 2, 3, 4


# ── Feature extraction (reused from train_faced.py) ──────────────────────────

def _de_to_features(de_4ch: np.ndarray) -> np.ndarray:
    """Convert a (4, 5) DE window into a 57-dimensional feature vector.

    Identical to train_faced._de_to_features — kept here to avoid import
    dependency from training script into training script.
    """
    eps = 1e-8
    t7, fp1, fp2, t8 = de_4ch[0], de_4ch[1], de_4ch[2], de_4ch[3]

    raw = de_4ch.flatten()   # (20,)

    ratios = []
    for ch in range(4):
        d = de_4ch[ch, _DELTA] + eps
        h = de_4ch[ch, _THETA] + eps
        a = de_4ch[ch, _ALPHA] + eps
        b = de_4ch[ch, _BETA]  + eps
        ratios.extend([a / b, h / b, a / h, d / h])
    ratios = np.array(ratios)   # (16,)

    f_dasm = fp2 - fp1                          # (5,)
    f_rasm = fp2 / (fp1 + eps)                  # (5,)
    t_dasm = t8  - t7                            # (5,)
    t_rasm = t8  / (t7  + eps)                  # (5,)
    faa    = np.array([fp2[_ALPHA] - fp1[_ALPHA]])  # (1,)

    return np.concatenate([raw, ratios, f_dasm, f_rasm, t_dasm, t_rasm, faa])


# ── Model ────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    log.error("PyTorch not installed. Run: pip install torch")


class DETransformer(nn.Module):
    """Temporal transformer over 1-second DE feature windows.

    Captures how band powers evolve across a 30-second EEG epoch.
    Works on FACED-style DE features extracted for Muse 2's 4-channel layout.

    Input shape:  (batch, seq_len, input_dim)  e.g. (32, 30, 57)
    Output shape: (batch, n_classes)            e.g. (32, 3)

    Why transformer over plain LGBM:
    - LGBM flattens temporal sequences, losing time-ordering information
    - Transformer self-attention can learn which seconds of a clip are diagnostic
    - Captures temporal patterns like alpha-build-up, beta-spike-at-onset, etc.
    """

    def __init__(
        self,
        input_dim: int  = INPUT_DIM,
        d_model:   int  = 64,
        n_heads:   int  = 4,
        n_layers:  int  = 2,
        n_classes: int  = N_CLASSES,
        dropout:   float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,         # pre-LN (more stable training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        # Store constructor args for serialisation
        self._cfg = dict(input_dim=input_dim, d_model=d_model, n_heads=n_heads,
                         n_layers=n_layers, n_classes=n_classes, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x   : (batch, seq_len, input_dim)
            mask: optional boolean padding mask (batch, seq_len) — True = pad token
        """
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        x = self.pos_dropout(x)
        x = self.encoder(x, src_key_padding_mask=mask)  # (batch, seq_len, d_model)
        x = x.mean(dim=1)               # mean-pool over time  (batch, d_model)
        return self.classifier(x)        # (batch, n_classes)

    @classmethod
    def from_config(cls, cfg: dict) -> "DETransformer":
        return cls(**cfg)


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_faced_sequences(de_dir: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load FACED DE features as temporal sequences for the DETransformer.

    Unlike train_faced.load_faced_de() which flattens to (n_sec, 57) and treats
    each second independently, this function keeps all 30 seconds together as a
    temporal sequence per video clip.

    Returns:
        X      : float32 (n_samples, 30, 57)  — temporal sequences
        y      : int32   (n_samples,)          — 3-class labels
        n_subj : int                           — subjects successfully loaded
    """
    import pickle

    pkl_files = sorted(de_dir.glob("sub*.pkl.pkl"))
    if not pkl_files:
        log.error(
            "No FACED DE files found in %s\n"
            "  Extract EEG_Features.zip (Synapse syn52368847) into ml/data/faced/\n"
            "  OR run: python training/train_reve_faced.py --download-torcheeg",
            de_dir,
        )
        return np.array([]), np.array([]), 0

    log.info("Found %d FACED subject files", len(pkl_files))
    # Collect per-subject arrays before merging so we can normalize per subject
    all_X_subj: List[np.ndarray] = []   # each entry: (28, 30, 57)
    all_y_subj: List[np.ndarray] = []   # each entry: (28,)
    n_ok = 0

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)   # (28, 32, 30, 5)
        except Exception as exc:
            log.debug("Skip %s: %s", pkl_path.name, exc)
            continue

        if data.ndim != 4 or data.shape != (28, 32, 30, 5):
            log.debug("Unexpected shape %s in %s — skipping", data.shape, pkl_path.name)
            continue

        subj_seqs = []
        subj_labels = []
        for vid_idx, raw_label in enumerate(FACED_VIDEO_LABELS_28):
            label = FACED_9_TO_3[raw_label]

            # Extract 4 Muse channels, all 30 seconds → (4, 30, 5)
            de_vid = data[vid_idx][_MUSE_CH, :, :]   # (4, 30, 5)
            de_vid = de_vid.transpose(1, 0, 2)         # (30, 4, 5)

            # Apply _de_to_features() to each 1-second window → sequence (30, 57)
            seq = np.stack([_de_to_features(de_vid[t]) for t in range(SEQ_LEN)])  # (30, 57)
            subj_seqs.append(seq)
            subj_labels.append(label)

        # Per-subject z-score normalization (the most important step for cross-subject accuracy).
        # Normalizes over all clips × seconds within this subject, removing inter-subject
        # amplitude variation caused by skull thickness, electrode impedance, etc.
        subj_arr = np.stack(subj_seqs).astype(np.float32)   # (28, 30, 57)
        flat = subj_arr.reshape(-1, INPUT_DIM)               # (840, 57)
        subj_mean = flat.mean(axis=0)                        # (57,)
        subj_std  = flat.std(axis=0) + 1e-8                  # (57,)
        subj_arr  = (subj_arr - subj_mean) / subj_std        # (28, 30, 57)

        all_X_subj.append(subj_arr)
        all_y_subj.append(np.array(subj_labels, dtype=np.int32))
        n_ok += 1

    if not all_X_subj:
        log.error("No sequences extracted — check DE directory.")
        return np.array([]), np.array([]), 0

    X = np.concatenate(all_X_subj, axis=0).astype(np.float32)  # (n, 30, 57)
    y = np.concatenate(all_y_subj, axis=0)

    counts = {c: int((y == c).sum()) for c in range(N_CLASSES)}
    log.info(
        "Loaded %d sequences from %d subjects | pos=%d neu=%d neg=%d",
        len(y), n_ok, counts.get(0, 0), counts.get(1, 0), counts.get(2, 0),
    )
    return X, y, n_ok


# ── Download helpers ──────────────────────────────────────────────────────────

def download_faced_torcheeg() -> bool:
    """Download FACED DE features via TorchEEG (no account required).

    TorchEEG stores data in its own LMDB format. This function converts it
    back to the sub*.pkl.pkl format expected by the loader above.
    """
    try:
        from torcheeg.datasets import FACEDFeatureDataset
    except ImportError:
        log.error("TorchEEG not installed. Run: pip install torcheeg")
        return False

    root = _ML_ROOT / "data" / "faced_torcheeg"
    log.info("Downloading FACED features via TorchEEG to %s (may take a few minutes)...", root)

    try:
        ds = FACEDFeatureDataset(
            root_path=str(root / "raw"),
            feature=["de_LDS"],
            num_worker=1,
            io_path=str(root / "io"),
            online_transform=None,
        )
        log.info("TorchEEG FACED download complete. %d samples.", len(ds))
        log.info("Note: TorchEEG stores data in LMDB format — running conversion…")
        _convert_torcheeg_to_pkl(ds, root)
        return True
    except Exception as exc:
        log.error("TorchEEG download failed: %s", exc)
        return False


def _convert_torcheeg_to_pkl(ds, torcheeg_root: Path) -> None:
    """Convert TorchEEG FACED dataset to sub*.pkl.pkl format."""
    import pickle

    out_dir = DE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Converting %d TorchEEG samples to pkl format in %s", len(ds), out_dir)

    # TorchEEG FACEDFeatureDataset returns (feature_dict, label) per sample.
    # Group by subject_id to reconstruct per-subject (28, 32, 30, 5) arrays.
    subj_data: Dict[int, Dict[int, np.ndarray]] = {}

    for i, (feat, label) in enumerate(ds):
        meta = ds.info.iloc[i]
        subj_id = int(meta.get("subject_id", 0))
        clip_id  = int(meta.get("trial_id",   0))
        de = feat.get("de_LDS", feat.get("de", None))
        if de is None:
            continue
        de = np.array(de)  # (32, 30, 5) or similar
        if subj_id not in subj_data:
            subj_data[subj_id] = {}
        subj_data[subj_id][clip_id] = de

    for subj_id, clips in subj_data.items():
        if len(clips) != 28:
            log.debug("Subject %d has %d clips (expected 28) — skipping", subj_id, len(clips))
            continue
        arr = np.stack([clips[c] for c in sorted(clips)])   # (28, 32, 30, 5)
        out_path = out_dir / f"sub{subj_id:03d}.pkl.pkl"
        with open(out_path, "wb") as fh:
            pickle.dump(arr, fh)

    n_converted = len(list(out_dir.glob("sub*.pkl.pkl")))
    log.info("Converted %d subject files to %s", n_converted, out_dir)


# ── Training ──────────────────────────────────────────────────────────────────

def _augment_gaussian(X: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Add small Gaussian noise for EEG feature augmentation (standard technique)."""
    return X + np.random.randn(*X.shape).astype(np.float32) * std


def train_de_transformer(
    X: np.ndarray,
    y: np.ndarray,
    d_model: int   = 128,
    n_layers: int  = 3,
    n_epochs: int  = 150,
    lr: float      = 2e-4,
    batch_size: int = 64,
    n_folds: int   = 5,
    aug_noise: float = 0.05,
) -> Dict:
    """Train DETransformer with 5-fold stratified CV.

    Key improvements over v1:
    - Per-subject normalization in data loader
    - d_model=128, n_layers=3 (3× capacity over original)
    - Gaussian noise augmentation (aug_noise=0.05) during training
    - lr=2e-4 with linear warmup + cosine decay
    - label_smoothing=0.05

    Returns dict with 'model', 'scaler', 'cv_mean', 'cv_folds'.
    """
    if not _TORCH_OK:
        raise RuntimeError("PyTorch not installed")

    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s | X=%s y=%s", device, X.shape, y.shape)

    # Normalise: fit scaler on flat (n, 30*57) view, reshape back
    flat = X.reshape(len(X), -1)
    scaler = StandardScaler()
    flat_norm = scaler.fit_transform(flat)
    X_norm = flat_norm.reshape(X.shape).astype(np.float32)

    X_t = torch.tensor(X_norm)
    y_t = torch.tensor(y.astype(np.int64))

    # Log class distribution
    class_counts = np.bincount(y, minlength=N_CLASSES)
    log.info("Class counts: pos=%d neu=%d neg=%d", class_counts[0], class_counts[1], class_counts[2])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs: List[float] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_norm, y)):
        model = DETransformer(d_model=d_model, n_layers=n_layers).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # Warmup for first 10% of epochs, then cosine decay
        warmup_steps = max(1, int(n_epochs * 0.10))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0,
                                                   total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs - warmup_steps),
            ],
            milestones=[warmup_steps],
        )
        crit  = nn.CrossEntropyLoss(label_smoothing=0.05)

        tr_ds  = TensorDataset(X_t[tr_idx], y_t[tr_idx])
        tr_dl  = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        best_val_acc = 0.0
        best_state   = None

        for epoch in range(n_epochs):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                # Gaussian noise augmentation during training
                if aug_noise > 0:
                    xb = xb + torch.randn_like(xb) * aug_noise
                loss = crit(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_t[val_idx].to(device))
                val_acc = (val_logits.argmax(1) == y_t[val_idx].to(device)).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        fold_accs.append(best_val_acc)
        log.info("Fold %d/%d  val_acc=%.3f%%", fold + 1, n_folds, best_val_acc * 100)

    cv_mean = float(np.mean(fold_accs))
    cv_std  = float(np.std(fold_accs))
    log.info("5-fold CV: %.2f%% ± %.2f%%", cv_mean * 100, cv_std * 100)

    # Retrain on full dataset
    log.info("Retraining final model on full dataset…")
    final_model = DETransformer(d_model=d_model, n_layers=n_layers).to(device)
    opt_f  = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_steps_f = max(1, int(n_epochs * 0.10))
    sched_f = torch.optim.lr_scheduler.SequentialLR(
        opt_f,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt_f, start_factor=0.1, end_factor=1.0,
                                               total_iters=warmup_steps_f),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=n_epochs - warmup_steps_f),
        ],
        milestones=[warmup_steps_f],
    )
    crit_f = nn.CrossEntropyLoss(label_smoothing=0.05)
    full_ds = TensorDataset(X_t, y_t)
    full_dl = DataLoader(full_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for epoch in range(n_epochs):
        final_model.train()
        for xb, yb in full_dl:
            xb, yb = xb.to(device), yb.to(device)
            if aug_noise > 0:
                xb = xb + torch.randn_like(xb) * aug_noise
            loss = crit_f(final_model(xb), yb)
            opt_f.zero_grad(); loss.backward(); opt_f.step()
        sched_f.step()

    return {
        "model":     final_model.cpu(),
        "scaler":    scaler,
        "cv_mean":   cv_mean,
        "cv_folds":  fold_accs,
        "cv_std":    cv_std,
    }


# ── Save / benchmark ──────────────────────────────────────────────────────────

def save_model(result: Dict, n_subjects: int, n_samples: int) -> Path:
    """Save model weights + benchmark JSON."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "reve_emotion_4ch.pt"
    bench_path = BENCHMARK_DIR / "reve_emotion_4ch_benchmark.json"

    payload = {
        "model_state":  result["model"].state_dict(),
        "model_cfg":    result["model"]._cfg,
        "scaler":       result["scaler"],
        "seq_len":      SEQ_LEN,
        "input_dim":    INPUT_DIM,
        "n_classes":    N_CLASSES,
        "class_names":  ["positive", "neutral", "negative"],
        "channel_map":  "T7(23)=TP9, FP1(0)=AF7, FP2(2)=AF8, T8(24)=TP10",
        "cv_accuracy":  result["cv_mean"],
    }

    torch.save(payload, model_path)
    log.info("Saved model → %s", model_path)

    benchmark = {
        "model":          "DETransformer (REVE-style temporal DE transformer)",
        "dataset":        "FACED",
        "n_subjects":     n_subjects,
        "n_samples":      n_samples,
        "cv_accuracy":    round(result["cv_mean"], 4),
        "cv_std":         round(result["cv_std"],  4),
        "cv_folds":       [round(f, 4) for f in result["cv_folds"]],
        "n_classes":      N_CLASSES,
        "classes":        ["positive", "neutral", "negative"],
        "seq_len":        SEQ_LEN,
        "input_dim":      INPUT_DIM,
        "architecture":   (
            f"DETransformer d_model={result['model']._cfg['d_model']}, "
            f"n_layers={result['model']._cfg['n_layers']}, "
            f"n_heads={result['model']._cfg['n_heads']}"
        ),
        "n_params":       sum(p.numel() for p in result["model"].parameters()),
        "channel_map":    "T7(23)=TP9, FP1(0)=AF7, FP2(2)=AF8, T8(24)=TP10",
    }
    bench_path.write_text(json.dumps(benchmark, indent=2))
    log.info("Saved benchmark → %s", bench_path)
    log.info("CV accuracy: %.2f%%", result["cv_mean"] * 100)
    return model_path


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train DETransformer on FACED DE features")
    parser.add_argument("--de-dir",       type=Path, default=DE_DIR,
                        help="Path to FACED DE directory (sub*.pkl.pkl files)")
    parser.add_argument("--epochs",       type=int,  default=150,
                        help="Training epochs (default 150)")
    parser.add_argument("--d-model",      type=int,  default=128,
                        help="Transformer hidden dim (default 128)")
    parser.add_argument("--n-layers",     type=int,  default=3,
                        help="Transformer encoder layers (default 3)")
    parser.add_argument("--lr",           type=float, default=2e-4,
                        help="Learning rate (default 1e-3)")
    parser.add_argument("--batch-size",   type=int,  default=64)
    parser.add_argument("--folds",        type=int,  default=5)
    parser.add_argument("--download-torcheeg", action="store_true",
                        help="Auto-download FACED features via TorchEEG (no Synapse account)")
    args = parser.parse_args(argv)

    if args.download_torcheeg:
        log.info("Attempting FACED download via TorchEEG…")
        ok = download_faced_torcheeg()
        if not ok:
            log.error("Download failed. Please download manually from Synapse syn52368847")
            sys.exit(1)

    if not _TORCH_OK:
        log.error("PyTorch required. Run: pip install torch")
        sys.exit(1)

    X, y, n_subj = _load_faced_sequences(args.de_dir)
    if len(X) == 0:
        log.error("No data loaded. Use --download-torcheeg or download FACED manually.")
        sys.exit(1)

    result = train_de_transformer(
        X, y,
        d_model=args.d_model,
        n_layers=getattr(args, "n_layers", 3),
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_folds=args.folds,
    )

    save_model(result, n_subjects=n_subj, n_samples=len(y))


if __name__ == "__main__":
    main()
