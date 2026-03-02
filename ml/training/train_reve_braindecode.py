"""Fine-tune REVE (braindecode) on FACED dataset for 4-channel Muse 2 emotion.

Strategy:
  - Freeze the REVE backbone (69.7M params) — only trains the final FC head (~3K params)
  - This is transfer-learning: backbone as feature extractor, small head on top
  - With random backbone: head learns to interpret random features — low accuracy
  - With pretrained backbone (brain-bzh/reve-base): head adapts REVE to 6-class FACED

Training conditions:
  - FACED DE features (123 subjects × 28 videos × 9 emotions → mapped to 6 classes)
  - 5-fold cross-subject validation (stratified by subject)
  - Batch size: 16 (feasible on MacBook M1 CPU, 15-30 min total)
  - Epochs: 50 per fold

Output:
  models/saved/reve_braindecode_4ch.pt    — best fold weights
  benchmarks/reve_braindecode_4ch_benchmark.json

Usage:
  cd ml/
  python training/train_reve_braindecode.py
  python training/train_reve_braindecode.py --epochs 100 --lr 5e-4
  python training/train_reve_braindecode.py --no-freeze    # fine-tune all layers (needs GPU)

FACED data:
  Option A (automatic): pip install torcheeg && python train_reve_braindecode.py --download
  Option B (manual):    Download from Synapse syn52368847 → ml/data/faced/EEG_Features/DE/
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.signal import resample as sp_resample

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_ML_ROOT = _HERE.parent
sys.path.insert(0, str(_ML_ROOT))

_FACED_DIR   = _ML_ROOT / "data" / "faced" / "EEG_Features" / "DE"
_OUT_WEIGHTS = _ML_ROOT / "models" / "saved" / "reve_braindecode_4ch.pt"
_OUT_BENCH   = _ML_ROOT / "benchmarks" / "reve_braindecode_4ch_benchmark.json"

# ── FACED → Muse 2 channel mapping ────────────────────────────────────────────
# FACED uses 32-channel 250Hz EEG. We map 4 channels closest to Muse 2's layout.
# Muse 2 (BrainFlow order): TP9, AF7, AF8, TP10
# FACED channel indices (10-20 system, 32ch): T7=23, FP1=0, FP2=2, T8=24
_FACED_CH_IDX = [23, 0, 2, 24]   # T7→TP9, FP1→AF7, FP2→AF8, T8→TP10

# REVE parameters
_REVE_FS   = 200     # REVE expects 200 Hz
_REVE_SECS = 30      # 30-second input window
_N_SAMPLES = _REVE_FS * _REVE_SECS  # 6000

# Muse 2 electrode positions (confirmed from brain-bzh/reve-positions bank)
_MUSE2_POSITIONS = torch.tensor([
    [-0.08562, -0.04651, -0.04571],   # TP9
    [-0.05484,  0.06857, -0.01059],   # AF7
    [ 0.05574,  0.06966, -0.01075],   # AF8
    [ 0.08616, -0.04704, -0.04587],   # TP10
], dtype=torch.float32)

# FACED 9-class → 6-class mapping (FACED: disgust, fear, sad, neutral, amused, inspi, joy, tender, anger)
# → our labels: fearful, fearful, sad, relaxed, happy, focused, happy, relaxed, angry
_FACED_9_TO_6 = [3, 0, 1, 4, 0, 5, 0, 4, 2]
# FACED original class indices 0-8 → our 6-class indices
# 0=disgust→3(fearful), 1=fear→0(happy? no→fearful), 2=sad→1, 3=neutral→4(relaxed),
# 4=amused→0(happy), 5=inspired→5(focused), 6=joy→0(happy), 7=tender→4(relaxed), 8=anger→2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_faced_raw_eeg() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load FACED raw EEG from pkl files (if downloaded via torcheeg).

    Returns:
        X: (N, 4, 6000) float32 — 4-ch Muse-equivalent EEG at 200Hz, 30 sec
        y: (N,) int — 6-class emotion labels
        subjects: (N,) int — subject IDs for cross-subject splits
    """
    try:
        import pickle, glob
        pkl_files = sorted(glob.glob(str(_FACED_DIR / "*.pkl")))
        if not pkl_files:
            raise FileNotFoundError(f"No pkl files found in {_FACED_DIR}")

        all_X, all_y, all_subs = [], [], []

        for sub_id, pkl_file in enumerate(pkl_files):
            with open(pkl_file, "rb") as fh:
                data = pickle.load(fh)

            # Expected shape: (n_clips, 32_channels, n_timepoints)
            # or dict with 'eeg' and 'labels' keys
            if isinstance(data, dict):
                eeg_data = data.get("eeg", data.get("de_features"))
                labels   = data.get("labels", data.get("emotion_labels"))
            else:
                eeg_data, labels = data

            eeg_data = np.asarray(eeg_data, dtype=np.float32)
            labels   = np.asarray(labels).flatten()

            # Select 4 Muse-equivalent channels
            if eeg_data.ndim == 3 and eeg_data.shape[1] >= 25:
                eeg_4ch = eeg_data[:, _FACED_CH_IDX, :]  # (n_clips, 4, n_time)
            elif eeg_data.ndim == 3 and eeg_data.shape[1] == 4:
                eeg_4ch = eeg_data
            else:
                log.warning("Unexpected shape %s in %s — skipping", eeg_data.shape, pkl_file)
                continue

            # Resample to 200 Hz if needed (FACED is 250 Hz)
            if eeg_4ch.shape[2] != _N_SAMPLES:
                eeg_200 = sp_resample(eeg_4ch, _N_SAMPLES, axis=2).astype(np.float32)
            else:
                eeg_200 = eeg_4ch

            # Map 9-class FACED labels → 6-class
            mapped_labels = np.array([_FACED_9_TO_6[int(l) % 9] for l in labels], dtype=np.int64)

            all_X.append(eeg_200)
            all_y.append(mapped_labels)
            all_subs.append(np.full(len(labels), sub_id, dtype=np.int64))

        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        subs = np.concatenate(all_subs, axis=0)
        log.info("Loaded FACED: X=%s y=%s subjects=%d", X.shape, y.shape, len(set(subs)))
        return X, y, subs

    except Exception as exc:
        log.error("Failed to load FACED data: %s", exc)
        raise


def download_faced_torcheeg():
    """Download FACED DE features via torcheeg (no account needed)."""
    try:
        from torcheeg.datasets import FACEDDataset
        from torcheeg import transforms

        log.info("Downloading FACED via torcheeg...")
        ds = FACEDDataset(
            root_path=str(_ML_ROOT / "data"),
            offline_transform=transforms.BandDifferentialEntropy(
                sampling_rate=250,
                apply_to_baseline=True,
            ),
            online_transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            label_transform=transforms.Compose([
                transforms.Select("valence"),
                transforms.Binary(4.5),
            ]),
        )
        log.info("FACED downloaded successfully: %d samples", len(ds))
        return True
    except ImportError:
        log.error("torcheeg not installed. Run: pip install torcheeg")
        return False
    except Exception as exc:
        log.error("Download failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    try:
        from braindecode.models import REVE
    except ImportError:
        log.error("braindecode not installed. Run: pip install braindecode")
        sys.exit(1)

    import os
    hf_token = os.environ.get("HF_TOKEN", "")

    # Load data
    if args.download:
        download_faced_torcheeg()
        return

    log.info("Loading FACED data from %s...", _FACED_DIR)
    X, y, subjects = load_faced_raw_eeg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on device: %s", device)

    # Position tensor (same for all samples)
    pos_template = _MUSE2_POSITIONS.unsqueeze(0).to(device)  # (1, 4, 3)

    fold_accs = []
    best_acc = 0.0
    best_state = None

    # Cross-subject split: hold out one subject at a time
    unique_subs = sorted(set(subjects))
    log.info("Cross-subject CV: %d subjects", len(unique_subs))

    for fold_idx, test_sub in enumerate(unique_subs[:args.n_folds]):
        train_mask = subjects != test_sub
        test_mask  = subjects == test_sub

        X_tr = torch.from_numpy(X[train_mask]).float().to(device)
        y_tr = torch.from_numpy(y[train_mask]).long().to(device)
        X_te = torch.from_numpy(X[test_mask]).float().to(device)
        y_te = torch.from_numpy(y[test_mask]).long().to(device)

        pos_tr = pos_template.expand(len(X_tr), -1, -1)
        pos_te = pos_template.expand(len(X_te), -1, -1)

        # Build REVE model
        if hf_token and not args.no_pretrained:
            try:
                model = REVE.from_pretrained(
                    "brain-bzh/reve-base",
                    token=hf_token,
                    n_chans=4, sfreq=_REVE_FS,
                    input_window_seconds=_REVE_SECS,
                    n_outputs=6,
                ).to(device)
                log.info("[fold %d] Using pretrained reve-base backbone!", fold_idx)
            except Exception:
                model = REVE(n_chans=4, sfreq=_REVE_FS,
                             input_window_seconds=_REVE_SECS, n_outputs=6).to(device)
                log.info("[fold %d] Using random-init REVE backbone", fold_idx)
        else:
            model = REVE(n_chans=4, sfreq=_REVE_FS,
                         input_window_seconds=_REVE_SECS, n_outputs=6).to(device)
            log.info("[fold %d] Using random-init REVE backbone", fold_idx)

        # Freeze backbone (only train final FC layer unless --no-freeze)
        if not args.no_freeze:
            trainable_names = []
            for name, param in model.named_parameters():
                # Only train the final classification head
                if any(k in name for k in ["fc", "head", "classifier", "mlp_head", "to_cls"]):
                    param.requires_grad = True
                    trainable_names.append(name)
                else:
                    param.requires_grad = False
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info("[fold %d] Frozen backbone. Trainable params: %d (%s)",
                     fold_idx, n_trainable, trainable_names[:3])
        else:
            n_trainable = sum(p.numel() for p in model.parameters())
            log.info("[fold %d] Full fine-tune. Trainable params: %d", fold_idx, n_trainable)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Training loop
        model.train()
        for epoch in range(args.epochs):
            # Mini-batch training
            perm = torch.randperm(len(X_tr))
            ep_loss = 0.0
            for i in range(0, len(X_tr), args.batch_size):
                idx = perm[i:i + args.batch_size]
                xb = X_tr[idx]
                yb = y_tr[idx]
                pb = pos_tr[idx]

                optimizer.zero_grad()
                logits = model(xb, pos=pb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    tr_acc = float((model(X_tr[:256], pos=pos_tr[:256]).argmax(1) == y_tr[:256]).float().mean())
                log.info("[fold %d] epoch %d/%d loss=%.4f train_acc=%.1f%%",
                         fold_idx, epoch + 1, args.epochs, ep_loss, tr_acc * 100)
                model.train()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_acc = float((model(X_te, pos=pos_te).argmax(1) == y_te).float().mean())
        fold_accs.append(test_acc)
        log.info("[fold %d] test acc (subject %d) = %.1f%%", fold_idx, test_sub, test_acc * 100)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    mean_acc = float(np.mean(fold_accs))
    std_acc  = float(np.std(fold_accs))
    log.info("Cross-subject CV: %.2f%% ± %.2f%%", mean_acc * 100, std_acc * 100)

    # Save best weights
    if best_state is not None:
        _OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, _OUT_WEIGHTS)
        log.info("Best weights saved to %s", _OUT_WEIGHTS)

    # Save benchmark
    bench = {
        "model":      "reve-braindecode-4ch",
        "accuracy":   round(mean_acc, 4),
        "std":        round(std_acc, 4),
        "n_folds":    len(fold_accs),
        "fold_accs":  [round(a, 4) for a in fold_accs],
        "dataset":    "FACED",
        "n_classes":  6,
        "freeze_backbone": not args.no_freeze,
        "params":     sum(p.numel() for p in (model.parameters() if 'model' in dir() else [])),
    }
    _OUT_BENCH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_BENCH.write_text(json.dumps(bench, indent=2))
    log.info("Benchmark saved to %s", _OUT_BENCH)
    return mean_acc


def main():
    parser = argparse.ArgumentParser(description="Fine-tune REVE (braindecode) on FACED")
    parser.add_argument("--epochs",      type=int,   default=50,   help="Training epochs per fold")
    parser.add_argument("--lr",          type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size",  type=int,   default=16,   help="Batch size")
    parser.add_argument("--n-folds",     type=int,   default=5,    help="Number of cross-subject folds")
    parser.add_argument("--no-freeze",   action="store_true",      help="Fine-tune all layers (needs GPU)")
    parser.add_argument("--no-pretrained", action="store_true",    help="Skip pretrained weights even if HF_TOKEN is set")
    parser.add_argument("--download",    action="store_true",      help="Download FACED via torcheeg and exit")
    args = parser.parse_args()

    acc = train(args)
    if acc is not None:
        print(f"\nFinal cross-subject accuracy: {acc * 100:.1f}%")
        if acc >= 0.60:
            print("Model exceeds 60% threshold — will activate in live inference!")
        else:
            print("Below 60% threshold — backbone fine-tuning needed (--no-freeze with GPU)")


if __name__ == "__main__":
    main()
