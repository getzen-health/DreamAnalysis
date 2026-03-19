"""Train CNN-KAN with Domain-Adversarial Neural Network (DANN) for cross-subject EEG.

Domain-adversarial training forces the shared CNN backbone to learn
subject-invariant features by adding a gradient reversal branch that tries —
and is prevented — from identifying which subject produced each epoch.

Architecture:
    Shared CNN → KAN emotion head  (emotion loss, cross-entropy)
              ↘ GradientReversal → DomainClassifier  (domain loss, cross-entropy)

    total_loss = emotion_loss + lambda_domain * domain_loss

    During back-prop, the domain loss gradients are NEGATED before reaching the
    shared CNN, pushing it toward representations that fool the domain classifier.

Alpha schedule (Ganin et al.):
    alpha ramps from 0 → 1.0 over training using:
        p     = epoch / total_epochs
        alpha = 2 / (1 + exp(-10 * p)) - 1

Cross-subject evaluation:
    Leave-one-subject-out (LOSO) CV — train on all subjects except one,
    evaluate on the held-out subject. Reported accuracy is the mean LOSO
    accuracy across all subjects, which is the correct metric for cross-subject
    generalisation (no data leakage between subjects).

Saved artefacts:
    ml/models/saved/cnn_kan_dann_emotion.pt        DANN model weights + config
    ml/models/saved/cnn_kan_dann_emotion_bench.txt  Mean LOSO accuracy (single float)

Usage examples:
    # Smoke-test with synthetic data (fast):
    python -m training.train_cnn_kan_dann --synthetic --epochs 30

    # Train on DEAP .dat files (32 subjects, full LOSO CV):
    python -m training.train_cnn_kan_dann --deap-dir data/deap --epochs 100

    # GPU / Apple Silicon:
    python -m training.train_cnn_kan_dann --synthetic --device mps

    # Adjust domain-loss weight:
    python -m training.train_cnn_kan_dann --synthetic --lambda-domain 0.5

    # Skip LOSO (use simple val split for speed):
    python -m training.train_cnn_kan_dann --deap-dir data/deap --no-loso --epochs 80
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_ML_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ML_ROOT))

# ── project imports ───────────────────────────────────────────────────────────
from models.cnn_kan import (
    DANNCNNKANModel,
    EMOTION_CLASSES,
    alpha_schedule,
)

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
N_CLASSES   = len(EMOTION_CLASSES)   # 3
N_CHANNELS  = 4
FS          = 256.0
EPOCH_SEC   = 4.0
N_SAMPLES   = int(EPOCH_SEC * FS)    # 1024
SAVED_DIR   = _ML_ROOT / "models" / "saved"


# ── synthetic data (smoke-test) ───────────────────────────────────────────────

def make_synthetic_data(
    n_subjects: int = 8,
    n_per_class_per_subject: int = 60,
    n_channels: int = N_CHANNELS,
    fs: float = FS,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate subject-labelled synthetic EEG epochs.

    Each subject has a slightly different amplitude bias to simulate realistic
    inter-subject variability.

    Returns:
        X:        (n_total, n_channels, n_samples) float32
        y_emotion:(n_total,) int64 — emotion labels 0/1/2
        y_subject:(n_total,) int64 — subject IDs 0..n_subjects-1
    """
    rng = np.random.default_rng(seed)
    n_samples = int(EPOCH_SEC * fs)
    t = np.linspace(0, EPOCH_SEC, n_samples, endpoint=False)

    # (alpha_amp, beta_amp, high_beta_amp, theta_amp, noise_std)
    signatures = [
        (2.5, 0.6, 0.2, 0.5, 0.4),   # positive: dominant alpha
        (1.0, 1.0, 0.5, 0.8, 0.5),   # neutral:  balanced
        (0.4, 2.0, 1.0, 0.4, 0.6),   # negative: dominant beta
    ]

    X_list, y_e_list, y_s_list = [], [], []

    for subj in range(n_subjects):
        # Inter-subject variability: random scale factor for amplitudes
        subj_scale = rng.uniform(0.7, 1.4)

        for label, (a_amp, b_amp, hb_amp, th_amp, noise) in enumerate(signatures):
            for _ in range(n_per_class_per_subject):
                epoch = np.zeros((n_channels, n_samples), dtype=np.float32)
                for ch in range(n_channels):
                    phase_a  = rng.uniform(0, 2 * np.pi)
                    phase_b  = rng.uniform(0, 2 * np.pi)
                    phase_hb = rng.uniform(0, 2 * np.pi)
                    phase_th = rng.uniform(0, 2 * np.pi)

                    faa_mod = 1.1 if (label == 0 and ch == 2) else 1.0
                    faa_mod = 0.9 if (label == 2 and ch == 2) else faa_mod

                    sig = (
                        faa_mod * a_amp  * np.sin(2 * np.pi * 10.0 * t + phase_a)
                        + b_amp          * np.sin(2 * np.pi * 18.0 * t + phase_b)
                        + hb_amp         * np.sin(2 * np.pi * 24.0 * t + phase_hb)
                        + th_amp         * np.sin(2 * np.pi *  6.0 * t + phase_th)
                        + rng.normal(0, noise, n_samples)
                    )
                    epoch[ch] = (sig * subj_scale).astype(np.float32)

                X_list.append(epoch)
                y_e_list.append(label)
                y_s_list.append(subj)

    X         = np.stack(X_list)
    y_emotion = np.array(y_e_list, dtype=np.int64)
    y_subject = np.array(y_s_list, dtype=np.int64)

    perm = rng.permutation(len(y_emotion))
    return X[perm], y_emotion[perm], y_subject[perm]


# ── DEAP data loading ─────────────────────────────────────────────────────────

def load_deap_with_subjects(
    data_dir: str,
    epoch_sec: float = EPOCH_SEC,
    target_fs: float = FS,
    n_classes: int = N_CLASSES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DEAP as raw 4-channel epochs with per-subject domain labels.

    Wraps training.data_loaders.load_deap_raw_4ch but also returns the
    subject ID for every epoch so DANN can train domain adversarially.

    DEAP: 32 subjects × 40 trials × multiple epochs per trial.

    Returns:
        X:         (n_total, 4, n_samples) float32
        y_emotion: (n_total,) int64 — 3-class valence labels
        y_subject: (n_total,) int64 — subject IDs 0..31
    """
    import pickle
    from scipy.signal import resample

    # DEAP channel indices for Muse 2-equivalent positions
    # Order: TP9(T7=7), AF7(AF3=1), AF8(AF4=17), TP10(T8=23)
    MUSE_INDICES   = [7, 1, 17, 23]
    DEAP_FS        = 128.0
    BASELINE_SAMP  = int(3 * DEAP_FS)   # 3 s pre-trial baseline

    deap_path = Path(data_dir)
    dat_files = sorted(deap_path.glob("s*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No DEAP .dat files in {data_dir}")

    samples_per_epoch      = int(epoch_sec * target_fs)
    deap_samples_per_epoch = int(epoch_sec * DEAP_FS)

    X_all, y_e_all, y_s_all = [], [], []

    for subj_idx, dat_file in enumerate(dat_files):
        try:
            with open(dat_file, "rb") as f:
                subject = pickle.load(f, encoding="latin1")
        except Exception as exc:
            log.warning("Skipping %s: %s", dat_file.name, exc)
            continue

        data   = subject["data"]     # (40, 40, 8064)
        labels = subject["labels"]   # (40, 4) — valence, arousal, dominance, liking

        for trial_idx in range(data.shape[0]):
            valence = labels[trial_idx, 0]   # 1–9 scale

            if n_classes == 3:
                if valence > 5.5:
                    label = 0  # positive
                elif valence < 4.5:
                    label = 2  # negative
                else:
                    label = 1  # neutral
            else:
                # 6-class circumplex
                arousal = labels[trial_idx, 1]
                if valence > 5.5 and arousal > 5.0:
                    label = 0
                elif valence < 4.5 and arousal < 4.5:
                    label = 1
                elif valence < 4.5 and arousal > 5.5:
                    label = 2
                elif valence < 4.0 and arousal > 5.0:
                    label = 3
                elif arousal > 7.0:
                    label = 4
                else:
                    label = 5

            trial_eeg = data[trial_idx, MUSE_INDICES, BASELINE_SAMP:]  # (4, ~7680)
            n_deap    = trial_eeg.shape[1]
            hop       = deap_samples_per_epoch

            for start in range(0, n_deap - deap_samples_per_epoch + 1, hop):
                chunk     = trial_eeg[:, start:start + deap_samples_per_epoch]
                resampled = resample(chunk, samples_per_epoch, axis=1)
                X_all.append(resampled.astype(np.float32))
                y_e_all.append(label)
                y_s_all.append(subj_idx)

    if not X_all:
        raise RuntimeError("No epochs extracted from DEAP")

    X         = np.array(X_all)
    y_emotion = np.array(y_e_all, dtype=np.int64)
    y_subject = np.array(y_s_all, dtype=np.int64)

    log.info(
        "DEAP DANN: %d epochs, %d subjects, class dist=%s",
        len(X), len(dat_files),
        dict(zip(*np.unique(y_emotion, return_counts=True))),
    )
    return X, y_emotion, y_subject


# ── preprocessing ─────────────────────────────────────────────────────────────

def normalize_epochs(X: np.ndarray) -> np.ndarray:
    """Per-channel, per-epoch z-score normalisation."""
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True) + 1e-7
    return (X - mean) / std


# ── DANN training loop ────────────────────────────────────────────────────────

def train_dann(
    X: np.ndarray,
    y_emotion: np.ndarray,
    y_subject: np.ndarray,
    n_channels: int = N_CHANNELS,
    grid_size: int = 5,
    spline_order: int = 3,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_domain: float = 0.3,
    val_split: float = 0.15,
    patience: int = 20,
    device: str = "cpu",
) -> Tuple["DANNCNNKANModel", float]:
    """Train DANNCNNKANModel with combined emotion + adversarial domain loss.

    Loss:
        emotion_loss = CrossEntropy(emotion_logits, y_emotion)
        domain_loss  = CrossEntropy(domain_logits,  y_subject)
        total        = emotion_loss + lambda_domain * domain_loss

    The gradient reversal layer inside DomainClassifier negates the domain
    loss gradients before they reach the shared CNN backbone, making the
    backbone learn subject-invariant representations.

    Alpha schedule: Ganin et al. ramp from 0 → 1.0 over all epochs.

    Args:
        X:             (n_total, n_channels, n_samples) float32
        y_emotion:     (n_total,) int64 — emotion class labels
        y_subject:     (n_total,) int64 — subject / domain IDs
        n_channels:    Number of EEG channels
        grid_size:     KAN B-spline grid size
        spline_order:  KAN B-spline polynomial order
        epochs:        Maximum training epochs
        batch_size:    Minibatch size
        lr:            Initial learning rate (AdamW)
        lambda_domain: Weight for domain adversarial loss (typically 0.1–1.0)
        val_split:     Fraction of data held out for validation
        patience:      Early stopping patience (epochs with no emotion val improvement)
        device:        PyTorch device ("cpu", "cuda", "mps")

    Returns:
        model:    Best DANNCNNKANModel (restored to best emotion-val checkpoint)
        best_acc: Best emotion validation accuracy
    """
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, TensorDataset, random_split

    n_subjects = int(y_subject.max()) + 1
    log.info(
        "DANN training — epochs=%d, n=%d, subjects=%d, "
        "lambda_domain=%.2f, grid_size=%d, device=%s",
        epochs, len(y_emotion), n_subjects, lambda_domain, grid_size, device,
    )

    X = normalize_epochs(X)

    # Class-balanced emotion weights
    counts = np.bincount(y_emotion, minlength=N_CLASSES).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    emotion_weights = torch.tensor(1.0 / counts, dtype=torch.float32, device=device)

    X_t       = torch.from_numpy(X)
    ye_t      = torch.from_numpy(y_emotion)
    ys_t      = torch.from_numpy(y_subject)
    dataset   = TensorDataset(X_t, ye_t, ys_t)

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(0)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = DANNCNNKANModel(
        n_channels=n_channels,
        n_classes=N_CLASSES,
        n_subjects=n_subjects,
        grid_size=grid_size,
        spline_order=spline_order,
    ).to(device)

    log.info(
        "DANNCNNKANModel: %d total params  "
        "(emotion head: %d, domain head: %d)",
        model.count_parameters(),
        sum(p.numel() for p in list(model.kan1.parameters())
            + list(model.kan2.parameters()) if p.requires_grad),
        sum(p.numel() for p in model.domain_classifier.parameters()
            if p.requires_grad),
    )

    optimizer  = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)
    emotion_crit = nn.CrossEntropyLoss(weight=emotion_weights)
    domain_crit  = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        alpha = alpha_schedule(epoch, epochs)

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        e_loss_sum = 0.0
        d_loss_sum = 0.0

        for xb, ye_b, ys_b in train_loader:
            xb   = xb.to(device)
            ye_b = ye_b.to(device)
            ys_b = ys_b.to(device)

            optimizer.zero_grad()

            emotion_logits, domain_logits = model(xb, alpha=alpha)

            e_loss = emotion_crit(emotion_logits, ye_b)
            d_loss = domain_crit(domain_logits, ys_b)
            loss   = e_loss + lambda_domain * d_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            e_loss_sum += e_loss.item()
            d_loss_sum += d_loss.item()

        scheduler.step()
        n_batches = max(len(train_loader), 1)

        # ── validate (emotion accuracy only — what matters for deployment) ─
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, ye_b, _ in val_loader:
                xb   = xb.to(device)
                ye_b = ye_b.to(device)
                preds = model.predict_emotion(xb).argmax(dim=1)
                correct += (preds == ye_b).sum().item()
                total   += len(ye_b)

        val_acc = correct / max(total, 1)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d  total_loss=%.4f  e_loss=%.4f  d_loss=%.4f  "
                "val_acc=%.1f%%  alpha=%.3f",
                epoch, epochs,
                total_loss / n_batches,
                e_loss_sum / n_batches,
                d_loss_sum / n_batches,
                val_acc * 100,
                alpha,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(
                    "Early stopping at epoch %d — best val_acc=%.1f%%",
                    epoch, best_val_acc * 100,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    log.info("DANN training complete — best val_acc=%.1f%%", best_val_acc * 100)
    return model, best_val_acc


# ── Leave-One-Subject-Out CV ──────────────────────────────────────────────────

def loso_cross_validation(
    X: np.ndarray,
    y_emotion: np.ndarray,
    y_subject: np.ndarray,
    n_channels: int = N_CHANNELS,
    grid_size: int = 5,
    spline_order: int = 3,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_domain: float = 0.3,
    device: str = "cpu",
) -> Dict[str, float]:
    """Leave-One-Subject-Out cross-validation for DANN.

    This is the gold-standard evaluation for cross-subject EEG models.
    For each subject S:
        - Train DANN on all subjects except S
        - Evaluate on subject S (unseen during training)

    Returns the mean accuracy and per-subject breakdown.

    Args:
        X:             (n_total, n_channels, n_samples)
        y_emotion:     (n_total,) int64
        y_subject:     (n_total,) int64
        (training hyperparams as in train_dann)

    Returns:
        results dict with keys:
            mean_accuracy:   float — average over all LOSO folds
            per_subject:     List[float] — accuracy for each held-out subject
            n_subjects:      int
    """
    import torch

    subjects = np.unique(y_subject)
    n_subj   = len(subjects)
    accs: List[float] = []

    log.info("LOSO CV: %d subjects, %d epochs per fold", n_subj, epochs)

    for fold, test_subj in enumerate(subjects):
        train_mask = y_subject != test_subj
        test_mask  = y_subject == test_subj

        X_train, ye_train, ys_train = (
            X[train_mask], y_emotion[train_mask], y_subject[train_mask]
        )
        X_test,  ye_test            = X[test_mask], y_emotion[test_mask]

        if len(X_test) == 0:
            log.warning("Subject %d has no test epochs — skipping", test_subj)
            continue

        # Re-index subject IDs for training fold (remove the gap left by
        # removing the test subject so IDs are contiguous 0..n_train_subj-1)
        unique_train_subj = np.unique(ys_train)
        subj_remap = {s: i for i, s in enumerate(unique_train_subj)}
        ys_train_reindexed = np.array(
            [subj_remap[s] for s in ys_train], dtype=np.int64
        )

        log.info(
            "Fold %2d/%d — test subject %d  train=%d  test=%d",
            fold + 1, n_subj, test_subj, len(X_train), len(X_test),
        )

        model, _ = train_dann(
            X_train, ye_train, ys_train_reindexed,
            n_channels=n_channels,
            grid_size=grid_size,
            spline_order=spline_order,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lambda_domain=lambda_domain,
            val_split=0.1,
            patience=15,
            device=device,
        )

        # Evaluate on the held-out subject
        X_test_norm = normalize_epochs(X_test)
        x_t  = torch.from_numpy(X_test_norm).to(device)
        ye_t = torch.from_numpy(ye_test).to(device)

        model.eval()
        with torch.no_grad():
            logits = model.predict_emotion(x_t)
            preds  = logits.argmax(dim=1)
            acc    = (preds == ye_t).float().mean().item()

        accs.append(acc)
        log.info(
            "  Subject %2d held-out accuracy: %.1f%%", test_subj, acc * 100
        )

    mean_acc = float(np.mean(accs)) if accs else 0.0
    log.info(
        "LOSO CV complete — mean accuracy=%.1f%% over %d subjects",
        mean_acc * 100, len(accs),
    )
    return {
        "mean_accuracy": mean_acc,
        "per_subject": accs,
        "n_subjects": len(accs),
    }


# ── save helpers ──────────────────────────────────────────────────────────────

def save_artefacts(model: "DANNCNNKANModel", accuracy: float) -> None:
    """Save DANN model weights and benchmark text to models/saved/."""
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    pt_path = SAVED_DIR / "cnn_kan_dann_emotion.pt"
    model.save(pt_path)
    print(f"DANN model saved  → {pt_path}")

    bench_path = SAVED_DIR / "cnn_kan_dann_emotion_bench.txt"
    bench_path.write_text(f"{accuracy:.6f}")
    print(f"Benchmark         → {bench_path}  ({accuracy * 100:.2f}%)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Train CNN-KAN with Domain-Adversarial Neural Network (DANN) "
            "for cross-subject EEG emotion classification."
        )
    )

    # Data sources
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic multi-subject EEG data (smoke-test)",
    )
    data_group.add_argument(
        "--deap-dir", type=Path, default=None,
        help="Path to DEAP .dat files (s01.dat … s32.dat)",
    )

    # DANN hyperparameters
    parser.add_argument(
        "--lambda-domain", type=float, default=0.3,
        help="Domain loss weight (default 0.3). Higher = stronger subject-invariance pressure.",
    )

    # Model hyperparameters
    parser.add_argument("--grid-size",    type=int,   default=5,  help="KAN B-spline grid size")
    parser.add_argument("--spline-order", type=int,   default=3,  help="KAN B-spline order")
    parser.add_argument("--channels",     type=int,   default=N_CHANNELS)

    # Training hyperparameters
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--patience",     type=int,   default=20)
    parser.add_argument(
        "--n-subjects-synth", type=int, default=8,
        help="Number of synthetic subjects (ignored when --deap-dir is used)",
    )
    parser.add_argument(
        "--n-per-class-synth", type=int, default=60,
        help="Synthetic epochs per class per subject",
    )

    # Evaluation mode
    parser.add_argument(
        "--no-loso", action="store_true",
        help=(
            "Skip leave-one-subject-out CV and use a simple val split instead. "
            "Much faster but does not measure true cross-subject generalisation."
        ),
    )
    parser.add_argument(
        "--loso-epochs", type=int, default=None,
        help=(
            "Epochs per LOSO fold (default: same as --epochs). "
            "Set lower than --epochs to run LOSO faster."
        ),
    )

    # Hardware
    parser.add_argument(
        "--device", type=str, default=None,
        help="PyTorch device: cpu, cuda, mps. Auto-detected if not set.",
    )

    args = parser.parse_args()

    # ── PyTorch check ──────────────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    # ── Device selection ──────────────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Using device: %s", device)

    # ── Load data ──────────────────────────────────────────────────────────
    if args.synthetic:
        log.info(
            "Generating synthetic DANN data — %d subjects, %d per class per subject",
            args.n_subjects_synth, args.n_per_class_synth,
        )
        X, y_emotion, y_subject = make_synthetic_data(
            n_subjects=args.n_subjects_synth,
            n_per_class_per_subject=args.n_per_class_synth,
            n_channels=args.channels,
        )
    else:
        log.info("Loading DEAP from %s", args.deap_dir)
        X, y_emotion, y_subject = load_deap_with_subjects(
            str(args.deap_dir),
            epoch_sec=EPOCH_SEC,
            target_fs=FS,
            n_classes=N_CLASSES,
        )

    if len(y_emotion) == 0:
        print("ERROR: No training data. Check --deap-dir path or use --synthetic.")
        sys.exit(1)

    n_subjects = int(y_subject.max()) + 1
    log.info(
        "Dataset: %d epochs, %d subjects | class dist: %s",
        len(y_emotion), n_subjects,
        dict(zip(*np.unique(y_emotion, return_counts=True))),
    )

    # Pad/trim to expected n_samples
    actual = X.shape[2]
    if actual != N_SAMPLES:
        if actual > N_SAMPLES:
            X = X[:, :, :N_SAMPLES]
        else:
            X = np.pad(X, ((0, 0), (0, 0), (0, N_SAMPLES - actual)), mode="edge")
        log.info("Resized epochs: %d → %d samples", actual, N_SAMPLES)

    # ── Train final DANN model on all data ─────────────────────────────────
    log.info("Training final DANN model on full dataset…")
    final_model, final_val_acc = train_dann(
        X, y_emotion, y_subject,
        n_channels=args.channels,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_domain=args.lambda_domain,
        val_split=0.15,
        patience=args.patience,
        device=device,
    )

    # ── Cross-subject evaluation ───────────────────────────────────────────
    if args.no_loso:
        log.info("Skipping LOSO CV (--no-loso set). Final val acc: %.1f%%", final_val_acc * 100)
        report_acc = final_val_acc
    else:
        loso_epochs = args.loso_epochs if args.loso_epochs else args.epochs
        log.info("Running LOSO CV (%d epochs per fold)…", loso_epochs)
        loso_results = loso_cross_validation(
            X, y_emotion, y_subject,
            n_channels=args.channels,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            epochs=loso_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_domain=args.lambda_domain,
            device=device,
        )
        report_acc = loso_results["mean_accuracy"]
        print("\n──── LOSO Cross-Subject Results ────")
        for i, acc in enumerate(loso_results["per_subject"]):
            print(f"  Subject {i:2d}: {acc * 100:.1f}%")
        print(f"  Mean LOSO accuracy: {report_acc * 100:.2f}%")
        print("────────────────────────────────────\n")

    # ── Save ───────────────────────────────────────────────────────────────
    save_artefacts(final_model, report_acc)

    print(
        f"\nDone. DANN cross-subject accuracy: {report_acc * 100:.2f}% "
        f"({'LOSO' if not args.no_loso else 'val split'})"
    )
    if report_acc < 0.45:
        print(
            "NOTE: Accuracy below 45% — training may need more data or hyperparameter "
            "tuning. Baseline chance is 33.3% for 3-class. "
            "Try --lambda-domain 0.5 or --epochs 150."
        )


if __name__ == "__main__":
    main()
