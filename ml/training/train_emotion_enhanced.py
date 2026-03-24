"""Training script for the enhanced emotion classifier.

Loads a pretrained SSL encoder, combines with 53-dim hand-crafted features,
and trains the classifier head on synthetic emotional EEG data.

Usage:
    python -m training.train_emotion_enhanced [--pretrained_encoder models/saved/eeg_ssl_encoder.pt]

Output:
    models/saved/enhanced_emotion_classifier.pt
    models/saved/enhanced_emotion_classifier.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.eeg_emotion_enhanced import EMOTIONS, EnhancedEmotionClassifier
from models.eeg_ssl import EEGContrastiveEncoder
from processing.emotion_features_enhanced import (
    ENHANCED_FEATURE_DIM,
    extract_enhanced_emotion_features,
    extract_temporal_features,
)
from processing.focal_loss import FocalLoss, focal_mixup_cross_entropy
from processing.mixup_augmentation import mixup_batch, mixup_cross_entropy, to_one_hot
from training.pretrain_eeg_ssl import _BAND_CENTERS, _generate_epoch

log = logging.getLogger(__name__)

# Emotion-specific frequency profiles for synthetic training data
_EMOTION_PROFILES = {
    "happy": {"delta": 0.08, "theta": 0.12, "alpha": 0.45, "beta": 0.25, "gamma": 0.10},
    "sad": {"delta": 0.20, "theta": 0.25, "alpha": 0.30, "beta": 0.15, "gamma": 0.10},
    "angry": {"delta": 0.05, "theta": 0.08, "alpha": 0.07, "beta": 0.55, "gamma": 0.25},
    "fearful": {"delta": 0.05, "theta": 0.15, "alpha": 0.05, "beta": 0.45, "gamma": 0.30},
    "relaxed": {"delta": 0.12, "theta": 0.15, "alpha": 0.50, "beta": 0.13, "gamma": 0.10},
    "focused": {"delta": 0.05, "theta": 0.08, "alpha": 0.12, "beta": 0.50, "gamma": 0.25},
}

# Asymmetry profiles (FAA sign and magnitude)
_EMOTION_ASYMMETRY = {
    "happy": 1.2,   # Strong left-frontal (positive FAA)
    "sad": -1.0,    # Right-frontal (negative FAA)
    "angry": 0.8,   # Left-frontal (approach motivation)
    "fearful": -0.8, # Right-frontal (withdrawal)
    "relaxed": 0.3,  # Slight left bias
    "focused": 0.0,  # Symmetric
}


def generate_emotional_eeg(
    emotion: str,
    n_samples: int = 1024,
    fs: float = 256.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 4-channel synthetic EEG epoch with emotion-specific features.

    Includes frontal asymmetry matching the emotion's valence profile.

    Args:
        emotion: One of the 6 emotion labels.
        n_samples: Samples per epoch.
        fs: Sampling rate.
        rng: NumPy random generator.

    Returns:
        (4, n_samples) array.
    """
    if rng is None:
        rng = np.random.default_rng()

    profile = _EMOTION_PROFILES.get(emotion, _EMOTION_PROFILES["relaxed"])
    asym = _EMOTION_ASYMMETRY.get(emotion, 0.0)

    # Generate base epoch from profile
    varied = {}
    for band, power in profile.items():
        varied[band] = max(0.01, power + rng.uniform(-0.08, 0.08))
    total = sum(varied.values())
    varied = {k: v / total for k, v in varied.items()}

    epoch = _generate_epoch(varied, n_channels=4, n_samples=n_samples, fs=fs, rng=rng)

    # Inject frontal alpha asymmetry
    # AF7 = ch1 (left), AF8 = ch2 (right)
    t = np.arange(n_samples) / fs
    alpha_freq = 10.0 + rng.uniform(-1.0, 1.0)

    # Positive FAA means more right alpha = less right activation = more left activation
    right_alpha_boost = asym * 8.0 * rng.uniform(0.5, 1.5)
    epoch[2] += right_alpha_boost * np.sin(2 * np.pi * alpha_freq * t + rng.uniform(0, 2 * np.pi))

    # Also inject temporal asymmetry for better discrimination
    temporal_asym = asym * 0.5 * rng.uniform(0.3, 1.0)
    epoch[3] += temporal_asym * 5.0 * np.sin(2 * np.pi * alpha_freq * t + rng.uniform(0, 2 * np.pi))

    return epoch


def generate_training_data(
    n_per_class: int = 500,
    n_samples: int = 1024,
    fs: int = 256,
    seed: int = 42,
    include_temporal: bool = True,
) -> tuple:
    """Generate labeled training data with hand-crafted features.

    Args:
        n_per_class: Number of epochs per emotion class.
        n_samples: Samples per epoch.
        fs: Sampling rate.
        seed: Random seed.
        include_temporal: Whether to generate temporal delta features.

    Returns:
        (raw_eeg, features, labels) tuple:
            raw_eeg: (N, 4, n_samples) array
            features: (N, 106) or (N, 53) array
            labels: (N,) integer labels
    """
    rng = np.random.default_rng(seed)
    n_total = n_per_class * len(EMOTIONS)

    raw_eeg = np.zeros((n_total, 4, n_samples), dtype=np.float32)
    feat_dim = ENHANCED_FEATURE_DIM * 2 if include_temporal else ENHANCED_FEATURE_DIM
    features = np.zeros((n_total, feat_dim), dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.int64)

    idx = 0
    for class_idx, emotion in enumerate(EMOTIONS):
        history = []
        for j in range(n_per_class):
            epoch = generate_emotional_eeg(emotion, n_samples, fs, rng)
            raw_eeg[idx] = epoch.astype(np.float32)

            # Extract hand-crafted features
            hc = extract_enhanced_emotion_features(epoch, fs=fs)

            if include_temporal:
                temporal = extract_temporal_features(hc, history=history if j > 0 else None)
                features[idx] = temporal.astype(np.float32)
                history.append(hc)
                # Keep only last 5 for memory
                if len(history) > 5:
                    history.pop(0)
            else:
                features[idx] = hc.astype(np.float32)

            labels[idx] = class_idx
            idx += 1

    return raw_eeg, features, labels


def train(
    pretrained_encoder_path: str | None = None,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_per_class: int = 500,
    include_temporal: bool = True,
    save_dir: str | None = None,
    export_onnx: bool = True,
    seed: int = 42,
    mixup_alpha: float = 0.4,
) -> dict:
    """Train the enhanced emotion classifier.

    Args:
        pretrained_encoder_path: Path to pretrained SSL encoder. If None, uses untrained encoder.
        n_epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        n_per_class: Samples per emotion class for training data.
        include_temporal: Include temporal delta features.
        save_dir: Directory to save models. Defaults to models/saved/.
        export_onnx: Whether to export ONNX after training.
        seed: Random seed.
        mixup_alpha: Mixup interpolation strength. 0 disables. 0.2-0.4 recommended.

    Returns:
        Dict with training stats.
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch required", "torch_available": False}

    if save_dir is None:
        save_dir = str(Path(__file__).parent.parent / "models" / "saved")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load pretrained encoder if available
    encoder = None
    if pretrained_encoder_path and Path(pretrained_encoder_path).exists():
        log.info("Loading pretrained encoder from %s", pretrained_encoder_path)
        encoder = EEGContrastiveEncoder.load(pretrained_encoder_path)

    # Generate training data
    log.info("Generating training data: %d samples per class...", n_per_class)
    raw_eeg, features, labels = generate_training_data(
        n_per_class=n_per_class,
        include_temporal=include_temporal,
        seed=seed,
    )

    # Split: 80% train, 20% val
    n_total = len(labels)
    indices = np.random.RandomState(seed).permutation(n_total)
    split = int(0.8 * n_total)
    train_idx, val_idx = indices[:split], indices[split:]

    train_eeg = torch.FloatTensor(raw_eeg[train_idx])
    train_feat = torch.FloatTensor(features[train_idx])
    train_labels = torch.LongTensor(labels[train_idx])

    val_eeg = torch.FloatTensor(raw_eeg[val_idx])
    val_feat = torch.FloatTensor(features[val_idx])
    val_labels = torch.LongTensor(labels[val_idx])

    train_ds = TensorDataset(train_eeg, train_feat, train_labels)
    val_ds = TensorDataset(val_eeg, val_feat, val_labels)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Build model
    model = EnhancedEmotionClassifier(
        n_classes=len(EMOTIONS),
        pretrained_encoder=encoder,
        include_temporal=include_temporal,
        freeze_encoder=True,
    )

    # Only train classifier head parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)  # focal loss for validation + non-mixup training
    use_mixup = mixup_alpha > 0.0

    best_val_acc = 0.0
    loss_history = []
    val_acc_history = []

    log.info("Training enhanced classifier: %d epochs, %d trainable params, mixup=%s",
             n_epochs, sum(p.numel() for p in trainable_params),
             f"alpha={mixup_alpha}" if use_mixup else "disabled")

    for epoch in range(n_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_correct = 0
        n_total_train = 0

        for batch_eeg, batch_feat, batch_labels in train_dl:
            if use_mixup:
                # For dual-input model: apply same mixup permutation to both inputs
                bs = batch_eeg.shape[0]
                if bs > 1:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    lam = max(lam, 1.0 - lam)
                    idx = torch.randperm(bs)
                    batch_eeg_mix = lam * batch_eeg + (1.0 - lam) * batch_eeg[idx]
                    batch_feat_mix = lam * batch_feat + (1.0 - lam) * batch_feat[idx]
                    y_onehot = F.one_hot(batch_labels, num_classes=len(EMOTIONS)).float()
                    y_mix = lam * y_onehot + (1.0 - lam) * y_onehot[idx]
                    logits = model(batch_eeg_mix, batch_feat_mix)
                    loss = focal_mixup_cross_entropy(logits, y_mix, gamma=2.0)
                else:
                    logits = model(batch_eeg, batch_feat)
                    loss = criterion(logits, batch_labels)
            else:
                logits = model(batch_eeg, batch_feat)
                loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(batch_labels)
            # For accuracy tracking, use argmax of logits vs original hard labels
            n_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            n_total_train += len(batch_labels)

        train_loss = epoch_loss / max(n_total_train, 1)
        train_acc = n_correct / max(n_total_train, 1)

        # Validate
        model.eval()
        val_correct = 0
        n_val = 0
        with torch.no_grad():
            for batch_eeg, batch_feat, batch_labels in val_dl:
                logits = model(batch_eeg, batch_feat)
                val_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                n_val += len(batch_labels)

        val_acc = val_correct / max(n_val, 1)
        scheduler.step(1.0 - val_acc)

        loss_history.append(train_loss)
        val_acc_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info(
                "Epoch %d/%d — loss: %.4f — train_acc: %.2f%% — val_acc: %.2f%%",
                epoch + 1, n_epochs, train_loss, train_acc * 100, val_acc * 100,
            )

    # Save model
    model_path = str(Path(save_dir) / "enhanced_emotion_classifier.pt")
    model.save(model_path)
    log.info("Model saved -> %s", model_path)

    # Export ONNX
    onnx_path = None
    if export_onnx:
        onnx_path = str(Path(save_dir) / "enhanced_emotion_classifier.onnx")
        try:
            model.export_onnx(onnx_path)
            log.info("ONNX exported -> %s", onnx_path)
        except Exception as e:
            log.warning("ONNX export failed: %s", e)
            onnx_path = None

    return {
        "best_val_acc": best_val_acc,
        "final_train_loss": loss_history[-1] if loss_history else None,
        "val_acc_history": val_acc_history,
        "loss_history": loss_history,
        "n_epochs_trained": n_epochs,
        "model_path": model_path,
        "onnx_path": onnx_path,
        "n_classes": len(EMOTIONS),
        "n_training_samples": len(train_labels),
        "n_val_samples": len(val_labels),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train enhanced emotion classifier")
    parser.add_argument("--pretrained_encoder", type=str, default=None,
                        help="Path to pretrained SSL encoder checkpoint")
    parser.add_argument("--n_epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n_per_class", type=int, default=500, help="Samples per class")
    parser.add_argument("--no_temporal", action="store_true", help="Disable temporal features")
    parser.add_argument("--no_onnx", action="store_true", help="Skip ONNX export")
    args = parser.parse_args()

    result = train(
        pretrained_encoder_path=args.pretrained_encoder,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_per_class=args.n_per_class,
        include_temporal=not args.no_temporal,
        export_onnx=not args.no_onnx,
    )

    print(f"\nTraining complete:")
    print(f"  Best val accuracy: {result.get('best_val_acc', 0) * 100:.2f}%")
    print(f"  Model saved: {result.get('model_path', 'N/A')}")
    print(f"  ONNX saved: {result.get('onnx_path', 'N/A')}")
