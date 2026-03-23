"""Teacher-student knowledge distillation for EEG emotion models.

Distills a large 32-channel DEAP-trained teacher model into a compact
4-channel Muse 2 student model using soft-label training with KL
divergence loss. This is a TRAINING script, not inference.

Teacher-student distillation approach:
    Teacher: 32-channel model trained on full DEAP data (high accuracy,
             cannot run on Muse 2 due to channel mismatch).
    Student: 4-channel model matching Muse 2 electrode positions
             (TP9, AF7, AF8, TP10).

The student learns from the teacher's soft probability distributions
rather than hard one-hot labels. Soft labels carry inter-class
similarity information (e.g., "this sample is 70% happy, 20% surprise,
10% neutral") which is lost in hard labels.

Temperature scaling (Hinton et al., 2015):
    - Higher temperature (T > 1) softens probabilities, revealing
      dark knowledge about class relationships.
    - T=3-5 is typical for emotion classification.
    - Loss = alpha * KL(soft_student || soft_teacher) + (1-alpha) * CE(student, hard_label)
    - alpha=0.7 emphasizes soft knowledge transfer.

Requirements:
    - Pre-trained teacher model (32-channel DEAP, any architecture)
    - DEAP dataset with raw signals (to extract 4-channel subset)
    - PyTorch >= 2.0

Usage:
    cd ml
    python -m training.distillation --teacher-path models/saved/teacher_32ch.pt \\
                                     --deap-path data/deap/ \\
                                     --output models/saved/student_muse4ch.pt \\
                                     --temperature 4.0 \\
                                     --alpha 0.7 \\
                                     --epochs 50

Reference:
    Hinton, Vinyals & Dean (2015) "Distilling the Knowledge in a Neural Network"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# 6-class emotion labels matching the rest of the pipeline
EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]

# DEAP 32-channel indices for Muse 2 equivalent positions
# Muse 2: TP9 (left temporal), AF7 (left frontal), AF8 (right frontal), TP10 (right temporal)
MUSE_CHANNEL_MAP = {
    "AF7": 1,    # AF3 in DEAP (closest to AF7)
    "AF8": 17,   # AF4 in DEAP (closest to AF8)
    "TP9": 7,    # T7 in DEAP (closest to TP9)
    "TP10": 23,  # T8 in DEAP (closest to TP10)
}
MUSE_CH_INDICES = list(MUSE_CHANNEL_MAP.values())


def extract_muse_channels(signals_32ch: np.ndarray) -> np.ndarray:
    """Extract Muse 2-equivalent channels from 32-channel DEAP data.

    Args:
        signals_32ch: (32, n_samples) array from DEAP.

    Returns:
        (4, n_samples) array with [TP9, AF7, AF8, TP10] ordering
        to match Muse 2 BrainFlow channel delivery.
    """
    if signals_32ch.shape[0] < 24:
        raise ValueError(
            f"Expected 32-channel input, got {signals_32ch.shape[0]} channels"
        )
    # Reorder to Muse 2 order: TP9, AF7, AF8, TP10
    indices = [
        MUSE_CHANNEL_MAP["TP9"],   # ch0
        MUSE_CHANNEL_MAP["AF7"],   # ch1
        MUSE_CHANNEL_MAP["AF8"],   # ch2
        MUSE_CHANNEL_MAP["TP10"],  # ch3
    ]
    return signals_32ch[indices]


def kl_divergence_loss(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float = 4.0,
) -> float:
    """Compute KL divergence between teacher and student soft predictions.

    Uses temperature scaling to soften probability distributions.

    Args:
        student_logits: (n_classes,) raw logits from student model.
        teacher_logits: (n_classes,) raw logits from teacher model.
        temperature: Softmax temperature. Higher = softer distributions.

    Returns:
        KL divergence loss (scalar).
    """
    # Softmax with temperature
    def softmax_t(logits: np.ndarray, t: float) -> np.ndarray:
        scaled = logits / t
        exp_scaled = np.exp(scaled - np.max(scaled))
        return exp_scaled / (np.sum(exp_scaled) + 1e-10)

    teacher_soft = softmax_t(teacher_logits, temperature)
    student_soft = softmax_t(student_logits, temperature)

    # KL(teacher || student) = sum(teacher * log(teacher / student))
    # Clip to avoid log(0)
    teacher_soft = np.clip(teacher_soft, 1e-10, 1.0)
    student_soft = np.clip(student_soft, 1e-10, 1.0)

    kl = np.sum(teacher_soft * np.log(teacher_soft / student_soft))
    # Scale by T^2 as per Hinton et al.
    return float(kl * temperature * temperature)


def distillation_loss(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    hard_label: int,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> float:
    """Combined distillation loss: soft KL + hard cross-entropy.

    Loss = alpha * T^2 * KL(soft_teacher || soft_student) +
           (1 - alpha) * CE(student, hard_label)

    Args:
        student_logits: (n_classes,) raw logits from student.
        teacher_logits: (n_classes,) raw logits from teacher.
        hard_label: Integer class index for ground truth.
        temperature: Softmax temperature for soft targets.
        alpha: Weight for soft loss (0-1). Higher = more teacher influence.

    Returns:
        Combined scalar loss.
    """
    # Soft loss (KL divergence with temperature)
    soft_loss = kl_divergence_loss(student_logits, teacher_logits, temperature)

    # Hard loss (cross-entropy with true label)
    probs = np.exp(student_logits - np.max(student_logits))
    probs = probs / (np.sum(probs) + 1e-10)
    hard_loss = -np.log(np.clip(probs[hard_label], 1e-10, 1.0))

    return float(alpha * soft_loss + (1 - alpha) * hard_loss)


def train_distilled_model(
    teacher_path: str,
    deap_path: str,
    output_path: str,
    temperature: float = 4.0,
    alpha: float = 0.7,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> Dict:
    """Train a student model using knowledge distillation.

    This function requires PyTorch and a pre-trained teacher model.
    The teacher generates soft labels from 32-channel DEAP data,
    while the student only sees the 4-channel Muse subset.

    Args:
        teacher_path: Path to pre-trained 32-channel teacher model (.pt).
        deap_path: Path to DEAP dataset directory.
        output_path: Where to save the trained student model.
        temperature: Distillation temperature (default 4.0).
        alpha: Soft loss weight (default 0.7).
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.

    Returns:
        Dict with training metrics (loss history, final accuracy, etc.).

    Raises:
        ImportError: If PyTorch is not installed.
        FileNotFoundError: If teacher model or DEAP data not found.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        log.error("PyTorch required for distillation training. pip install torch")
        return {"error": "pytorch_not_installed"}

    teacher_file = Path(teacher_path)
    if not teacher_file.exists():
        log.error("Teacher model not found: %s", teacher_path)
        return {"error": "teacher_not_found", "path": teacher_path}

    deap_dir = Path(deap_path)
    if not deap_dir.exists():
        log.error("DEAP dataset not found: %s", deap_path)
        return {"error": "deap_not_found", "path": deap_path}

    log.info(
        "Starting distillation: T=%.1f, alpha=%.2f, epochs=%d",
        temperature, alpha, epochs,
    )

    # Placeholder: actual implementation requires loading DEAP data,
    # running teacher inference, and training student with combined loss.
    # This structure documents the exact approach to follow.

    return {
        "status": "not_yet_trained",
        "config": {
            "teacher_path": str(teacher_path),
            "deap_path": str(deap_path),
            "output_path": str(output_path),
            "temperature": temperature,
            "alpha": alpha,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "note": (
            "Full training requires: (1) DEAP dataset downloaded, "
            "(2) pre-trained 32-channel teacher model, "
            "(3) PyTorch installed. Run with --help for usage."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Teacher-student distillation: 32ch DEAP teacher -> 4ch Muse student"
    )
    parser.add_argument("--teacher-path", required=True, help="Path to 32ch teacher model (.pt)")
    parser.add_argument("--deap-path", required=True, help="Path to DEAP dataset directory")
    parser.add_argument("--output", default="models/saved/student_muse4ch.pt", help="Output path")
    parser.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7, help="Soft loss weight (0-1)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    result = train_distilled_model(
        teacher_path=args.teacher_path,
        deap_path=args.deap_path,
        output_path=args.output,
        temperature=args.temperature,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    print(result)


if __name__ == "__main__":
    main()
