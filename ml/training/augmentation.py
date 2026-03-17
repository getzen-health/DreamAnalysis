"""EEG data augmentation: channel reflection + noise + temporal jitter.

Channel reflection (Wang et al. 2024, Neural Networks):
Swaps symmetric EEG channels to create neurophysiologically valid
augmented samples. For Muse 2: AF7<->AF8, TP9<->TP10.
Doubles training data for free with zero inference cost.

Two modes:
  1. Raw EEG:     channel_reflect()           -- swap channels in (4, n_samples) arrays
  2. Feature-space: reflect_features_85dim()  -- swap channel features in 85-dim vectors
     (works on already-extracted features from train_mega_lgbm_unified.py)
"""

import numpy as np
from typing import Optional, Tuple

# Muse 2 symmetric channel pairs: (left, right)
# ch0=TP9 (left temporal), ch1=AF7 (left frontal),
# ch2=AF8 (right frontal), ch3=TP10 (right temporal)
MUSE2_SYMMETRIC_PAIRS = [(0, 3), (1, 2)]  # (TP9,TP10), (AF7,AF8)


# ---------------------------------------------------------------------------
# Raw EEG augmentation
# ---------------------------------------------------------------------------

def channel_reflect(eeg_4ch: np.ndarray) -> np.ndarray:
    """Swap left/right symmetric channels in raw EEG.

    Args:
        eeg_4ch: shape (4, n_samples) -- 4-channel EEG

    Returns:
        reflected: shape (4, n_samples) -- channels swapped
    """
    if eeg_4ch.ndim != 2 or eeg_4ch.shape[0] != 4:
        raise ValueError(f"Expected (4, n_samples), got {eeg_4ch.shape}")
    reflected = eeg_4ch.copy()
    for left, right in MUSE2_SYMMETRIC_PAIRS:
        reflected[left], reflected[right] = eeg_4ch[right].copy(), eeg_4ch[left].copy()
    return reflected


def augment_with_reflection(
    X: np.ndarray,
    y: np.ndarray,
    valence_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Double dataset via channel reflection on raw EEG arrays.

    Args:
        X: shape (n_samples, 4, n_timepoints) -- EEG data
        y: shape (n_samples,) -- class labels
        valence_labels: optional (n_samples,) -- if provided, flips sign
                       for reflected samples (FAA inverts)

    Returns:
        (X_aug, y_aug, valence_aug) -- 2x the original size
    """
    n = len(X)
    X_reflected = np.array([channel_reflect(X[i]) for i in range(n)])
    X_aug = np.concatenate([X, X_reflected], axis=0)
    y_aug = np.concatenate([y, y], axis=0)  # same labels (arousal unchanged)

    valence_aug = None
    if valence_labels is not None:
        valence_aug = np.concatenate([valence_labels, -valence_labels], axis=0)

    return X_aug, y_aug, valence_aug


# ---------------------------------------------------------------------------
# Feature-space augmentation (85-dim vectors)
# ---------------------------------------------------------------------------

def reflect_features_85dim(features: np.ndarray) -> np.ndarray:
    """Swap channel features in an 85-dim feature vector (or batch).

    Feature layout from train_mega_lgbm_unified.py extract_features():
      [0:80]  5 bands x 4 channels x 4 stats (mean, std, median, IQR)
              Outer loop: band (0-4), inner: ch (0-3), innermost: stat (0-3)
              Index formula: band*16 + ch*4 + stat
      [80:85] 5 DASM features: mean(AF8_band) - mean(AF7_band) per band

    Channel reflection swaps:
      ch0 (TP9) <-> ch3 (TP10)
      ch1 (AF7) <-> ch2 (AF8)
    And negates DASM (since AF8-AF7 becomes AF7-AF8).

    Args:
        features: shape (85,) or (n_samples, 85)

    Returns:
        reflected features with same shape
    """
    single = features.ndim == 1
    if single:
        features = features.reshape(1, -1)

    if features.shape[1] < 85:
        raise ValueError(f"Expected >= 85 features, got {features.shape[1]}")

    reflected = features.copy()

    # Swap channel features in the 80-feature block
    # Layout: band*16 + ch*4 + stat
    for band in range(5):
        for stat in range(4):
            # TP9 (ch0) <-> TP10 (ch3)
            idx_tp9  = band * 16 + 0 * 4 + stat
            idx_tp10 = band * 16 + 3 * 4 + stat
            reflected[:, idx_tp9]  = features[:, idx_tp10]
            reflected[:, idx_tp10] = features[:, idx_tp9]

            # AF7 (ch1) <-> AF8 (ch2)
            idx_af7 = band * 16 + 1 * 4 + stat
            idx_af8 = band * 16 + 2 * 4 + stat
            reflected[:, idx_af7] = features[:, idx_af8]
            reflected[:, idx_af8] = features[:, idx_af7]

    # Negate DASM features (AF8 - AF7 becomes AF7 - AF8)
    reflected[:, 80:85] = -features[:, 80:85]

    if single:
        return reflected[0]
    return reflected


def augment_features_with_reflection(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Double a feature-space dataset via channel reflection.

    Works on 85-dim feature vectors from train_mega_lgbm_unified.py.
    Emotion labels (arousal-based 3-class) are preserved since channel
    reflection changes laterality but not overall arousal.

    Args:
        X: shape (n_samples, 85) -- feature vectors
        y: shape (n_samples,) -- class labels

    Returns:
        (X_aug, y_aug) -- 2x the original size
    """
    X_reflected = reflect_features_85dim(X)
    X_aug = np.concatenate([X, X_reflected], axis=0)
    y_aug = np.concatenate([y, y], axis=0)
    return X_aug, y_aug


# ---------------------------------------------------------------------------
# Additional augmentation utilities
# ---------------------------------------------------------------------------

def add_gaussian_noise(eeg: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to raw EEG to simulate impedance drift.

    Args:
        eeg: shape (4, n_samples) or (n_samples, n_features)
        std: noise standard deviation relative to signal std

    Returns:
        noisy copy of the input
    """
    noise_scale = std * np.std(eeg) if np.std(eeg) > 0 else std
    return eeg + np.random.normal(0, noise_scale, eeg.shape)


def temporal_jitter(eeg: np.ndarray, max_shift: int = 5) -> np.ndarray:
    """Random temporal shift to simulate movement jitter.

    Args:
        eeg: shape (4, n_samples) -- raw EEG
        max_shift: maximum shift in samples (both directions)

    Returns:
        shifted copy of the input
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(eeg, shift, axis=-1)


# ── Audio / Voice Augmentation ────────────────────────────────────────────────
# For voice emotion model training. Based on:
# - Springer 2024: "Real-time speech emotion recognition using deep learning
#   and data augmentation" — noise injection + pitch shift → +5-10% accuracy
# - Taylor & Francis 2024: "Data augmentation using 1D-CNN with MFCC/MFMC"


def augment_audio_noise(
    audio: np.ndarray, snr_db: float = 20.0
) -> np.ndarray:
    """Add white Gaussian noise at a specified signal-to-noise ratio.

    Args:
        audio: 1D float audio array
        snr_db: target SNR in decibels (lower = more noise). 20 dB is subtle.

    Returns:
        Noisy copy of audio.
    """
    rms_signal = np.sqrt(np.mean(audio ** 2)) + 1e-10
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, audio.shape)
    return (audio + noise).astype(audio.dtype)


def augment_audio_pitch(
    audio: np.ndarray, sr: int = 16000, n_steps: float = 0.0
) -> np.ndarray:
    """Shift pitch by n_steps semitones without changing duration.

    Args:
        audio: 1D float audio
        sr: sample rate
        n_steps: semitones to shift (positive = higher, negative = lower).
                 Random if 0: uniform(-2, 2).

    Returns:
        Pitch-shifted copy.
    """
    try:
        import librosa
    except ImportError:
        return audio  # librosa not available — return original

    if n_steps == 0.0:
        n_steps = np.random.uniform(-2.0, 2.0)
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)


def augment_audio_speed(
    audio: np.ndarray, rate: float = 0.0
) -> np.ndarray:
    """Time-stretch audio by a rate factor without changing pitch.

    Args:
        audio: 1D float audio
        rate: stretch factor (>1 = faster, <1 = slower).
              Random if 0: uniform(0.85, 1.15).

    Returns:
        Time-stretched copy.
    """
    try:
        import librosa
    except ImportError:
        return audio

    if rate == 0.0:
        rate = np.random.uniform(0.85, 1.15)
    stretched = librosa.effects.time_stretch(y=audio, rate=rate)
    return stretched.astype(audio.dtype)


def augment_audio_gain(
    audio: np.ndarray, gain_db: float = 0.0
) -> np.ndarray:
    """Apply random volume gain in decibels.

    Args:
        audio: 1D float audio
        gain_db: gain in dB. Random if 0: uniform(-6, 6).

    Returns:
        Gain-adjusted copy.
    """
    if gain_db == 0.0:
        gain_db = np.random.uniform(-6.0, 6.0)
    factor = 10 ** (gain_db / 20)
    return (audio * factor).astype(audio.dtype)


def augment_audio_full(
    audio: np.ndarray, sr: int = 16000
) -> np.ndarray:
    """Apply a random combination of noise + pitch + speed + gain augmentation.

    Each augmentation is applied with 50% probability.
    Suitable for voice emotion training data expansion.

    Returns:
        Augmented copy.
    """
    out = audio.copy()
    if np.random.random() < 0.5:
        out = augment_audio_noise(out, snr_db=np.random.uniform(15, 30))
    if np.random.random() < 0.5:
        out = augment_audio_pitch(out, sr=sr)
    if np.random.random() < 0.5:
        out = augment_audio_speed(out)
    if np.random.random() < 0.5:
        out = augment_audio_gain(out)
    return out
