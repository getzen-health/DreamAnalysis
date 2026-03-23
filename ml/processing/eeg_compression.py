"""
EEG data compression utilities.

Delta encoding stores differences between consecutive EEG samples. Since
adjacent samples at 256 Hz differ by only a fraction of a microvolt, the
deltas cluster near zero and compress extremely well with general-purpose
compressors (zstd, zlib).

Functions:
    delta_encode / delta_decode — core codec
    compress_eeg_frame / decompress_eeg_frame — full frame with metadata
    downsample_to_128hz — reduce archival storage by 2x with anti-alias filter
"""

from __future__ import annotations

import json
import struct
import zlib
from typing import Tuple

import numpy as np
from scipy.signal import decimate


# ── Delta Encoding ───────────────────────────────────────────────────────────


def delta_encode(samples: np.ndarray) -> np.ndarray:
    """Delta-encode a 1-D array.

    Returns an array of the same length where:
        out[0] = samples[0]            (baseline)
        out[i] = samples[i] - samples[i-1]   for i > 0
    """
    if len(samples) == 0:
        return np.array([], dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    out = np.empty_like(samples)
    out[0] = samples[0]
    out[1:] = np.diff(samples)
    return out


def delta_decode(deltas: np.ndarray) -> np.ndarray:
    """Reverse delta encoding to reconstruct the original signal."""
    if len(deltas) == 0:
        return np.array([], dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    return np.cumsum(deltas)


# ── Frame Compression ────────────────────────────────────────────────────────


def compress_eeg_frame(
    channels: np.ndarray,
    timestamp: float,
) -> bytes:
    """Compress a multi-channel EEG frame to bytes.

    Args:
        channels: (n_channels, n_samples) float64 array.
        timestamp: UNIX timestamp (seconds or ms).

    Returns:
        Compressed bytes containing all data + metadata.
    """
    channels = np.asarray(channels, dtype=np.float64)
    if channels.ndim == 1:
        channels = channels.reshape(1, -1)

    n_ch, n_samples = channels.shape

    # Delta-encode each channel
    encoded = np.empty_like(channels)
    for i in range(n_ch):
        encoded[i] = delta_encode(channels[i])

    # Serialize: header (JSON) + encoded data (zlib-compressed raw bytes)
    header = json.dumps({
        "n_channels": n_ch,
        "n_samples": n_samples,
        "timestamp": timestamp,
    }).encode("utf-8")

    raw_data = encoded.tobytes()
    compressed_data = zlib.compress(raw_data, level=6)

    # Pack: [header_len (4 bytes)] [header] [compressed_data]
    header_len = struct.pack("<I", len(header))
    return header_len + header + compressed_data


def decompress_eeg_frame(data: bytes) -> Tuple[np.ndarray, float]:
    """Decompress bytes back to (channels, timestamp).

    Returns:
        (channels, timestamp) where channels is (n_channels, n_samples) float64.
    """
    # Unpack header
    header_len = struct.unpack("<I", data[:4])[0]
    header = json.loads(data[4 : 4 + header_len].decode("utf-8"))

    n_ch = header["n_channels"]
    n_samples = header["n_samples"]
    timestamp = header["timestamp"]

    # Decompress data
    compressed_data = data[4 + header_len :]
    raw_data = zlib.decompress(compressed_data)

    # Reconstruct
    encoded = np.frombuffer(raw_data, dtype=np.float64).reshape(n_ch, n_samples)

    channels = np.empty_like(encoded)
    for i in range(n_ch):
        channels[i] = delta_decode(encoded[i])

    return channels, timestamp


# ── Downsampling ─────────────────────────────────────────────────────────────


def downsample_to_128hz(
    signal: np.ndarray,
    original_fs: int = 256,
) -> np.ndarray:
    """Downsample a signal to 128 Hz for archival storage.

    Uses scipy.signal.decimate which applies an anti-aliasing filter before
    decimation to prevent spectral aliasing. The built-in Chebyshev type I
    filter has a cutoff just below the new Nyquist frequency (64 Hz).

    Args:
        signal: 1-D array of samples at original_fs.
        original_fs: Original sampling rate in Hz.

    Returns:
        Downsampled 1-D array at 128 Hz.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if original_fs == 128:
        return signal.copy()

    if original_fs <= 0:
        raise ValueError(f"original_fs must be positive, got {original_fs}")

    factor = original_fs // 128
    if factor < 1:
        raise ValueError(
            f"Cannot downsample from {original_fs} Hz to 128 Hz "
            f"(original rate must be >= 128 Hz)"
        )

    if factor == 1:
        return signal.copy()

    return decimate(signal, factor, ftype="iir", zero_phase=True)
