"""Device-specific EEG channel index maps.

Maps device name → channel roles so frontal asymmetry computations
(compute_frontal_asymmetry, compute_dasm_rasm) automatically use the
correct left/right frontal electrode indices instead of hardcoding
Muse 2 positions.

Usage:
    from processing.channel_maps import get_channel_map

    cmap = get_channel_map("openbci_cyton", n_channels=8)
    left_ch  = cmap["left_frontal"]
    right_ch = cmap["right_frontal"]
"""

from typing import Dict, Optional

# Keys per device entry:
#   left_frontal   — index of left-frontal electrode  (e.g. AF7, F3, FC5)
#   right_frontal  — index of right-frontal electrode (e.g. AF8, F4, FC6)
#   left_temporal  — index of left-temporal electrode (e.g. TP9, T7)
#   right_temporal — index of right-temporal electrode (e.g. TP10, T8)
#   midline        — index of midline frontal channel for FMT, or None
CHANNEL_MAPS: Dict[str, Dict] = {
    # ── Muse 2 / Muse S ─────────────────────────────────────────────
    # BrainFlow delivery order (board_id 22/38):
    #   ch0=TP9 (left temporal), ch1=AF7 (left frontal),
    #   ch2=AF8 (right frontal), ch3=TP10 (right temporal)
    "muse_2": {
        "left_frontal": 1, "right_frontal": 2,
        "left_temporal": 0, "right_temporal": 3,
        "midline": None,
    },
    "muse_2_bled": {
        "left_frontal": 1, "right_frontal": 2,
        "left_temporal": 0, "right_temporal": 3,
        "midline": None,
    },
    "muse_s": {
        "left_frontal": 1, "right_frontal": 2,
        "left_temporal": 0, "right_temporal": 3,
        "midline": None,
    },
    "muse_s_bled": {
        "left_frontal": 1, "right_frontal": 2,
        "left_temporal": 0, "right_temporal": 3,
        "midline": None,
    },
    # ── OpenBCI Ganglion — 4 channels ───────────────────────────────
    # Default emotion-research cap placement:
    #   ch0=Fp1 (left frontal), ch1=Fp2 (right frontal),
    #   ch2=C3  (left central), ch3=C4  (right central)
    "openbci_ganglion": {
        "left_frontal": 0, "right_frontal": 1,
        "left_temporal": 2, "right_temporal": 3,
        "midline": None,
    },
    # ── OpenBCI Cyton — 8 channels ──────────────────────────────────
    # Standard 10-20 cap for emotion research:
    #   ch0=Fp1, ch1=Fp2, ch2=F3, ch3=F4,
    #   ch4=C3,  ch5=C4,  ch6=P3, ch7=P4
    "openbci_cyton": {
        "left_frontal": 2, "right_frontal": 3,   # F3, F4
        "left_temporal": 4, "right_temporal": 5,  # C3, C4 (nearest temporal)
        "midline": None,
    },
    # ── OpenBCI Cyton+Daisy — 16 channels ───────────────────────────
    # Standard 10-20:
    #   Frontal: F3=ch3, F4=ch5 — Temporal: T7=ch9, T8=ch10
    "openbci_cyton_daisy": {
        "left_frontal": 3, "right_frontal": 5,
        "left_temporal": 9, "right_temporal": 10,
        "midline": None,
    },
    # ── Neurosity Crown — 8 channels ────────────────────────────────
    # Temporal-parietal focus:
    #   ch0=CP3, ch1=C3, ch2=F5, ch3=PO3,
    #   ch4=PO4, ch5=F6, ch6=C4, ch7=CP4
    "neurosity_crown": {
        "left_frontal": 2, "right_frontal": 5,   # F5, F6
        "left_temporal": 1, "right_temporal": 6,  # C3, C4
        "midline": None,
    },
    # ── Emotiv EPOC / EPOC X — 14 channels ──────────────────────────
    # Channel order:
    #   AF3(0), F7(1), F3(2), FC5(3), T7(4), P7(5), O1(6),
    #   O2(7),  P8(8), T8(9), FC6(10), F4(11), F8(12), AF4(13)
    "emotiv_epoc": {
        "left_frontal": 2, "right_frontal": 11,  # F3, F4
        "left_temporal": 4, "right_temporal": 9,  # T7, T8
        "midline": None,
    },
    "emotiv_epoc_x": {
        "left_frontal": 2, "right_frontal": 11,
        "left_temporal": 4, "right_temporal": 9,
        "midline": None,
    },
    # ── Synthetic (default Muse 2 layout) ───────────────────────────
    "synthetic": {
        "left_frontal": 1, "right_frontal": 2,
        "left_temporal": 0, "right_temporal": 3,
        "midline": None,
    },
}

# Default fallback — Muse 2 layout
_DEFAULT_MAP: Dict = CHANNEL_MAPS["muse_2"]


def get_channel_map(device: str, n_channels: int = 4) -> Dict:
    """Return channel index map for a given device.

    Falls back to Muse 2 layout if device is unknown.
    Clamps indices to [0, n_channels-1] to guard against wrong configs.

    Args:
        device:     Device name string (e.g. "muse_2", "openbci_cyton").
        n_channels: Number of available channels (for bounds-checking).

    Returns:
        Dict with keys: left_frontal, right_frontal, left_temporal,
        right_temporal, midline.
    """
    cmap = CHANNEL_MAPS.get(device, _DEFAULT_MAP)
    # Clamp each index so we never index out of bounds on unexpected configs
    safe: Dict = {}
    for key, idx in cmap.items():
        if idx is None:
            safe[key] = None
        else:
            safe[key] = min(idx, max(0, n_channels - 1))
    return safe
