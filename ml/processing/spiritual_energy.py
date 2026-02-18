"""Spiritual Energy Analysis Module.

Maps EEG brainwave patterns to spiritual energy frameworks including
chakra activation, meditation depth, aura visualization, kundalini
flow, prana balance, and consciousness level estimation.

Grounded in neuroscience:
- Alpha power correlates with calm, meditative states
- Theta bursts correlate with deep meditation / transcendence
- Gamma bursts correlate with peak spiritual experiences
- Hemispheric coherence correlates with unified awareness
- Band ratios map naturally to traditional energy center descriptions
"""

import numpy as np
from typing import Dict, Optional
from scipy import signal as scipy_signal

from processing.eeg_processor import (
    extract_band_powers,
    spectral_entropy,
    BANDS,
)

# Numpy 2.x compat
_trapezoid = getattr(np, "trapezoid", np.trapz)


# --- Chakra Definitions ---
# Each chakra maps to a frequency range, color, element, and qualities
CHAKRAS = {
    "root": {
        "sanskrit": "Muladhara",
        "frequency_band": (0.5, 4.0),   # Delta
        "color": "#FF0000",
        "element": "Earth",
        "qualities": ["grounding", "stability", "survival", "security"],
        "location": "base of spine",
        "mantra": "LAM",
    },
    "sacral": {
        "sanskrit": "Svadhisthana",
        "frequency_band": (4.0, 8.0),   # Theta
        "color": "#FF8C00",
        "element": "Water",
        "qualities": ["creativity", "emotion", "pleasure", "flow"],
        "location": "lower abdomen",
        "mantra": "VAM",
    },
    "solar_plexus": {
        "sanskrit": "Manipura",
        "frequency_band": (8.0, 10.0),  # Low Alpha
        "color": "#FFD700",
        "element": "Fire",
        "qualities": ["willpower", "confidence", "personal power", "transformation"],
        "location": "upper abdomen",
        "mantra": "RAM",
    },
    "heart": {
        "sanskrit": "Anahata",
        "frequency_band": (10.0, 12.0), # High Alpha
        "color": "#00FF00",
        "element": "Air",
        "qualities": ["love", "compassion", "connection", "harmony"],
        "location": "center of chest",
        "mantra": "YAM",
    },
    "throat": {
        "sanskrit": "Vishuddha",
        "frequency_band": (12.0, 20.0), # Low Beta
        "color": "#00BFFF",
        "element": "Ether",
        "qualities": ["expression", "communication", "truth", "clarity"],
        "location": "throat",
        "mantra": "HAM",
    },
    "third_eye": {
        "sanskrit": "Ajna",
        "frequency_band": (20.0, 40.0), # High Beta / Low Gamma
        "color": "#4B0082",
        "element": "Light",
        "qualities": ["intuition", "insight", "wisdom", "perception"],
        "location": "between eyebrows",
        "mantra": "OM",
    },
    "crown": {
        "sanskrit": "Sahasrara",
        "frequency_band": (40.0, 100.0), # Gamma
        "color": "#9400D3",
        "element": "Cosmic Energy",
        "qualities": ["transcendence", "unity", "enlightenment", "bliss"],
        "location": "top of head",
        "mantra": "AUM",
    },
}

# Consciousness levels inspired by David Hawkins' Map of Consciousness
# mapped to EEG signatures
CONSCIOUSNESS_LEVELS = [
    {"level": "Deep Sleep", "range": (0, 100), "dominant": "delta", "description": "Unconscious restoration"},
    {"level": "Drowsy / Hypnagogic", "range": (100, 200), "dominant": "theta", "description": "Dream-like imagery, subconscious access"},
    {"level": "Relaxed Awareness", "range": (200, 350), "dominant": "alpha", "description": "Calm presence, open awareness"},
    {"level": "Focused Attention", "range": (350, 500), "dominant": "low_beta", "description": "Engaged mind, active thinking"},
    {"level": "Heightened Perception", "range": (500, 650), "dominant": "high_beta", "description": "Sharp awareness, analytical insight"},
    {"level": "Meditative Absorption", "range": (650, 800), "dominant": "alpha_theta", "description": "Deep meditation, inner stillness"},
    {"level": "Transcendent Awareness", "range": (800, 900), "dominant": "gamma", "description": "Peak experience, unity consciousness"},
    {"level": "Cosmic Consciousness", "range": (900, 1000), "dominant": "gamma_burst", "description": "Non-dual awareness, enlightenment"},
]


def compute_chakra_activations(
    eeg: np.ndarray, fs: float = 256.0
) -> Dict[str, Dict]:
    """Compute activation level for each chakra based on EEG frequency content.

    Each chakra maps to a specific frequency range. Activation is the
    relative power in that range, normalized and scaled 0-100.

    Args:
        eeg: 1D EEG signal array.
        fs: Sampling frequency in Hz.

    Returns:
        Dict mapping chakra name to activation details.
    """
    freqs, psd = scipy_signal.welch(eeg, fs=fs, nperseg=min(len(eeg), int(fs * 2)))
    total_power = _trapezoid(psd, freqs)
    if total_power == 0:
        total_power = 1e-10

    activations = {}
    raw_values = {}

    for name, info in CHAKRAS.items():
        low, high = info["frequency_band"]
        mask = (freqs >= low) & (freqs <= high)
        if mask.any():
            band_power = _trapezoid(psd[mask], freqs[mask])
        else:
            band_power = 0.0
        raw_values[name] = band_power / total_power

    # Normalize across chakras so max is ~100
    max_raw = max(raw_values.values()) if raw_values else 1e-10
    if max_raw == 0:
        max_raw = 1e-10

    for name, info in CHAKRAS.items():
        raw = raw_values[name]
        activation = min(100.0, (raw / max_raw) * 100.0)

        # Determine if this chakra is "open" (above 40%), "balanced" (40-70%), or "active" (>70%)
        if activation >= 70:
            status = "highly active"
        elif activation >= 40:
            status = "balanced"
        elif activation >= 15:
            status = "low activity"
        else:
            status = "dormant"

        activations[name] = {
            "activation": round(activation, 1),
            "status": status,
            "raw_power": round(raw, 4),
            "sanskrit": info["sanskrit"],
            "color": info["color"],
            "element": info["element"],
            "qualities": info["qualities"],
            "location": info["location"],
            "mantra": info["mantra"],
        }

    return activations


def compute_chakra_balance(chakra_activations: Dict) -> Dict:
    """Analyze overall chakra system balance.

    Returns balance metrics including:
    - Overall harmony score (0-100)
    - Which chakras need attention
    - Energy flow direction (ascending/descending/balanced)
    """
    values = [c["activation"] for c in chakra_activations.values()]
    names = list(chakra_activations.keys())

    if not values:
        return {"harmony": 0, "flow": "unknown", "guidance": []}

    mean_activation = np.mean(values)
    std_activation = np.std(values)

    # Harmony = inverse of variance (more uniform = more harmonious)
    harmony = max(0, 100 - std_activation * 2)

    # Energy flow: compare lower chakras vs upper chakras
    lower = np.mean(values[:3])  # root, sacral, solar plexus
    upper = np.mean(values[4:])  # throat, third eye, crown
    middle = values[3]           # heart

    if upper > lower * 1.3:
        flow = "ascending"
        flow_description = "Energy is flowing upward — spiritual seeking, intellectual focus"
    elif lower > upper * 1.3:
        flow = "descending"
        flow_description = "Energy is grounded — physical vitality, earthly focus"
    else:
        flow = "balanced"
        flow_description = "Energy is balanced between earth and spirit"

    # Find which chakras need attention (below 25%)
    needs_attention = []
    for name, data in chakra_activations.items():
        if data["activation"] < 25:
            needs_attention.append({
                "chakra": name,
                "sanskrit": data["sanskrit"],
                "suggestion": f"Practice {data['mantra']} mantra meditation to activate {data['sanskrit']}",
            })

    # Dominant chakra
    dominant_idx = int(np.argmax(values))
    dominant = names[dominant_idx]

    return {
        "harmony_score": round(harmony, 1),
        "mean_activation": round(mean_activation, 1),
        "energy_flow": flow,
        "flow_description": flow_description,
        "dominant_chakra": dominant,
        "dominant_sanskrit": chakra_activations[dominant]["sanskrit"],
        "heart_center_strength": round(middle, 1),
        "needs_attention": needs_attention,
    }


def compute_meditation_depth(
    eeg: np.ndarray, fs: float = 256.0
) -> Dict:
    """Estimate meditation depth from EEG patterns.

    Uses alpha/theta ratio, alpha power, and spectral entropy to
    classify meditation depth on a 0-10 scale.

    Neuroscience basis:
    - Light meditation: increased alpha, reduced beta
    - Medium meditation: alpha-theta crossover
    - Deep meditation: theta dominance, reduced complexity
    - Transcendent: gamma bursts with theta/delta base
    """
    bands = extract_band_powers(eeg, fs)
    se = spectral_entropy(eeg, fs)

    delta = bands.get("delta", 0)
    theta = bands.get("theta", 0)
    alpha = bands.get("alpha", 0)
    beta = bands.get("beta", 0)
    gamma = bands.get("gamma", 0)

    # Meditation indicators
    alpha_strength = alpha / max(beta, 1e-10)
    theta_strength = theta / max(beta, 1e-10)
    calm_ratio = (alpha + theta) / max(beta + gamma, 1e-10)
    gamma_bursts = gamma > 0.15  # elevated gamma suggests peak experience

    # Compute depth score (0-10)
    depth = 0.0

    # Alpha dominance adds up to 3 points
    depth += min(3.0, alpha_strength * 1.5)

    # Theta presence adds up to 3 points
    depth += min(3.0, theta_strength * 1.0)

    # Low beta (calm mind) adds up to 2 points
    if beta < 0.2:
        depth += 2.0 * (0.2 - beta) / 0.2

    # Low spectral entropy (focused state) adds up to 1 point
    if se < 0.7:
        depth += 1.0 * (0.7 - se) / 0.7

    # Gamma bursts indicate transcendent states (+1)
    if gamma_bursts:
        depth += 1.0

    depth = min(10.0, max(0.0, depth))

    # Classify into meditation stages
    if depth >= 8.5:
        stage = "Transcendent"
        description = "Deep absorption — unity awareness, ego dissolution"
        guidance = "You are in a profound meditative state. Simply be."
    elif depth >= 7.0:
        stage = "Deep Meditation"
        description = "Theta-dominant stillness — inner silence, witness consciousness"
        guidance = "Maintain this stillness. Let awareness rest in itself."
    elif depth >= 5.0:
        stage = "Meditation"
        description = "Alpha-theta crossover — calm focus, expanded awareness"
        guidance = "Gently deepen by releasing any remaining thoughts."
    elif depth >= 3.0:
        stage = "Light Meditation"
        description = "Alpha dominance — relaxed alertness, settling mind"
        guidance = "Focus on your breath or mantra to deepen."
    elif depth >= 1.5:
        stage = "Relaxation"
        description = "Mild alpha increase — beginning to calm"
        guidance = "Close your eyes, slow your breathing, release tension."
    else:
        stage = "Active Mind"
        description = "Beta-dominant — thinking, planning, analyzing"
        guidance = "Start with 5 deep breaths to begin calming the mind."

    return {
        "depth_score": round(depth, 2),
        "stage": stage,
        "description": description,
        "guidance": guidance,
        "indicators": {
            "alpha_strength": round(alpha_strength, 3),
            "theta_strength": round(theta_strength, 3),
            "calm_ratio": round(calm_ratio, 3),
            "spectral_entropy": round(se, 3),
            "gamma_burst_detected": gamma_bursts,
        },
    }


def compute_aura_energy(
    eeg: np.ndarray, fs: float = 256.0
) -> Dict:
    """Generate aura color and intensity from EEG patterns.

    Maps the dominant frequency band to traditional aura colors
    and computes overall energy intensity.
    """
    bands = extract_band_powers(eeg, fs)

    # Aura color mapping based on dominant brain activity
    aura_colors = {
        "delta": {"color": "#8B0000", "name": "Deep Red", "meaning": "Physical healing, deep rest"},
        "theta": {"color": "#FF6347", "name": "Orange-Red", "meaning": "Creative energy, emotional processing"},
        "alpha": {"color": "#32CD32", "name": "Green", "meaning": "Heart-centered, balanced, healing"},
        "beta": {"color": "#4169E1", "name": "Royal Blue", "meaning": "Mental activity, communication"},
        "gamma": {"color": "#9370DB", "name": "Violet", "meaning": "Spiritual awareness, higher consciousness"},
    }

    # Find dominant band
    dominant_band = max(bands, key=bands.get)
    dominant_power = bands[dominant_band]

    # Compute blended aura color (weighted RGB mix of all bands)
    color_weights = {
        "delta": np.array([139, 0, 0]),       # deep red
        "theta": np.array([255, 140, 0]),      # orange
        "alpha": np.array([50, 205, 50]),      # green
        "beta": np.array([65, 105, 225]),      # blue
        "gamma": np.array([148, 0, 211]),      # violet
    }

    blended = np.zeros(3)
    for band, power in bands.items():
        if band in color_weights:
            blended += color_weights[band] * power

    blended = np.clip(blended, 0, 255).astype(int)
    blended_hex = "#{:02x}{:02x}{:02x}".format(*blended)

    # Energy intensity (total absolute power proxy)
    energy_level = min(100, sum(bands.values()) * 100)

    # Aura layers (inner, middle, outer)
    sorted_bands = sorted(bands.items(), key=lambda x: x[1], reverse=True)

    layers = []
    for i, (band, power) in enumerate(sorted_bands[:3]):
        layer_names = ["inner", "middle", "outer"]
        info = aura_colors.get(band, {"color": "#FFFFFF", "name": "White", "meaning": "Pure energy"})
        layers.append({
            "layer": layer_names[i],
            "band": band,
            "color": info["color"],
            "color_name": info["name"],
            "meaning": info["meaning"],
            "intensity": round(power * 100, 1),
        })

    return {
        "dominant_color": aura_colors[dominant_band]["color"],
        "dominant_color_name": aura_colors[dominant_band]["name"],
        "dominant_meaning": aura_colors[dominant_band]["meaning"],
        "blended_aura_color": blended_hex,
        "energy_level": round(energy_level, 1),
        "layers": layers,
    }


def compute_kundalini_flow(
    eeg: np.ndarray, fs: float = 256.0
) -> Dict:
    """Track kundalini energy flow through the chakra system.

    Kundalini is traditionally described as energy rising from the
    base of the spine to the crown. We track this as progressive
    activation from lower to higher frequency bands.
    """
    chakra_activations = compute_chakra_activations(eeg, fs)
    chakra_order = ["root", "sacral", "solar_plexus", "heart", "throat", "third_eye", "crown"]
    values = [chakra_activations[c]["activation"] for c in chakra_order]

    # Find the "highest reached" chakra (last one above threshold)
    threshold = 30.0
    highest_reached = 0
    for i, v in enumerate(values):
        if v >= threshold:
            highest_reached = i

    # Check for continuous flow (each chakra feeds the next)
    flow_continuity = 0
    for i in range(len(values) - 1):
        if values[i] >= threshold and values[i + 1] >= threshold:
            flow_continuity += 1
    flow_continuity_pct = (flow_continuity / (len(values) - 1)) * 100

    # Rising energy pattern: are values generally increasing?
    diffs = np.diff(values)
    rising_count = sum(1 for d in diffs if d > 0)
    rising_pct = (rising_count / len(diffs)) * 100

    # Kundalini awakening indicators
    crown_active = values[6] >= 50
    third_eye_active = values[5] >= 50
    heart_open = values[3] >= 40
    base_grounded = values[0] >= 30

    if crown_active and base_grounded and flow_continuity_pct >= 70:
        awakening_status = "Full kundalini flow"
        description = "Energy flows freely from root to crown — integrated spiritual state"
    elif third_eye_active and heart_open:
        awakening_status = "Upper chakra activation"
        description = "Intuitive awareness expanding — maintain heart connection"
    elif heart_open and base_grounded:
        awakening_status = "Heart-centered grounding"
        description = "Balanced foundation — heart bridges earth and spirit"
    elif base_grounded:
        awakening_status = "Grounded foundation"
        description = "Strong root energy — stable base for spiritual growth"
    else:
        awakening_status = "Gathering energy"
        description = "Energy building — practice grounding exercises first"

    return {
        "highest_chakra_reached": chakra_order[highest_reached],
        "highest_chakra_sanskrit": CHAKRAS[chakra_order[highest_reached]]["sanskrit"],
        "flow_continuity_pct": round(flow_continuity_pct, 1),
        "rising_energy_pct": round(rising_pct, 1),
        "awakening_status": awakening_status,
        "description": description,
        "chakra_progression": [
            {
                "chakra": chakra_order[i],
                "sanskrit": CHAKRAS[chakra_order[i]]["sanskrit"],
                "activation": round(values[i], 1),
                "color": CHAKRAS[chakra_order[i]]["color"],
            }
            for i in range(len(chakra_order))
        ],
    }


def compute_prana_balance(
    eeg_left: np.ndarray, eeg_right: np.ndarray, fs: float = 256.0
) -> Dict:
    """Compute prana/chi energy balance from bilateral EEG.

    Uses hemispheric asymmetry to determine balance between:
    - Ida (left/lunar/yin) - right hemisphere
    - Pingala (right/solar/yang) - left hemisphere
    - Sushumna (central/balanced) - when both are equal

    Neuroscience: frontal alpha asymmetry is a well-established
    marker of approach/withdrawal motivation and emotional valence.
    """
    left_bands = extract_band_powers(eeg_left, fs)
    right_bands = extract_band_powers(eeg_right, fs)

    # Alpha asymmetry (classic FAA metric)
    left_alpha = left_bands.get("alpha", 0)
    right_alpha = right_bands.get("alpha", 0)

    # FAA: log(right) - log(left); positive = more left activation (approach)
    faa = np.log(max(right_alpha, 1e-10)) - np.log(max(left_alpha, 1e-10))

    # Overall power per hemisphere
    left_total = sum(left_bands.values())
    right_total = sum(right_bands.values())
    total = left_total + right_total
    if total == 0:
        total = 1e-10

    left_pct = (left_total / total) * 100
    right_pct = (right_total / total) * 100

    # Determine dominant nadi (energy channel)
    asymmetry = abs(left_pct - right_pct)
    if asymmetry < 5:
        dominant_nadi = "sushumna"
        nadi_description = "Central channel active — balanced, meditative state"
        balance_quality = "Harmonized"
    elif left_pct > right_pct:
        dominant_nadi = "pingala"
        nadi_description = "Solar/yang energy dominant — active, analytical, warming"
        balance_quality = "Yang-dominant"
    else:
        dominant_nadi = "ida"
        nadi_description = "Lunar/yin energy dominant — receptive, intuitive, cooling"
        balance_quality = "Yin-dominant"

    # Band-by-band laterality
    band_balance = {}
    for band in BANDS:
        l = left_bands.get(band, 0)
        r = right_bands.get(band, 0)
        band_total = l + r
        if band_total > 0:
            band_balance[band] = {
                "left_pct": round((l / band_total) * 100, 1),
                "right_pct": round((r / band_total) * 100, 1),
                "laterality": "left" if l > r else "right" if r > l else "balanced",
            }
        else:
            band_balance[band] = {"left_pct": 50.0, "right_pct": 50.0, "laterality": "balanced"}

    return {
        "dominant_nadi": dominant_nadi,
        "nadi_description": nadi_description,
        "balance_quality": balance_quality,
        "frontal_alpha_asymmetry": round(faa, 4),
        "left_hemisphere_pct": round(left_pct, 1),
        "right_hemisphere_pct": round(right_pct, 1),
        "asymmetry_magnitude": round(asymmetry, 1),
        "band_balance": band_balance,
        "guidance": _prana_guidance(dominant_nadi, asymmetry),
    }


def _prana_guidance(nadi: str, asymmetry: float) -> str:
    if nadi == "sushumna":
        return "Beautiful balance! This is ideal for meditation. Energy flows through the central channel."
    elif nadi == "pingala":
        if asymmetry > 15:
            return "Strong yang energy. Try left-nostril breathing (Chandra Bhedana) to restore balance."
        return "Mild yang dominance. Good for focused tasks. Practice alternate nostril breathing for balance."
    else:
        if asymmetry > 15:
            return "Strong yin energy. Try right-nostril breathing (Surya Bhedana) to energize."
        return "Mild yin dominance. Good for creative work. Practice alternate nostril breathing for balance."


def compute_consciousness_level(
    eeg: np.ndarray, fs: float = 256.0
) -> Dict:
    """Estimate consciousness level from EEG patterns.

    Maps brain activity to a 0-1000 consciousness scale inspired by
    contemplative traditions, using verifiable EEG correlates.
    """
    bands = extract_band_powers(eeg, fs)
    se = spectral_entropy(eeg, fs)

    delta = bands.get("delta", 0)
    theta = bands.get("theta", 0)
    alpha = bands.get("alpha", 0)
    beta = bands.get("beta", 0)
    gamma = bands.get("gamma", 0)

    # Compute consciousness score components
    # Base: weighted combination reflecting awareness level
    base_score = (
        delta * 50 +       # deep unconscious
        theta * 200 +      # subconscious/dreaming
        alpha * 400 +      # relaxed awareness
        beta * 500 +       # focused attention
        gamma * 800        # heightened/transcendent
    )

    # Meditation bonus: high alpha/theta with low beta
    calm_ratio = (alpha + theta) / max(beta, 1e-10)
    if calm_ratio > 2.0:
        base_score += min(200, calm_ratio * 50)

    # Gamma burst bonus
    if gamma > 0.15:
        base_score += 150

    # Entropy factor: low entropy = focused = higher consciousness
    entropy_bonus = (1 - se) * 100
    base_score += entropy_bonus

    score = min(1000, max(0, base_score))

    # Find matching consciousness level
    matched_level = CONSCIOUSNESS_LEVELS[0]
    for level in CONSCIOUSNESS_LEVELS:
        low, high = level["range"]
        if low <= score < high:
            matched_level = level
            break
    else:
        if score >= 900:
            matched_level = CONSCIOUSNESS_LEVELS[-1]

    return {
        "score": round(score, 1),
        "level": matched_level["level"],
        "description": matched_level["description"],
        "band_contributions": {
            "delta": round(delta * 50, 1),
            "theta": round(theta * 200, 1),
            "alpha": round(alpha * 400, 1),
            "beta": round(beta * 500, 1),
            "gamma": round(gamma * 800, 1),
        },
        "calm_ratio": round(calm_ratio, 3),
        "spectral_entropy": round(se, 3),
    }


def compute_third_eye_activation(
    eeg: np.ndarray, fs: float = 256.0
) -> Dict:
    """Measure third eye (Ajna) activation through gamma and high-beta analysis.

    High gamma activity in frontal regions, especially when combined
    with alpha baseline, correlates with reported intuitive experiences
    and heightened perception in contemplative traditions.
    """
    bands = extract_band_powers(eeg, fs)

    # High beta (20-40 Hz) and gamma (40+ Hz) are Ajna indicators
    freqs, psd = scipy_signal.welch(eeg, fs=fs, nperseg=min(len(eeg), int(fs * 2)))
    total_power = _trapezoid(psd, freqs)
    if total_power == 0:
        total_power = 1e-10

    # Sub-band analysis for third eye region
    high_beta_mask = (freqs >= 20) & (freqs < 40)
    gamma_mask = (freqs >= 40) & (freqs < 100)

    high_beta_power = _trapezoid(psd[high_beta_mask], freqs[high_beta_mask]) / total_power if high_beta_mask.any() else 0
    gamma_power = _trapezoid(psd[gamma_mask], freqs[gamma_mask]) / total_power if gamma_mask.any() else 0

    # Ajna activation score
    ajna_raw = high_beta_power * 0.4 + gamma_power * 0.6
    alpha = bands.get("alpha", 0)

    # Alpha base enhances the score (calm + gamma = spiritual insight)
    if alpha > 0.15:
        ajna_raw *= 1.3

    activation_pct = min(100, ajna_raw * 500)

    if activation_pct >= 75:
        status = "Highly active"
        insight = "Strong intuitive awareness. Trust your inner knowing."
    elif activation_pct >= 50:
        status = "Active"
        insight = "Intuition channel opening. Pay attention to subtle perceptions."
    elif activation_pct >= 25:
        status = "Awakening"
        insight = "Third eye beginning to activate. Practice visualization meditation."
    else:
        status = "Resting"
        insight = "Focus on trataka (candle gazing) or third eye meditation to activate."

    return {
        "activation_pct": round(activation_pct, 1),
        "status": status,
        "insight": insight,
        "high_beta_power": round(high_beta_power, 4),
        "gamma_power": round(gamma_power, 4),
        "alpha_base": round(alpha, 4),
        "alpha_enhanced": alpha > 0.15,
    }


def full_spiritual_analysis(
    eeg: np.ndarray,
    fs: float = 256.0,
    eeg_left: Optional[np.ndarray] = None,
    eeg_right: Optional[np.ndarray] = None,
) -> Dict:
    """Run complete spiritual energy analysis on EEG data.

    Combines all spiritual metrics into a unified report.

    Args:
        eeg: Primary EEG signal (1D array).
        fs: Sampling frequency.
        eeg_left: Optional left-hemisphere EEG for prana balance.
        eeg_right: Optional right-hemisphere EEG for prana balance.

    Returns:
        Complete spiritual energy analysis dictionary.
    """
    chakras = compute_chakra_activations(eeg, fs)
    chakra_bal = compute_chakra_balance(chakras)
    meditation = compute_meditation_depth(eeg, fs)
    aura = compute_aura_energy(eeg, fs)
    kundalini = compute_kundalini_flow(eeg, fs)
    consciousness = compute_consciousness_level(eeg, fs)
    third_eye = compute_third_eye_activation(eeg, fs)

    result = {
        "chakras": chakras,
        "chakra_balance": chakra_bal,
        "meditation_depth": meditation,
        "aura": aura,
        "kundalini": kundalini,
        "consciousness": consciousness,
        "third_eye": third_eye,
    }

    # Add prana balance if bilateral EEG available
    if eeg_left is not None and eeg_right is not None:
        result["prana_balance"] = compute_prana_balance(eeg_left, eeg_right, fs)

    # Generate overall spiritual insight
    result["insight"] = _generate_spiritual_insight(
        chakra_bal, meditation, consciousness, kundalini
    )

    return result


def _generate_spiritual_insight(
    chakra_balance: Dict,
    meditation: Dict,
    consciousness: Dict,
    kundalini: Dict,
) -> Dict:
    """Generate personalized spiritual insight from all metrics."""
    insights = []
    practices = []

    # Meditation depth insight
    depth = meditation["depth_score"]
    if depth >= 7:
        insights.append("Your mind has reached deep stillness — a rare and beautiful state.")
    elif depth >= 4:
        insights.append("You are settling into meditation. The mind is quieting.")
    else:
        insights.append("Your mind is currently active. This is natural — observe without judgment.")

    # Chakra insight
    dominant = chakra_balance.get("dominant_chakra", "heart")
    harmony = chakra_balance.get("harmony_score", 50)
    if harmony >= 70:
        insights.append("Your energy centers are well balanced — inner harmony is present.")
    else:
        needs = chakra_balance.get("needs_attention", [])
        if needs:
            chakra_name = needs[0]["sanskrit"]
            insights.append(f"Your {chakra_name} chakra could benefit from attention.")
            practices.append(needs[0]["suggestion"])

    # Kundalini insight
    if kundalini.get("flow_continuity_pct", 0) >= 60:
        insights.append("Energy is flowing smoothly through your system.")
    elif kundalini.get("highest_chakra_reached") in ("crown", "third_eye"):
        insights.append("Upper energy centers are active — stay grounded through the heart.")

    # Consciousness insight
    c_score = consciousness.get("score", 0)
    if c_score >= 700:
        insights.append("You are experiencing an elevated state of awareness.")
        practices.append("Rest in this awareness. No effort needed.")
    elif c_score >= 400:
        practices.append("Deepen with slow, rhythmic breathing — 4 counts in, 8 counts out.")
    else:
        practices.append("Begin with 10 minutes of breath awareness to center yourself.")

    return {
        "summary": " ".join(insights),
        "recommended_practices": practices,
        "overall_energy_state": _classify_overall_state(depth, harmony, c_score),
    }


def _classify_overall_state(depth: float, harmony: float, consciousness: float) -> str:
    if depth >= 7 and harmony >= 60 and consciousness >= 600:
        return "Transcendent Harmony"
    elif depth >= 5 and harmony >= 50:
        return "Meditative Balance"
    elif harmony >= 70:
        return "Energetic Harmony"
    elif depth >= 3:
        return "Settling Awareness"
    elif consciousness >= 400:
        return "Alert Presence"
    else:
        return "Active Mind"
