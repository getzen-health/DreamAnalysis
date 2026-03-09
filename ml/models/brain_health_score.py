"""Brain Health Score Calculator.

Computes a composite brain health score (0-100) from EEG signals across
five signal-analysis domains:

1. Spectral Health    -- alpha/theta ratio, alpha peak frequency, spectral entropy
2. Connectivity Health -- inter-channel coherence (alpha), frontal-temporal coupling
3. Complexity Health   -- spectral entropy, Hjorth mobility (signal variability)
4. Stability Health    -- coefficient of variation of band powers, stationarity
5. Asymmetry Health    -- FAA balance, temporal symmetry (TP9 vs TP10)

Each domain is scored 0-100. The overall score is the unweighted mean of all
five domains, mapped to a letter grade (A/B/C/D/F).

Supports per-user baselines, session history, and actionable recommendations.

Standard Muse 2 channel layout:
    ch0 = TP9  (left temporal)
    ch1 = AF7  (left frontal)
    ch2 = AF8  (right frontal)
    ch3 = TP10 (right temporal)

References:
    Klimesch (1999) -- Alpha peak frequency and cognitive performance
    Davidson (1992) -- Frontal alpha asymmetry and emotional regulation
    Hjorth (1970)   -- EEG descriptive parameters (activity, mobility, complexity)
    Tononi et al.   -- Neural complexity and consciousness
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal

# NumPy 2.0 renamed np.trapz -> np.trapezoid; 1.x only has np.trapz
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

# Five health domains
DOMAINS = [
    "spectral",
    "connectivity",
    "complexity",
    "stability",
    "asymmetry",
]

# Grade thresholds (issue spec: A>=80, B>=65, C>=50, D>=35, F<35)
GRADE_THRESHOLDS = [
    (80, "A"),
    (65, "B"),
    (50, "C"),
    (35, "D"),
    (0, "F"),
]

# EEG frequency bands
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "high_beta": (20.0, 30.0),
}


def _grade_from_score(score: float) -> str:
    """Map a 0-100 score to a letter grade."""
    for threshold, letter in GRADE_THRESHOLDS:
        if score >= threshold:
            return letter
    return "F"


class BrainHealthScore:
    """Compute a composite brain health score from EEG data.

    Five domains (spectral, connectivity, complexity, stability,
    asymmetry) are each scored 0-100. The overall score is their
    unweighted mean, mapped to a letter grade.

    Supports per-user baselines, session tracking, and history.

    Usage:
        scorer = BrainHealthScore(fs=256.0)
        scorer.set_baseline(signals_4ch, user_id="alice")
        result = scorer.assess(signals_4ch, user_id="alice")
        print(result["overall_score"], result["grade"])
    """

    def __init__(self, fs: float = 256.0) -> None:
        self._fs = fs
        # Per-user state
        self._baselines: Dict[str, Dict] = {}
        self._history: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_baseline(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Record a resting-state baseline for a user.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate (defaults to constructor value).
            user_id: User identifier for multi-user support.

        Returns:
            Dict with ``baseline_set`` flag and ``domain_scores``.
        """
        fs = fs or self._fs
        signals = self._prepare_signals(signals)

        domain_scores = self._compute_all_domains(
            signals, fs, user_id=user_id, is_baseline=True,
        )

        self._baselines[user_id] = {
            "band_powers": self._compute_band_powers_per_channel(signals, fs),
            "domain_scores": dict(domain_scores),
        }

        return {
            "baseline_set": True,
            "domain_scores": {k: round(v, 1) for k, v in domain_scores.items()},
        }

    def assess(
        self,
        signals: np.ndarray,
        fs: Optional[float] = None,
        user_id: str = "default",
    ) -> Dict:
        """Assess current brain health from an EEG epoch.

        Args:
            signals: (n_channels, n_samples) or (n_samples,) EEG array.
            fs: Sampling rate (defaults to constructor value).
            user_id: User identifier.

        Returns:
            Dict with overall_score, grade, domain_scores,
            recommendations, and has_baseline.
        """
        fs = fs or self._fs
        signals = self._prepare_signals(signals)

        domain_scores = self._compute_all_domains(
            signals, fs, user_id=user_id, is_baseline=False,
        )

        overall = float(np.mean(list(domain_scores.values())))
        overall = float(np.clip(overall, 0.0, 100.0))

        grade = _grade_from_score(overall)
        recommendations = self._generate_recommendations(domain_scores)
        has_baseline = user_id in self._baselines

        result = {
            "overall_score": round(overall, 1),
            "grade": grade,
            "domain_scores": {k: round(v, 1) for k, v in domain_scores.items()},
            "recommendations": recommendations,
            "has_baseline": has_baseline,
        }

        # Track history
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].append(result)

        return result

    def get_session_stats(self, user_id: str = "default") -> Dict:
        """Get summary statistics for the current session.

        Returns:
            Dict with n_epochs, has_baseline, and (if epochs > 0)
            mean_score, best_domain, worst_domain.
        """
        has_baseline = user_id in self._baselines
        history = self._history.get(user_id, [])
        n_epochs = len(history)

        stats: Dict = {
            "n_epochs": n_epochs,
            "has_baseline": has_baseline,
        }

        if n_epochs > 0:
            scores = [h["overall_score"] for h in history]
            stats["mean_score"] = round(float(np.mean(scores)), 1)

            # Aggregate domain scores across epochs
            domain_means: Dict[str, float] = {}
            for domain in DOMAINS:
                vals = [h["domain_scores"][domain] for h in history]
                domain_means[domain] = float(np.mean(vals))

            stats["best_domain"] = max(
                domain_means, key=domain_means.get,  # type: ignore[arg-type]
            )
            stats["worst_domain"] = min(
                domain_means, key=domain_means.get,  # type: ignore[arg-type]
            )

        return stats

    def get_history(
        self,
        user_id: str = "default",
        last_n: Optional[int] = None,
    ) -> List[Dict]:
        """Return assessment history for a user.

        Args:
            user_id: User identifier.
            last_n: If set, return only the last N entries.

        Returns:
            List of past ``assess()`` result dicts.
        """
        history = self._history.get(user_id, [])
        if last_n is not None:
            return list(history[-last_n:])
        return list(history)

    def reset(self, user_id: str = "default") -> None:
        """Clear baseline and history for a user."""
        self._baselines.pop(user_id, None)
        self._history.pop(user_id, None)

    # ------------------------------------------------------------------ #
    #  Domain scoring                                                      #
    # ------------------------------------------------------------------ #

    def _compute_all_domains(
        self,
        signals: np.ndarray,
        fs: float,
        user_id: str = "default",
        is_baseline: bool = False,
    ) -> Dict[str, float]:
        """Score all five domains."""
        return {
            "spectral": self._score_spectral(signals, fs),
            "connectivity": self._score_connectivity(signals, fs),
            "complexity": self._score_complexity(signals, fs),
            "stability": self._score_stability(
                signals, fs, user_id, is_baseline,
            ),
            "asymmetry": self._score_asymmetry(signals, fs),
        }

    # -- 1. Spectral Health ------------------------------------------- #

    def _score_spectral(self, signals: np.ndarray, fs: float) -> float:
        """Score spectral health (0-100).

        Components:
        - Alpha/theta ratio: higher = better resting cognition
        - Alpha peak frequency: 9.5-11 Hz optimal (Klimesch 1999)
        - Spectral entropy: moderate (0.5-0.8) is healthy
        """
        avg_signal = np.mean(signals, axis=0)

        powers = self._compute_band_powers(avg_signal, fs)
        alpha = powers.get("alpha", 0.0)
        theta = powers.get("theta", 0.0)

        # Alpha/theta ratio: optimal ~1.0-2.5
        atr = alpha / max(theta, 1e-10)
        if atr >= 1.0:
            atr_score = float(np.clip(atr / 2.5 * 100.0, 0, 100))
        else:
            atr_score = float(np.clip(atr / 1.0 * 60.0, 0, 60))

        # Alpha peak frequency: 9.5-11 Hz optimal
        apf = self._compute_alpha_peak_freq(avg_signal, fs)
        if 9.5 <= apf <= 11.0:
            apf_score = 100.0
        elif 8.0 <= apf < 9.5:
            apf_score = 60.0 + (apf - 8.0) / 1.5 * 40.0
        elif 11.0 < apf <= 13.0:
            apf_score = 100.0 - (apf - 11.0) / 2.0 * 40.0
        else:
            apf_score = 30.0

        # Spectral entropy: healthy range 0.5-0.85
        se = self._compute_spectral_entropy(avg_signal, fs)
        if 0.5 <= se <= 0.85:
            se_score = 80.0 + (se - 0.5) / 0.35 * 20.0
        elif se < 0.5:
            se_score = float(np.clip(se / 0.5 * 80.0, 0, 80))
        else:
            se_score = float(np.clip((1.0 - se) / 0.15 * 80.0, 40, 100))

        return float(np.clip(
            0.40 * atr_score + 0.35 * apf_score + 0.25 * se_score,
            0, 100,
        ))

    # -- 2. Connectivity Health --------------------------------------- #

    def _score_connectivity(self, signals: np.ndarray, fs: float) -> float:
        """Score connectivity health (0-100).

        Components:
        - Inter-channel coherence in alpha band
        - Frontal-temporal coupling (AF7/AF8 vs TP9/TP10)

        Falls back to 50 for single-channel data.
        """
        n_ch = signals.shape[0]
        if n_ch < 2:
            return 50.0

        alpha_coh = self._compute_coherence(signals, fs, "alpha")
        # Moderate coherence (0.3-0.7) is optimal
        if 0.3 <= alpha_coh <= 0.7:
            coh_score = 80.0 + (alpha_coh - 0.3) / 0.4 * 20.0
        elif alpha_coh < 0.3:
            coh_score = float(np.clip(alpha_coh / 0.3 * 80.0, 0, 80))
        else:
            coh_score = float(np.clip(
                (1.0 - alpha_coh) / 0.3 * 80.0, 30, 100,
            ))

        if n_ch >= 4:
            ft_coh = self._compute_frontal_temporal_coupling(signals, fs)
            if 0.2 <= ft_coh <= 0.6:
                ft_score = 80.0 + (ft_coh - 0.2) / 0.4 * 20.0
            elif ft_coh < 0.2:
                ft_score = float(np.clip(ft_coh / 0.2 * 80.0, 0, 80))
            else:
                ft_score = float(np.clip(
                    (1.0 - ft_coh) / 0.4 * 80.0, 30, 100,
                ))
            return float(np.clip(
                0.55 * coh_score + 0.45 * ft_score, 0, 100,
            ))

        return float(np.clip(coh_score, 0, 100))

    # -- 3. Complexity Health ----------------------------------------- #

    def _score_complexity(self, signals: np.ndarray, fs: float) -> float:
        """Score complexity health (0-100).

        Components:
        - Spectral entropy (neural complexity)
        - Hjorth mobility (signal variability)
        """
        avg_signal = np.mean(signals, axis=0)

        se = self._compute_spectral_entropy(avg_signal, fs)
        se_score = float(np.clip(se * 120.0, 0, 100))

        mobility = self._compute_hjorth_mobility(avg_signal)
        if 0.02 <= mobility <= 0.5:
            mob_score = 70.0 + (mobility - 0.02) / 0.48 * 30.0
        elif mobility < 0.02:
            mob_score = float(np.clip(mobility / 0.02 * 70.0, 0, 70))
        else:
            mob_score = float(np.clip(
                (1.0 - mobility) / 0.5 * 70.0, 20, 100,
            ))

        return float(np.clip(
            0.55 * se_score + 0.45 * mob_score, 0, 100,
        ))

    # -- 4. Stability Health ------------------------------------------ #

    def _score_stability(
        self,
        signals: np.ndarray,
        fs: float,
        user_id: str,
        is_baseline: bool,
    ) -> float:
        """Score stability health (0-100).

        Components:
        - Within-epoch stationarity (CV of sub-epoch band powers)
        - Cross-epoch consistency vs baseline (if available)
        """
        avg_signal = np.mean(signals, axis=0)
        n_samples = len(avg_signal)
        min_sub_epoch = int(fs * 0.5)  # 0.5 second minimum

        if n_samples >= min_sub_epoch * 2:
            n_sub = min(4, n_samples // min_sub_epoch)
            sub_len = n_samples // n_sub
            sub_powers = []
            for i in range(n_sub):
                start = i * sub_len
                end = start + sub_len
                sp = self._compute_band_powers(avg_signal[start:end], fs)
                sub_powers.append(sp)

            alpha_vals = [sp.get("alpha", 0.0) for sp in sub_powers]
            beta_vals = [sp.get("beta", 0.0) for sp in sub_powers]

            cv_alpha = self._safe_cv(alpha_vals)
            cv_beta = self._safe_cv(beta_vals)
            avg_cv = (cv_alpha + cv_beta) / 2.0

            stationarity_score = float(np.clip(
                (1.0 - avg_cv) * 100.0, 0, 100,
            ))
        else:
            stationarity_score = 50.0

        baseline_stability_score = 50.0
        if not is_baseline and user_id in self._baselines:
            baseline = self._baselines[user_id]
            current_powers = self._compute_band_powers_per_channel(
                signals, fs,
            )
            baseline_powers = baseline["band_powers"]

            deviations = []
            for band in ["alpha", "beta", "theta"]:
                curr = current_powers.get(band, 0.0)
                base = baseline_powers.get(band, 0.0)
                if base > 1e-10:
                    dev = abs(curr - base) / base
                    deviations.append(dev)

            if deviations:
                mean_dev = float(np.mean(deviations))
                baseline_stability_score = float(np.clip(
                    (1.0 - min(mean_dev, 1.0)) * 100.0, 0, 100,
                ))

        return float(np.clip(
            0.55 * stationarity_score + 0.45 * baseline_stability_score,
            0, 100,
        ))

    # -- 5. Asymmetry Health ------------------------------------------ #

    def _score_asymmetry(self, signals: np.ndarray, fs: float) -> float:
        """Score asymmetry health (0-100).

        Components:
        - FAA balance: mild positive asymmetry is optimal
        - Temporal symmetry: TP9 vs TP10 power similarity

        Falls back to 50 for single-channel data.
        """
        n_ch = signals.shape[0]
        if n_ch < 2:
            return 50.0

        if n_ch >= 4:
            left_ch, right_ch = 1, 2   # AF7, AF8
        else:
            left_ch, right_ch = 0, 1

        left_alpha = self._get_band_power(signals[left_ch], fs, "alpha")
        right_alpha = self._get_band_power(signals[right_ch], fs, "alpha")

        faa = (
            np.log(max(right_alpha, 1e-10))
            - np.log(max(left_alpha, 1e-10))
        )

        abs_faa = abs(faa)
        if abs_faa <= 0.3:
            faa_score = 90.0 + (0.3 - abs_faa) / 0.3 * 10.0
        elif abs_faa <= 0.6:
            faa_score = 70.0
        elif abs_faa <= 1.0:
            faa_score = 50.0
        else:
            faa_score = float(np.clip(
                30.0 - (abs_faa - 1.0) * 10.0, 0, 30,
            ))

        temporal_score = 70.0
        if n_ch >= 4:
            tp9_alpha = self._get_band_power(signals[0], fs, "alpha")
            tp10_alpha = self._get_band_power(signals[3], fs, "alpha")
            denom = max(tp9_alpha, tp10_alpha)
            if denom > 1e-10:
                ratio = min(tp9_alpha, tp10_alpha) / denom
                temporal_score = float(np.clip(ratio * 100.0, 0, 100))

        return float(np.clip(
            0.60 * faa_score + 0.40 * temporal_score, 0, 100,
        ))

    # ------------------------------------------------------------------ #
    #  Low-level computation helpers                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prepare_signals(signals: np.ndarray) -> np.ndarray:
        """Ensure signals is 2D (n_channels, n_samples)."""
        signals = np.asarray(signals, dtype=float)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        return signals

    def _compute_band_powers(
        self, eeg: np.ndarray, fs: float,
    ) -> Dict[str, float]:
        """Compute normalized band powers via Welch PSD for a 1D signal."""
        nperseg = min(len(eeg), int(fs * 2))
        if nperseg < 4:
            return {band: 0.0 for band in _BANDS}

        try:
            freqs, psd = scipy_signal.welch(eeg, fs=fs, nperseg=nperseg)
        except Exception:
            return {band: 0.0 for band in _BANDS}

        total = _trapezoid(psd, freqs)
        if total < 1e-10:
            return {band: 0.0 for band in _BANDS}

        powers: Dict[str, float] = {}
        for band_name, (low, high) in _BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                powers[band_name] = float(
                    _trapezoid(psd[mask], freqs[mask]) / total,
                )
            else:
                powers[band_name] = 0.0

        return powers

    def _compute_band_powers_per_channel(
        self, signals: np.ndarray, fs: float,
    ) -> Dict[str, float]:
        """Average band powers across all channels."""
        all_powers: Dict[str, List[float]] = {b: [] for b in _BANDS}
        for ch in range(signals.shape[0]):
            ch_powers = self._compute_band_powers(signals[ch], fs)
            for band in _BANDS:
                all_powers[band].append(ch_powers[band])

        return {
            band: float(np.mean(vals)) for band, vals in all_powers.items()
        }

    def _compute_alpha_peak_freq(
        self, eeg: np.ndarray, fs: float,
    ) -> float:
        """Find peak frequency in the alpha band (8-12 Hz)."""
        nperseg = min(len(eeg), int(fs * 2))
        if nperseg < 4:
            return 10.0

        try:
            freqs, psd = scipy_signal.welch(eeg, fs=fs, nperseg=nperseg)
        except Exception:
            return 10.0

        alpha_mask = (freqs >= 8) & (freqs <= 12)
        if not np.any(alpha_mask):
            return 10.0

        return float(freqs[alpha_mask][np.argmax(psd[alpha_mask])])

    def _compute_spectral_entropy(
        self, eeg: np.ndarray, fs: float,
    ) -> float:
        """Compute normalized spectral entropy."""
        nperseg = min(len(eeg), int(fs * 2))
        if nperseg < 4:
            return 0.5

        try:
            _, psd = scipy_signal.welch(eeg, fs=fs, nperseg=nperseg)
        except Exception:
            return 0.5

        psd_sum = np.sum(psd)
        if psd_sum < 1e-10:
            return 0.0

        psd_norm = psd / psd_sum
        psd_positive = psd_norm[psd_norm > 0]
        if len(psd_positive) == 0:
            return 0.0

        se = -np.sum(psd_positive * np.log2(psd_positive + 1e-10))
        max_entropy = np.log2(len(psd_positive))
        if max_entropy < 1e-10:
            return 0.0

        return float(np.clip(se / max_entropy, 0.0, 1.0))

    @staticmethod
    def _compute_hjorth_mobility(eeg: np.ndarray) -> float:
        """Compute Hjorth mobility parameter."""
        if len(eeg) < 3:
            return 0.0

        activity = np.var(eeg)
        if activity < 1e-10:
            return 0.0

        diff1 = np.diff(eeg)
        return float(np.sqrt(np.var(diff1) / activity))

    def _compute_coherence(
        self,
        signals: np.ndarray,
        fs: float,
        band: str = "alpha",
    ) -> float:
        """Compute mean inter-channel coherence in a band."""
        low, high = _BANDS.get(band, (8.0, 12.0))
        n_ch = signals.shape[0]

        if n_ch < 2:
            return 1.0

        coh_values: List[float] = []
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                nperseg = min(len(signals[i]), int(fs * 2))
                if nperseg < 4:
                    continue
                try:
                    freqs, coh = scipy_signal.coherence(
                        signals[i], signals[j], fs=fs, nperseg=nperseg,
                    )
                    mask = (freqs >= low) & (freqs <= high)
                    if mask.any():
                        val = float(np.nanmean(coh[mask]))
                        if np.isfinite(val):
                            coh_values.append(val)
                except Exception:
                    continue

        return float(np.mean(coh_values)) if coh_values else 0.0

    def _compute_frontal_temporal_coupling(
        self, signals: np.ndarray, fs: float,
    ) -> float:
        """Average alpha coherence between frontal and temporal channels."""
        pairs = [(0, 1), (0, 2), (3, 1), (3, 2)]  # TP-AF pairs
        low, high = _BANDS["alpha"]

        coh_values: List[float] = []
        for i, j in pairs:
            if i >= signals.shape[0] or j >= signals.shape[0]:
                continue
            nperseg = min(len(signals[i]), int(fs * 2))
            if nperseg < 4:
                continue
            try:
                freqs, coh = scipy_signal.coherence(
                    signals[i], signals[j], fs=fs, nperseg=nperseg,
                )
                mask = (freqs >= low) & (freqs <= high)
                if mask.any():
                    val = float(np.nanmean(coh[mask]))
                    if np.isfinite(val):
                        coh_values.append(val)
            except Exception:
                continue

        return float(np.mean(coh_values)) if coh_values else 0.0

    def _get_band_power(
        self, eeg: np.ndarray, fs: float, band: str,
    ) -> float:
        """Get absolute (un-normalized) band power for a single channel."""
        low, high = _BANDS.get(band, (8.0, 12.0))
        nperseg = min(len(eeg), int(fs * 2))
        if nperseg < 4:
            return 1e-10

        try:
            freqs, psd = scipy_signal.welch(eeg, fs=fs, nperseg=nperseg)
        except Exception:
            return 1e-10

        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 1e-10

        return float(max(_trapezoid(psd[mask], freqs[mask]), 1e-10))

    @staticmethod
    def _safe_cv(values: List[float]) -> float:
        """Coefficient of variation, safe for near-zero means."""
        arr = np.array(values)
        mean_val = np.mean(arr)
        if abs(mean_val) < 1e-10:
            return 1.0  # maximally unstable
        return float(np.std(arr) / abs(mean_val))

    # ------------------------------------------------------------------ #
    #  Recommendations                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _generate_recommendations(
        domain_scores: Dict[str, float],
    ) -> List[str]:
        """Generate actionable recommendations based on weak domains."""
        recs: List[str] = []

        if domain_scores.get("spectral", 100) < 60:
            recs.append(
                "spectral: practice eyes-closed relaxation for 10 minutes "
                "daily to strengthen alpha rhythms and improve "
                "alpha/theta ratio"
            )
        if domain_scores.get("connectivity", 100) < 60:
            recs.append(
                "connectivity: engage in tasks requiring cross-brain "
                "coordination (music, bilateral exercises) to improve "
                "inter-hemispheric coherence"
            )
        if domain_scores.get("complexity", 100) < 60:
            recs.append(
                "complexity: challenge your brain with novel stimuli "
                "(puzzles, new learning) to increase neural complexity "
                "and spectral entropy"
            )
        if domain_scores.get("stability", 100) < 60:
            recs.append(
                "stability: ensure consistent sleep schedule and reduce "
                "caffeine/stimulants to stabilize EEG band power patterns"
            )
        if domain_scores.get("asymmetry", 100) < 60:
            recs.append(
                "asymmetry: regular aerobic exercise and mindfulness "
                "meditation can help normalize frontal alpha asymmetry"
            )

        if not recs:
            recs.append(
                "overall: brain health metrics are in good range; "
                "maintain current healthy habits"
            )

        return recs
