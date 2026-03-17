"""Data loading utilities for real EEG datasets.

Provides loaders for:
- Sleep-EDF (PhysioNet) — 5-class sleep staging
- DENS (OpenNeuro ds003751) — emotion recognition with 128-channel EEG
- DEAP — emotion recognition (with simulation fallback)
- REM detection — binary REM vs non-REM from Sleep-EDF
- PhysioNet Motor Imagery (EEGBCI) — motor execution/imagery classification
- MOABB — standardized BCI benchmark datasets
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from processing.eeg_processor import extract_features, preprocess

# AASM sleep stage mapping from Sleep-EDF annotations
_SLEEP_EDF_STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # merge S3+S4 into N3
    "Sleep stage R": 4,
}

# DEAP circumplex model thresholds for emotion mapping
_DEAP_EMOTIONS = {
    0: "happy",    # high valence, high arousal
    1: "sad",      # low valence, low arousal
    2: "angry",    # low valence, high arousal
    3: "fearful",  # low valence, mid-high arousal
    4: "relaxed",  # high valence, low arousal
    5: "focused",  # mid valence, mid arousal
}


def load_sleep_edf(
    n_subjects: int = 20, epoch_sec: float = 30.0, target_fs: float = 256.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Sleep-EDF dataset from PhysioNet via MNE.

    Downloads automatically on first use. Extracts Fpz-Cz channel,
    resamples to target_fs, and maps to AASM 5-class labels.

    Args:
        n_subjects: Number of subjects to load (max 20 for age group).
        epoch_sec: Epoch duration in seconds.
        target_fs: Target sampling frequency after resampling.

    Returns:
        (X, y) where X is (n_epochs, n_samples) and y is (n_epochs,) with
        labels 0-4 mapping to [Wake, N1, N2, N3, REM].
    """
    import mne

    mne.set_log_level("WARNING")

    subjects = list(range(min(n_subjects, 20)))
    recording = [1]  # night 1

    X_all, y_all = [], []

    for subj in subjects:
        try:
            files = mne.datasets.sleep_physionet.age.fetch_data(
                subjects=[subj], recording=recording
            )
        except Exception as e:
            print(f"  Skipping subject {subj}: {e}")
            continue

        for raw_fname, annot_fname in files:
            raw = mne.io.read_raw_edf(raw_fname, preload=True, verbose=False)
            annots = mne.read_annotations(annot_fname)
            raw.set_annotations(annots)

            # Pick Fpz-Cz channel
            channel = "EEG Fpz-Cz"
            if channel not in raw.ch_names:
                channel = raw.ch_names[0]
            raw.pick([channel])

            # Resample to target frequency
            if raw.info["sfreq"] != target_fs:
                raw.resample(target_fs)

            events, event_id = mne.events_from_annotations(
                raw, chunk_duration=epoch_sec, verbose=False
            )

            # Map event IDs to our 5-class scheme
            for description, eid in event_id.items():
                if description not in _SLEEP_EDF_STAGE_MAP:
                    continue
                stage_label = _SLEEP_EDF_STAGE_MAP[description]
                stage_events = events[events[:, 2] == eid]

                for event in stage_events:
                    start_sample = event[0]
                    n_samples = int(epoch_sec * target_fs)
                    end_sample = start_sample + n_samples

                    if end_sample > raw.n_times:
                        continue

                    epoch_data = raw.get_data(
                        start=start_sample, stop=end_sample
                    )[0]

                    if len(epoch_data) == n_samples:
                        X_all.append(epoch_data)
                        y_all.append(stage_label)

        print(f"  Subject {subj}: {len(y_all)} epochs total so far")

    if len(X_all) == 0:
        raise RuntimeError("No epochs loaded from Sleep-EDF. Check MNE data download.")

    return np.array(X_all), np.array(y_all)


def load_dens(
    data_dir: str = "data/dens",
    target_fs: float = 256.0,
    epoch_sec: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DENS dataset (OpenNeuro ds003751) — real 128-channel EEG with emotion labels.

    Dataset: 'Dataset on Emotion with Naturalistic Stimuli'
    - 128 EEG channels at 250Hz (EEGLAB .set/.fdt format)
    - Valence/arousal ratings (1-9) per trial
    - Emotion categories: Joyous, Afraid, Angry, Sad, etc.

    Download: aws s3 sync s3://openneuro.org/ds003751/ data/dens/ --no-sign-request

    Args:
        data_dir: Path to DENS subject directories.
        target_fs: Target sampling frequency after resampling.
        epoch_sec: Duration of each epoch to extract (seconds).

    Returns:
        (X, y, groups) where X is feature vectors, y is emotion labels 0-5,
        and groups is an array of subject IDs (for LOSO cross-validation).
    """
    import mne
    import csv

    mne.set_log_level("WARNING")

    dens_path = Path(data_dir)
    subject_dirs = sorted(dens_path.glob("sub-*"))

    if not subject_dirs:
        raise FileNotFoundError(f"No DENS subjects found in {data_dir}")

    X_all, y_all, groups_all = [], [], []

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        print(f"  Loading {subj_id}...")

        # Find EEG file (.set format)
        set_files = list((subj_dir / "eeg").glob("*_eeg.set"))
        if not set_files:
            print("    No .set file found, skipping")
            continue

        # Find behavioral file with emotion ratings
        beh_files = list((subj_dir / "beh").glob("*_beh.tsv"))
        if not beh_files:
            print("    No behavioral file found, skipping")
            continue

        # Load behavioral data (valence/arousal ratings per trial)
        trials = []
        with open(beh_files[0], "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    valence = float(row.get("valence", 0))
                    arousal = float(row.get("arousal", 0))
                    trials.append({"valence": valence, "arousal": arousal})
                except (ValueError, TypeError):
                    continue

        if not trials:
            print("    No valid trials in behavioral data, skipping")
            continue

        # Load events file to find stimulus onsets
        events_files = list((subj_dir / "eeg").glob("*_events.tsv"))
        stim_onsets = []
        if events_files:
            with open(events_files[0], "r") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    if row.get("trial_type") == "stm":
                        try:
                            onset_ms = float(row["onset"])
                            stim_onsets.append(onset_ms / 1000.0)  # convert to seconds
                        except (ValueError, KeyError):
                            continue

        # Load raw EEG
        try:
            raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)
        except Exception as e:
            print(f"    Failed to load EEG: {e}")
            continue

        fs_orig = raw.info["sfreq"]

        # Pick only EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(eeg_picks) == 0:
            print("    No EEG channels found, skipping")
            continue
        raw.pick(eeg_picks)

        # Resample if needed
        if fs_orig != target_fs:
            raw.resample(target_fs)

        data = raw.get_data()  # (n_channels, n_samples)
        n_samples_epoch = int(epoch_sec * target_fs)

        # Extract overlapping windows from each trial for more samples
        # Each stimulus lasts ~10-20s; extract sliding windows with 50% overlap
        step_samples = n_samples_epoch // 2  # 50% overlap
        n_extracted = 0

        for trial_idx, onset_sec in enumerate(stim_onsets):
            if trial_idx >= len(trials):
                break

            # Determine trial duration (until next stim or +20s max)
            if trial_idx + 1 < len(stim_onsets):
                trial_end_sec = min(onset_sec + 20.0, stim_onsets[trial_idx + 1])
            else:
                trial_end_sec = onset_sec + 20.0

            trial_start = int(onset_sec * target_fs)
            trial_end = int(trial_end_sec * target_fs)

            # Map valence/arousal to emotion label
            valence = trials[trial_idx]["valence"]
            arousal = trials[trial_idx]["arousal"]
            emotion_label = _circumplex_to_emotion(valence, arousal)

            # Slide windows across the trial
            pos = trial_start
            while pos + n_samples_epoch <= min(trial_end, data.shape[1]):
                epoch_data = data[0, pos:pos + n_samples_epoch]

                if len(epoch_data) == n_samples_epoch:
                    features = extract_features(preprocess(epoch_data, target_fs), target_fs)
                    X_all.append(list(features.values()))
                    y_all.append(emotion_label)
                    groups_all.append(subj_id)
                    n_extracted += 1

                pos += step_samples

        print(f"    Extracted {n_extracted} epochs, {len(y_all)} total")

    if len(X_all) == 0:
        raise RuntimeError("No epochs extracted from DENS dataset.")

    return np.array(X_all), np.array(y_all), np.array(groups_all)


def load_deap_or_simulate(
    data_dir: str = "data/deap",
    n_samples_per_class: int = 1500,
    fs: float = 256.0,
    epoch_sec: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load real EEG emotion data, with cascading fallback.

    Priority: DENS (OpenNeuro) → DEAP → enhanced simulation.

    Args:
        data_dir: Path to DEAP .dat files.
        n_samples_per_class: Samples per class for simulation fallback.
        fs: Sampling frequency.
        epoch_sec: Epoch duration for simulation.

    Returns:
        (X, y) where X is feature vectors and y is emotion labels 0-5.
    """
    # Try DENS first (freely downloadable from OpenNeuro)
    dens_path = Path("data/dens")
    if dens_path.exists() and list(dens_path.glob("sub-*")):
        try:
            print("  Found DENS dataset, loading real EEG data...")
            X, y, _ = load_dens(data_dir="data/dens", target_fs=fs, epoch_sec=epoch_sec)
            return X, y
        except Exception as e:
            print(f"  DENS loading failed: {e}, trying DEAP...")

    # Try DEAP (local files or Kaggle auto-download)
    deap_path = Path(data_dir)
    if deap_path.exists() and list(deap_path.glob("s*.dat")):
        X, y, _ = _load_deap_real(deap_path, fs)
        return X, y

    # Try auto-downloading DEAP from Kaggle
    try:
        print("  Attempting DEAP download from Kaggle...")
        deap_path = download_deap_kaggle(target_dir=data_dir)
        if list(deap_path.glob("s*.dat")):
            X, y, _ = _load_deap_real(deap_path, fs)
            return X, y
    except Exception as e:
        print(f"  DEAP Kaggle download failed: {e}")

    print("  No real EEG dataset found, using enhanced simulation fallback...")
    return _simulate_emotion_data(n_samples_per_class, fs, epoch_sec)


def _load_deap_real(
    data_dir: Path, fs: float = 128.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real DEAP .dat files and map to 6 emotions via circumplex model.

    DEAP preprocessed data is at 128 Hz with 3s baseline + 60s trial.
    Uses average of frontal EEG channels for better emotion discrimination.

    Returns:
        X: Feature matrix (n_epochs, n_features).
        y: Emotion labels 0-5 (n_epochs,).
        groups: Subject IDs per epoch (n_epochs,) — for LOSO cross-validation.
    """
    import pickle

    # Frontal channels most relevant for emotion (indices in DEAP 32-ch layout)
    # Fp1=0, AF3=1, F3=2, F7=3, FC5=4, FC1=5, Fp2=16, AF4=17, F4=18, F8=19
    FRONTAL_CH = [0, 1, 2, 3, 4, 5, 16, 17, 18, 19]

    X_all, y_all, groups_all = [], [], []
    dat_files = sorted(data_dir.glob("s*.dat"))

    for dat_file in dat_files:
        subj_id = dat_file.stem  # e.g. "s01", "s02", ...
        with open(dat_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        # data["data"]: (40 trials, 40 channels, 8064 samples @ 128 Hz)
        # data["labels"]: (40 trials, 4) -> [valence, arousal, dominance, liking]
        eeg_data = data["data"][:, :32, :]  # first 32 channels are EEG
        labels = data["labels"]

        for trial_idx in range(eeg_data.shape[0]):
            valence = labels[trial_idx, 0]  # 1-9 scale
            arousal = labels[trial_idx, 1]  # 1-9 scale
            emotion_label = _circumplex_to_emotion(valence, arousal)

            # Skip 3s baseline (384 samples @ 128 Hz), use trial data
            trial_eeg = eeg_data[trial_idx, :, 384:]

            # Average frontal channels for emotion-relevant signal
            frontal_avg = trial_eeg[FRONTAL_CH, :].mean(axis=0)

            features = extract_features(preprocess(frontal_avg, fs), fs)
            X_all.append(list(features.values()))
            y_all.append(emotion_label)
            groups_all.append(subj_id)

    return np.array(X_all), np.array(y_all), np.array(groups_all)


def _circumplex_to_emotion(valence: float, arousal: float) -> int:
    """Map valence-arousal to one of 6 emotion classes using circumplex model.

    Uses median-split thresholds (5.0) for balanced distribution across
    the valence-arousal quadrants.
    """
    v_high = valence >= 5.0
    a_high = arousal >= 5.0

    if v_high and a_high:
        # High valence, high arousal -> happy or focused
        return 0 if valence >= 6.0 else 5  # happy vs focused
    elif not v_high and not a_high:
        # Low valence, low arousal -> sad or relaxed
        return 1 if valence < 4.0 else 4  # sad vs relaxed
    elif not v_high and a_high:
        # Low valence, high arousal -> angry or fearful
        return 2 if arousal >= 6.0 else 3  # angry vs fearful
    else:
        # High valence, low arousal -> relaxed or focused
        return 4 if arousal < 4.0 else 5  # relaxed vs focused


def _simulate_emotion_data(
    n_samples_per_class: int, fs: float, epoch_sec: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced simulation fallback with more samples and noise."""
    from simulation.eeg_simulator import simulate_eeg

    emotion_to_state = {
        0: "rest",
        1: "deep_sleep",
        2: "stress",
        3: "stress",
        4: "meditation",
        5: "focus",
    }

    X_all, y_all = [], []

    for emotion_idx, state_name in emotion_to_state.items():
        for i in range(n_samples_per_class):
            result = simulate_eeg(state=state_name, duration=epoch_sec, fs=fs)
            eeg = np.array(result["signals"][0])

            # Add extra noise for variety
            noise_level = 0.1 + 0.05 * (i % 5)
            eeg = eeg + np.random.normal(0, noise_level * np.std(eeg), len(eeg))

            features = extract_features(preprocess(eeg, fs), fs)
            X_all.append(list(features.values()))
            y_all.append(emotion_idx)

    return np.array(X_all), np.array(y_all)


def load_rem_detection(
    n_subjects: int = 20, epoch_sec: float = 30.0, target_fs: float = 256.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Sleep-EDF data with binary labels: REM=1, non-REM=0.

    Reuses load_sleep_edf and remaps 5-class labels to binary.

    Args:
        n_subjects: Number of subjects.
        epoch_sec: Epoch duration.
        target_fs: Target sampling frequency.

    Returns:
        (X, y) where y is binary: 1=REM, 0=non-REM.
    """
    X, y = load_sleep_edf(n_subjects, epoch_sec, target_fs)

    # Remap: REM (class 4) = 1, everything else = 0
    y_binary = (y == 4).astype(int)

    return X, y_binary


# ---------------------------------------------------------------------------
# Motor Imagery task labels
# ---------------------------------------------------------------------------
_MI_TASK_MAP = {
    "hands": 0,
    "feet": 1,
}


def load_motor_imagery(
    subjects: Optional[list] = None,
    runs_motor_imagery: Optional[list] = None,
    target_fs: float = 160.0,
    epoch_sec: float = 3.0,
    fmin: float = 7.0,
    fmax: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load PhysioNet EEG Motor Movement/Imagery dataset via MNE.

    Downloads automatically on first use. Extracts motor imagery epochs
    (hands vs feet) from 64-channel EEG.

    Dataset: PhysioNet EEGMMIDB — 109 subjects, 64 EEG channels at 160 Hz.
    Reference: Schalk et al., BCI2000, IEEE TBME, 2004.

    Args:
        subjects: List of subject IDs (1-109). Defaults to first 10.
        runs_motor_imagery: EDF run numbers for MI tasks.
            Defaults to [6, 10, 14] (imagine opening/closing hands vs feet).
        target_fs: Target sampling frequency (native is 160 Hz).
        epoch_sec: Duration of each epoch in seconds after event onset.
        fmin: Bandpass filter low cutoff (Hz).
        fmax: Bandpass filter high cutoff (Hz).

    Returns:
        (X, y) where X is (n_epochs, n_channels, n_samples) and y is
        (n_epochs,) with labels 0=hands, 1=feet.
    """
    import mne
    from mne.datasets import eegbci
    from mne.io import concatenate_raws, read_raw_edf

    mne.set_log_level("WARNING")

    if subjects is None:
        subjects = list(range(1, 11))  # first 10 subjects
    if runs_motor_imagery is None:
        runs_motor_imagery = [6, 10, 14]  # imagine hands vs feet

    X_all, y_all = [], []

    for subj in subjects:
        try:
            raw_fnames = eegbci.load_data(subj, runs_motor_imagery)
        except Exception as e:
            print(f"  Skipping subject {subj}: {e}")
            continue

        raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
        raw = concatenate_raws(raws)

        eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore")
        raw.set_eeg_reference(projection=True)
        raw.filter(fmin, fmax, fir_design="firwin", verbose=False)

        # Rename annotations: T1 = hands, T2 = feet
        raw.annotations.rename({"T1": "hands", "T2": "feet"})

        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Keep only hands/feet events
        kept_events = {k: v for k, v in event_id.items() if k in _MI_TASK_MAP}
        if not kept_events:
            print(f"  Subject {subj}: no MI events found, skipping")
            continue

        tmin, tmax = 0.0, epoch_sec
        epochs = mne.Epochs(
            raw, events, event_id=kept_events,
            tmin=tmin, tmax=tmax, baseline=None,
            preload=True, verbose=False,
        )

        for label_name, label_id in _MI_TASK_MAP.items():
            if label_name not in kept_events:
                continue
            subset = epochs[label_name].get_data()  # (n_trials, n_channels, n_samples)
            for trial in subset:
                X_all.append(trial)
                y_all.append(label_id)

        print(f"  Subject {subj}: {len(y_all)} epochs total so far")

    if len(X_all) == 0:
        raise RuntimeError(
            "No MI epochs loaded. Check MNE eegbci data download."
        )

    return np.array(X_all), np.array(y_all)


def load_motor_imagery_features(
    subjects: Optional[list] = None,
    target_fs: float = 160.0,
    epoch_sec: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load Motor Imagery data and extract feature vectors (like other loaders).

    Extracts the same 17-feature set used by emotion/sleep models so MI data
    can be used with the existing sklearn pipeline.

    Args:
        subjects: List of subject IDs (1-109). Defaults to first 10.
        target_fs: Target sampling frequency.
        epoch_sec: Epoch duration in seconds.

    Returns:
        (X, y) where X is (n_epochs, n_features) and y is (n_epochs,).
    """
    X_raw, y = load_motor_imagery(
        subjects=subjects, target_fs=target_fs, epoch_sec=epoch_sec
    )

    X_feat = []
    for epoch in X_raw:
        # Average across channels then extract features
        avg_signal = epoch.mean(axis=0)
        features = extract_features(preprocess(avg_signal, target_fs), target_fs)
        X_feat.append(list(features.values()))

    return np.array(X_feat), y


def load_moabb_dataset(
    dataset_name: str = "BNCI2014_001",
    paradigm_name: str = "MotorImagery",
    subjects: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a BCI benchmark dataset via MOABB (Mother of All BCI Benchmarks).

    MOABB provides standardized access to dozens of public BCI datasets.
    All data is auto-downloaded on first use.

    Supported dataset_name values (most common):
        - "BNCI2014_001" — BCI Competition IV 2a, 9 subjects, 22 EEG ch,
          4-class MI (left hand, right hand, feet, tongue)
        - "BNCI2014_004" — BCI Competition IV 2b, 9 subjects, 3 EEG ch,
          2-class MI (left hand, right hand)
        - "BNCI2015_001" — 12 subjects, 13 EEG ch, 2-class MI
        - "Zhou2016" — 4 subjects, 14 EEG ch, 3-class MI
        - "PhysionetMI" — PhysioNet EEGMMIDB via MOABB, 109 subjects

    Supported paradigm_name values:
        - "MotorImagery" — motor imagery classification
        - "P300" — P300 event-related potential
        - "SSVEP" — steady-state visual evoked potential

    Args:
        dataset_name: Name of the MOABB dataset class.
        paradigm_name: Name of the MOABB paradigm class.
        subjects: List of subject IDs (1-based). None = all subjects.

    Returns:
        (X, y) where X is (n_epochs, n_channels, n_samples) and y is
        (n_epochs,) with string labels (e.g. "left_hand", "right_hand").
    """
    import moabb
    from moabb import datasets as moabb_datasets
    from moabb import paradigms as moabb_paradigms

    moabb.set_log_level("WARNING")

    # Resolve dataset class
    dataset_cls = getattr(moabb_datasets, dataset_name, None)
    if dataset_cls is None:
        available = [
            n for n in dir(moabb_datasets)
            if not n.startswith("_") and n[0].isupper()
        ]
        raise ValueError(
            f"Unknown MOABB dataset '{dataset_name}'. "
            f"Available: {available[:20]}..."
        )
    dataset = dataset_cls()

    # Resolve paradigm class
    paradigm_cls = getattr(moabb_paradigms, paradigm_name, None)
    if paradigm_cls is None:
        raise ValueError(
            f"Unknown MOABB paradigm '{paradigm_name}'. "
            f"Available: MotorImagery, P300, SSVEP, LeftRightImagery, "
            f"FilterBankMotorImagery, FilterBankSSVEP"
        )
    paradigm = paradigm_cls()

    if subjects is not None:
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=subjects)
    else:
        X, y, meta = paradigm.get_data(dataset=dataset)

    print(f"  MOABB {dataset_name}: {X.shape[0]} epochs, "
          f"{X.shape[1]} channels, {X.shape[2]} samples, "
          f"classes={set(y)}")

    return X, y


def load_moabb_features(
    dataset_name: str = "BNCI2014_001",
    paradigm_name: str = "MotorImagery",
    subjects: Optional[list] = None,
    target_fs: float = 250.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load MOABB dataset and extract feature vectors for sklearn pipelines.

    Args:
        dataset_name: MOABB dataset class name.
        paradigm_name: MOABB paradigm class name.
        subjects: List of subject IDs. None = all.
        target_fs: Sampling rate to use for feature extraction.

    Returns:
        (X, y) where X is (n_epochs, n_features) and y is (n_epochs,)
        with integer labels.
    """
    X_raw, y_str = load_moabb_dataset(dataset_name, paradigm_name, subjects)

    # Encode string labels to integers
    label_set = sorted(set(y_str))
    label_map = {label: idx for idx, label in enumerate(label_set)}
    y = np.array([label_map[l] for l in y_str])
    print(f"  Label mapping: {label_map}")

    X_feat = []
    for epoch in X_raw:
        avg_signal = epoch.mean(axis=0)
        features = extract_features(preprocess(avg_signal, target_fs), target_fs)
        X_feat.append(list(features.values()))

    return np.array(X_feat), y


def download_deap_kaggle(
    target_dir: str = "data/deap",
) -> Path:
    """Download DEAP dataset from Kaggle via kagglehub or kaggle CLI.

    Uses kagglehub (preferred) or falls back to kaggle CLI.
    Kaggle slug: manh123df/deap-dataset (~2.93 GB preprocessed .dat files)

    Args:
        target_dir: Directory to copy the .dat files into.

    Returns:
        Path to the directory containing s01.dat ... s32.dat
    """
    import shutil

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(target.glob("s*.dat"))
    if len(existing) >= 32:
        print(f"  DEAP already downloaded: {len(existing)} .dat files in {target}")
        return target

    # Try kagglehub first (simplest, handles auth via browser)
    try:
        import kagglehub

        print("  Downloading DEAP via kagglehub (manh123df/deap-dataset, ~2.93 GB)...")
        cache_path = Path(kagglehub.dataset_download("manh123df/deap-dataset"))

        # Find .dat files (may be nested)
        dat_files = list(cache_path.rglob("s*.dat"))
        if dat_files:
            for f in dat_files:
                dest = target / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
            dat_count = len(list(target.glob("s*.dat")))
            print(f"  DEAP download complete: {dat_count} .dat files")
            return target
    except ImportError:
        print("  kagglehub not installed, trying kaggle CLI...")
    except Exception as e:
        print(f"  kagglehub download failed: {e}, trying kaggle CLI...")

    # Fallback: kaggle CLI
    import subprocess
    import zipfile

    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "Neither kagglehub nor kaggle CLI available.\n"
            "Install with: pip install kagglehub\n"
            "Or: pip install kaggle (+ ~/.kaggle/kaggle.json)"
        )

    print("  Downloading DEAP via kaggle CLI...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "manh123df/deap-dataset",
         "-p", str(target)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")

    for zf in target.glob("*.zip"):
        print(f"  Extracting {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(target)
        zf.unlink()

    # Move files up if nested
    for subdir in target.iterdir():
        if subdir.is_dir():
            for dat_file in subdir.glob("s*.dat"):
                dat_file.rename(target / dat_file.name)

    dat_count = len(list(target.glob("s*.dat")))
    print(f"  DEAP download complete: {dat_count} .dat files")
    return target


def download_dens_openneuro(
    target_dir: str = "data/dens",
) -> Path:
    """Download DENS dataset from OpenNeuro (ds003751).

    128-channel EEG emotion dataset, freely available, no registration needed.

    Args:
        target_dir: Directory to download into.

    Returns:
        Path to the dataset directory.
    """
    import openneuro

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    existing_subjects = list(target.glob("sub-*"))
    print(f"  Found {len(existing_subjects)} existing subjects, syncing...")

    openneuro.download(
        dataset="ds003751",
        target_dir=str(target),
        verify_size=False,
    )

    updated_subjects = list(target.glob("sub-*"))
    print(f"  DENS download complete: {len(updated_subjects)} subjects")
    return target


# ---------------------------------------------------------------------------
# EEGdenoiseNet — paired clean/noisy data for denoiser training
# ---------------------------------------------------------------------------

def load_eegdenoisenet(
    data_dir: str = "data/eegdenoisenet",
    target_fs: float = 256.0,
) -> dict:
    """Load EEGdenoiseNet paired clean/noisy EEG epochs.

    Dataset: 4,514 clean EEG + 3,400 EOG artifact + 5,598 EMG artifact epochs.
    Source: https://github.com/ncclabsustech/EEGdenoiseNet

    Returns:
        Dict with 'clean_eeg', 'eog_artifacts', 'emg_artifacts' as numpy arrays.
    """
    data_path = Path(data_dir)
    result = {}

    # Try loading .npy files (preferred format)
    for key, filename in [
        ("clean_eeg", "EEG_all_epochs.npy"),
        ("eog_artifacts", "EOG_all_epochs.npy"),
        ("emg_artifacts", "EMG_all_epochs.npy"),
    ]:
        filepath = data_path / filename
        if filepath.exists():
            arr = np.load(str(filepath))
            result[key] = arr
            print(f"  Loaded {key}: {arr.shape}")
        else:
            # Try .mat format
            mat_path = data_path / filename.replace(".npy", ".mat")
            if mat_path.exists():
                from scipy.io import loadmat
                mat = loadmat(str(mat_path))
                # EEGdenoiseNet .mat files typically have a single variable
                for k, v in mat.items():
                    if not k.startswith("_") and isinstance(v, np.ndarray):
                        result[key] = v
                        print(f"  Loaded {key} from .mat: {v.shape}")
                        break

    if not result:
        raise FileNotFoundError(
            f"No EEGdenoiseNet data found in {data_dir}. "
            "Download from: https://gin.g-node.org/NCClab/EEGdenoiseNet"
        )

    return result


def generate_paired_denoise_data(
    n_pairs: int = 5000,
    fs: float = 256.0,
    epoch_sec: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic paired clean/noisy data for denoiser training.

    Uses the noise augmentation module to create realistic training pairs
    when EEGdenoiseNet is not available.

    Returns:
        (noisy, clean) arrays of shape (n_pairs, n_samples).
    """
    from simulation.eeg_simulator import simulate_eeg
    from processing.noise_augmentation import augment_eeg

    n_samples = int(epoch_sec * fs)
    clean_all, noisy_all = [], []

    states = ["rest", "focus", "meditation", "deep_sleep", "rem", "stress"]
    for i in range(n_pairs):
        state = states[i % len(states)]
        result = simulate_eeg(state=state, duration=epoch_sec, fs=fs)
        clean = np.array(result["signals"][0])[:n_samples]

        # Apply random noise augmentation
        difficulty = np.random.choice(["easy", "medium", "hard"], p=[0.2, 0.5, 0.3])
        noisy = augment_eeg(clean, fs=fs, difficulty=difficulty)

        clean_all.append(clean)
        noisy_all.append(noisy)

    return np.array(noisy_all), np.array(clean_all)


# ---------------------------------------------------------------------------
# SHHS — Sleep Heart Health Study (5,800+ subjects)
# ---------------------------------------------------------------------------

def load_shhs(
    data_dir: str = "data/shhs",
    n_subjects: int = 100,
    epoch_sec: float = 30.0,
    target_fs: float = 256.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load SHHS dataset for robust sleep staging.

    The Sleep Heart Health Study has 5,793 full-night PSGs from a
    multi-center cardiovascular cohort. Subjects aged 40+, many with
    sleep apnea — inherently noisy and clinically representative.

    Download: requires NSRR account at https://sleepdata.org/datasets/shhs

    Args:
        data_dir: Path to SHHS EDF files and annotation XMLs.
        n_subjects: Max number of subjects to load.
        epoch_sec: Epoch duration in seconds.
        target_fs: Target sampling frequency.

    Returns:
        (X, y) where X is (n_epochs, n_samples) and y is labels 0-4
        mapping to [Wake, N1, N2, N3, REM].
    """
    import mne
    import xml.etree.ElementTree as ET

    mne.set_log_level("WARNING")

    shhs_path = Path(data_dir)
    edf_dir = shhs_path / "edfs"
    annot_dir = shhs_path / "annotations-events-nsrr"

    if not edf_dir.exists():
        # Try flat structure
        edf_files = sorted(shhs_path.glob("shhs1-*.edf"))[:n_subjects]
        annot_files_map = {
            f.stem: shhs_path / f"{f.stem}-nsrr.xml"
            for f in edf_files
        }
    else:
        edf_files = sorted(edf_dir.glob("shhs1-*.edf"))[:n_subjects]
        annot_files_map = {
            f.stem: annot_dir / f"{f.stem}-nsrr.xml"
            for f in edf_files
        }

    if not edf_files:
        raise FileNotFoundError(
            f"No SHHS EDF files found in {data_dir}. "
            "Register at https://sleepdata.org/datasets/shhs to download."
        )

    # SHHS annotation stage mapping
    shhs_stage_map = {
        "0": 0,  # Wake
        "1": 1,  # N1
        "2": 2,  # N2
        "3": 3,  # N3
        "4": 3,  # N3 (S4 merged)
        "5": 4,  # REM
    }

    X_all, y_all = [], []

    for edf_file in edf_files:
        annot_file = annot_files_map.get(edf_file.stem)
        if annot_file is None or not annot_file.exists():
            continue

        try:
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
        except Exception as e:
            print(f"  Skipping {edf_file.name}: {e}")
            continue

        # Pick EEG channel (SHHS uses C4-A1 or C3-A2)
        eeg_ch = None
        for ch_name in ["EEG(sec)", "EEG", "C4-A1", "C3-A2", "EEG2"]:
            if ch_name in raw.ch_names:
                eeg_ch = ch_name
                break
        if eeg_ch is None:
            eeg_ch = raw.ch_names[0]
        raw.pick([eeg_ch])

        if raw.info["sfreq"] != target_fs:
            raw.resample(target_fs)

        data = raw.get_data()[0]

        # Parse NSRR XML annotations
        try:
            tree = ET.parse(str(annot_file))
            root = tree.getroot()
        except Exception:
            continue

        n_samples_epoch = int(epoch_sec * target_fs)

        for event in root.iter("ScoredEvent"):
            event_type = event.findtext("EventType", "")
            if "Stages" not in event_type:
                continue

            stage_str = event.findtext("EventConcept", "")
            # Extract stage number from strings like "Stage 2 sleep|2"
            stage_num = stage_str.split("|")[-1].strip() if "|" in stage_str else ""
            if stage_num not in shhs_stage_map:
                continue

            start_sec = float(event.findtext("Start", "0"))
            duration = float(event.findtext("Duration", "0"))

            stage_label = shhs_stage_map[stage_num]
            start_sample = int(start_sec * target_fs)

            # Extract 30-second epochs within this scored segment
            n_epochs_in_segment = int(duration / epoch_sec)
            for ep in range(n_epochs_in_segment):
                s = start_sample + ep * n_samples_epoch
                e = s + n_samples_epoch
                if e > len(data):
                    break
                X_all.append(data[s:e])
                y_all.append(stage_label)

        print(f"  {edf_file.name}: {len(y_all)} epochs total")

    if not X_all:
        raise RuntimeError("No epochs loaded from SHHS. Check data format.")

    return np.array(X_all), np.array(y_all)


# ---------------------------------------------------------------------------
# FACED — 123 subjects, 9 fine-grained emotions
# ---------------------------------------------------------------------------

def load_faced(
    data_dir: str = "data/faced",
    target_fs: float = 256.0,
    epoch_sec: float = 4.0,
    return_groups: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load FACED dataset for cross-subject emotion recognition.

    123 subjects, 30 EEG channels, 250 Hz, 9 emotion categories:
    amusement, inspiration, joy, tenderness, anger, fear, disgust, sadness, neutral.

    Source: https://doi.org/10.7303/syn50614194

    We map 9 categories → 6 classes for compatibility with our emotion classifier:
    - happy: amusement, joy
    - relaxed: tenderness, inspiration
    - angry: anger
    - fearful: fear, disgust
    - sad: sadness
    - focused: neutral

    Returns:
        (X, y) where X is feature vectors and y is labels 0-5.
    """
    faced_path = Path(data_dir)

    # FACED can be stored as .mat files or .npy files
    mat_files = sorted(faced_path.glob("sub_*.mat")) + sorted(faced_path.glob("*.mat"))
    npy_files = sorted(faced_path.glob("*.npy"))

    faced_emotion_map = {
        "amusement": 0, "joy": 0,          # happy
        "tenderness": 4, "inspiration": 4,  # relaxed
        "anger": 2,                          # angry
        "fear": 3, "disgust": 3,            # fearful
        "sadness": 1,                        # sad
        "neutral": 5,                        # focused
    }

    X_all, y_all, groups = [], [], []

    if npy_files:
        # Pre-processed numpy format
        for npy_file in npy_files:
            if "data" in npy_file.stem.lower() or "eeg" in npy_file.stem.lower():
                X_raw = np.load(str(npy_file))
                print(f"  Loaded {npy_file.name}: {X_raw.shape}")
            elif "label" in npy_file.stem.lower():
                y_raw = np.load(str(npy_file))
                print(f"  Loaded {npy_file.name}: {y_raw.shape}")

    elif mat_files:
        from scipy.io import loadmat

        for mat_file in mat_files:
            try:
                mat = loadmat(str(mat_file))
            except Exception as e:
                print(f"  Skipping {mat_file.name}: {e}")
                continue

            # Find EEG data and labels in the .mat file
            eeg_data = None
            labels = None
            for key, val in mat.items():
                if key.startswith("_"):
                    continue
                if isinstance(val, np.ndarray):
                    if "label" in key.lower() or "y" == key.lower():
                        labels = val.flatten()
                    elif val.ndim >= 2 and val.shape[-1] > 100:
                        eeg_data = val

            if eeg_data is None:
                continue

            n_samples_epoch = int(epoch_sec * target_fs)

            if labels is not None and eeg_data.ndim == 3:
                # Shape: (n_trials, n_channels, n_samples)
                    for trial_idx in range(eeg_data.shape[0]):
                        if trial_idx >= len(labels):
                            break
                    label = int(labels[trial_idx])
                    if label < 0 or label > 5:
                        continue

                    # Average channels and extract features
                    trial = eeg_data[trial_idx]
                    if trial.ndim == 2:
                        trial_avg = trial.mean(axis=0)
                    else:
                        trial_avg = trial

                    # Slide windows
                        for pos in range(0, len(trial_avg) - n_samples_epoch, n_samples_epoch // 2):
                            epoch = trial_avg[pos:pos + n_samples_epoch]
                            features = extract_features(preprocess(epoch, target_fs), target_fs)
                            X_all.append(list(features.values()))
                            y_all.append(label)
                            if return_groups:
                                groups.append(mat_file.stem)

            print(f"  {mat_file.name}: {len(y_all)} epochs total")

    if not X_all:
        raise FileNotFoundError(
            f"No FACED data found in {data_dir}. "
            "Download from: https://doi.org/10.7303/syn50614194"
        )

    if return_groups:
        return np.array(X_all), np.array(y_all), np.array(groups)
    return np.array(X_all), np.array(y_all)


# ---------------------------------------------------------------------------
# DREAM Database — 505 subjects, 2643 awakenings for dream detection
# ---------------------------------------------------------------------------

def load_dream_database(
    data_dir: str = "data/dream_database",
    target_fs: float = 256.0,
    pre_awakening_sec: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load DREAM Database for supervised dream detection.

    505 participants, 2,643 awakenings with dream/no-dream reports.
    Each sample is a pre-awakening EEG segment with a binary label.

    Source: https://bridges.monash.edu/projects/The_Dream_EEG_and_Mentation_DREAM_database/158987

    Returns:
        (X, y) where X is (n_epochs, n_samples) and y is binary:
        1 = dream reported, 0 = no dream reported.
    """
    import mne

    mne.set_log_level("WARNING")
    dream_path = Path(data_dir)

    X_all, y_all = [], []
    n_samples_epoch = int(pre_awakening_sec * target_fs)

    # The DREAM database has multiple sub-datasets
    # Try various common organization patterns
    for pattern in ["**/*awakening*.edf", "**/*.edf", "**/*.set", "**/*.fif"]:
        edf_files = sorted(dream_path.glob(pattern))
        if edf_files:
            break

    # Also look for pre-processed numpy format
    npy_data = dream_path / "dream_eeg.npy"
    npy_labels = dream_path / "dream_labels.npy"

    if npy_data.exists() and npy_labels.exists():
        X_raw = np.load(str(npy_data))
        y_raw = np.load(str(npy_labels))
        print(f"  Loaded preprocessed DREAM data: {X_raw.shape}, {y_raw.shape}")
        return X_raw, y_raw

    # Look for CSV/TSV metadata with dream reports
    meta_files = list(dream_path.glob("**/*metadata*.csv")) + \
                 list(dream_path.glob("**/*awakening*.tsv")) + \
                 list(dream_path.glob("**/*participants*.tsv"))

    dream_labels = {}
    if meta_files:
        import csv
        for meta_file in meta_files:
            delimiter = "\t" if meta_file.suffix == ".tsv" else ","
            with open(meta_file, "r") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    # Look for dream report columns
                    awake_id = row.get("awakening_id", row.get("id", ""))
                    dream_report = row.get("dream_report", row.get("report", ""))
                    has_dream = row.get("dream", row.get("has_dream", ""))

                    if has_dream:
                        dream_labels[awake_id] = int(has_dream) if has_dream.isdigit() else (1 if has_dream.lower() in ("yes", "true", "1") else 0)
                    elif dream_report:
                        dream_labels[awake_id] = 1 if len(dream_report.strip()) > 10 else 0

    if not edf_files and not dream_labels:
        raise FileNotFoundError(
            f"No DREAM database files found in {data_dir}. "
            "Download from: https://bridges.monash.edu/projects/"
            "The_Dream_EEG_and_Mentation_DREAM_database/158987"
        )

    for edf_file in edf_files:
        try:
            if edf_file.suffix == ".set":
                raw = mne.io.read_raw_eeglab(str(edf_file), preload=True, verbose=False)
            elif edf_file.suffix == ".fif":
                raw = mne.io.read_raw_fif(str(edf_file), preload=True, verbose=False)
            else:
                raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
        except Exception as e:
            print(f"  Skipping {edf_file.name}: {e}")
            continue

        # Pick EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(eeg_picks) == 0:
            continue
        raw.pick(eeg_picks[:1])  # Single channel for consistency

        if raw.info["sfreq"] != target_fs:
            raw.resample(target_fs)

        data = raw.get_data()[0]

        # Extract pre-awakening segment (last N seconds)
        if len(data) >= n_samples_epoch:
            epoch = data[-n_samples_epoch:]
            X_all.append(epoch)

            # Get label from metadata
            file_id = edf_file.stem
            label = dream_labels.get(file_id, 1)  # Default to dream if no metadata
            y_all.append(label)

        if len(X_all) % 100 == 0 and X_all:
            print(f"  Loaded {len(X_all)} awakenings...")

    if not X_all:
        raise RuntimeError("No epochs loaded from DREAM database.")

    print(f"  DREAM database: {len(X_all)} awakenings "
          f"({sum(y_all)} dream, {len(y_all) - sum(y_all)} no-dream)")

    return np.array(X_all), np.array(y_all)


# ---------------------------------------------------------------------------
# DREAMT — Wearable + PSG paired for consumer device validation
# ---------------------------------------------------------------------------

def load_dreamt(
    data_dir: str = "data/dreamt",
    epoch_sec: float = 30.0,
    target_fs: float = 256.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load DREAMT dataset — wearable signals alongside PSG.

    Paired Empatica E4 wearable + full PSG recordings for validating
    sleep staging models on consumer devices.

    Source: https://physionet.org/content/dreamt/2.0.0/

    Returns:
        (X, y) where X is (n_epochs, n_samples) and y is labels 0-4.
    """
    import mne

    mne.set_log_level("WARNING")
    dreamt_path = Path(data_dir)

    edf_files = sorted(dreamt_path.glob("**/*psg*.edf")) + \
                sorted(dreamt_path.glob("**/*.edf"))

    if not edf_files:
        raise FileNotFoundError(
            f"No DREAMT files found in {data_dir}. "
            "Download from: https://physionet.org/content/dreamt/2.0.0/"
        )

    # Reuse sleep staging labels
    stage_map = {
        "W": 0, "Wake": 0,
        "N1": 1, "1": 1,
        "N2": 2, "2": 2,
        "N3": 3, "3": 3,
        "R": 4, "REM": 4,
    }

    X_all, y_all = [], []
    n_samples_epoch = int(epoch_sec * target_fs)

    for edf_file in edf_files:
        try:
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
        except Exception:
            continue

        # Look for matching annotation file
        annot_candidates = [
            edf_file.with_suffix(".tsv"),
            edf_file.parent / (edf_file.stem + "_hypnogram.tsv"),
            edf_file.parent / (edf_file.stem + "_scoring.csv"),
        ]

        annot_file = None
        for cand in annot_candidates:
            if cand.exists():
                annot_file = cand
                break

        if annot_file is None:
            # Try MNE annotations
            try:
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                for desc, eid in event_id.items():
                    stage_key = desc.split()[-1] if " " in desc else desc
                    if stage_key in stage_map:
                        stage_events = events[events[:, 2] == eid]
                        for ev in stage_events:
                            s = ev[0]
                            e = s + n_samples_epoch
                            if e > raw.n_times:
                                continue
                            eeg_ch = raw.ch_names[0]
                            raw_pick = raw.copy().pick([eeg_ch])
                            if raw_pick.info["sfreq"] != target_fs:
                                raw_pick.resample(target_fs)
                            data = raw_pick.get_data()[0]
                            s_rs = int(s * target_fs / raw.info["sfreq"])
                            e_rs = s_rs + n_samples_epoch
                            if e_rs <= len(data):
                                X_all.append(data[s_rs:e_rs])
                                y_all.append(stage_map[stage_key])
            except Exception:
                pass
            continue

        print(f"  {edf_file.name}: loaded, {len(y_all)} epochs total")

    if not X_all:
        raise RuntimeError("No epochs loaded from DREAMT.")

    return np.array(X_all), np.array(y_all)


def list_available_datasets() -> dict:
    """List all available datasets and their status (downloaded or not).

    Returns:
        Dict mapping dataset name to status info.
    """
    datasets = {}

    # Sleep-EDF (always available via MNE auto-download)
    datasets["sleep_edf"] = {
        "name": "Sleep-EDF (PhysioNet)",
        "status": "available (auto-download via MNE)",
        "task": "5-class sleep staging",
        "loader": "load_sleep_edf()",
    }

    # DENS
    dens_path = Path("data/dens")
    dens_subjects = list(dens_path.glob("sub-*")) if dens_path.exists() else []
    dens_has_eeg = any(
        list((s / "eeg").glob("*.set")) for s in dens_subjects
    ) if dens_subjects else False
    datasets["dens"] = {
        "name": "DENS (OpenNeuro ds003751)",
        "status": f"{'ready' if dens_has_eeg else 'metadata only'} "
                  f"({len(dens_subjects)} subjects)",
        "task": "emotion recognition (128-ch EEG)",
        "loader": "load_dens()",
        "download": "download_dens_openneuro()",
    }

    # DEAP
    deap_path = Path("data/deap")
    deap_files = list(deap_path.glob("s*.dat")) if deap_path.exists() else []
    datasets["deap"] = {
        "name": "DEAP (Kaggle)",
        "status": f"{'ready' if deap_files else 'not downloaded'} "
                  f"({len(deap_files)}/32 files)",
        "task": "emotion recognition (32-ch EEG + peripheral)",
        "loader": "load_deap_or_simulate()",
        "download": "download_deap_kaggle()",
    }

    # PhysioNet Motor Imagery
    datasets["motor_imagery"] = {
        "name": "PhysioNet EEGMMIDB",
        "status": "available (auto-download via MNE)",
        "task": "motor imagery classification (64-ch EEG, 109 subjects)",
        "loader": "load_motor_imagery()",
    }

    # MOABB
    try:
        import moabb
        moabb_status = f"installed (v{moabb.__version__})"
    except ImportError:
        moabb_status = "not installed (pip install moabb)"
    datasets["moabb"] = {
        "name": "MOABB BCI Benchmarks",
        "status": moabb_status,
        "task": "MI, P300, SSVEP (dozens of datasets)",
        "loader": "load_moabb_dataset()",
    }

    # DREAMER
    dreamer_path = Path("data/DREAMER.mat")
    dreamer_exists = dreamer_path.exists() and dreamer_path.stat().st_size > 1_000_000
    datasets["dreamer"] = {
        "name": "DREAMER (Zenodo)",
        "status": "ready" if dreamer_exists else "not downloaded or corrupt",
        "task": "emotion recognition (14-ch EEG, 23 subjects, 18 film clips)",
        "loader": "load_dreamer()",
        "download": "curl -L -o data/DREAMER.mat https://zenodo.org/records/546113/files/DREAMER.mat",
    }

    # STEW
    stew_path = Path(os.path.expanduser('~/.cache/kagglehub/datasets/mitulahirwal/mental-cognitive-workload-eeg-data-stew-dataset/versions/1/dataset.mat'))
    datasets["stew"] = {
        "name": "STEW (Kaggle)",
        "status": "ready" if stew_path.exists() else "not downloaded",
        "task": "cognitive workload (14-ch EEG, 48 subjects, 3-class)",
        "loader": "load_stew()",
        "download": "kagglehub.dataset_download('mitulahirwal/mental-cognitive-workload-eeg-data-stew-dataset')",
    }

    # EEGdenoiseNet
    denoisenet_path = Path("data/eegdenoisenet")
    denoisenet_files = (
        list(denoisenet_path.glob("*.npy")) + list(denoisenet_path.glob("*.mat"))
        if denoisenet_path.exists() else []
    )
    datasets["eegdenoisenet"] = {
        "name": "EEGdenoiseNet (G-Node)",
        "status": f"{'ready' if denoisenet_files else 'not downloaded'} "
                  f"({len(denoisenet_files)} files)",
        "task": "denoising (paired clean/noisy EEG epochs)",
        "loader": "load_eegdenoisenet()",
        "download": "git clone https://gin.g-node.org/NCClab/EEGdenoiseNet data/eegdenoisenet",
    }

    # SHHS
    shhs_path = Path("data/shhs")
    shhs_files = (
        list(shhs_path.glob("**/*.edf"))
        if shhs_path.exists() else []
    )
    datasets["shhs"] = {
        "name": "SHHS (NSRR)",
        "status": f"{'ready' if shhs_files else 'not downloaded'} "
                  f"({len(shhs_files)} EDFs)",
        "task": "sleep staging (5,800 subjects, multi-center PSG)",
        "loader": "load_shhs()",
        "download": "Register at https://sleepdata.org/datasets/shhs",
    }

    # FACED
    faced_path = Path("data/faced")
    faced_files = (
        list(faced_path.glob("*.mat")) + list(faced_path.glob("*.npy"))
        if faced_path.exists() else []
    )
    datasets["faced"] = {
        "name": "FACED (Synapse)",
        "status": f"{'ready' if faced_files else 'not downloaded'} "
                  f"({len(faced_files)} files)",
        "task": "emotion recognition (123 subjects, 9 emotions, 30-ch EEG)",
        "loader": "load_faced()",
        "download": "Download from https://doi.org/10.7303/syn50614194",
    }

    # DREAM Database
    dream_path = Path("data/dream_database")
    dream_files = (
        list(dream_path.glob("**/*.edf")) + list(dream_path.glob("*.npy"))
        if dream_path.exists() else []
    )
    datasets["dream_database"] = {
        "name": "DREAM Database (Monash)",
        "status": f"{'ready' if dream_files else 'not downloaded'} "
                  f"({len(dream_files)} files)",
        "task": "dream detection (505 subjects, 2643 awakenings)",
        "loader": "load_dream_database()",
        "download": "Download from https://bridges.monash.edu/projects/The_Dream_EEG_and_Mentation_DREAM_database/158987",
    }

    # DREAMT
    dreamt_path = Path("data/dreamt")
    dreamt_files = (
        list(dreamt_path.glob("**/*.edf"))
        if dreamt_path.exists() else []
    )
    datasets["dreamt"] = {
        "name": "DREAMT (PhysioNet)",
        "status": f"{'ready' if dreamt_files else 'not downloaded'} "
                  f"({len(dreamt_files)} EDFs)",
        "task": "sleep staging from wearables (paired PSG + Empatica E4)",
        "loader": "load_dreamt()",
        "download": "Download from https://physionet.org/content/dreamt/2.0.0/",
    }

    return datasets


def load_deap_raw_4ch(
    data_dir: str = "data/deap",
    epoch_sec: float = 4.0,
    target_fs: float = 256.0,
    n_classes: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load DEAP as raw 4-channel waveform epochs for CNN-KAN training.

    Extracts Muse 2-equivalent channels (T7≈TP9, AF3≈AF7, AF4≈AF8, T8≈TP10)
    from DEAP's 32-channel data. Resamples 128→256Hz. Epochs into windows.

    Args:
        data_dir: Path to DEAP .dat files
        epoch_sec: Epoch length in seconds (default 4.0)
        target_fs: Target sampling rate (default 256.0 to match Muse 2)
        n_classes: 3 (pos/neu/neg) or 6 (happy/sad/angry/fear/surprise/neutral)

    Returns:
        X: (n_epochs, 4, n_samples) raw EEG waveforms
        y: (n_epochs,) integer labels
    """
    import pickle
    from scipy.signal import resample

    # DEAP channel indices for Muse 2-equivalent positions
    # Order: TP9(T7=7), AF7(AF3=1), AF8(AF4=17), TP10(T8=23)
    MUSE_INDICES = [7, 1, 17, 23]
    DEAP_FS = 128.0
    BASELINE_SAMPLES = int(3 * DEAP_FS)  # 3s pre-trial baseline

    deap_path = Path(data_dir)
    dat_files = sorted(deap_path.glob("s*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No DEAP .dat files in {data_dir}")

    samples_per_epoch = int(epoch_sec * target_fs)
    deap_samples_per_epoch = int(epoch_sec * DEAP_FS)

    X_all, y_all = [], []

    for dat_file in dat_files:
        try:
            with open(dat_file, "rb") as f:
                subject = pickle.load(f, encoding="latin1")
        except Exception:
            continue

        data = subject["data"]      # (40, 40, 8064)
        labels = subject["labels"]  # (40, 4) — valence, arousal, dominance, liking

        for trial_idx in range(data.shape[0]):
            valence = labels[trial_idx, 0]  # 1-9
            arousal = labels[trial_idx, 1]  # 1-9

            # Label mapping
            if n_classes == 3:
                if valence > 5.5:
                    label = 0  # positive
                elif valence < 4.5:
                    label = 2  # negative
                else:
                    label = 1  # neutral
            else:
                # 6-class using circumplex quadrants
                if valence > 5.5 and arousal > 5.0:
                    label = 0  # happy
                elif valence < 4.5 and arousal < 4.5:
                    label = 1  # sad
                elif valence < 4.5 and arousal > 5.5:
                    label = 2  # angry
                elif valence < 4.0 and arousal > 5.0:
                    label = 3  # fear
                elif arousal > 7.0:
                    label = 4  # surprise
                else:
                    label = 5  # neutral

            # Extract Muse-equivalent channels, skip 3s baseline
            trial_eeg = data[trial_idx, MUSE_INDICES, BASELINE_SAMPLES:]  # (4, ~7680)

            # Epoch into windows
            n_deap_samples = trial_eeg.shape[1]
            hop = deap_samples_per_epoch  # no overlap for training
            for start in range(0, n_deap_samples - deap_samples_per_epoch + 1, hop):
                chunk = trial_eeg[:, start:start + deap_samples_per_epoch]  # (4, deap_epoch)

                # Resample 128→256Hz
                resampled = resample(chunk, samples_per_epoch, axis=1)
                X_all.append(resampled.astype(np.float32))
                y_all.append(label)

    if not X_all:
        raise RuntimeError("No epochs extracted from DEAP")

    X = np.array(X_all)  # (n_epochs, 4, samples_per_epoch)
    y = np.array(y_all, dtype=np.int64)
    print(f"DEAP raw 4ch: {X.shape[0]} epochs, {len(dat_files)} subjects, "
          f"classes={dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y
