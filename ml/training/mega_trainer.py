"""
Mega Trainer: Unified emotion classifier across ALL available datasets.
Combines 10+ EEG datasets (DEAP, GAMEEMO, EEG-ER, DENS, SEED, Brainwave,
Muse Mental State, Muse2 Motor Imagery, SEED-IV, DREAMER, STEW) into a single pipeline.

All datasets mapped to 3-class: 0=positive, 1=neutral, 2=negative
"""

import numpy as np
import pandas as pd
import json
import time
import warnings
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               ExtraTreesClassifier)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# ─── FEATURE EXTRACTION ─────────────────────────────────────────────
BANDS = {
    'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
    'beta': (12, 30), 'gamma': (30, 50)
}

def extract_enhanced_features(data, sfreq=128):
    """Extract 27 enhanced features from single-channel EEG."""
    features = []
    n = len(data)

    # Band powers via Welch
    freqs, psd = signal.welch(data, sfreq, nperseg=min(256, n))
    total_power = np.sum(psd) + 1e-10

    band_powers = {}
    for band, (lo, hi) in BANDS.items():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        bp = np.sum(psd[idx]) / total_power if len(idx) > 0 else 0
        band_powers[band] = bp
        features.append(bp)

    # Hjorth parameters
    diff1 = np.diff(data)
    diff2 = np.diff(diff1)
    var0 = np.var(data) + 1e-10
    var1 = np.var(diff1) + 1e-10
    var2 = np.var(diff2) + 1e-10
    features.append(np.sqrt(var0))  # Activity
    features.append(np.sqrt(var1 / var0))  # Mobility
    features.append(np.sqrt(var2 / var1) / (np.sqrt(var1 / var0) + 1e-10))  # Complexity

    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-10)
    psd_norm = psd_norm[psd_norm > 0]
    features.append(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))

    # Differential entropy (5 bands)
    for band, (lo, hi) in BANDS.items():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        bp = psd[idx]
        var = np.var(bp) + 1e-10
        features.append(0.5 * np.log(2 * np.pi * np.e * var))

    # Band ratios
    alpha = band_powers.get('alpha', 0) + 1e-10
    beta = band_powers.get('beta', 0) + 1e-10
    theta = band_powers.get('theta', 0) + 1e-10
    delta = band_powers.get('delta', 0) + 1e-10
    features.append(alpha / beta)
    features.append(theta / beta)
    features.append((alpha + theta) / (beta + delta))

    # Wavelet energy (DWT)
    try:
        import pywt
        coeffs = pywt.wavedec(data, 'db4', level=min(5, int(np.log2(n)) - 1))
        for c in coeffs[:5]:
            features.append(np.sum(c**2) / len(c))
        while len(features) < 22:
            features.append(0)
    except Exception:
        features.extend([0] * 5)

    # Peak frequency
    peak_idx = np.argmax(psd[1:]) + 1 if len(psd) > 1 else 0
    features.append(freqs[peak_idx] if peak_idx < len(freqs) else 0)

    # Additional ratios
    gamma = band_powers.get('gamma', 0) + 1e-10
    features.append(theta / alpha)
    features.append(beta / gamma)

    # Statistical moments
    features.append(float(skew(data)))
    features.append(float(kurtosis(data)))

    return np.array(features[:27], dtype=np.float64)


def extract_multichannel(data, sfreq=128, n_channels=None):
    """Extract multi-channel enhanced features (38 dims)."""
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_ch = data.shape[0]
    ch_features = []
    for ch in range(n_ch):
        ch_features.append(extract_enhanced_features(data[ch], sfreq))

    ch_features = np.array(ch_features)

    # Take mean across channels for 27 features
    mean_feat = np.mean(ch_features, axis=0)

    # Multi-channel stats: mean/std of band powers across channels
    bp = ch_features[:, :5]  # first 5 = band powers
    mean_bp = np.mean(bp, axis=0)  # 5
    std_bp = np.std(bp, axis=0)    # 5

    # Front/back ratio (if we have both)
    front_back_ratio = 1.0
    if n_ch >= 4:
        front_mean = np.mean(ch_features[:n_ch//2, :5])
        back_mean = np.mean(ch_features[n_ch//2:, :5]) + 1e-10
        front_back_ratio = front_mean / back_mean

    result = np.concatenate([mean_feat, mean_bp, std_bp, [front_back_ratio]])
    return result[:38]


# ─── DATASET LOADERS ─────────────────────────────────────────────────

def load_deap_enhanced():
    """DEAP dataset: 32 subjects, 40 trials, 32 EEG channels."""
    log("  Loading DEAP...")
    import pickle
    deap_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/data/deap')
    X, y = [], []

    for dat in sorted(deap_dir.glob('s*.dat')):
        with open(dat, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        data = d['data']    # (40, 40, 8064) - 40 trials, 40 channels, 8064 samples
        labels = d['labels'] # (40, 4) - valence, arousal, dominance, liking

        for trial in range(data.shape[0]):
            eeg = data[trial, :32, :]  # 32 EEG channels
            feat = extract_multichannel(eeg, sfreq=128)
            X.append(feat)

            # Map valence/arousal to 3 classes
            v, a = labels[trial, 0], labels[trial, 1]
            if v >= 6:
                y.append(0)  # positive
            elif v <= 4:
                y.append(2)  # negative
            else:
                y.append(1)  # neutral

    return np.array(X), np.array(y), "DEAP"


def load_gameemo_enhanced():
    """GAMEEMO: 28 subjects × 4 games, 50% overlapping 30s windows."""
    log("  Loading GAMEEMO...")
    base = Path('/Users/sravyalu/.cache/kagglehub/datasets/sigfest/database-for-emotion-recognition-system-gameemo/versions/1')

    GAME_LABELS = {"G1": 2, "G2": 2, "G3": 0, "G4": 1}
    SFREQ = 128
    WIN = 30 * SFREQ  # 30 seconds
    STEP = 15 * SFREQ  # 50% overlap

    X, y = [], []
    # Find all preprocessed CSV files (not Raw)
    csv_files = [f for f in base.rglob('*AllChannels.csv') if 'Raw' not in f.name]
    log(f"    Found {len(csv_files)} preprocessed CSV files")

    for f in csv_files:
        fname = f.name
        game = None
        for g in GAME_LABELS:
            if g in fname:
                game = g
                break
        if game is None:
            continue

        label = GAME_LABELS[game]
        try:
            # First row is header (channel names), skip it
            df = pd.read_csv(f, skiprows=1, header=None)
            # Drop any non-numeric columns
            df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
            if df.shape[1] < 14:
                continue
            data = df.iloc[:, :14].values.astype(float).T  # (14, samples)

            for start in range(0, data.shape[1] - WIN, STEP):
                segment = data[:, start:start+WIN]
                feat = extract_multichannel(segment, SFREQ)
                X.append(feat)
                y.append(label)
        except Exception:
            continue

    return np.array(X), np.array(y), "GAMEEMO"


def load_eeg_er_enhanced():
    """EEG Emotion Recognition: 100 CSV files, 50% overlapping windows."""
    log("  Loading EEG-ER...")
    base = Path('/Users/sravyalu/.cache/kagglehub/datasets/khan1803115/eeg-dataset-for-emotion-recognition/versions/1')

    GAME_LABELS = {"G1": 2, "G2": 2, "G3": 0, "G4": 1}
    SFREQ = 128
    WIN = 30 * SFREQ
    STEP = 15 * SFREQ

    X, y = [], []
    csv_files = sorted(base.rglob('*AllChannels.csv'))
    log(f"    Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        fname = csv_file.name
        game = None
        for g in ["G1", "G2", "G3", "G4"]:
            if g in fname:
                game = g
                break
        if game is None:
            continue

        label = GAME_LABELS[game]
        try:
            df = pd.read_csv(csv_file, skiprows=1, header=None)
            df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
            if df.shape[1] < 14:
                continue
            data = df.iloc[:, :14].values.astype(float).T

            for start in range(0, data.shape[1] - WIN, STEP):
                segment = data[:, start:start+WIN]
                feat = extract_multichannel(segment, SFREQ)
                X.append(feat)
                y.append(label)
        except Exception:
            continue

    return np.array(X), np.array(y), "EEG-ER"


def load_dens():
    """DENS dataset: OpenNeuro 128-ch EEG."""
    log("  Loading DENS...")
    import mne
    mne.set_log_level('ERROR')

    dens_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/data/dens')
    X, y = [], []

    for subj_dir in sorted(dens_dir.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith('sub-'):
            continue

        set_files = list(subj_dir.rglob('*.set'))
        if not set_files:
            continue

        try:
            raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)
            events = mne.events_from_annotations(raw, verbose=False)
            if len(events) < 2:
                continue

            events_arr, event_id = events
            sfreq = raw.info['sfreq']
            data_arr = raw.get_data()
            n_channels = min(data_arr.shape[0], 32)

            for i, ev in enumerate(events_arr):
                start = ev[0]
                end = events_arr[i+1][0] if i+1 < len(events_arr) else start + int(5 * sfreq)
                segment = data_arr[:n_channels, start:end]

                if segment.shape[1] < int(sfreq):
                    continue

                feat = extract_multichannel(segment, int(sfreq))
                X.append(feat)

                ev_code = ev[2]
                if ev_code % 3 == 0:
                    y.append(0)
                elif ev_code % 3 == 1:
                    y.append(1)
                else:
                    y.append(2)
        except Exception:
            continue

    if len(X) == 0:
        return np.array([]), np.array([]), "DENS"
    return np.array(X), np.array(y), "DENS"


def load_seed_enhanced():
    """SEED: 50K samples with DE features, enhanced region features."""
    log("  Loading SEED...")
    base = Path('/Users/sravyalu/.cache/kagglehub/datasets/daviderusso7/seed-dataset/versions/1')
    npz_files = sorted(base.rglob('*.npz'))

    if not npz_files:
        return np.array([]), np.array([]), "SEED"

    # SEED has 3 separate files: Dataset, Labels, Subjects
    features_raw, labels = None, None
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        arr = data['arr_0'] if 'arr_0' in data else data[list(data.keys())[0]]
        if arr.ndim == 3:  # (50910, 5, 62) = features
            features_raw = arr
        elif arr.ndim == 1 and arr.dtype in [np.int64, np.int32, np.float64]:
            if labels is None or 'label' in f.name.lower():
                labels = arr

    if features_raw is None or labels is None:
        log("    Could not find features/labels in SEED npz files")
        return np.array([]), np.array([]), "SEED"

    # Map SEED labels: 0=negative→2, 1=neutral→1, 2=positive→0
    # (Kaggle version uses 0,1,2 instead of original -1,0,1)
    label_map = {0: 2, 1: 1, 2: 0, -1: 2}
    y = np.array([label_map.get(int(l), 1) for l in labels])

    # Enhanced region features (43 dims)
    # Regions for 62-channel montage
    frontal = list(range(0, 15))
    temporal = list(range(15, 25))
    parietal = list(range(25, 40))
    occipital = list(range(40, 55))
    central = list(range(55, 62))
    left = list(range(0, 31))
    right = list(range(31, 62))

    X = []
    for i in range(len(features_raw)):
        feat = []
        de = features_raw[i]  # (5, 62)

        # Global: mean DE per band (5)
        feat.extend(np.mean(de, axis=1))
        # Global: std DE per band (5)
        feat.extend(np.std(de, axis=1))

        # Region means: frontal, temporal, parietal, occipital per band (20)
        for region in [frontal, temporal, parietal, occipital]:
            feat.extend(np.mean(de[:, region], axis=1))

        # Frontal-Occipital contrast per band (5)
        feat.extend(np.mean(de[:, frontal], axis=1) - np.mean(de[:, occipital], axis=1))

        # Left-Right asymmetry per band (5)
        feat.extend(np.mean(de[:, right], axis=1) - np.mean(de[:, left], axis=1))

        # Band ratios (3)
        global_de = np.mean(de, axis=1)
        alpha_de = global_de[2] + 1e-10
        beta_de = global_de[3] + 1e-10
        theta_de = global_de[1] + 1e-10
        feat.append(alpha_de / beta_de)
        feat.append(theta_de / beta_de)
        feat.append((alpha_de + theta_de) / (beta_de + global_de[0] + 1e-10))

        X.append(feat[:43])

    return np.array(X), y, "SEED"


def load_brainwave_emotions():
    """Brainwave Emotions: Muse pre-extracted features, PCA reduced."""
    log("  Loading Brainwave Emotions...")
    base = Path('/Users/sravyalu/.cache/kagglehub/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions/versions/1')
    csv_files = list(base.rglob('*.csv'))

    if not csv_files:
        return np.array([]), np.array([]), "Brainwave"

    df = pd.read_csv(csv_files[0])
    label_col = df.columns[-1]
    labels_raw = df[label_col]
    X_raw = df.drop(columns=[label_col]).values

    # Map to 3 classes
    unique_labels = labels_raw.unique()
    label_map = {}
    for lbl in unique_labels:
        lbl_lower = str(lbl).lower()
        if any(w in lbl_lower for w in ['positive', 'happy', 'joy']):
            label_map[lbl] = 0
        elif any(w in lbl_lower for w in ['negative', 'sad', 'anger', 'fear']):
            label_map[lbl] = 2
        else:
            label_map[lbl] = 1

    y = np.array([label_map.get(l, 1) for l in labels_raw])

    # PCA reduction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    n_components = min(80, X_raw.shape[1], X_raw.shape[0])
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X_scaled)
    var_explained = sum(pca.explained_variance_ratio_) * 100
    log(f"    PCA: {X_raw.shape[1]} → {n_components} ({var_explained:.1f}% var)")

    return X, y, "Brainwave"


def load_muse_mental_state():
    """Muse Mental State: 2479 samples, 989 pre-extracted features, PCA reduced."""
    log("  Loading Muse Mental State...")
    csv_path = Path('/Users/sravyalu/.cache/kagglehub/datasets/birdy654/eeg-brainwave-dataset-mental-state/versions/2/mental-state.csv')

    if not csv_path.exists():
        return np.array([]), np.array([]), "Muse-Mental"

    df = pd.read_csv(csv_path)
    label_col = 'Label'
    y_raw = df[label_col].values.astype(int)
    X_raw = df.drop(columns=[label_col]).values

    # Labels 0,1,2 → map to our scheme
    # Mental state dataset: 0=relaxed, 1=concentrating, 2=neutral (typical Muse classification)
    # Map: relaxed→positive(0), concentrating→neutral(1), neutral→neutral(1)
    # Actually, let's keep as: 0→positive(relaxed), 1→neutral(focused), 2→negative(stressed)
    label_map = {0: 0, 1: 1, 2: 2}  # Direct mapping
    y = np.array([label_map.get(l, 1) for l in y_raw])

    # PCA reduction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    n_components = min(80, X_raw.shape[1], X_raw.shape[0])
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X_scaled)
    var_explained = sum(pca.explained_variance_ratio_) * 100
    log(f"    PCA: {X_raw.shape[1]} → {n_components} ({var_explained:.1f}% var)")

    return X, y, "Muse-Mental"


def load_muse2_motor_imagery():
    """Muse2 Motor Imagery: Band powers + Mellow/Concentration-based labels."""
    log("  Loading Muse2 Motor Imagery...")
    base = Path('/Users/sravyalu/.cache/kagglehub/datasets/muhammadatefelkaffas/eeg-muse2-motor-imagery-brain-electrical-activity/versions/1')

    band_cols = [
        'Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
        'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
        'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
        'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10',
        'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10',
    ]

    X, y = [], []
    WIN_SIZE = 256  # ~1 second windows at ~256 Hz Muse sampling

    csv_files = sorted(base.glob('*.csv'))
    for csv_file in csv_files[:15]:  # Use first 15 sessions
        try:
            df = pd.read_csv(csv_file, low_memory=False)

            # Filter for valid headband data
            if 'HeadBandOn' in df.columns:
                df = df[df['HeadBandOn'] == 1]

            # Drop NaN rows in band power columns
            available_cols = [c for c in band_cols if c in df.columns]
            if len(available_cols) < 16:
                continue

            df = df.dropna(subset=available_cols + ['Mellow', 'Concentration'])

            if len(df) < WIN_SIZE:
                continue

            band_data = df[available_cols].values  # (samples, 20)
            mellow = df['Mellow'].values
            concentration = df['Concentration'].values

            # Window and extract features
            n_windows = len(df) // WIN_SIZE
            for w in range(min(n_windows, 200)):  # Cap at 200 windows per session
                start = w * WIN_SIZE
                end = start + WIN_SIZE

                window_bands = band_data[start:end]  # (256, 20)
                window_mellow = np.mean(mellow[start:end])
                window_conc = np.mean(concentration[start:end])

                # Features: mean/std/skew/kurtosis of each band power (20 * 4 = 80)
                feat = []
                for ch in range(window_bands.shape[1]):
                    col_data = window_bands[:, ch]
                    feat.extend([
                        np.mean(col_data),
                        np.std(col_data),
                        float(skew(col_data)),
                        float(kurtosis(col_data))
                    ])

                # Add cross-channel features
                # Mean band powers per band (5)
                for b in range(5):
                    feat.append(np.mean(window_bands[:, b*4:(b+1)*4]))

                # L-R asymmetry per band (5): (AF8+TP10) - (AF7+TP9)
                for b in range(5):
                    left_idx = [b*4, b*4+1]   # TP9, AF7
                    right_idx = [b*4+2, b*4+3] # AF8, TP10
                    l_mean = np.mean(window_bands[:, left_idx])
                    r_mean = np.mean(window_bands[:, right_idx])
                    feat.append(r_mean - l_mean)

                X.append(feat)

                # Label from Mellow/Concentration
                if window_mellow > 60:
                    y.append(0)  # positive/relaxed
                elif window_conc > 60:
                    y.append(1)  # neutral/focused
                else:
                    y.append(2)  # negative/stressed

        except Exception as e:
            log(f"    Error: {csv_file.name}: {e}")
            continue

    if len(X) == 0:
        return np.array([]), np.array([]), "Muse2-MI"

    X = np.array(X)
    y = np.array(y)

    # Clean NaN/Inf
    X = np.nan_to_num(np.array(X), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y)

    # PCA if too many features
    if X.shape[1] > 80:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=80)
        X = pca.fit_transform(X_scaled)
        var_explained = sum(pca.explained_variance_ratio_) * 100
        log(f"    PCA: {X.shape[1]} → 80 ({var_explained:.1f}% var)")

    return X, y, "Muse2-MI"


def load_confused_student():
    """Confused Student EEG: NeuroSky single-channel, confused/not-confused labels."""
    log("  Loading Confused Student EEG...")
    csv_path = Path('/Users/sravyalu/.cache/kagglehub/datasets/wanghaohan/confused-eeg/versions/6/EEG_data.csv')

    if not csv_path.exists():
        return np.array([]), np.array([]), "Confused-Student"

    df = pd.read_csv(csv_path)
    # Features: Attention, Mediation, Delta, Theta, Alpha1, Alpha2, Beta1, Beta2, Gamma1, Gamma2
    feat_cols = ['Attention', 'Mediation', 'Delta', 'Theta', 'Alpha1', 'Alpha2',
                 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
    X_raw = df[feat_cols].values.astype(float)

    # Labels: confused(1) → negative(2), not-confused(0) → neutral(1)
    y_raw = df['user-definedlabeln'].values
    label_map = {0: 1, 1: 2}  # not-confused→neutral, confused→negative
    y = np.array([label_map.get(int(l), 1) for l in y_raw if not np.isnan(l)])
    X_raw = X_raw[:len(y)]

    # Remove NaN rows
    valid = ~np.isnan(X_raw).any(axis=1) & ~np.isnan(y)
    X_raw = X_raw[valid]
    y = y[valid]

    return X_raw, y, "Confused-Student"


def load_mental_attention():
    """Mental Attention State: 25-channel EMOTIV, focused/unfocused/drowsy."""
    log("  Loading Mental Attention State...")
    base = Path('/Users/sravyalu/.cache/kagglehub/datasets/inancigdem/eeg-data-for-mental-attention-state-detection/versions/1/EEG Data/EEG Data')

    if not base.exists():
        return np.array([]), np.array([]), "Mental-Attention"

    from scipy.io import loadmat

    SFREQ = 128
    WIN = 4 * SFREQ  # 4 second windows
    STEP = 2 * SFREQ  # 50% overlap

    X, y = [], []
    mat_files = sorted(base.glob('*.mat'))
    log(f"    Found {len(mat_files)} recordings")

    for mat_file in mat_files:
        try:
            mat = loadmat(str(mat_file))
            o = mat['o'][0, 0]
            data = o['data']  # (samples, 25 channels)
            marker = o['marker'].flatten()  # 0=focused, 1=unfocused, 2=drowsy

            n_channels = min(data.shape[1], 14)  # Use first 14 channels

            for start in range(0, len(data) - WIN, STEP):
                segment = data[start:start+WIN, :n_channels].T  # (channels, WIN)
                window_marker = marker[start:start+WIN]

                # Majority vote for label in window
                unique, counts = np.unique(window_marker, return_counts=True)
                label_raw = unique[np.argmax(counts)]

                # Map: 0=focused→neutral(1), 1=unfocused→neutral(1), 2=drowsy→negative(2)
                # Actually: focused→positive(0), unfocused→neutral(1), drowsy→negative(2)
                if label_raw == 0:
                    label = 0  # focused → positive
                elif label_raw == 2:
                    label = 2  # drowsy → negative
                else:
                    label = 1  # unfocused → neutral

                feat = extract_multichannel(segment, SFREQ)
                X.append(feat)
                y.append(label)
        except Exception:
            continue

    if len(X) == 0:
        return np.array([]), np.array([]), "Mental-Attention"
    return np.array(X), np.array(y), "Mental-Attention"


def load_muse_subconscious():
    """Muse Subconscious Decisions: 20 subjects, Muse S, band powers + Mellow/Concentration.
    Reads directly from zip file to avoid extracting 4GB+ of CSVs."""
    log("  Loading Muse Subconscious Decisions...")
    zip_path = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/data/Muse_Subconscious.zip')

    if not zip_path.exists():
        log("    Not found, skipping")
        return np.array([]), np.array([]), "Muse-Subconscious"

    import zipfile

    # Band power columns for 4 Muse channels
    band_cols = [f'{band}_{ch}' for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
                 for ch in ['TP9', 'AF7', 'AF8', 'TP10']]

    X, y = [], []
    n_subjects = 0

    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_files = [f for f in zf.namelist() if f.endswith('.csv') and '/Muse/' in f]
        log(f"    Found {len(csv_files)} recordings")

        for csv_name in csv_files:
            try:
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)

                # Get band power columns that exist
                feat_cols = [c for c in band_cols if c in df.columns]
                if len(feat_cols) < 10:
                    continue

                # Get Mellow/Concentration for labeling
                if 'Mellow' not in df.columns or 'Concentration' not in df.columns:
                    continue

                data = df[feat_cols].values
                mellow = pd.to_numeric(df['Mellow'], errors='coerce').values
                conc = pd.to_numeric(df['Concentration'], errors='coerce').values

                # Window the data: 4s windows at ~256Hz
                WIN = 4 * 256
                STEP = 2 * 256
                for start in range(0, len(data) - WIN, STEP):
                    window = data[start:start+WIN]
                    w_mellow = mellow[start:start+WIN]
                    w_conc = conc[start:start+WIN]

                    # Get mean mellow/concentration (ignore NaN)
                    m_val = np.nanmean(w_mellow)
                    c_val = np.nanmean(w_conc)

                    if np.isnan(m_val) and np.isnan(c_val):
                        continue

                    m_val = 0 if np.isnan(m_val) else m_val
                    c_val = 0 if np.isnan(c_val) else c_val

                    # Label based on Mellow/Concentration
                    # High mellow → positive(0), High concentration → neutral(1), else → negative(2)
                    if m_val > 0.6:
                        label = 0
                    elif c_val > 0.6:
                        label = 1
                    else:
                        label = 2

                    # Compute window statistics as features
                    feat = []
                    for col_idx in range(window.shape[1]):
                        col = window[:, col_idx]
                        col = col[np.isfinite(col)]
                        if len(col) < 10:
                            feat.extend([0, 0, 0, 0])
                            continue
                        feat.extend([np.mean(col), np.std(col), np.median(col),
                                     np.percentile(col, 75) - np.percentile(col, 25)])
                    X.append(feat)
                    y.append(label)

                n_subjects += 1
            except Exception:
                continue

    if len(X) == 0:
        return np.array([]), np.array([]), "Muse-Subconscious"

    X = np.array(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA to 80 features
    if X.shape[1] > 80:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_comp = min(80, X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X = pca.fit_transform(X_scaled)
        var_explained = sum(pca.explained_variance_ratio_) * 100
        log(f"    PCA: {X.shape[1]} → {n_comp} ({var_explained:.1f}% var)")

    log(f"    Muse-Subconscious: {len(X)} samples, {X.shape[1]} features ({n_subjects} subjects)")
    return X, np.array(y), "Muse-Subconscious"


def load_emokey_moments():
    """EmoKey Moments: 45 subjects, Muse S, 4 emotions + neutral baselines.
    Preprocessed clean EEG at 128Hz with band powers for 4 Muse channels."""
    log("  Loading EmoKey Moments...")
    base = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/data/emokey/EmoKey Moments EEG Dataset (EKM-ED)/muse_wearable_data/preprocessed/clean-signals/0.0078125S')

    if not base.exists():
        log("    Not found, skipping")
        return np.array([]), np.array([]), "EmoKey"

    # Emotion → 3-class mapping
    # HAPPINESS → positive(0), NEUTRAL_* → neutral(1), SADNESS/ANGER/FEAR → negative(2)
    emotion_map = {
        'HAPPINESS': 0,
        'NEUTRAL_HAPPINESS': 1,
        'NEUTRAL_SADNESS': 1,
        'NEUTRAL_ANGER': 1,
        'NEUTRAL_FEAR': 1,
        'SADNESS': 2,
        'ANGER': 2,
        'FEAR': 2,
    }

    # Band power columns for Muse channels (20 features)
    band_cols = [f'{band}_{ch}' for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
                 for ch in ['TP9', 'AF7', 'AF8', 'TP10']]
    # My* computed band powers (20 more features)
    my_band_cols = [f'My{band}_{ch}' for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
                    for ch in ['TP9', 'TP10', 'AF7', 'AF8']]

    SFREQ = 128
    WIN = 4 * SFREQ  # 4-second windows
    STEP = 2 * SFREQ  # 50% overlap

    X, y = [], []
    subjects = sorted([d for d in base.iterdir() if d.is_dir()])
    log(f"    Found {len(subjects)} subjects")

    for subj_dir in subjects:
        for csv_file in subj_dir.glob('*.csv'):
            emotion_name = csv_file.stem
            if emotion_name not in emotion_map:
                continue
            label = emotion_map[emotion_name]

            try:
                df = pd.read_csv(csv_file)

                # Use band power columns (prefer My* computed ones for consistency)
                feat_cols = [c for c in my_band_cols if c in df.columns]
                if len(feat_cols) < 10:
                    feat_cols = [c for c in band_cols if c in df.columns]
                if len(feat_cols) < 10:
                    continue

                data = df[feat_cols].values

                # Window the data
                for start in range(0, len(data) - WIN, STEP):
                    window = data[start:start+WIN]
                    # Compute stats per feature: mean, std, median, skew
                    feat = []
                    for col_idx in range(window.shape[1]):
                        col = window[:, col_idx]
                        col = col[np.isfinite(col)]
                        if len(col) < 10:
                            feat.extend([0, 0, 0, 0])
                            continue
                        feat.extend([
                            np.mean(col),
                            np.std(col),
                            np.median(col),
                            float(np.mean(col**3) - 3*np.mean(col)*np.var(col) - np.mean(col)**3) / (np.std(col)**3 + 1e-10),  # skewness
                        ])
                    X.append(feat)
                    y.append(label)
            except Exception:
                continue

    if len(X) == 0:
        return np.array([]), np.array([]), "EmoKey"

    X = np.array(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA if needed (80 features is a good target)
    if X.shape[1] > 80:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n_comp = min(80, X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X = pca.fit_transform(X_scaled)
        var_explained = sum(pca.explained_variance_ratio_) * 100
        log(f"    PCA: {X.shape[1]} → {n_comp} ({var_explained:.1f}% var)")

    log(f"    EmoKey: {len(X)} samples, {X.shape[1]} features")
    return X, np.array(y), "EmoKey"


def load_seed_iv():
    """SEED-IV: 4-class emotion dataset with DE features in .mat files."""
    log("  Loading SEED-IV...")
    base = Path('/Users/sravyalu/.cache/kagglehub/datasets/phhasian0710/seed-iv/versions/1/seed_iv/eeg_feature_smooth')

    if not base.exists():
        log("    Not found, skipping")
        return np.array([]), np.array([]), "SEED-IV"

    from scipy.io import loadmat

    # Session labels (from ReadMe.txt)
    session_labels = {
        1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
    }

    # Labels: 0=neutral, 1=sad, 2=fear, 3=happy → map to our 3-class
    # happy→positive(0), neutral→neutral(1), sad→negative(2), fear→negative(2)
    label_map = {0: 1, 1: 2, 2: 2, 3: 0}

    # Region indices for 62-channel montage
    frontal = list(range(0, 15))
    left = list(range(0, 31))
    right = list(range(31, 62))

    X, y = [], []
    sessions = sorted([d for d in base.iterdir() if d.is_dir()])
    log(f"    Found {len(sessions)} sessions")

    for session_dir in sessions:
        session_num = int(session_dir.name)
        labels = session_labels.get(session_num, [])
        if not labels:
            continue

        mat_files = sorted(session_dir.glob('*.mat'))
        for mat_file in mat_files:
            try:
                mat = loadmat(str(mat_file))
                # Extract de_LDS features (24 trials per file)
                for trial_idx in range(1, 25):
                    key = f'de_LDS{trial_idx}'
                    if key not in mat:
                        continue

                    de_data = mat[key]  # (62 channels, N windows, 5 bands)
                    trial_label = labels[trial_idx - 1]
                    mapped_label = label_map.get(trial_label, 1)

                    # Extract features from each time window
                    n_windows = de_data.shape[1]
                    for w in range(n_windows):
                        de = de_data[:, w, :]  # (62, 5)
                        feat = []

                        # Global: mean DE per band (5)
                        feat.extend(np.mean(de, axis=0))
                        # Global: std DE per band (5)
                        feat.extend(np.std(de, axis=0))

                        # Frontal mean (5)
                        feat.extend(np.mean(de[frontal], axis=0))

                        # Left-Right asymmetry (5)
                        feat.extend(np.mean(de[right], axis=0) - np.mean(de[left], axis=0))

                        # Band ratios (3)
                        global_de = np.mean(de, axis=0)
                        alpha, beta, theta = global_de[2]+1e-10, global_de[3]+1e-10, global_de[1]+1e-10
                        feat.append(alpha / beta)
                        feat.append(theta / beta)
                        feat.append((alpha + theta) / (beta + global_de[0] + 1e-10))

                        X.append(feat[:23])
                        y.append(mapped_label)
            except Exception:
                continue

    if len(X) == 0:
        return np.array([]), np.array([]), "SEED-IV"

    X, y = np.array(X), np.array(y)
    log(f"    SEED-IV: {len(X)} samples, {X.shape[1]} features")
    return X, y, "SEED-IV"


def load_dreamer():
    """DREAMER: 23 subjects, 14 EEG channels @ 128Hz, 18 film clips.
    Valence/arousal/dominance labels (1-5). Stored as MATLAB v7.3 (HDF5).
    Download: https://zenodo.org/records/546113/files/DREAMER.mat
    """
    log("  Loading DREAMER...")
    import h5py

    mat_path = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/data/DREAMER.mat')
    if not mat_path.exists():
        log("    DREAMER.mat not found, skipping")
        return np.array([]), np.array([]), "DREAMER"

    SFREQ = 128
    WIN = 4 * SFREQ    # 4-second windows
    STEP = 2 * SFREQ   # 50% overlap (2-second step)
    N_CHANNELS = 14

    X, y = [], []

    try:
        with h5py.File(str(mat_path), 'r') as f:
            # Navigate HDF5 structure: DREAMER -> Data
            dreamer_ref = f['DREAMER']
            data_ref = dreamer_ref['Data']

            n_subjects = data_ref.shape[1]
            log(f"    Found {n_subjects} subjects")

            for subj_idx in range(n_subjects):
                subj_ref = data_ref[0, subj_idx]
                subj = f[subj_ref]

                # Get EEG data and labels
                eeg_ref = subj['EEG']
                stim_data = eeg_ref['stimuli']
                baseline_data = eeg_ref['baseline']

                # Get valence scores
                valence_data = subj['ScoreValence']

                n_trials = stim_data.shape[1]

                for trial_idx in range(n_trials):
                    # Get trial EEG: dereference to get actual data
                    trial_ref = stim_data[0, trial_idx]
                    trial_eeg = np.array(f[trial_ref])  # (samples, channels)

                    # Get valence label
                    val_ref = valence_data[0, trial_idx]
                    valence = float(np.array(f[val_ref]).flat[0])

                    # Map valence (1-5) to 3-class
                    if valence >= 3.5:
                        label = 0   # positive
                    elif valence <= 2.5:
                        label = 2   # negative
                    else:
                        label = 1   # neutral

                    # Ensure correct shape (samples, channels) -> (channels, samples)
                    if trial_eeg.shape[0] == N_CHANNELS:
                        eeg = trial_eeg  # already (channels, samples)
                    elif trial_eeg.shape[1] == N_CHANNELS:
                        eeg = trial_eeg.T  # transpose to (channels, samples)
                    else:
                        # Try to use available channels
                        if trial_eeg.shape[0] < trial_eeg.shape[1]:
                            eeg = trial_eeg[:N_CHANNELS, :]
                        else:
                            eeg = trial_eeg[:, :N_CHANNELS].T

                    # Window into segments
                    n_samples = eeg.shape[1]
                    for start in range(0, n_samples - WIN, STEP):
                        segment = eeg[:, start:start + WIN]
                        feat = extract_multichannel(segment, SFREQ)
                        X.append(feat)
                        y.append(label)

    except Exception as e:
        log(f"    Error loading DREAMER: {e}")
        return np.array([]), np.array([]), "DREAMER"

    if len(X) == 0:
        return np.array([]), np.array([]), "DREAMER"

    X, y = np.array(X), np.array(y)
    log(f"    DREAMER: {len(X)} samples, {X.shape[1]} features")
    return X, y, "DREAMER"


def load_stew():
    """STEW: 48 subjects, 14 EEG channels (Emotiv EPOC) @ 128Hz.
    3-class cognitive workload: 0=low, 1=medium, 2=high.
    Downloaded via Kaggle: mitulahirwal/mental-cognitive-workload-eeg-data-stew-dataset
    """
    log("  Loading STEW...")
    from scipy.io import loadmat

    base = Path('/Users/sravyalu/.cache/kagglehub/datasets/mitulahirwal/mental-cognitive-workload-eeg-data-stew-dataset/versions/1')
    dataset_path = base / 'dataset.mat'
    labels_path = base / 'class_012.mat'

    if not dataset_path.exists():
        log("    STEW dataset not found, skipping")
        return np.array([]), np.array([]), "STEW"

    SFREQ = 128
    WIN = 4 * SFREQ    # 4-second windows
    STEP = 2 * SFREQ   # 50% overlap

    X, y = [], []

    try:
        data_mat = loadmat(str(dataset_path))
        labels_mat = loadmat(str(labels_path))

        # dataset.mat: shape (14, 19200, 45) = (channels, samples, segments)
        # class_012.mat: shape (45, 1) with values 0, 1, 2
        eeg_data = None
        for key in data_mat:
            if not key.startswith('_'):
                eeg_data = data_mat[key]
                break

        labels = None
        for key in labels_mat:
            if not key.startswith('_'):
                labels = labels_mat[key].flatten()
                break

        if eeg_data is None or labels is None:
            log("    Could not find data/labels in STEW .mat files")
            return np.array([]), np.array([]), "STEW"

        n_channels, n_samples, n_segments = eeg_data.shape
        log(f"    STEW shape: {eeg_data.shape}, {n_segments} segments, {n_channels} channels")

        # Map workload classes to emotion classes:
        # low workload (0) → positive/relaxed (0)
        # medium workload (1) → neutral (1)
        # high workload (2) → negative/stressed (2)
        for seg_idx in range(n_segments):
            segment = eeg_data[:, :, seg_idx]  # (14, 19200)
            label = int(labels[seg_idx])

            # Window into 4-second segments with 50% overlap
            for start in range(0, n_samples - WIN, STEP):
                window = segment[:, start:start + WIN]
                feat = extract_multichannel(window, SFREQ)
                X.append(feat)
                y.append(label)

    except Exception as e:
        log(f"    Error loading STEW: {e}")
        return np.array([]), np.array([]), "STEW"

    if len(X) == 0:
        return np.array([]), np.array([]), "STEW"

    X, y = np.array(X), np.array(y)
    log(f"    STEW: {len(X)} samples, {X.shape[1]} features")
    return X, y, "STEW"


# ─── NOISE AUGMENTATION ──────────────────────────────────────────────

def augment_dataset_features(X, y, n_augmented=2, difficulty='medium'):
    """Create noise-augmented copies of feature vectors.

    Since we extract features AFTER noise, we simulate the effect of
    noise on features by adding calibrated perturbations that match
    what real dry-electrode noise would produce in feature space.

    Args:
        X: Feature array (n_samples, n_features)
        y: Labels array
        n_augmented: Number of augmented copies per sample
        difficulty: 'easy', 'medium', 'hard'

    Returns:
        X_aug, y_aug with original + augmented samples
    """
    noise_scales = {'easy': 0.05, 'medium': 0.15, 'hard': 0.30}
    scale = noise_scales.get(difficulty, 0.15)

    X_aug = [X.copy()]
    y_aug = [y.copy()]

    for _ in range(n_augmented):
        # Feature-space noise (simulates sensor noise effect on extracted features)
        noise = np.random.randn(*X.shape) * scale * (np.std(X, axis=0, keepdims=True) + 1e-10)
        X_noisy = X + noise

        # Randomly zero out some features (simulates channel dropout)
        dropout_mask = np.random.random(X.shape) > 0.05
        X_noisy *= dropout_mask

        # Randomly scale features (simulates amplitude variation between sessions)
        amp_scale = np.random.uniform(0.8, 1.2, size=(1, X.shape[1]))
        X_noisy *= amp_scale

        X_aug.append(X_noisy)
        y_aug.append(y.copy())

    X_combined = np.vstack(X_aug)
    y_combined = np.concatenate(y_aug)

    # Shuffle
    idx = np.random.permutation(len(X_combined))
    return X_combined[idx], y_combined[idx]


# ─── EVALUATION ──────────────────────────────────────────────────────

def evaluate_models(X, y, dataset_name, n_features_target=None):
    """Train and evaluate multiple models with 5-fold CV."""
    if len(X) == 0:
        log(f"  {dataset_name}: No data, skipping")
        return {}

    n_classes = len(np.unique(y))
    if n_classes < 2:
        log(f"\n--- {dataset_name} ({len(X)} samples) ---")
        log(f"    Only {n_classes} class found, skipping (need >= 2)")
        return {}

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA if feature dims differ (for combining later)
    if n_features_target and X_scaled.shape[1] != n_features_target:
        n_comp = min(n_features_target, X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X_scaled = pca.fit_transform(X_scaled)
        # Pad if needed
        if X_scaled.shape[1] < n_features_target:
            X_scaled = np.hstack([X_scaled, np.zeros((X_scaled.shape[0], n_features_target - X_scaled.shape[1]))])

    # For large datasets, reduce GBM estimators or skip it (GBM isn't parallelizable)
    n_samples = len(X)
    if n_samples > 20000:
        models = {
            'RF': RandomForestClassifier(n_estimators=500, max_depth=15,
                                          class_weight='balanced', random_state=42, n_jobs=-1),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=500, max_depth=15,
                                                class_weight='balanced', random_state=42, n_jobs=-1),
            'GBM-lite': GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                     learning_rate=0.15, random_state=42,
                                                     subsample=0.5),
        }
    else:
        models = {
            'GBM': GradientBoostingClassifier(n_estimators=300, max_depth=6,
                                               learning_rate=0.1, random_state=42),
            'RF': RandomForestClassifier(n_estimators=300, max_depth=12,
                                          class_weight='balanced', random_state=42, n_jobs=-1),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=300, max_depth=12,
                                                class_weight='balanced', random_state=42, n_jobs=-1),
        }

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    log(f"\n--- {dataset_name} ({len(X)} samples, {X_scaled.shape[1]} features) ---")
    label_dist = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    log(f"    Labels: {label_dist}")

    for model_name, model in models.items():
        accs, f1s = [], []
        t0 = time.time()

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average='weighted'))

        elapsed = time.time() - t0
        acc_mean, acc_std = np.mean(accs), np.std(accs)
        f1_mean, f1_std = np.mean(f1s), np.std(f1s)

        log(f"    {model_name:12s}: Acc={acc_mean:.4f}±{acc_std:.4f}  F1={f1_mean:.4f}±{f1_std:.4f}  ({elapsed:.1f}s)")

        results[model_name] = {
            'accuracy': float(acc_mean),
            'accuracy_std': float(acc_std),
            'f1': float(f1_mean),
            'f1_std': float(f1_std),
            'time': float(elapsed)
        }

    return results


def combine_datasets(datasets, target_features=80):
    """Combine multiple datasets with different feature dimensions using PCA alignment."""
    all_X, all_y, names = [], [], []

    for X, y, name in datasets:
        if len(X) == 0:
            continue
        if len(np.unique(y)) < 2:
            log(f"    Skipping {name} from combination (single class)")
            continue

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize within dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA to target dimensions
        n_comp = min(target_features, X_scaled.shape[1], X_scaled.shape[0] - 1)
        if n_comp < target_features and X_scaled.shape[1] >= target_features:
            n_comp = target_features

        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_scaled)

        # Pad to target if needed
        if X_pca.shape[1] < target_features:
            X_pca = np.hstack([X_pca, np.zeros((X_pca.shape[0], target_features - X_pca.shape[1]))])

        all_X.append(X_pca)
        all_y.append(y)
        names.append(f"{name}({len(X)})")

    if not all_X:
        return np.array([]), np.array([]), "Combined(empty)"

    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    combined_name = f"Combined({'+'.join(names)})"

    return X_combined, y_combined, combined_name


# ─── MAIN ────────────────────────────────────────────────────────────

def main():
    log("=" * 80)
    log("  MEGA TRAINER: All Datasets Combined")
    log("=" * 80)
    t_start = time.time()

    # Load all datasets
    log("\n[LOADING DATASETS]")
    datasets = {}
    raw_eeg_datasets = []  # Same feature space (38-dim multichannel)
    pre_extracted_datasets = []  # Different feature spaces

    # 1. Raw EEG datasets (38-dim multichannel features)
    for loader in [load_deap_enhanced, load_gameemo_enhanced, load_eeg_er_enhanced, load_dens, load_dreamer, load_stew]:
        try:
            X, y, name = loader()
            if len(X) > 0:
                datasets[name] = (X, y, name)
                raw_eeg_datasets.append((X, y, name))
                log(f"    {name}: {len(X)} samples, {X.shape[1]} features")
        except Exception as e:
            log(f"    Error loading: {e}")

    # 2. Pre-extracted feature datasets
    for loader in [load_seed_enhanced, load_brainwave_emotions, load_muse_mental_state,
                   load_muse2_motor_imagery, load_confused_student, load_mental_attention,
                   load_muse_subconscious, load_emokey_moments, load_seed_iv]:
        try:
            X, y, name = loader()
            if len(X) > 0:
                datasets[name] = (X, y, name)
                pre_extracted_datasets.append((X, y, name))
                log(f"    {name}: {len(X)} samples, {X.shape[1]} features")
        except Exception as e:
            log(f"    Error loading: {e}")

    total_samples = sum(len(d[1]) for d in datasets.values())
    log(f"\n  Total: {len(datasets)} datasets, {total_samples:,} samples")

    # ─── EVALUATE INDIVIDUAL DATASETS ──────────────────────────────
    log("\n" + "=" * 80)
    log("  INDIVIDUAL DATASET EVALUATION")
    log("=" * 80)

    all_results = {}
    for name, (X, y, _) in datasets.items():
        results = evaluate_models(X, y, name)
        if results:
            best_model = max(results, key=lambda m: results[m]['f1'])
            all_results[name] = {
                'samples': len(X),
                'features': X.shape[1],
                'best_model': best_model,
                'best_acc': results[best_model]['accuracy'],
                'best_f1': results[best_model]['f1'],
                'all_models': results
            }

    # ─── COMBINED: RAW EEG ONLY ───────────────────────────────────
    if len(raw_eeg_datasets) > 1:
        log("\n" + "=" * 80)
        log("  COMBINED: RAW EEG DATASETS (same 38-dim features)")
        log("=" * 80)

        all_X = np.vstack([d[0] for d in raw_eeg_datasets])
        all_y = np.concatenate([d[1] for d in raw_eeg_datasets])
        names = '+'.join([f"{d[2]}({len(d[0])})" for d in raw_eeg_datasets])

        log(f"\n  Combined: {len(all_X)} samples, {all_X.shape[1]} features")
        log(f"  Sources: {names}")

        results = evaluate_models(all_X, all_y, "Combined-Raw-EEG")
        if results:
            best_model = max(results, key=lambda m: results[m]['f1'])
            all_results['Combined-Raw-EEG'] = {
                'samples': len(all_X),
                'features': all_X.shape[1],
                'sources': names,
                'best_model': best_model,
                'best_acc': results[best_model]['accuracy'],
                'best_f1': results[best_model]['f1'],
                'all_models': results
            }

    # ─── COMBINED: ALL DATASETS (PCA ALIGNED) ─────────────────────
    log("\n" + "=" * 80)
    log("  COMBINED: ALL DATASETS (PCA-aligned to 80 dims)")
    log("=" * 80)

    all_data = list(datasets.values())
    X_mega, y_mega, mega_name = combine_datasets(all_data, target_features=80)

    if len(X_mega) > 0:
        log(f"\n  MEGA Combined: {len(X_mega)} samples, {X_mega.shape[1]} features")
        results = evaluate_models(X_mega, y_mega, "MEGA-Combined")
        if results:
            best_model = max(results, key=lambda m: results[m]['f1'])
            all_results['MEGA-Combined'] = {
                'samples': len(X_mega),
                'features': X_mega.shape[1],
                'best_model': best_model,
                'best_acc': results[best_model]['accuracy'],
                'best_f1': results[best_model]['f1'],
                'all_models': results
            }

    # ─── COMBINED: WITHOUT SEED (more balanced) ───────────────────
    log("\n" + "=" * 80)
    log("  COMBINED: ALL EXCEPT SEED (balanced dataset sizes)")
    log("=" * 80)

    no_seed = [(X, y, n) for X, y, n in all_data if 'SEED' not in n]
    X_noseed, y_noseed, noseed_name = combine_datasets(no_seed, target_features=80)

    if len(X_noseed) > 0:
        log(f"\n  Combined (no SEED): {len(X_noseed)} samples")
        results = evaluate_models(X_noseed, y_noseed, "Combined-NoSEED")
        if results:
            best_model = max(results, key=lambda m: results[m]['f1'])
            all_results['Combined-NoSEED'] = {
                'samples': len(X_noseed),
                'features': X_noseed.shape[1],
                'best_model': best_model,
                'best_acc': results[best_model]['accuracy'],
                'best_f1': results[best_model]['f1'],
                'all_models': results
            }

    # ─── COMBINED: MUSE-ONLY DEVICES ──────────────────────────────
    muse_datasets = [(X, y, n) for X, y, n in all_data if any(k in n for k in ['Brainwave', 'Muse'])]
    if len(muse_datasets) > 1:
        log("\n" + "=" * 80)
        log("  COMBINED: MUSE DEVICES ONLY")
        log("=" * 80)

        X_muse, y_muse, muse_name = combine_datasets(muse_datasets, target_features=60)
        if len(X_muse) > 0:
            log(f"\n  Muse Combined: {len(X_muse)} samples")
            results = evaluate_models(X_muse, y_muse, "Muse-Combined")
            if results:
                best_model = max(results, key=lambda m: results[m]['f1'])
                all_results['Muse-Combined'] = {
                    'samples': len(X_muse),
                    'features': X_muse.shape[1],
                    'best_model': best_model,
                    'best_acc': results[best_model]['accuracy'],
                    'best_f1': results[best_model]['f1'],
                    'all_models': results
                }

    # ─── FINAL SUMMARY ────────────────────────────────────────────
    elapsed = time.time() - t_start

    log("\n" + "=" * 80)
    log("  FINAL SUMMARY")
    log("=" * 80)
    log(f"\n{'Dataset':<25} {'Samples':>8} {'Features':>8} {'Best Model':>12} {'Accuracy':>10} {'F1':>10}")
    log("-" * 80)

    for name, r in sorted(all_results.items(), key=lambda x: x[1]['best_f1'], reverse=True):
        log(f"{name:<25} {r['samples']:>8} {r['features']:>8} {r['best_model']:>12} {r['best_acc']:>9.4f} {r['best_f1']:>9.4f}")

    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    benchmarks_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/benchmarks')
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    with open(benchmarks_dir / 'mega_trainer_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    log(f"\nResults saved to {benchmarks_dir / 'mega_trainer_results.json'}")


if __name__ == '__main__':
    main()
