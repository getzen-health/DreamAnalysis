"""Train all new cognitive models: drowsiness, cognitive load, attention, stress,
lucid dream detection, and meditation depth.

Also retrains the flow state model with improved features.

Usage:
    python -m training.train_all_new_models
    python -m training.train_all_new_models --models drowsiness,attention
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

from processing.eeg_processor import extract_features, preprocess
from simulation.eeg_simulator import simulate_eeg
from processing.noise_augmentation import augment_eeg

MODEL_DIR = Path("models/saved")
REPORT_PATH = MODEL_DIR / "new_models_report.json"


def generate_state_data(states_config, n_per_class=500, fs=256.0, epoch_sec=10.0):
    """Generate labeled training data by simulating EEG states.

    Args:
        states_config: Dict mapping class_label -> list of simulation states
        n_per_class: Samples per class
        fs: Sampling frequency
        epoch_sec: Epoch duration

    Returns:
        (X, y, feature_names)
    """
    X, y = [], []
    feature_names = None

    for class_label, sim_states in states_config.items():
        for i in range(n_per_class):
            state = sim_states[i % len(sim_states)]
            result = simulate_eeg(state=state, duration=epoch_sec, fs=fs)
            eeg = np.array(result["signals"][0])

            # Add noise augmentation for robustness
            difficulty = np.random.choice(["easy", "medium", "hard"], p=[0.3, 0.5, 0.2])
            eeg = augment_eeg(eeg, fs=fs, difficulty=difficulty)

            features = extract_features(preprocess(eeg, fs), fs)
            if feature_names is None:
                feature_names = list(features.keys())
            X.append(list(features.values()))
            y.append(class_label)

    return np.array(X), np.array(y), feature_names


def train_model(X, y, feature_names, model_name, n_classes):
    """Train a LightGBM or RandomForest classifier with cross-validation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try LightGBM first
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=800,
            max_depth=8,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        model_type = "LightGBM"
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model_type = "RandomForest"

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

    # Train on full data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    # Save
    model_path = MODEL_DIR / f"{model_name}_model.pkl"
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        "scaler": scaler,
    }, model_path)

    result = {
        "model_name": model_name,
        "model_type": model_type,
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "test_accuracy": float(test_acc),
        "n_samples": len(y),
        "n_classes": n_classes,
        "model_size_mb": round(model_path.stat().st_size / 1e6, 1),
    }

    print(f"  {model_name}: CV={cv_scores.mean():.4f}+/-{cv_scores.std():.4f}, "
          f"Test={test_acc:.4f}, {model_type}")

    return result


def try_load_real_data(dataset_name):
    """Try to load real dataset if available."""
    if dataset_name == "mental_attention":
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from mega_trainer import load_mental_attention
            X, y, name = load_mental_attention()
            if len(X) > 0:
                return X, y, name
        except Exception:
            pass
    elif dataset_name == "stew":
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from mega_trainer import load_stew
            X, y, name = load_stew()
            if len(X) > 0:
                return X, y, name
        except Exception:
            pass
    return None, None, None


def train_drowsiness(n_samples=600):
    """Train drowsiness detector: alert / drowsy / sleepy."""
    print("\n=== Training Drowsiness Detector ===")

    # Try real data first (Mental Attention dataset)
    X, y, name = try_load_real_data("mental_attention")
    if X is not None and len(X) > 0:
        print(f"  Using real data: {name} ({len(X)} samples)")
        # Map: focused(0)→alert(0), unfocused(1)→drowsy(1), drowsy(2)→sleepy(2)
        dummy_feat = extract_features(preprocess(np.random.randn(2560), 256), 256)
        feature_names = list(dummy_feat.keys())

        # Re-extract features to match our pipeline format
        # Mental Attention uses multichannel features, we need single-channel
        X_feat, y_clean = [], []
        for i in range(len(X)):
            feat = X[i]
            if len(feat) >= len(feature_names):
                X_feat.append(feat[:len(feature_names)])
            else:
                X_feat.append(np.pad(feat, (0, len(feature_names) - len(feat))))
            y_clean.append(int(y[i]))
        X_real = np.array(X_feat)
        y_real = np.array(y_clean)
    else:
        X_real, y_real = None, None

    # Generate synthetic data
    states_config = {
        0: ["focus", "rest"],           # alert
        1: ["meditation", "rest"],      # drowsy (relaxed→drowsy transition)
        2: ["deep_sleep", "rem"],       # sleepy
    }
    X_syn, y_syn, feature_names = generate_state_data(states_config, n_per_class=n_samples)

    # Combine if real data available
    if X_real is not None:
        X = np.vstack([X_syn, X_real[:, :X_syn.shape[1]]]) if X_real.shape[1] >= X_syn.shape[1] else X_syn
        y = np.concatenate([y_syn, y_real]) if X_real.shape[1] >= X_syn.shape[1] else y_syn
    else:
        X, y = X_syn, y_syn

    return train_model(X, y, feature_names, "drowsiness", 3)


def train_cognitive_load(n_samples=600):
    """Train cognitive load estimator: low / moderate / high."""
    print("\n=== Training Cognitive Load Estimator ===")

    # Try STEW dataset
    X, y, name = try_load_real_data("stew")
    if X is not None and len(X) > 0:
        print(f"  Using real data: {name} ({len(X)} samples)")
        dummy_feat = extract_features(preprocess(np.random.randn(2560), 256), 256)
        feature_names = list(dummy_feat.keys())
        X_feat = []
        for i in range(len(X)):
            feat = X[i]
            if len(feat) >= len(feature_names):
                X_feat.append(feat[:len(feature_names)])
            else:
                X_feat.append(np.pad(feat, (0, len(feature_names) - len(feat))))
        X_real, y_real = np.array(X_feat), np.array(y, dtype=int)
    else:
        X_real, y_real = None, None

    states_config = {
        0: ["rest", "meditation"],      # low cognitive load
        1: ["focus", "rest"],           # moderate
        2: ["stress", "focus"],         # high cognitive load
    }
    X_syn, y_syn, feature_names = generate_state_data(states_config, n_per_class=n_samples)

    if X_real is not None:
        X = np.vstack([X_syn, X_real[:, :X_syn.shape[1]]]) if X_real.shape[1] >= X_syn.shape[1] else X_syn
        y = np.concatenate([y_syn, y_real]) if X_real.shape[1] >= X_syn.shape[1] else y_syn
    else:
        X, y = X_syn, y_syn

    return train_model(X, y, feature_names, "cognitive_load", 3)


def _load_deap_proxy(data_dir: str = "data/deap") -> tuple:
    """Load DEAP .dat files and extract stress + attention proxy labels.

    DEAP arousal/valence ratings are on a 1-9 scale.
    Stress proxy:  high arousal + low valence  → stressed
    Attention proxy: high arousal + high valence → hyperfocused/focused

    Returns:
        (X_stress, y_stress, X_attn, y_attn, feature_names) or all-None on failure.
    """
    import pickle
    from pathlib import Path

    deap_path = Path(data_dir)
    dat_files = sorted(deap_path.glob("s*.dat"))
    if not dat_files:
        return None, None, None, None, None

    print(f"  Loading {len(dat_files)} DEAP subjects for stress/attention proxy...")

    dummy_feat = extract_features(preprocess(np.random.randn(2560), 256.0), 256.0)
    feature_names = list(dummy_feat.keys())

    X_stress, y_stress = [], []
    X_attn, y_attn = [], []

    for dat_file in dat_files[:20]:  # limit to first 20 subjects for speed
        try:
            with open(dat_file, "rb") as f:
                subj = pickle.load(f, encoding="latin1")
            # data: (40 trials, 40 channels, 8064 samples)
            # labels: (40 trials, 4) — [valence, arousal, dominance, liking], 1–9 scale
            data = subj["data"]        # (40, 40, 8064)
            labels = subj["labels"]    # (40, 4)

            n_trials = data.shape[0]
            fs = 128.0   # DEAP sampling rate
            epoch_samples = int(fs * 10.0)   # 10-second epochs
            step = epoch_samples // 2         # 50% overlap

            for trial_idx in range(n_trials):
                valence = float(labels[trial_idx, 0])   # 1-9
                arousal = float(labels[trial_idx, 1])   # 1-9

                # Stress proxy labels (4-class)
                if arousal > 6.0 and valence < 4.5:
                    stress_label = 3   # high stress
                elif arousal > 5.0 and valence < 5.5:
                    stress_label = 2   # moderate
                elif arousal > 4.0:
                    stress_label = 1   # mild
                else:
                    stress_label = 0   # relaxed

                # Attention proxy labels (4-class)
                if arousal > 6.5 and valence > 6.0:
                    attn_label = 3    # hyperfocused
                elif arousal > 5.5 and valence > 4.5:
                    attn_label = 2    # focused
                elif arousal > 4.0:
                    attn_label = 1    # passive
                else:
                    attn_label = 0    # distracted

                # Use channel 0 (AF7-equivalent frontal channel in DEAP)
                trial_sig = data[trial_idx, 0, :]   # (8064,)

                pos = 0
                while pos + epoch_samples <= len(trial_sig):
                    epoch = trial_sig[pos:pos + epoch_samples]
                    feat = extract_features(preprocess(epoch, fs), fs)
                    fv = [feat.get(k, 0.0) for k in feature_names]
                    X_stress.append(fv)
                    y_stress.append(stress_label)
                    X_attn.append(fv)
                    y_attn.append(attn_label)
                    pos += step

        except Exception as e:
            print(f"    Skipping {dat_file.name}: {e}")
            continue

    if len(X_stress) == 0:
        return None, None, None, None, None

    print(f"  Loaded {len(X_stress)} DEAP samples for stress/attention proxy")
    return (
        np.array(X_stress), np.array(y_stress),
        np.array(X_attn), np.array(y_attn),
        feature_names,
    )


def train_attention(n_samples=500):
    """Train attention classifier: distracted / passive / focused / hyperfocused."""
    print("\n=== Training Attention Classifier ===")

    states_config = {
        0: ["rest", "deep_sleep"],      # distracted
        1: ["rest", "meditation"],      # passive
        2: ["focus", "rest"],           # focused
        3: ["focus", "stress"],         # hyperfocused
    }
    X_syn, y_syn, feature_names = generate_state_data(states_config, n_per_class=n_samples)

    # Try DEAP proxy
    _, _, X_deap, y_deap, deap_feat = _load_deap_proxy()
    if X_deap is not None:
        # Align feature count
        min_feat = min(X_syn.shape[1], X_deap.shape[1])
        X = np.vstack([X_syn[:, :min_feat], X_deap[:, :min_feat]])
        y = np.concatenate([y_syn, y_deap])
        feature_names = feature_names[:min_feat]
        print(f"  Combined synthetic ({len(y_syn)}) + DEAP proxy ({len(y_deap)}) = {len(y)} samples")
    else:
        X, y = X_syn, y_syn

    return train_model(X, y, feature_names, "attention", 4)


def train_stress(n_samples=500):
    """Train stress detector: relaxed / mild / moderate / high."""
    print("\n=== Training Stress Detector ===")

    states_config = {
        0: ["meditation", "rest"],      # relaxed
        1: ["rest", "focus"],           # mild
        2: ["stress", "focus"],         # moderate
        3: ["stress"],                  # high stress
    }
    X_syn, y_syn, feature_names = generate_state_data(states_config, n_per_class=n_samples)

    # Try DEAP proxy
    X_deap, y_deap, _, _, _ = _load_deap_proxy()
    if X_deap is not None:
        min_feat = min(X_syn.shape[1], X_deap.shape[1])
        X = np.vstack([X_syn[:, :min_feat], X_deap[:, :min_feat]])
        y = np.concatenate([y_syn, y_deap])
        feature_names = feature_names[:min_feat]
        print(f"  Combined synthetic ({len(y_syn)}) + DEAP proxy ({len(y_deap)}) = {len(y)} samples")
    else:
        X, y = X_syn, y_syn

    return train_model(X, y, feature_names, "stress", 4)


def train_lucid_dream(n_samples=500):
    """Train lucid dream detector: non_lucid / pre_lucid / lucid / controlled."""
    print("\n=== Training Lucid Dream Detector ===")

    # Lucid dreams are characterized by REM + gamma bursts + high beta
    states_config = {
        0: ["rem", "deep_sleep"],       # non_lucid (standard dreaming)
        1: ["rem", "rest"],             # pre_lucid (transitional)
        2: ["rem", "focus"],            # lucid (REM + alertness markers)
        3: ["focus", "stress"],         # controlled (high metacognition)
    }
    X, y, feature_names = generate_state_data(states_config, n_per_class=n_samples)
    return train_model(X, y, feature_names, "lucid_dream", 4)


def train_meditation(n_samples=800):
    """Train meditation depth classifier: relaxed / meditating / deep.

    Reduced from 5 classes to 3 to improve cross-subject accuracy.
    52% CV with 5 classes (5×500 synthetic) → target ~70%+ with 3 classes (3×800).

    Classes:
      0: relaxed     — eyes-closed rest, alpha dominant
      1: meditating  — sustained meditation, theta increasing
      2: deep        — deep theta dominance, beta suppressed
    """
    print("\n=== Training Meditation Classifier (3-class) ===")

    states_config = {
        0: ["rest"],                         # relaxed (eyes-closed baseline)
        1: ["meditation", "rest"],           # meditating (theta + alpha mix)
        2: ["meditation", "deep_sleep"],     # deep (theta + delta, low beta)
    }
    X, y, feature_names = generate_state_data(states_config, n_per_class=n_samples)
    return train_model(X, y, feature_names, "meditation", 3)


def retrain_flow_state(n_samples=800):
    """Retrain flow state with more data and noise augmentation."""
    print("\n=== Retraining Flow State Detector ===")

    states_config = {
        0: ["rest", "deep_sleep"],      # no_flow
        1: ["rest", "focus"],           # micro_flow
        2: ["focus", "meditation"],     # flow
        3: ["focus"],                   # deep_flow
    }
    X, y, feature_names = generate_state_data(states_config, n_per_class=n_samples)
    return train_model(X, y, feature_names, "flow_state", 4)


ALL_TRAINERS = {
    "drowsiness": train_drowsiness,
    "cognitive_load": train_cognitive_load,
    "attention": train_attention,
    "stress": train_stress,
    "lucid_dream": train_lucid_dream,
    "meditation": train_meditation,
    "flow_state": retrain_flow_state,
}


def main(models=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    trainers = {}
    if models:
        for m in models:
            if m in ALL_TRAINERS:
                trainers[m] = ALL_TRAINERS[m]
            else:
                print(f"Unknown model: {m}. Available: {list(ALL_TRAINERS.keys())}")
    else:
        trainers = ALL_TRAINERS

    print(f"Training {len(trainers)} models: {list(trainers.keys())}")

    results = []
    for name, trainer in trainers.items():
        try:
            result = trainer()
            results.append(result)
        except Exception as e:
            print(f"  ERROR training {name}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_sec": round(elapsed, 1),
        "models": results,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete — {len(results)} models in {elapsed:.1f}s")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['model_name']:20s} CV={r['cv_accuracy']:.4f} "
              f"Test={r['test_accuracy']:.4f} ({r['model_type']})")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all new cognitive models")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of models to train (default: all)"
    )
    args = parser.parse_args()

    model_list = args.models.split(",") if args.models else None
    main(model_list)
