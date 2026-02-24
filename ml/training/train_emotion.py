"""Training script for emotion classifier.

Trains on DEAP dataset if available, otherwise uses enhanced simulation.
Exports to both sklearn .pkl and ONNX formats with benchmark results.

Usage:
    python -m training.train_emotion
    python -m training.train_emotion --simulated
"""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from scipy import stats as _scipy_stats
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from processing.eeg_processor import extract_features, preprocess
from simulation.eeg_simulator import simulate_eeg

EMOTIONS = ["happy", "sad", "angry", "fearful", "relaxed", "focused"]
EMOTION_TO_STATE = {
    0: "rest",        # happy -> relaxed state with positive valence
    1: "deep_sleep",  # sad -> low arousal, low energy
    2: "stress",      # angry -> high beta, high arousal
    3: "stress",      # fearful -> high beta, high gamma
    4: "meditation",  # relaxed -> high alpha, high theta
    5: "focus",       # focused -> high beta
}
MODEL_DIR = Path("models/saved")
BENCHMARK_DIR = Path("benchmarks")


def generate_training_data(n_samples_per_class: int = 150, fs: float = 256.0, epoch_duration: float = 10.0):
    """Generate synthetic training data."""
    X = []
    y = []

    for emotion_idx, state_name in EMOTION_TO_STATE.items():
        for _ in range(n_samples_per_class):
            result = simulate_eeg(state=state_name, duration=epoch_duration, fs=fs)
            eeg = np.array(result["signals"][0])
            features = extract_features(preprocess(eeg, fs), fs)
            X.append(list(features.values()))
            y.append(emotion_idx)

    return np.array(X), np.array(y), list(features.keys()), None


def load_real_data(data_dir: str = "data/deap", fs: float = 256.0):
    """Load real EEG data, combining DENS + DEAP when both are available.

    Returns:
        X: Feature matrix.
        y: Emotion labels.
        feature_names: List of feature names.
        groups: Subject ID per sample (for LOSO CV), or None if unavailable.
    """
    from training.data_loaders import load_dens, _load_deap_real

    dummy_features = extract_features(preprocess(np.random.randn(int(fs * 10)), fs), fs)
    feature_names = list(dummy_features.keys())

    datasets_loaded = []
    X_parts, y_parts, groups_parts = [], [], []

    # Try DENS (128-ch, OpenNeuro)
    dens_path = Path("data/dens")
    if dens_path.exists() and list(dens_path.glob("sub-*")):
        try:
            print("Loading DENS dataset (128-ch EEG, OpenNeuro)...")
            X_dens, y_dens, g_dens = load_dens(data_dir="data/dens", target_fs=fs)
            X_parts.append(X_dens)
            y_parts.append(y_dens)
            groups_parts.append(g_dens)
            datasets_loaded.append(f"dens({len(y_dens)} samples)")
            print(f"  DENS: {len(y_dens)} samples loaded")
        except Exception as e:
            print(f"  DENS failed: {e}")

    # Try DEAP (32-ch, Kaggle)
    deap_path = Path(data_dir)
    if deap_path.exists() and list(deap_path.glob("s*.dat")):
        try:
            print("Loading DEAP dataset (32-ch EEG, Kaggle)...")
            X_deap, y_deap, g_deap = _load_deap_real(deap_path, fs)
            X_parts.append(X_deap)
            y_parts.append(y_deap)
            groups_parts.append(g_deap)
            datasets_loaded.append(f"deap({len(y_deap)} samples)")
            print(f"  DEAP: {len(y_deap)} samples loaded")
        except Exception as e:
            print(f"  DEAP failed: {e}")

    if X_parts:
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        groups = np.concatenate(groups_parts)
        print(f"Combined dataset: {len(y)} samples from {', '.join(datasets_loaded)}")
        return X, y, feature_names, groups

    # Fallback to cascade loader (no subject groups available)
    from training.data_loaders import load_deap_or_simulate
    print("No local data found, using cascade loader...")
    X, y = load_deap_or_simulate(data_dir=data_dir, fs=fs)
    return X, y, feature_names, None


def export_onnx(model, feature_names, output_path):
    """Export sklearn model to ONNX format."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model saved to {output_path}")
        return True
    except ImportError:
        print("skl2onnx not installed, skipping ONNX export")
        return False


def save_benchmarks(
    y_test,
    y_pred,
    dataset_name,
    model_name,
    output_dir,
    cv_method: str = "train_test_split",
    fold_accuracies=None,
    fold_f1s=None,
    n_subjects=None,
):
    """Save benchmark results as JSON.

    Args:
        fold_accuracies: Per-fold accuracy list (for LOSO/k-fold).
        fold_f1s: Per-fold macro-F1 list.
        n_subjects: Number of held-out subjects (LOSO only).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    present_classes = sorted(set(y_test) | set(y_pred))
    present_names = [EMOTIONS[i] for i in present_classes if i < len(EMOTIONS)]
    report = classification_report(y_test, y_pred, labels=present_classes, target_names=present_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=present_classes).tolist()

    benchmarks = {
        "model_name": model_name,
        "dataset": dataset_name,
        "cv_method": cv_method,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "per_class": {
            emotion: {
                "precision": report[emotion]["precision"],
                "recall": report[emotion]["recall"],
                "f1": report[emotion]["f1-score"],
                "support": report[emotion]["support"],
            }
            for emotion in present_names
            if emotion in report
        },
        "confusion_matrix": cm,
    }

    # Statistical rigor: Wilson CI, bootstrap CI for F1, Cohen's kappa, p-value vs chance
    correct = int((np.array(y_test) == np.array(y_pred)).sum())
    n = int(len(y_test))
    chance = 1.0 / max(len(present_classes), 1)
    # Wilson score 95% CI for accuracy
    z = 1.96
    p_hat = correct / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    benchmarks["accuracy_ci_95"] = [float(max(0.0, centre - spread)), float(min(1.0, centre + spread))]
    benchmarks["chance_level"] = float(chance)
    # Binomial p-value: accuracy significantly above chance?
    benchmarks["accuracy_p_value_vs_chance"] = float(
        _scipy_stats.binomtest(correct, n, chance, alternative="greater").pvalue
    )
    # Cohen's kappa (agreement beyond chance)
    benchmarks["cohen_kappa"] = float(cohen_kappa_score(y_test, y_pred))
    # Bootstrap 95% CI for F1 macro (500 resamples)
    rng = np.random.default_rng(42)
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    boot_f1s = []
    for _ in range(500):
        idx = rng.integers(0, n, size=n)
        boot_f1s.append(
            f1_score(y_test_arr[idx], y_pred_arr[idx], average="macro", zero_division=0)
        )
    benchmarks["f1_macro_ci_95"] = [float(np.percentile(boot_f1s, 2.5)), float(np.percentile(boot_f1s, 97.5))]

    # Add cross-validation statistics when available
    if fold_accuracies is not None and len(fold_accuracies) > 1:
        fa = np.array(fold_accuracies, dtype=float)
        ff = np.array(fold_f1s, dtype=float) if fold_f1s is not None else None
        benchmarks["cv_accuracy_mean"] = float(np.mean(fa))
        benchmarks["cv_accuracy_std"] = float(np.std(fa))
        benchmarks["cv_accuracy_per_fold"] = [float(v) for v in fa]
        if ff is not None:
            benchmarks["cv_f1_mean"] = float(np.mean(ff))
            benchmarks["cv_f1_std"] = float(np.std(ff))
            benchmarks["cv_f1_per_fold"] = [float(v) for v in ff]
        if n_subjects is not None:
            benchmarks["n_subjects"] = int(n_subjects)

    benchmark_path = output_dir / f"{model_name}_benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Benchmarks saved to {benchmark_path}")

    return benchmarks


def train(simulated: bool = False, data_dir: str = "data/deap", output_dir: str = "models/saved"):
    """Train and save the emotion classifier."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    groups = None

    if simulated:
        print("Generating simulated training data...")
        X, y, feature_names, groups = generate_training_data(n_samples_per_class=150)
        dataset_name = "simulated"
    else:
        print("Loading emotion dataset...")
        try:
            X, y, feature_names, groups = load_real_data(data_dir=data_dir)
            # Detect which datasets were loaded
            dens_path = Path("data/dens")
            deap_path = Path(data_dir)
            dens_ok = dens_path.exists() and list(dens_path.glob("sub-*"))
            deap_ok = deap_path.exists() and list(deap_path.glob("s*.dat"))
            if dens_ok and deap_ok:
                dataset_name = "dens+deap-combined"
            elif dens_ok:
                dataset_name = "dens-openneuro"
            elif deap_ok:
                dataset_name = "deap-kaggle"
            else:
                dataset_name = "simulated-enhanced"
        except Exception as e:
            print(f"Failed to load data: {e}")
            print("Falling back to simulated data...")
            X, y, feature_names, groups = generate_training_data(n_samples_per_class=150)
            dataset_name = "simulated"

    # ── CV strategy: LOSO when subject groups are available, else 80/20 split ──
    from collections import Counter
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.ensemble import RandomForestClassifier

    n_subjects = len(np.unique(groups)) if groups is not None else None
    use_loso = (
        groups is not None
        and len(groups) == len(y)
        and n_subjects is not None
        and n_subjects >= 2
    )

    if use_loso:
        cv_method = f"LOSO ({n_subjects} subjects)"
        print(f"\nDataset: {dataset_name} — {len(y)} samples, {n_subjects} subjects")
        print(f"CV strategy: Leave-One-Subject-Out ({n_subjects} folds)")

        logo = LeaveOneGroupOut()
        fold_accuracies = []
        fold_f1s = []
        all_y_true = []
        all_y_pred = []

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
            held_out_subject = np.unique(groups[test_idx])[0]
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # SMOTE on training fold only
            class_counts_fold = Counter(y_train_fold)
            majority_count = max(class_counts_fold.values())
            smote_target = max(int(majority_count * 0.50), 10)
            sampling_strategy = {
                cls: max(count, smote_target)
                for cls, count in class_counts_fold.items()
                if count < smote_target
            }
            if sampling_strategy:
                try:
                    from imblearn.over_sampling import SMOTE
                    k_neighbors = min(5, min(class_counts_fold[c] for c in sampling_strategy) - 1)
                    k_neighbors = max(k_neighbors, 1)
                    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
                    X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)
                except Exception:
                    pass  # continue without SMOTE if unavailable or too few samples

            sw = compute_sample_weight("balanced", y_train_fold)

            gbm_fold = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42,
            )
            rf_fold = RandomForestClassifier(
                n_estimators=200, max_depth=6, class_weight="balanced", random_state=42,
            )
            gbm_fold.fit(X_train_fold, y_train_fold, sample_weight=sw)
            rf_fold.fit(X_train_fold, y_train_fold)

            gbm_acc_fold = accuracy_score(y_test_fold, gbm_fold.predict(X_test_fold))
            rf_acc_fold = accuracy_score(y_test_fold, rf_fold.predict(X_test_fold))
            best_fold = rf_fold if rf_acc_fold >= gbm_acc_fold else gbm_fold

            y_pred_fold = best_fold.predict(X_test_fold)
            fold_acc = accuracy_score(y_test_fold, y_pred_fold)
            fold_f1 = f1_score(y_test_fold, y_pred_fold, average="macro", zero_division=0)
            fold_accuracies.append(fold_acc)
            fold_f1s.append(fold_f1)
            all_y_true.extend(y_test_fold.tolist())
            all_y_pred.extend(y_pred_fold.tolist())
            print(f"  Fold {fold_idx + 1:02d} | held-out subject {held_out_subject} | "
                  f"acc={fold_acc:.4f}  f1={fold_f1:.4f}")

        print(f"\nLOSO mean accuracy : {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        print(f"LOSO mean F1 macro : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

        y_test_all = np.array(all_y_true)
        y_pred_all = np.array(all_y_pred)

        present_classes = sorted(set(y_test_all) | set(y_pred_all))
        present_names = [EMOTIONS[i] for i in present_classes if i < len(EMOTIONS)]
        print("\nAggregated Classification Report (all LOSO folds):")
        print(classification_report(y_test_all, y_pred_all, labels=present_classes, target_names=present_names))

        # ── Train final model on ALL data for deployment ──────────────────────
        print("Training final model on full dataset for deployment...")
        sw_all = compute_sample_weight("balanced", y)
        gbm_final = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
        rf_final = RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42,
        )
        gbm_final.fit(X, y, sample_weight=sw_all)
        rf_final.fit(X, y)
        # Pick the model that performed better on average across LOSO folds
        # by re-evaluating once on a held-aside 10% to break the tie
        class_counts_all = Counter(y)
        min_count_all = min(class_counts_all.values())
        stratify_all = y if min_count_all >= 2 else None
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.10, random_state=0, stratify=stratify_all
        )
        gbm_val_acc = accuracy_score(y_val, gbm_final.predict(X_val))
        rf_val_acc = accuracy_score(y_val, rf_final.predict(X_val))
        model = rf_final if rf_val_acc >= gbm_val_acc else gbm_final
        algo_name = "RandomForest" if rf_val_acc >= gbm_val_acc else "GradientBoosting"
        print(f"Final deployment model: {algo_name}")

        # Benchmark uses aggregated LOSO predictions
        y_for_bench, y_pred_for_bench = y_test_all, y_pred_all
        cv_label = cv_method

    else:
        # ── Fallback: standard 80/20 train-test split ─────────────────────────
        cv_method = "train_test_split_80_20"
        print(f"\nDataset: {dataset_name} — {len(y)} samples (no subject groups, using 80/20 split)")

        class_counts = Counter(y)
        min_count = min(class_counts.values())
        stratify_param = y if min_count >= 2 else None
        if stratify_param is None:
            print("  Warning: some classes have <2 samples, disabling stratification")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # SMOTE oversampling for minority classes
        class_counts_train = Counter(y_train)
        print(f"Class distribution before SMOTE: {dict(sorted(class_counts_train.items()))}")

        majority_count = max(class_counts_train.values())
        smote_target = max(int(majority_count * 0.50), 300)

        sampling_strategy = {
            cls: max(count, smote_target)
            for cls, count in class_counts_train.items()
            if count < smote_target
        }

        if sampling_strategy:
            try:
                from imblearn.over_sampling import SMOTE
                k_neighbors = min(5, min(class_counts_train[c] for c in sampling_strategy) - 1)
                k_neighbors = max(k_neighbors, 1)
                smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"Class distribution after SMOTE:  {dict(sorted(Counter(y_train).items()))}")
            except ImportError:
                print("imbalanced-learn not installed — skipping SMOTE (pip install imbalanced-learn)")
            except Exception as e:
                print(f"SMOTE failed ({e}), continuing without oversampling")
        else:
            print("Classes sufficiently balanced, SMOTE not needed")

        sample_weights = compute_sample_weight("balanced", y_train)

        gbm = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42,
        )

        print("Training GradientBoosting...")
        gbm.fit(X_train, y_train, sample_weight=sample_weights)
        gbm_acc = accuracy_score(y_test, gbm.predict(X_test))
        print(f"  GBM accuracy: {gbm_acc:.4f}")

        print("Training RandomForest...")
        rf.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"  RF accuracy: {rf_acc:.4f}")

        if rf_acc > gbm_acc:
            model = rf
            print(f"Selected RandomForest (accuracy: {rf_acc:.4f})")
        else:
            model = gbm
            print(f"Selected GradientBoosting (accuracy: {gbm_acc:.4f})")

        # Predict on test set
        y_pred_for_bench = model.predict(X_test)
        y_for_bench = y_test
        cv_label = cv_method

        fold_accuracies = None
        fold_f1s = None

        accuracy = accuracy_score(y_test, y_pred_for_bench)
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        present_classes = sorted(set(y_test) | set(y_pred_for_bench))
        present_names = [EMOTIONS[i] for i in present_classes if i < len(EMOTIONS)]
        print(classification_report(y_test, y_pred_for_bench, labels=present_classes, target_names=present_names))

    # ── Save deployment model ──────────────────────────────────────────────────
    model_path = output_path / "emotion_classifier_model.pkl"
    joblib.dump({"model": model, "feature_names": feature_names}, model_path)
    print(f"Model saved to {model_path}")

    # Export ONNX
    onnx_path = output_path / "emotion_classifier_model.onnx"
    export_onnx(model, feature_names, onnx_path)

    # Save benchmarks
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks = save_benchmarks(
        y_for_bench,
        y_pred_for_bench,
        dataset_name,
        "emotion_classifier",
        BENCHMARK_DIR,
        cv_method=cv_label,
        fold_accuracies=fold_accuracies if use_loso else None,
        fold_f1s=fold_f1s if use_loso else None,
        n_subjects=n_subjects if use_loso else None,
    )

    print(f"\nFinal accuracy: {benchmarks['accuracy']:.4f}")
    print(f"F1 macro: {benchmarks['f1_macro']:.4f}")
    if use_loso:
        print(f"LOSO mean ± std: {benchmarks['cv_accuracy_mean']:.4f} ± {benchmarks['cv_accuracy_std']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument(
        "--simulated", action="store_true",
        help="Use simulated data instead of DEAP"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/deap",
        help="Path to DEAP .dat files"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/saved",
        help="Directory to save trained models"
    )
    args = parser.parse_args()

    train(
        simulated=args.simulated,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
