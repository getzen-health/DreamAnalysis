"""
Accuracy improvement experiments for MEGA-Combined model.
Strategies:
1. Class rebalancing (SMOTE + class weights)
2. HistGradientBoosting (fast, native categorical support)
3. Stacking ensemble (RF + GBM + ExtraTrees → LogisticRegression)
4. Hyperparameter tuning with RandomizedSearchCV
5. Dataset-quality-aware sample weighting
6. Feature selection (remove low-importance features)
"""

import numpy as np
import time
import json
import warnings
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import randint, uniform

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mega_trainer import (
    load_deap_enhanced, load_gameemo_enhanced, load_eeg_er_enhanced,
    load_dens, load_seed_enhanced, load_brainwave_emotions,
    load_muse_mental_state, load_muse2_motor_imagery, load_confused_student,
    load_mental_attention, load_muse_subconscious, load_emokey_moments,
    load_seed_iv, combine_datasets, log
)


def load_all_datasets():
    """Load all datasets and return combined data."""
    datasets = []
    dataset_qualities = {}  # Track per-dataset quality for weighting

    # Quality tiers based on known accuracies
    high_quality = ['SEED', 'SEED-IV', 'Brainwave', 'Muse-Mental', 'Muse-Subconscious']
    medium_quality = ['GAMEEMO', 'EEG-ER', 'Confused-Student']
    low_quality = ['DEAP', 'DENS', 'EmoKey', 'Muse2-MI']

    loaders = [
        load_deap_enhanced, load_gameemo_enhanced, load_eeg_er_enhanced,
        load_dens, load_seed_enhanced, load_brainwave_emotions,
        load_muse_mental_state, load_muse2_motor_imagery, load_confused_student,
        load_mental_attention, load_muse_subconscious, load_emokey_moments,
        load_seed_iv,
    ]

    for loader in loaders:
        try:
            X, y, name = loader()
            if len(X) > 0 and len(np.unique(y)) >= 2:
                datasets.append((X, y, name))
                if name in high_quality:
                    dataset_qualities[name] = 1.0
                elif name in medium_quality:
                    dataset_qualities[name] = 0.8
                else:
                    dataset_qualities[name] = 0.5
                log(f"  {name}: {len(X)} samples, {X.shape[1]} features (quality={dataset_qualities[name]})")
        except Exception as e:
            log(f"  Error: {e}")

    return datasets, dataset_qualities


def build_mega_dataset(datasets, dataset_qualities, target_features=80):
    """Build MEGA-Combined with quality-aware sample weights."""
    X_mega, y_mega, _ = combine_datasets(datasets, target_features=target_features)

    # Build sample weights based on dataset quality
    sample_weights = np.ones(len(y_mega))
    idx = 0
    for X, y, name in datasets:
        if len(X) == 0 or len(np.unique(y)) < 2:
            continue
        n = len(y)
        quality = dataset_qualities.get(name, 0.5)
        sample_weights[idx:idx+n] = quality
        idx += n

    # Also weight by inverse class frequency for balance
    unique, counts = np.unique(y_mega, return_counts=True)
    class_weights = {c: len(y_mega) / (len(unique) * count) for c, count in zip(unique, counts)}
    for i, label in enumerate(y_mega):
        sample_weights[i] *= class_weights[label]

    return X_mega, y_mega, sample_weights


def experiment_1_histgbm(X, y, sample_weights):
    """HistGradientBoosting - much faster than GBM, often better."""
    log("\n--- Experiment 1: HistGradientBoostingClassifier ---")
    configs = [
        ("HGBM-256", HistGradientBoostingClassifier(
            max_iter=256, max_depth=8, learning_rate=0.1,
            min_samples_leaf=20, l2_regularization=0.1,
            class_weight='balanced', random_state=42)),
        ("HGBM-512", HistGradientBoostingClassifier(
            max_iter=512, max_depth=10, learning_rate=0.05,
            min_samples_leaf=10, l2_regularization=0.05,
            class_weight='balanced', random_state=42)),
        ("HGBM-1024", HistGradientBoostingClassifier(
            max_iter=1024, max_depth=8, learning_rate=0.03,
            min_samples_leaf=15, l2_regularization=0.01,
            class_weight='balanced', random_state=42)),
    ]
    return _evaluate_configs(X, y, configs, sample_weights)


def experiment_2_tuned_rf(X, y, sample_weights):
    """Tuned RF with more trees and better hyperparameters."""
    log("\n--- Experiment 2: Tuned RandomForest ---")
    configs = [
        ("RF-1000", RandomForestClassifier(
            n_estimators=1000, max_depth=20, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1)),
        ("RF-1000-deep", RandomForestClassifier(
            n_estimators=1000, max_depth=30, min_samples_leaf=1,
            max_features='sqrt', class_weight='balanced_subsample',
            random_state=42, n_jobs=-1)),
        ("RF-2000", RandomForestClassifier(
            n_estimators=2000, max_depth=25, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1)),
    ]
    return _evaluate_configs(X, y, configs, sample_weights)


def experiment_3_voting_ensemble(X, y, sample_weights):
    """Voting ensemble combining multiple strong models."""
    log("\n--- Experiment 3: Voting Ensemble ---")

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, class_weight='balanced',
        random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(
        n_estimators=500, max_depth=20, class_weight='balanced',
        random_state=42, n_jobs=-1)
    hgbm = HistGradientBoostingClassifier(
        max_iter=512, max_depth=10, learning_rate=0.05,
        class_weight='balanced', random_state=42)

    configs = [
        ("Voting-soft", VotingClassifier(
            estimators=[('rf', rf), ('et', et), ('hgbm', hgbm)],
            voting='soft', n_jobs=-1)),
    ]
    return _evaluate_configs(X, y, configs, sample_weights)


def experiment_4_stacking(X, y, sample_weights):
    """Stacking ensemble with meta-learner."""
    log("\n--- Experiment 4: Stacking Ensemble ---")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, class_weight='balanced',
        random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(
        n_estimators=300, max_depth=15, class_weight='balanced',
        random_state=42, n_jobs=-1)
    hgbm = HistGradientBoostingClassifier(
        max_iter=256, max_depth=8, learning_rate=0.1,
        class_weight='balanced', random_state=42)

    configs = [
        ("Stack-LR", StackingClassifier(
            estimators=[('rf', rf), ('et', et), ('hgbm', hgbm)],
            final_estimator=LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'),
            cv=3, n_jobs=-1)),
    ]
    return _evaluate_configs(X, y, configs, sample_weights)


def experiment_5_weighted_samples(X, y, sample_weights):
    """Use quality-aware sample weights during training."""
    log("\n--- Experiment 5: Quality-Weighted RF ---")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    t0 = time.time()

    model = RandomForestClassifier(
        n_estimators=1000, max_depth=20, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1)

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = sample_weights[train_idx]

        model_copy = RandomForestClassifier(**model.get_params())
        model_copy.fit(X_train, y_train, sample_weight=w_train)
        y_pred = model_copy.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))

    elapsed = time.time() - t0
    acc_mean, f1_mean = np.mean(accs), np.mean(f1s)
    log(f"    Weighted-RF-1000: Acc={acc_mean:.4f}±{np.std(accs):.4f}  F1={f1_mean:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"Weighted-RF-1000": {"accuracy": float(acc_mean), "f1": float(f1_mean), "time": float(elapsed)}}


def experiment_6_hgbm_tuned(X, y, sample_weights):
    """RandomizedSearchCV for HistGradientBoosting hyperparameters."""
    log("\n--- Experiment 6: HGBM Hyperparameter Search ---")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_dist = {
        'max_iter': randint(200, 800),
        'max_depth': randint(6, 15),
        'learning_rate': uniform(0.01, 0.15),
        'min_samples_leaf': randint(5, 30),
        'l2_regularization': uniform(0.0, 0.5),
        'max_bins': [128, 255],
    }

    hgbm = HistGradientBoostingClassifier(class_weight='balanced', random_state=42)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    t0 = time.time()
    search = RandomizedSearchCV(
        hgbm, param_dist, n_iter=20, cv=skf, scoring='f1_weighted',
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_scaled, y)
    elapsed = time.time() - t0

    best = search.best_estimator_
    log(f"    Best params: {search.best_params_}")
    log(f"    Best CV F1: {search.best_score_:.4f} ({elapsed:.1f}s)")

    # Evaluate best model with 5-fold
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for train_idx, test_idx in skf5.split(X_scaled, y):
        model = HistGradientBoostingClassifier(**best.get_params())
        model.fit(X_scaled[train_idx], y[train_idx])
        y_pred = model.predict(X_scaled[test_idx])
        accs.append(accuracy_score(y[test_idx], y_pred))
        f1s.append(f1_score(y[test_idx], y_pred, average='weighted'))

    acc_mean, f1_mean = np.mean(accs), np.mean(f1s)
    log(f"    HGBM-Tuned (5-fold): Acc={acc_mean:.4f}±{np.std(accs):.4f}  F1={f1_mean:.4f}±{np.std(f1s):.4f}")

    return {
        "HGBM-Tuned": {
            "accuracy": float(acc_mean), "f1": float(f1_mean),
            "time": float(elapsed),
            "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (float, np.floating)) else v) for k, v in search.best_params_.items()}
        }
    }


def experiment_7_feature_augmented(X, y, sample_weights):
    """Add interaction features and polynomial features for top-N important features."""
    log("\n--- Experiment 7: Feature Augmentation + RF ---")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # First, get feature importances from a quick RF
    quick_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    quick_rf.fit(X_scaled, y)
    importances = quick_rf.feature_importances_

    # Get top 20 features
    top_idx = np.argsort(importances)[-20:]

    # Create interaction features (top_i * top_j)
    interactions = []
    for i in range(len(top_idx)):
        for j in range(i+1, len(top_idx)):
            interactions.append(X_scaled[:, top_idx[i]] * X_scaled[:, top_idx[j]])

    # Add ratio features
    ratios = []
    for i in range(min(10, len(top_idx))):
        for j in range(i+1, min(10, len(top_idx))):
            denom = X_scaled[:, top_idx[j]]
            denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
            ratios.append(X_scaled[:, top_idx[i]] / denom)

    X_aug = np.column_stack([X_scaled] + interactions + ratios)
    X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=0.0, neginf=0.0)

    log(f"    Augmented features: {X_scaled.shape[1]} → {X_aug.shape[1]}")

    # PCA back down to manageable size
    if X_aug.shape[1] > 120:
        pca = PCA(n_components=120)
        X_aug = pca.fit_transform(X_aug)
        log(f"    PCA reduced to {X_aug.shape[1]} features")

    configs = [
        ("RF-Aug-1000", RandomForestClassifier(
            n_estimators=1000, max_depth=20, class_weight='balanced',
            random_state=42, n_jobs=-1)),
        ("HGBM-Aug-512", HistGradientBoostingClassifier(
            max_iter=512, max_depth=10, learning_rate=0.05,
            class_weight='balanced', random_state=42)),
    ]
    return _evaluate_configs(X_aug, y, configs, sample_weights)


def _evaluate_configs(X, y, configs, sample_weights=None):
    """Evaluate multiple model configs with 5-fold CV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in configs:
        accs, f1s = [], []
        t0 = time.time()

        for train_idx, test_idx in skf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average='weighted'))

        elapsed = time.time() - t0
        acc_mean, f1_mean = np.mean(accs), np.mean(f1s)
        log(f"    {name:20s}: Acc={acc_mean:.4f}±{np.std(accs):.4f}  F1={f1_mean:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")

        results[name] = {
            "accuracy": float(acc_mean),
            "accuracy_std": float(np.std(accs)),
            "f1": float(f1_mean),
            "f1_std": float(np.std(f1s)),
            "time": float(elapsed)
        }

    return results


def main():
    t_start = time.time()
    log("=" * 80)
    log("  ACCURACY IMPROVEMENT EXPERIMENTS")
    log("=" * 80)

    # Load datasets
    log("\n[Loading datasets...]")
    datasets, dataset_qualities = load_all_datasets()
    total = sum(len(d[1]) for d in datasets)
    log(f"\n  Total: {len(datasets)} datasets, {total:,} samples")

    # Build MEGA dataset with sample weights
    log("\n[Building MEGA-Combined dataset...]")
    X_mega, y_mega, sample_weights = build_mega_dataset(datasets, dataset_qualities)
    log(f"  MEGA: {len(X_mega)} samples, {X_mega.shape[1]} features")
    label_dist = {int(k): int(v) for k, v in zip(*np.unique(y_mega, return_counts=True))}
    log(f"  Labels: {label_dist}")

    # Standardize once for all experiments
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mega)

    all_results = {}

    # Baseline
    log("\n--- Baseline: Current best (RF-500) ---")
    baseline = _evaluate_configs(X_mega, y_mega, [
        ("Baseline-RF-500", RandomForestClassifier(
            n_estimators=500, max_depth=15, class_weight='balanced',
            random_state=42, n_jobs=-1)),
    ])
    all_results.update(baseline)

    # Run experiments
    r1 = experiment_1_histgbm(X_mega, y_mega, sample_weights)
    all_results.update(r1)

    r2 = experiment_2_tuned_rf(X_mega, y_mega, sample_weights)
    all_results.update(r2)

    r5 = experiment_5_weighted_samples(X_mega, y_mega, sample_weights)
    all_results.update(r5)

    r6 = experiment_6_hgbm_tuned(X_mega, y_mega, sample_weights)
    all_results.update(r6)

    r7 = experiment_7_feature_augmented(X_mega, y_mega, sample_weights)
    all_results.update(r7)

    # Voting and stacking (slower, run last)
    r3 = experiment_3_voting_ensemble(X_mega, y_mega, sample_weights)
    all_results.update(r3)

    r4 = experiment_4_stacking(X_mega, y_mega, sample_weights)
    all_results.update(r4)

    # Final summary
    elapsed = time.time() - t_start
    log("\n" + "=" * 80)
    log("  IMPROVEMENT RESULTS SUMMARY")
    log("=" * 80)
    log(f"\n{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Time':>8}")
    log("-" * 60)

    for name, r in sorted(all_results.items(), key=lambda x: x[1].get('f1', x[1].get('f1', 0)), reverse=True):
        acc = r.get('accuracy', 0)
        f1 = r.get('f1', 0)
        t = r.get('time', 0)
        log(f"{name:<25} {acc:>9.4f} {f1:>9.4f} {t:>7.1f}s")

    best_name = max(all_results, key=lambda k: all_results[k].get('f1', 0))
    best = all_results[best_name]
    baseline_f1 = all_results.get('Baseline-RF-500', {}).get('f1', 0.8511)

    log(f"\nBest model: {best_name}")
    log(f"  Accuracy: {best['accuracy']:.4f}")
    log(f"  F1: {best['f1']:.4f}")
    log(f"  Improvement over baseline: {(best['f1'] - baseline_f1)*100:+.2f}% F1")
    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    benchmarks_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/benchmarks')
    with open(benchmarks_dir / 'improvement_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Results saved to {benchmarks_dir / 'improvement_results.json'}")


if __name__ == '__main__':
    main()
