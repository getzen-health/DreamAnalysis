"""
Accuracy improvement experiments v2 - targeting >86.07% CV accuracy.
New strategies:
1. XGBoost (gradient boosting, often best-in-class)
2. LightGBM (fast, accurate gradient boosting)
3. ExtraTrees (randomized splitting, can beat RF)
4. MLP neural network (captures non-linear patterns)
5. Drop noisy datasets (DEAP 55%, DENS 47%) - may reduce noise
6. Smart voting ensemble (XGBoost + LightGBM + RF)
7. RF-3000 (even more trees)
"""

import numpy as np
import time
import json
import warnings
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mega_trainer import (
    load_deap_enhanced, load_gameemo_enhanced, load_eeg_er_enhanced,
    load_dens, load_seed_enhanced, load_brainwave_emotions,
    load_muse_mental_state, load_muse2_motor_imagery, load_confused_student,
    load_muse_subconscious, load_emokey_moments, load_seed_iv,
    combine_datasets, log
)


def load_datasets(exclude_noisy=False):
    """Load all datasets, optionally excluding noisy ones."""
    loaders = [
        load_deap_enhanced, load_gameemo_enhanced, load_eeg_er_enhanced,
        load_dens, load_seed_enhanced, load_brainwave_emotions,
        load_muse_mental_state, load_muse2_motor_imagery, load_confused_student,
        load_muse_subconscious, load_emokey_moments, load_seed_iv,
    ]

    noisy_datasets = {'DEAP', 'DENS'}  # 55% and 47% individual accuracy

    datasets = []
    for loader in loaders:
        try:
            X, y, name = loader()
            if len(X) > 0 and len(np.unique(y)) >= 2:
                if exclude_noisy and name in noisy_datasets:
                    log(f"  {name}: EXCLUDED (noisy)")
                    continue
                datasets.append((X, y, name))
                log(f"  {name}: {len(X)} samples, {X.shape[1]} features")
        except Exception as e:
            log(f"  Error loading: {e}")

    return datasets


def build_dataset(datasets, target_features=80):
    """Build combined dataset."""
    X, y, _ = combine_datasets(datasets, target_features=target_features)
    return X, y


def evaluate(X, y, name, model, n_splits=5):
    """5-fold CV evaluation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    t0 = time.time()

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        m = type(model)(**model.get_params())
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))

    elapsed = time.time() - t0
    acc_mean, f1_mean = np.mean(accs), np.mean(f1s)
    acc_std, f1_std = np.std(accs), np.std(f1s)
    log(f"    {name:25s}: Acc={acc_mean:.4f}±{acc_std:.4f}  F1={f1_mean:.4f}±{f1_std:.4f}  ({elapsed:.1f}s)")

    return {
        "accuracy": float(acc_mean), "accuracy_std": float(acc_std),
        "f1": float(f1_mean), "f1_std": float(f1_std),
        "time": float(elapsed)
    }


def evaluate_voting(X, y, name, estimators, n_splits=5):
    """Evaluate a voting ensemble (needs special handling for clone)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    t0 = time.time()

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        voter = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        voter.fit(X_train, y_train)
        y_pred = voter.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))

    elapsed = time.time() - t0
    acc_mean, f1_mean = np.mean(accs), np.mean(f1s)
    acc_std, f1_std = np.std(accs), np.std(f1s)
    log(f"    {name:25s}: Acc={acc_mean:.4f}±{acc_std:.4f}  F1={f1_mean:.4f}±{f1_std:.4f}  ({elapsed:.1f}s)")

    return {
        "accuracy": float(acc_mean), "accuracy_std": float(acc_std),
        "f1": float(f1_mean), "f1_std": float(f1_std),
        "time": float(elapsed)
    }


def main():
    t_start = time.time()
    log("=" * 80)
    log("  ACCURACY IMPROVEMENT v2 - XGBoost, LightGBM, MLP, Ensembles")
    log("  Current best: RF-2000 = 86.07% CV accuracy")
    log("=" * 80)

    # === Load ALL datasets ===
    log("\n[Loading all datasets...]")
    all_datasets = load_datasets(exclude_noisy=False)
    total = sum(len(d[1]) for d in all_datasets)
    log(f"  Total: {len(all_datasets)} datasets, {total:,} samples")

    log("\n[Building combined dataset...]")
    X_all, y_all = build_dataset(all_datasets)
    log(f"  Combined: {X_all.shape[0]} samples, {X_all.shape[1]} features")
    label_dist = {int(k): int(v) for k, v in zip(*np.unique(y_all, return_counts=True))}
    log(f"  Labels: {label_dist}")

    all_results = {}

    # === 1. XGBoost ===
    log("\n" + "=" * 60)
    log("  EXPERIMENT 1: XGBoost")
    log("=" * 60)

    # Compute scale_pos_weight for class imbalance
    unique, counts = np.unique(y_all, return_counts=True)
    n_classes = len(unique)

    xgb_configs = [
        ("XGB-500", xgb.XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob', num_class=n_classes,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ("XGB-1000", xgb.XGBClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            objective='multi:softprob', num_class=n_classes,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ("XGB-1500-deep", xgb.XGBClassifier(
            n_estimators=1500, max_depth=12, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            objective='multi:softprob', num_class=n_classes,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
    ]

    for name, model in xgb_configs:
        r = evaluate(X_all, y_all, name, model)
        all_results[name] = r

    # === 2. LightGBM ===
    log("\n" + "=" * 60)
    log("  EXPERIMENT 2: LightGBM")
    log("=" * 60)

    lgbm_configs = [
        ("LGBM-500", lgb.LGBMClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, num_leaves=63,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
        ("LGBM-1000", lgb.LGBMClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, num_leaves=127,
            min_child_samples=20, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1)),
        ("LGBM-1500-deep", lgb.LGBMClassifier(
            n_estimators=1500, max_depth=12, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, num_leaves=255,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
    ]

    for name, model in lgbm_configs:
        r = evaluate(X_all, y_all, name, model)
        all_results[name] = r

    # === 3. ExtraTrees (more trees) ===
    log("\n" + "=" * 60)
    log("  EXPERIMENT 3: ExtraTrees")
    log("=" * 60)

    et_configs = [
        ("ET-1000", ExtraTreesClassifier(
            n_estimators=1000, max_depth=25, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1)),
        ("ET-2000", ExtraTreesClassifier(
            n_estimators=2000, max_depth=30, min_samples_leaf=1,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1)),
    ]

    for name, model in et_configs:
        r = evaluate(X_all, y_all, name, model)
        all_results[name] = r

    # === 4. RF-3000 (even more trees) ===
    log("\n" + "=" * 60)
    log("  EXPERIMENT 4: RF with even more trees")
    log("=" * 60)

    rf_configs = [
        ("RF-3000", RandomForestClassifier(
            n_estimators=3000, max_depth=25, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1)),
        ("RF-2000-deep30", RandomForestClassifier(
            n_estimators=2000, max_depth=30, min_samples_leaf=1,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1)),
    ]

    for name, model in rf_configs:
        r = evaluate(X_all, y_all, name, model)
        all_results[name] = r

    # === 5. MLP Neural Network ===
    log("\n" + "=" * 60)
    log("  EXPERIMENT 5: MLP Neural Network")
    log("=" * 60)

    mlp_configs = [
        ("MLP-256-128", MLPClassifier(
            hidden_layer_sizes=(256, 128), activation='relu',
            solver='adam', learning_rate='adaptive', learning_rate_init=0.001,
            max_iter=300, batch_size=512, early_stopping=True,
            validation_fraction=0.1, random_state=42)),
        ("MLP-512-256-128", MLPClassifier(
            hidden_layer_sizes=(512, 256, 128), activation='relu',
            solver='adam', learning_rate='adaptive', learning_rate_init=0.0005,
            max_iter=500, batch_size=1024, early_stopping=True,
            validation_fraction=0.1, random_state=42)),
    ]

    for name, model in mlp_configs:
        r = evaluate(X_all, y_all, name, model)
        all_results[name] = r

    # === 6. Drop noisy datasets ===
    log("\n" + "=" * 60)
    log("  EXPERIMENT 6: Drop noisy datasets (DEAP, DENS)")
    log("=" * 60)

    log("[Loading clean datasets...]")
    clean_datasets = load_datasets(exclude_noisy=True)
    clean_total = sum(len(d[1]) for d in clean_datasets)
    log(f"  Clean total: {len(clean_datasets)} datasets, {clean_total:,} samples")

    X_clean, y_clean = build_dataset(clean_datasets)
    log(f"  Clean combined: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")

    clean_configs = [
        ("RF-2000-clean", RandomForestClassifier(
            n_estimators=2000, max_depth=25, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1)),
        ("XGB-1000-clean", xgb.XGBClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ("LGBM-1000-clean", lgb.LGBMClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, num_leaves=127,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
    ]

    for name, model in clean_configs:
        r = evaluate(X_clean, y_clean, name, model)
        all_results[name] = r

    # === 7. Smart Voting Ensemble ===
    log("\n" + "=" * 60)
    log("  EXPERIMENT 7: Smart Voting Ensemble")
    log("=" * 60)

    # Best models from each family
    estimators_full = [
        ('rf', RandomForestClassifier(
            n_estimators=1000, max_depth=25, min_samples_leaf=2,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1)),
        ('xgb', xgb.XGBClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ('lgbm', lgb.LGBMClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            num_leaves=127, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1)),
        ('et', ExtraTreesClassifier(
            n_estimators=500, max_depth=25, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=-1)),
    ]

    r = evaluate_voting(X_all, y_all, "Vote-RF+XGB+LGBM+ET", estimators_full)
    all_results["Vote-RF+XGB+LGBM+ET"] = r

    # Smaller, faster ensemble
    estimators_fast = [
        ('rf', RandomForestClassifier(
            n_estimators=500, max_depth=20, class_weight='balanced',
            random_state=42, n_jobs=-1)),
        ('xgb', xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ('lgbm', lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
    ]

    r = evaluate_voting(X_all, y_all, "Vote-RF+XGB+LGBM", estimators_fast)
    all_results["Vote-RF+XGB+LGBM"] = r

    # === FINAL SUMMARY ===
    elapsed = time.time() - t_start
    log("\n" + "=" * 80)
    log("  FINAL RESULTS (sorted by F1)")
    log("  Baseline: RF-2000 = 86.07% accuracy")
    log("=" * 80)
    log(f"\n{'Model':<28} {'Accuracy':>10} {'F1':>10} {'Time':>8}")
    log("-" * 62)

    for name, r in sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True):
        acc = r['accuracy']
        f1 = r['f1']
        t = r['time']
        marker = " ***" if f1 > 0.8607 else ""
        log(f"{name:<28} {acc:>9.4f} {f1:>9.4f} {t:>7.1f}s{marker}")

    best_name = max(all_results, key=lambda k: all_results[k]['f1'])
    best = all_results[best_name]
    improvement = (best['f1'] - 0.8600) * 100

    log(f"\nBest model: {best_name}")
    log(f"  Accuracy: {best['accuracy']:.4f}")
    log(f"  F1: {best['f1']:.4f}")
    log(f"  Improvement over RF-2000: {improvement:+.2f}% F1")
    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    benchmarks_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/benchmarks')
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    with open(benchmarks_dir / 'improvement_v2_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nResults saved to {benchmarks_dir / 'improvement_v2_results.json'}")


if __name__ == '__main__':
    main()
