"""
Accuracy improvement experiments v3 - targeting >88.22% CV accuracy.
Focus on clean data + gradient boosting fine-tuning.
"""

import numpy as np
import time
import json
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mega_trainer import (
    load_gameemo_enhanced, load_eeg_er_enhanced,
    load_seed_enhanced, load_brainwave_emotions,
    load_muse_mental_state, load_muse2_motor_imagery, load_confused_student,
    load_muse_subconscious, load_emokey_moments, load_seed_iv,
    log
)


def load_clean_datasets():
    """Load clean datasets (excluding DEAP, DENS)."""
    loaders = [
        load_gameemo_enhanced, load_eeg_er_enhanced,
        load_seed_enhanced, load_brainwave_emotions,
        load_muse_mental_state, load_muse2_motor_imagery, load_confused_student,
        load_muse_subconscious, load_emokey_moments, load_seed_iv,
    ]
    datasets = []
    for loader in loaders:
        try:
            X, y, name = loader()
            if len(X) > 0 and len(np.unique(y)) >= 2:
                datasets.append((X, y, name))
                log(f"  {name}: {len(X)} samples, {X.shape[1]} features")
        except Exception as e:
            log(f"  Error: {e}")
    return datasets


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
    log(f"    {name:30s}: Acc={acc_mean:.4f}±{np.std(accs):.4f}  F1={f1_mean:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {
        "accuracy": float(acc_mean), "accuracy_std": float(np.std(accs)),
        "f1": float(f1_mean), "f1_std": float(np.std(f1s)),
        "time": float(elapsed)
    }


def evaluate_voting(X, y, name, estimators, n_splits=5):
    """Evaluate soft voting ensemble."""
    from sklearn.ensemble import VotingClassifier
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    t0 = time.time()

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        voter = VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)
        voter.fit(X_train, y_train)
        y_pred = voter.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))

    elapsed = time.time() - t0
    acc_mean, f1_mean = np.mean(accs), np.mean(f1s)
    log(f"    {name:30s}: Acc={acc_mean:.4f}±{np.std(accs):.4f}  F1={f1_mean:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {
        "accuracy": float(acc_mean), "accuracy_std": float(np.std(accs)),
        "f1": float(f1_mean), "f1_std": float(np.std(f1s)),
        "time": float(elapsed)
    }


def main():
    t_start = time.time()
    log("=" * 80)
    log("  ACCURACY IMPROVEMENT v3 - Fine-tuning on clean data")
    log("  Current best: XGB-1000-clean = 88.22% CV accuracy")
    log("=" * 80)

    log("\n[Loading clean datasets...]")
    datasets = load_clean_datasets()
    total = sum(len(d[1]) for d in datasets)
    log(f"  Total: {len(datasets)} datasets, {total:,} samples")

    log("\n[Building combined dataset...]")
    X, y = [], []
    TARGET_FEATURES = 80
    for Xi, yi, name in datasets:
        Xi = np.nan_to_num(Xi, nan=0.0, posinf=0.0, neginf=0.0)
        s = StandardScaler()
        Xi_s = s.fit_transform(Xi)
        n_comp = min(TARGET_FEATURES, Xi_s.shape[1], Xi_s.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        Xi_p = pca.fit_transform(Xi_s)
        if Xi_p.shape[1] < TARGET_FEATURES:
            Xi_p = np.hstack([Xi_p, np.zeros((Xi_p.shape[0], TARGET_FEATURES - Xi_p.shape[1]))])
        X.append(Xi_p)
        y.append(yi)
    X = np.vstack(X)
    y = np.concatenate(y)
    log(f"  Combined: {X.shape[0]} samples, {X.shape[1]} features")
    label_dist = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    log(f"  Labels: {label_dist}")

    n_classes = len(np.unique(y))
    all_results = {}

    # === 1. XGBoost fine-tuning ===
    log("\n" + "=" * 60)
    log("  XGBoost hyperparameter variations")
    log("=" * 60)

    xgb_configs = [
        ("XGB-1500-d12-lr03", xgb.XGBClassifier(
            n_estimators=1500, max_depth=12, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ("XGB-2000-d10-lr02", xgb.XGBClassifier(
            n_estimators=2000, max_depth=10, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            gamma=0.05, reg_alpha=0.05, reg_lambda=0.5,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ("XGB-1000-d15-lr05", xgb.XGBClassifier(
            n_estimators=1000, max_depth=15, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ("XGB-1500-d10-lr05", xgb.XGBClassifier(
            n_estimators=1500, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
    ]

    for name, model in xgb_configs:
        r = evaluate(X, y, name, model)
        all_results[name] = r

    # === 2. LightGBM fine-tuning ===
    log("\n" + "=" * 60)
    log("  LightGBM hyperparameter variations")
    log("=" * 60)

    lgbm_configs = [
        ("LGBM-2000-d12-lr02", lgb.LGBMClassifier(
            n_estimators=2000, max_depth=12, learning_rate=0.02,
            subsample=0.85, colsample_bytree=0.7, num_leaves=255,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
        ("LGBM-1500-d15-lr03", lgb.LGBMClassifier(
            n_estimators=1500, max_depth=15, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, num_leaves=512,
            min_child_samples=5, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1)),
        ("LGBM-2000-d10-lr03", lgb.LGBMClassifier(
            n_estimators=2000, max_depth=10, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, num_leaves=127,
            min_child_samples=20, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1)),
    ]

    for name, model in lgbm_configs:
        r = evaluate(X, y, name, model)
        all_results[name] = r

    # === 3. XGB + LGBM voting on clean data ===
    log("\n" + "=" * 60)
    log("  XGB + LGBM voting ensembles on clean data")
    log("=" * 60)

    estimators_xgb_lgbm = [
        ('xgb', xgb.XGBClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ('lgbm', lgb.LGBMClassifier(
            n_estimators=1500, max_depth=12, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, num_leaves=255,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
    ]
    r = evaluate_voting(X, y, "Vote-XGB+LGBM", estimators_xgb_lgbm)
    all_results["Vote-XGB+LGBM"] = r

    # Triple voting
    estimators_triple = [
        ('xgb1', xgb.XGBClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ('xgb2', xgb.XGBClassifier(
            n_estimators=1500, max_depth=12, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, min_child_weight=3,
            gamma=0.1, tree_method='hist', random_state=43, n_jobs=-1, verbosity=0)),
        ('lgbm', lgb.LGBMClassifier(
            n_estimators=1500, max_depth=12, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, num_leaves=255,
            min_child_samples=10, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1)),
    ]
    r = evaluate_voting(X, y, "Vote-2xXGB+LGBM", estimators_triple)
    all_results["Vote-2xXGB+LGBM"] = r

    # === 4. Drop even more borderline datasets ===
    log("\n" + "=" * 60)
    log("  Experiment: Drop Muse2-MI and EmoKey (weaker datasets)")
    log("=" * 60)

    # Build dataset without Muse2-MI and EmoKey
    weak_datasets = {'Muse2-MI', 'EmoKey'}
    X_clean, y_clean = [], []
    for Xi, yi, name in datasets:
        if name in weak_datasets:
            log(f"  Excluding {name}")
            continue
        Xi = np.nan_to_num(Xi, nan=0.0, posinf=0.0, neginf=0.0)
        s = StandardScaler()
        Xi_s = s.fit_transform(Xi)
        n_comp = min(TARGET_FEATURES, Xi_s.shape[1], Xi_s.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        Xi_p = pca.fit_transform(Xi_s)
        if Xi_p.shape[1] < TARGET_FEATURES:
            Xi_p = np.hstack([Xi_p, np.zeros((Xi_p.shape[0], TARGET_FEATURES - Xi_p.shape[1]))])
        X_clean.append(Xi_p)
        y_clean.append(yi)
    X_clean = np.vstack(X_clean)
    y_clean = np.concatenate(y_clean)
    log(f"  Ultra-clean: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")

    ultra_configs = [
        ("XGB-1000-ultraclean", xgb.XGBClassifier(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ("LGBM-1500-ultraclean", lgb.LGBMClassifier(
            n_estimators=1500, max_depth=12, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, num_leaves=255,
            min_child_samples=10, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1)),
    ]

    for name, model in ultra_configs:
        r = evaluate(X_clean, y_clean, name, model)
        all_results[name] = r

    # === FINAL SUMMARY ===
    elapsed = time.time() - t_start
    log("\n" + "=" * 80)
    log("  FINAL RESULTS (sorted by F1)")
    log("  Baseline: XGB-1000-clean = 88.22% / 88.13% F1")
    log("=" * 80)
    log(f"\n{'Model':<33} {'Accuracy':>10} {'F1':>10} {'Time':>8}")
    log("-" * 67)

    for name, r in sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True):
        acc = r['accuracy']
        f1 = r['f1']
        t = r['time']
        marker = " ***" if f1 > 0.8813 else ""
        log(f"{name:<33} {acc:>9.4f} {f1:>9.4f} {t:>7.1f}s{marker}")

    best_name = max(all_results, key=lambda k: all_results[k]['f1'])
    best = all_results[best_name]
    log(f"\nBest: {best_name} = {best['accuracy']:.4f} accuracy, {best['f1']:.4f} F1")
    log(f"Improvement over XGB-1000-clean: {(best['f1'] - 0.8813)*100:+.2f}% F1")
    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save
    benchmarks_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/benchmarks')
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    with open(benchmarks_dir / 'improvement_v3_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Results saved to {benchmarks_dir / 'improvement_v3_results.json'}")


if __name__ == '__main__':
    main()
