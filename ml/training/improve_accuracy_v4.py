"""
Accuracy improvement v4 - targeting >95% CV accuracy.
Current best: LGBM-1500-ultraclean = 94.23%
Strategies:
1. PCA dimension optimization (try 60, 100, 120 instead of 80)
2. LGBM hyperparameter grid around the best config
3. Drop more weak datasets (try removing Confused-Student, EEG-ER)
4. Multi-seed ensemble (average predictions from multiple LGBM seeds)
5. XGBoost fine-tuning on ultra-clean data
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
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mega_trainer import (
    load_gameemo_enhanced, load_eeg_er_enhanced,
    load_seed_enhanced, load_brainwave_emotions,
    load_muse_mental_state, load_confused_student,
    load_muse_subconscious, load_seed_iv,
    log
)

ALL_LOADERS = {
    'GAMEEMO': load_gameemo_enhanced,
    'EEG-ER': load_eeg_er_enhanced,
    'SEED': load_seed_enhanced,
    'Brainwave': load_brainwave_emotions,
    'Muse-Mental': load_muse_mental_state,
    'Confused-Student': load_confused_student,
    'Muse-Subconscious': load_muse_subconscious,
    'SEED-IV': load_seed_iv,
}


def load_datasets(exclude=None):
    """Load datasets, optionally excluding some."""
    exclude = exclude or set()
    datasets = []
    for name, loader in ALL_LOADERS.items():
        if name in exclude:
            continue
        try:
            X, y, dname = loader()
            if len(X) > 0 and len(np.unique(y)) >= 2:
                datasets.append((X, y, dname))
        except Exception as e:
            log(f"  Error loading {name}: {e}")
    return datasets


def build_dataset(datasets, target_features=80):
    """PCA-align and combine datasets."""
    all_X, all_y = [], []
    for X, y, name in datasets:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        s = StandardScaler()
        X_s = s.fit_transform(X)
        n_comp = min(target_features, X_s.shape[1], X_s.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X_p = pca.fit_transform(X_s)
        if X_p.shape[1] < target_features:
            X_p = np.hstack([X_p, np.zeros((X_p.shape[0], target_features - X_p.shape[1]))])
        all_X.append(X_p)
        all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)


def evaluate(X, y, name, model, n_splits=5):
    """5-fold CV evaluation."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    t0 = time.time()
    for tr, te in skf.split(X_s, y):
        m = type(model)(**model.get_params())
        m.fit(X_s[tr], y[tr])
        yp = m.predict(X_s[te])
        accs.append(accuracy_score(y[te], yp))
        f1s.append(f1_score(y[te], yp, average='weighted'))
    elapsed = time.time() - t0
    am, fm = np.mean(accs), np.mean(f1s)
    log(f"    {name:35s}: Acc={am:.4f}±{np.std(accs):.4f}  F1={fm:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "accuracy_std": float(np.std(accs)),
            "f1": float(fm), "f1_std": float(np.std(f1s)), "time": float(elapsed)}


def evaluate_multiseed(X, y, name, base_params, n_seeds=5, n_splits=5):
    """Multi-seed ensemble: average probability predictions from models with different seeds."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    t0 = time.time()

    for tr, te in skf.split(X_s, y):
        probs = []
        for seed in range(n_seeds):
            params = dict(base_params)
            params['random_state'] = 42 + seed
            m = lgb.LGBMClassifier(**params)
            m.fit(X_s[tr], y[tr])
            probs.append(m.predict_proba(X_s[te]))
        avg_probs = np.mean(probs, axis=0)
        yp = np.argmax(avg_probs, axis=1)
        accs.append(accuracy_score(y[te], yp))
        f1s.append(f1_score(y[te], yp, average='weighted'))

    elapsed = time.time() - t0
    am, fm = np.mean(accs), np.mean(f1s)
    log(f"    {name:35s}: Acc={am:.4f}±{np.std(accs):.4f}  F1={fm:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "f1": float(fm), "time": float(elapsed)}


def main():
    t_start = time.time()
    log("=" * 80)
    log("  ACCURACY IMPROVEMENT v4 - Targeting >95%")
    log("  Current best: LGBM-1500-ultraclean = 94.23%")
    log("=" * 80)

    # Load all ultra-clean datasets once
    log("\n[Loading ultra-clean datasets...]")
    datasets = load_datasets()
    for X, y, name in datasets:
        log(f"  {name}: {len(X)} samples")
    total = sum(len(d[1]) for d in datasets)
    log(f"  Total: {len(datasets)} datasets, {total:,} samples")

    all_results = {}

    # === 1. PCA dimension optimization ===
    log("\n" + "=" * 60)
    log("  EXP 1: PCA dimension optimization")
    log("=" * 60)

    best_lgbm_params = dict(
        n_estimators=1500, max_depth=12, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.7, num_leaves=255,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)

    for dim in [60, 100, 120]:
        log(f"\n  PCA={dim} dimensions:")
        X, y = build_dataset(datasets, target_features=dim)
        log(f"    Data: {X.shape[0]} x {X.shape[1]}")
        r = evaluate(X, y, f"LGBM-1500-pca{dim}", lgb.LGBMClassifier(**best_lgbm_params))
        all_results[f"LGBM-1500-pca{dim}"] = r

    # Baseline PCA=80
    log("\n  PCA=80 (baseline):")
    X80, y80 = build_dataset(datasets, target_features=80)
    r = evaluate(X80, y80, "LGBM-1500-pca80 (baseline)", lgb.LGBMClassifier(**best_lgbm_params))
    all_results["LGBM-1500-pca80"] = r

    # === 2. LGBM hyperparameter grid ===
    log("\n" + "=" * 60)
    log("  EXP 2: LGBM hyperparameter grid (on PCA=80)")
    log("=" * 60)

    lgbm_configs = [
        ("LGBM-2000-d15-nl512", dict(
            n_estimators=2000, max_depth=15, learning_rate=0.02,
            subsample=0.85, colsample_bytree=0.7, num_leaves=512,
            min_child_samples=5, reg_alpha=0.05, reg_lambda=0.5,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
        ("LGBM-3000-d12-lr01", dict(
            n_estimators=3000, max_depth=12, learning_rate=0.01,
            subsample=0.85, colsample_bytree=0.7, num_leaves=255,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
        ("LGBM-2000-d12-nl255-nobal", dict(
            n_estimators=2000, max_depth=12, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, num_leaves=255,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1)),
        ("LGBM-1500-d20-nl1024", dict(
            n_estimators=1500, max_depth=20, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, num_leaves=1024,
            min_child_samples=5, reg_alpha=0.01, reg_lambda=0.1,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
        ("LGBM-2500-d12-lr02", dict(
            n_estimators=2500, max_depth=12, learning_rate=0.02,
            subsample=0.85, colsample_bytree=0.7, num_leaves=255,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)),
    ]

    for name, params in lgbm_configs:
        r = evaluate(X80, y80, name, lgb.LGBMClassifier(**params))
        all_results[name] = r

    # === 3. Dataset ablation ===
    log("\n" + "=" * 60)
    log("  EXP 3: Dataset ablation (drop one at a time)")
    log("=" * 60)

    for drop_name in ALL_LOADERS.keys():
        ds = load_datasets(exclude={drop_name})
        X_abl, y_abl = build_dataset(ds, target_features=80)
        n = X_abl.shape[0]
        r = evaluate(X_abl, y_abl, f"drop-{drop_name} ({n:,})", lgb.LGBMClassifier(**best_lgbm_params))
        all_results[f"drop-{drop_name}"] = r

    # === 4. Best subset combinations ===
    log("\n" + "=" * 60)
    log("  EXP 4: Best dataset subsets")
    log("=" * 60)

    # Try top-quality only: SEED + SEED-IV + Brainwave + Muse-Mental + Muse-Subconscious
    subsets = [
        ("top5-research", {'GAMEEMO', 'EEG-ER', 'Confused-Student'}),
        ("top4-seed-muse", {'GAMEEMO', 'EEG-ER', 'Confused-Student', 'Brainwave'}),
        ("seed-family-only", {'GAMEEMO', 'EEG-ER', 'Confused-Student', 'Brainwave', 'Muse-Mental', 'Muse-Subconscious'}),
    ]

    for subset_name, exclude_set in subsets:
        ds = load_datasets(exclude=exclude_set)
        names = [n for _, _, n in ds]
        X_sub, y_sub = build_dataset(ds, target_features=80)
        log(f"  {subset_name}: {names}, {X_sub.shape[0]} samples")
        r = evaluate(X_sub, y_sub, f"LGBM-{subset_name}", lgb.LGBMClassifier(**best_lgbm_params))
        all_results[f"LGBM-{subset_name}"] = r

    # === 5. Multi-seed ensemble ===
    log("\n" + "=" * 60)
    log("  EXP 5: Multi-seed LGBM ensemble")
    log("=" * 60)

    multiseed_params = dict(
        n_estimators=1500, max_depth=12, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.7, num_leaves=255,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
        class_weight='balanced', n_jobs=-1, verbose=-1)

    r = evaluate_multiseed(X80, y80, "LGBM-5seed-ensemble", multiseed_params, n_seeds=5)
    all_results["LGBM-5seed-ensemble"] = r

    r = evaluate_multiseed(X80, y80, "LGBM-10seed-ensemble", multiseed_params, n_seeds=10)
    all_results["LGBM-10seed-ensemble"] = r

    # === 6. XGBoost fine-tuning on ultraclean ===
    log("\n" + "=" * 60)
    log("  EXP 6: XGBoost fine-tuning on ultra-clean")
    log("=" * 60)

    xgb_configs = [
        ("XGB-2000-d15-lr03", xgb.XGBClassifier(
            n_estimators=2000, max_depth=15, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, min_child_weight=3,
            gamma=0.05, reg_alpha=0.05, reg_lambda=0.5,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
        ("XGB-1500-d12-lr05", xgb.XGBClassifier(
            n_estimators=1500, max_depth=12, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)),
    ]

    for name, model in xgb_configs:
        r = evaluate(X80, y80, name, model)
        all_results[name] = r

    # === FINAL SUMMARY ===
    elapsed = time.time() - t_start
    log("\n" + "=" * 80)
    log("  FINAL RESULTS (sorted by F1)")
    log("  Baseline: LGBM-1500-ultraclean = 94.23%")
    log("=" * 80)
    log(f"\n{'Model':<38} {'Accuracy':>10} {'F1':>10} {'Time':>8}")
    log("-" * 72)

    for name, r in sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True):
        acc = r['accuracy']
        f1 = r['f1']
        t = r['time']
        marker = " ***" if acc > 0.9423 else " ++" if acc > 0.9400 else ""
        log(f"{name:<38} {acc:>9.4f} {f1:>9.4f} {t:>7.1f}s{marker}")

    best_name = max(all_results, key=lambda k: all_results[k]['f1'])
    best = all_results[best_name]
    log(f"\nBest: {best_name}")
    log(f"  Accuracy: {best['accuracy']:.4f}")
    log(f"  F1: {best['f1']:.4f}")
    log(f"  vs baseline: {(best['accuracy'] - 0.9423)*100:+.2f}%")
    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    benchmarks_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/benchmarks')
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    with open(benchmarks_dir / 'improvement_v4_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Results saved to {benchmarks_dir / 'improvement_v4_results.json'}")


if __name__ == '__main__':
    main()
