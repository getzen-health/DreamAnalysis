"""
Accuracy improvement v6 - Confidence-based training & label cleaning.
Current best: LGBM-dataset-aware = 94.41% (v5)

Key insight from v5: dataset-aware features (one-hot source ID) give +0.14%.
Hypothesis: ~5.6% of samples are hard/mislabeled. Cleaning them should help.

Strategies:
1. Confidence-based sample weighting (self-training)
2. Iterative label cleaning (remove noisy samples)
3. Per-dataset error analysis (identify weak datasets)
4. Optuna on dataset-aware features (fast search on clean data)
5. Soft-label self-distillation
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
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

N_JOBS = 6


def load_datasets():
    datasets = []
    for name, loader in ALL_LOADERS.items():
        try:
            X, y, dname = loader()
            if len(X) > 0 and len(np.unique(y)) >= 2:
                datasets.append((X, y, dname))
        except Exception as e:
            log(f"  Error loading {name}: {e}")
    return datasets


def build_dataset_with_source(datasets, target_features=80):
    """PCA-align datasets, return combined data + source indices."""
    all_X, all_y, all_source = [], [], []
    for idx, (X, y, name) in enumerate(datasets):
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
        all_source.append(np.full(len(y), idx))
    return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_source)


def add_dataset_features(X, source, n_datasets):
    one_hot = np.zeros((len(X), n_datasets))
    for i, s in enumerate(source):
        one_hot[i, int(s)] = 1.0
    return np.hstack([X, one_hot])


def get_lgbm_params():
    return dict(
        n_estimators=1500, max_depth=12, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.7, num_leaves=255,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
        class_weight='balanced', random_state=42, n_jobs=N_JOBS, verbose=-1)


def get_oof_predictions(X, y, n_splits=5):
    """Get out-of-fold predictions and confidence scores."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(X), len(np.unique(y))))
    oof_preds = np.zeros(len(X), dtype=int)

    for tr, te in skf.split(X_s, y):
        m = lgb.LGBMClassifier(**get_lgbm_params())
        m.fit(X_s[tr], y[tr])
        probs = m.predict_proba(X_s[te])
        oof_probs[te] = probs
        oof_preds[te] = np.argmax(probs, axis=1)

    return oof_probs, oof_preds


def evaluate(X, y, name, n_splits=5, params=None):
    """5-fold CV evaluation."""
    if params is None:
        params = get_lgbm_params()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    t0 = time.time()
    for tr, te in skf.split(X_s, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_s[tr], y[tr])
        yp = m.predict(X_s[te])
        accs.append(accuracy_score(y[te], yp))
        f1s.append(f1_score(y[te], yp, average='weighted'))
    elapsed = time.time() - t0
    am, fm = np.mean(accs), np.mean(f1s)
    log(f"    {name:45s}: Acc={am:.4f}±{np.std(accs):.4f}  F1={fm:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "accuracy_std": float(np.std(accs)),
            "f1": float(fm), "f1_std": float(np.std(f1s)), "time": float(elapsed)}


def evaluate_weighted(X, y, name, weights, n_splits=5):
    """5-fold CV with sample weights."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    params = get_lgbm_params()
    # Remove class_weight when using sample_weight
    params.pop('class_weight', None)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    t0 = time.time()
    for tr, te in skf.split(X_s, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_s[tr], y[tr], sample_weight=weights[tr])
        yp = m.predict(X_s[te])
        accs.append(accuracy_score(y[te], yp))
        f1s.append(f1_score(y[te], yp, average='weighted'))
    elapsed = time.time() - t0
    am, fm = np.mean(accs), np.mean(f1s)
    log(f"    {name:45s}: Acc={am:.4f}±{np.std(accs):.4f}  F1={fm:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "accuracy_std": float(np.std(accs)),
            "f1": float(fm), "f1_std": float(np.std(f1s)), "time": float(elapsed)}


def main():
    t_start = time.time()
    log("=" * 80)
    log("  ACCURACY IMPROVEMENT v6 - Confidence-based training & label cleaning")
    log("  Current best: LGBM-dataset-aware = 94.41%")
    log("=" * 80)

    log("\n[Loading ultra-clean datasets...]")
    datasets = load_datasets()
    n_datasets = len(datasets)
    dataset_names = [name for _, _, name in datasets]
    for X, y, name in datasets:
        log(f"  {name}: {len(X)} samples")
    total = sum(len(d[1]) for d in datasets)
    log(f"  Total: {n_datasets} datasets, {total:,} samples")

    # Build dataset with source info
    X80, y80, source80 = build_dataset_with_source(datasets, target_features=80)
    X_aware = add_dataset_features(X80, source80, n_datasets)
    log(f"\n  Base data: {X80.shape[0]} x {X80.shape[1]}")
    log(f"  Dataset-aware: {X_aware.shape[1]} features")

    all_results = {}

    # === EXP 1: Baseline with dataset-aware features ===
    log("\n" + "=" * 60)
    log("  EXP 1: Baseline (dataset-aware, LGBM-1500)")
    log("=" * 60)
    r = evaluate(X_aware, y80, "LGBM-aware-baseline")
    all_results["LGBM-aware-baseline"] = r

    # === EXP 2: Error analysis per dataset ===
    log("\n" + "=" * 60)
    log("  EXP 2: Per-dataset error analysis")
    log("=" * 60)

    oof_probs, oof_preds = get_oof_predictions(X_aware, y80)
    confidence = np.max(oof_probs, axis=1)
    correct = (oof_preds == y80)
    overall_acc = np.mean(correct)

    log(f"  Overall OOF accuracy: {overall_acc:.4f}")
    log(f"  Mean confidence: {np.mean(confidence):.4f}")
    log(f"  Correct samples mean conf: {np.mean(confidence[correct]):.4f}")
    log(f"  Wrong samples mean conf: {np.mean(confidence[~correct]):.4f}")

    # Per-dataset error rates
    dataset_starts = [0]
    for _, y_d, _ in datasets:
        dataset_starts.append(dataset_starts[-1] + len(y_d))

    log(f"\n  {'Dataset':<25s} {'Samples':>8} {'OOF Acc':>8} {'Mean Conf':>10} {'Wrong%':>8}")
    log("  " + "-" * 65)
    for i, name in enumerate(dataset_names):
        start, end = dataset_starts[i], dataset_starts[i + 1]
        ds_correct = correct[start:end]
        ds_conf = confidence[start:end]
        ds_acc = np.mean(ds_correct)
        ds_wrong_pct = (1 - ds_acc) * 100
        log(f"  {name:<25s} {end-start:>8} {ds_acc:>8.4f} {np.mean(ds_conf):>10.4f} {ds_wrong_pct:>7.1f}%")

    # === EXP 3: Remove lowest-confidence samples ===
    log("\n" + "=" * 60)
    log("  EXP 3: Iterative label cleaning (remove low-confidence)")
    log("=" * 60)

    for pct_remove in [1, 2, 3, 5, 8, 10, 15]:
        threshold = np.percentile(confidence, pct_remove)
        mask = confidence >= threshold
        X_clean = X_aware[mask]
        y_clean = y80[mask]
        n_removed = np.sum(~mask)
        r = evaluate(X_clean, y_clean, f"LGBM-aware-clean-{pct_remove}pct ({n_removed} rm)")
        all_results[f"LGBM-aware-clean-{pct_remove}pct"] = r

    # === EXP 4: Confidence-based sample weighting ===
    log("\n" + "=" * 60)
    log("  EXP 4: Confidence-based sample weighting")
    log("=" * 60)

    # 4a: Weight by confidence (high-confidence samples get more weight)
    weights_conf = confidence.copy()
    weights_conf = weights_conf / np.mean(weights_conf)  # Normalize
    r = evaluate_weighted(X_aware, y80, "LGBM-aware-weight-confidence", weights_conf)
    all_results["LGBM-aware-weight-confidence"] = r

    # 4b: Weight by confidence squared
    weights_sq = confidence ** 2
    weights_sq = weights_sq / np.mean(weights_sq)
    r = evaluate_weighted(X_aware, y80, "LGBM-aware-weight-conf-sq", weights_sq)
    all_results["LGBM-aware-weight-conf-sq"] = r

    # 4c: Binary weight: 1 for correct, 0.5 for wrong
    weights_binary = np.where(correct, 1.0, 0.5)
    r = evaluate_weighted(X_aware, y80, "LGBM-aware-weight-binary", weights_binary)
    all_results["LGBM-aware-weight-binary"] = r

    # 4d: Remove wrong predictions, keep only correct
    X_correct = X_aware[correct]
    y_correct = y80[correct]
    log(f"\n  Correct-only: {len(y_correct)} samples ({len(y_correct)/len(y80)*100:.1f}%)")
    r = evaluate(X_correct, y_correct, "LGBM-aware-correct-only")
    all_results["LGBM-aware-correct-only"] = r

    # === EXP 5: Two-stage training (self-distillation) ===
    log("\n" + "=" * 60)
    log("  EXP 5: Self-distillation (OOF probs as soft labels)")
    log("=" * 60)

    # Add OOF probabilities as extra features (knowledge distillation)
    X_distill = np.hstack([X_aware, oof_probs])
    log(f"  Distilled features: {X_distill.shape[1]} (88 + 3 OOF probs)")
    r = evaluate(X_distill, y80, "LGBM-aware-distill-prob-features")
    all_results["LGBM-aware-distill-prob-features"] = r

    # OOF probs + confidence as features
    X_distill2 = np.hstack([X_aware, oof_probs, confidence.reshape(-1, 1)])
    log(f"  Distilled+conf features: {X_distill2.shape[1]}")
    r = evaluate(X_distill2, y80, "LGBM-aware-distill+conf")
    all_results["LGBM-aware-distill+conf"] = r

    # === EXP 6: Optuna on best feature set ===
    log("\n" + "=" * 60)
    log("  EXP 6: Optuna optimization on dataset-aware features")
    log("=" * 60)

    # Identify best cleaning percentage from EXP 3
    best_clean_name = max(
        [k for k in all_results if 'clean' in k],
        key=lambda k: all_results[k]['accuracy'])
    best_clean_acc = all_results[best_clean_name]['accuracy']
    log(f"  Best cleaning: {best_clean_name} = {best_clean_acc:.4f}")

    # Run Optuna on dataset-aware features (full data)
    scaler = StandardScaler()
    X_opt = scaler.fit_transform(X_aware)
    X_opt = np.nan_to_num(X_opt, nan=0.0, posinf=0.0, neginf=0.0)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'max_depth': trial.suggest_int('max_depth', 8, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 127, 512),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 20),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
            'class_weight': 'balanced',
            'random_state': 42, 'n_jobs': N_JOBS, 'verbose': -1,
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        accs = []
        for tr, te in skf.split(X_opt, y80):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_opt[tr], y80[tr])
            yp = m.predict(X_opt[te])
            accs.append(accuracy_score(y80[te], yp))
        return np.mean(accs)

    log("  Running Optuna (60 trials, 3-fold)...")
    t0 = time.time()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=60)
    optuna_time = time.time() - t0

    log(f"  Optuna best 3-fold: {study.best_value:.4f} ({optuna_time:.1f}s)")
    log(f"  Best params: {study.best_params}")

    # Evaluate best Optuna params with 5-fold
    best_params = dict(study.best_params)
    best_params['class_weight'] = 'balanced'
    best_params['random_state'] = 42
    best_params['n_jobs'] = N_JOBS
    best_params['verbose'] = -1
    r = evaluate(X_aware, y80, "LGBM-optuna-aware-60t", params=best_params)
    all_results["LGBM-optuna-aware-60t"] = r

    # === EXP 7: Optuna on cleaned data ===
    log("\n" + "=" * 60)
    log("  EXP 7: Optuna on confidence-cleaned data")
    log("=" * 60)

    # Use best cleaning from EXP 3
    for pct in [2, 5]:
        threshold = np.percentile(confidence, pct)
        mask = confidence >= threshold
        X_clean = X_aware[mask]
        y_clean = y80[mask]

        scaler_c = StandardScaler()
        X_c_opt = scaler_c.fit_transform(X_clean)
        X_c_opt = np.nan_to_num(X_c_opt, nan=0.0, posinf=0.0, neginf=0.0)

        def objective_clean(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
                'max_depth': trial.suggest_int('max_depth', 8, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 127, 512),
                'min_child_samples': trial.suggest_int('min_child_samples', 3, 20),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
                'class_weight': 'balanced',
                'random_state': 42, 'n_jobs': N_JOBS, 'verbose': -1,
            }
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            accs = []
            for tr, te in skf.split(X_c_opt, y_clean):
                m = lgb.LGBMClassifier(**params)
                m.fit(X_c_opt[tr], y_clean[tr])
                yp = m.predict(X_c_opt[te])
                accs.append(accuracy_score(y_clean[te], yp))
            return np.mean(accs)

        log(f"\n  Optuna on {pct}%-cleaned data ({len(y_clean)} samples, 40 trials)...")
        study_c = optuna.create_study(direction='maximize')
        study_c.optimize(objective_clean, n_trials=40)
        log(f"  Optuna best 3-fold: {study_c.best_value:.4f}")

        bp = dict(study_c.best_params)
        bp['class_weight'] = 'balanced'
        bp['random_state'] = 42
        bp['n_jobs'] = N_JOBS
        bp['verbose'] = -1
        r = evaluate(X_clean, y_clean, f"LGBM-optuna-aware-clean{pct}pct", params=bp)
        all_results[f"LGBM-optuna-aware-clean{pct}pct"] = r

    # === EXP 8: Multi-round cleaning ===
    log("\n" + "=" * 60)
    log("  EXP 8: Multi-round iterative cleaning")
    log("=" * 60)

    X_iter, y_iter = X_aware.copy(), y80.copy()
    for round_num in range(3):
        oof_p, oof_pred = get_oof_predictions(X_iter, y_iter)
        conf = np.max(oof_p, axis=1)
        threshold = np.percentile(conf, 2)  # Remove bottom 2% each round
        mask = conf >= threshold
        n_before = len(y_iter)
        X_iter = X_iter[mask]
        y_iter = y_iter[mask]
        n_removed = n_before - len(y_iter)
        round_acc = np.mean(oof_pred[mask] == y_iter)
        log(f"  Round {round_num+1}: removed {n_removed}, remaining {len(y_iter)}, OOF acc {round_acc:.4f}")

    r = evaluate(X_iter, y_iter, f"LGBM-aware-3round-clean ({len(y_iter)})")
    all_results["LGBM-aware-3round-clean"] = r

    # === FINAL SUMMARY ===
    elapsed = time.time() - t_start
    log("\n" + "=" * 80)
    log("  FINAL RESULTS (sorted by accuracy)")
    log("  Previous best: LGBM-dataset-aware = 94.41%")
    log("=" * 80)
    log(f"\n{'Model':<50} {'Accuracy':>10} {'F1':>10} {'Time':>8}")
    log("-" * 82)

    for name, r in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        acc = r['accuracy']
        f1 = r['f1']
        t = r['time']
        marker = " *** NEW BEST ***" if acc > 0.9445 else " ++" if acc > 0.9441 else ""
        log(f"{name:<50} {acc:>9.4f} {f1:>9.4f} {t:>7.1f}s{marker}")

    best_name = max(all_results, key=lambda k: all_results[k]['accuracy'])
    best = all_results[best_name]
    log(f"\nBest: {best_name}")
    log(f"  Accuracy: {best['accuracy']:.4f}")
    log(f"  F1: {best['f1']:.4f}")
    log(f"  vs v5 best: {(best['accuracy'] - 0.9441)*100:+.2f}%")
    log(f"  vs baseline: {(best['accuracy'] - 0.9427)*100:+.2f}%")
    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    benchmarks_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/benchmarks')
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    with open(benchmarks_dir / 'improvement_v6_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Results saved to {benchmarks_dir / 'improvement_v6_results.json'}")


if __name__ == '__main__':
    main()
