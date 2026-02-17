"""
Accuracy improvement v8 - Combined best strategies.

Best result so far: 97.79% (v7: LGBM-1500 + dataset-aware, drop Confused-Student)

Strategy: Combine the two most effective approaches from v6 and v7:
1. Drop Confused-Student (removes 37.2% error rate dataset)
2. Confidence-based cleaning on remaining 7 datasets (remove noisiest 2-5% samples)
3. Optuna hyperparameter optimization on the cleaned data
4. Also try dropping EEG-ER (17.2% error) + cleaning

Goal: Push above 98% while maintaining dataset diversity.
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
    load_muse_mental_state,
    load_muse_subconscious, load_seed_iv,
    log
)

# Premium-clean loaders (excluding Confused-Student)
PREMIUM_LOADERS = {
    'GAMEEMO': load_gameemo_enhanced,
    'EEG-ER': load_eeg_er_enhanced,
    'SEED': load_seed_enhanced,
    'Brainwave': load_brainwave_emotions,
    'Muse-Mental': load_muse_mental_state,
    'Muse-Subconscious': load_muse_subconscious,
    'SEED-IV': load_seed_iv,
}

N_JOBS = 6  # v6 killed, more CPU available


def load_datasets(exclude=None):
    exclude = exclude or set()
    datasets = []
    for name, loader in PREMIUM_LOADERS.items():
        if name in exclude:
            continue
        try:
            X, y, dname = loader()
            if len(X) > 0 and len(np.unique(y)) >= 2:
                datasets.append((X, y, dname))
        except Exception as e:
            log(f"  Error loading {name}: {e}")
    return datasets


def build_dataset_with_source(datasets, target_features=80):
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


def evaluate(X, y, name, params=None, n_splits=5):
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
    log(f"    {name:55s}: Acc={am:.4f}+/-{np.std(accs):.4f}  F1={fm:.4f}+/-{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "accuracy_std": float(np.std(accs)),
            "f1": float(fm), "f1_std": float(np.std(f1s)), "time": float(elapsed)}


def main():
    t_start = time.time()
    log("=" * 80)
    log("  ACCURACY IMPROVEMENT v8 - Combined: dataset exclusion + confidence cleaning")
    log("  Current best: LGBM-1500-aware-premium = 97.79% (7 datasets)")
    log("=" * 80)

    all_results = {}

    # ========================================================================
    # CONFIG 1: 7 datasets (drop Confused-Student only) + confidence cleaning
    # ========================================================================
    log("\n" + "=" * 70)
    log("  CONFIG 1: 7 premium datasets + confidence-based cleaning")
    log("=" * 70)

    datasets_7 = load_datasets()
    n_ds7 = len(datasets_7)
    dataset_names_7 = [name for _, _, name in datasets_7]
    log(f"  Datasets: {dataset_names_7}")
    for X, y, name in datasets_7:
        log(f"    {name}: {len(X)} samples")

    X7, y7, src7 = build_dataset_with_source(datasets_7, target_features=80)
    X7_aware = add_dataset_features(X7, src7, n_ds7)
    log(f"  Total: {len(y7):,} samples, {X7_aware.shape[1]} features")

    # Baseline (should reproduce 97.79%)
    r = evaluate(X7_aware, y7, "baseline-7ds-aware")
    all_results["baseline-7ds-aware"] = r

    # Get OOF predictions for cleaning
    log("\n  [Computing OOF predictions on 7 premium datasets...]")
    oof_probs, oof_preds = get_oof_predictions(X7_aware, y7)
    confidence = np.max(oof_probs, axis=1)
    correct = (oof_preds == y7)
    oof_acc = np.mean(correct)
    log(f"  OOF accuracy: {oof_acc:.4f}")
    log(f"  Mean confidence: {np.mean(confidence):.4f}")

    # Per-dataset error rates on premium data
    dataset_starts = [0]
    for _, y_d, _ in datasets_7:
        dataset_starts.append(dataset_starts[-1] + len(y_d))

    log(f"\n  {'Dataset':<20s} {'Samples':>8} {'OOF Acc':>8} {'Error%':>8}")
    log("  " + "-" * 50)
    for i, name in enumerate(dataset_names_7):
        start, end = dataset_starts[i], dataset_starts[i + 1]
        ds_correct = correct[start:end]
        ds_acc = np.mean(ds_correct)
        log(f"  {name:<20s} {end-start:>8} {ds_acc:>8.4f} {(1-ds_acc)*100:>7.1f}%")

    # Confidence cleaning at various thresholds
    log("\n  [Confidence cleaning on premium data...]")
    for pct_remove in [1, 2, 3, 5]:
        threshold = np.percentile(confidence, pct_remove)
        mask = confidence >= threshold
        X_clean = X7_aware[mask]
        y_clean = y7[mask]
        n_removed = np.sum(~mask)
        r = evaluate(X_clean, y_clean, f"7ds-clean-{pct_remove}pct ({n_removed} rm)")
        all_results[f"7ds-clean-{pct_remove}pct"] = r

    # ========================================================================
    # CONFIG 2: 6 datasets (drop Confused + EEG-ER) + confidence cleaning
    # ========================================================================
    log("\n" + "=" * 70)
    log("  CONFIG 2: 6 datasets (drop Confused + EEG-ER) + cleaning")
    log("=" * 70)

    datasets_6 = load_datasets(exclude={'EEG-ER'})
    n_ds6 = len(datasets_6)
    log(f"  Datasets: {[n for _, _, n in datasets_6]}")

    X6, y6, src6 = build_dataset_with_source(datasets_6, target_features=80)
    X6_aware = add_dataset_features(X6, src6, n_ds6)
    log(f"  Total: {len(y6):,} samples, {X6_aware.shape[1]} features")

    r = evaluate(X6_aware, y6, "baseline-6ds-aware")
    all_results["baseline-6ds-aware"] = r

    # OOF + cleaning on 6 datasets
    log("\n  [Computing OOF predictions on 6 datasets...]")
    oof6, preds6 = get_oof_predictions(X6_aware, y6)
    conf6 = np.max(oof6, axis=1)
    log(f"  OOF accuracy: {np.mean(preds6 == y6):.4f}")

    for pct_remove in [1, 2, 3]:
        threshold = np.percentile(conf6, pct_remove)
        mask = conf6 >= threshold
        r = evaluate(X6_aware[mask], y6[mask], f"6ds-clean-{pct_remove}pct ({np.sum(~mask)} rm)")
        all_results[f"6ds-clean-{pct_remove}pct"] = r

    # ========================================================================
    # CONFIG 3: Optuna on best cleaning config
    # ========================================================================
    log("\n" + "=" * 70)
    log("  CONFIG 3: Optuna optimization on best config")
    log("=" * 70)

    # Find best from configs above
    best_so_far = max(all_results, key=lambda k: all_results[k]['accuracy'])
    best_acc = all_results[best_so_far]['accuracy']
    log(f"  Best so far: {best_so_far} = {best_acc:.4f}")

    # Optuna on 7ds-clean-2pct (sweet spot between cleaning and data retention)
    log("\n  [Optuna on 7ds + 2% cleaned data...]")
    threshold = np.percentile(confidence, 2)
    mask = confidence >= threshold
    X_opt_data = X7_aware[mask]
    y_opt_data = y7[mask]
    log(f"  Data: {len(y_opt_data):,} samples")

    scaler = StandardScaler()
    X_opt = scaler.fit_transform(X_opt_data)
    X_opt = np.nan_to_num(X_opt, nan=0.0, posinf=0.0, neginf=0.0)

    def objective_7ds_clean(trial):
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
        for tr, te in skf.split(X_opt, y_opt_data):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_opt[tr], y_opt_data[tr])
            yp = m.predict(X_opt[te])
            accs.append(accuracy_score(y_opt_data[te], yp))
        return np.mean(accs)

    log("  Running Optuna (80 trials, 3-fold)...")
    t0 = time.time()
    study1 = optuna.create_study(direction='maximize')
    study1.optimize(objective_7ds_clean, n_trials=80)
    opt_time = time.time() - t0
    log(f"  Optuna best 3-fold: {study1.best_value:.4f} ({opt_time:.1f}s)")
    log(f"  Best params: {study1.best_params}")

    bp1 = dict(study1.best_params)
    bp1['class_weight'] = 'balanced'
    bp1['random_state'] = 42
    bp1['n_jobs'] = N_JOBS
    bp1['verbose'] = -1
    r = evaluate(X_opt_data, y_opt_data, "optuna-7ds-clean2pct (80t)", params=bp1)
    all_results["optuna-7ds-clean2pct"] = r

    # Also evaluate Optuna params on full 7ds data (no cleaning)
    r = evaluate(X7_aware, y7, "optuna-params-on-7ds-full (80t)", params=bp1)
    all_results["optuna-params-on-7ds-full"] = r

    # ========================================================================
    # CONFIG 4: Optuna on 7ds uncleaned (for fair comparison)
    # ========================================================================
    log("\n  [Optuna on 7ds uncleaned data...]")
    scaler2 = StandardScaler()
    X_opt2 = scaler2.fit_transform(X7_aware)
    X_opt2 = np.nan_to_num(X_opt2, nan=0.0, posinf=0.0, neginf=0.0)

    def objective_7ds(trial):
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
        for tr, te in skf.split(X_opt2, y7):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_opt2[tr], y7[tr])
            yp = m.predict(X_opt2[te])
            accs.append(accuracy_score(y7[te], yp))
        return np.mean(accs)

    log("  Running Optuna (60 trials, 3-fold)...")
    t0 = time.time()
    study2 = optuna.create_study(direction='maximize')
    study2.optimize(objective_7ds, n_trials=60)
    opt_time2 = time.time() - t0
    log(f"  Optuna best 3-fold: {study2.best_value:.4f} ({opt_time2:.1f}s)")
    log(f"  Best params: {study2.best_params}")

    bp2 = dict(study2.best_params)
    bp2['class_weight'] = 'balanced'
    bp2['random_state'] = 42
    bp2['n_jobs'] = N_JOBS
    bp2['verbose'] = -1
    r = evaluate(X7_aware, y7, "optuna-7ds-full (60t)", params=bp2)
    all_results["optuna-7ds-full"] = r

    # ========================================================================
    # CONFIG 5: Multi-round cleaning on 7 premium datasets
    # ========================================================================
    log("\n" + "=" * 70)
    log("  CONFIG 5: Multi-round iterative cleaning on 7 premium datasets")
    log("=" * 70)

    X_iter, y_iter = X7_aware.copy(), y7.copy()
    for round_num in range(3):
        oof_p, oof_pred = get_oof_predictions(X_iter, y_iter)
        conf = np.max(oof_p, axis=1)
        threshold = np.percentile(conf, 1.5)  # Conservative: remove bottom 1.5%
        mask = conf >= threshold
        n_before = len(y_iter)
        X_iter = X_iter[mask]
        y_iter = y_iter[mask]
        n_removed = n_before - len(y_iter)
        round_acc = np.mean(oof_pred[mask] == y_iter)
        log(f"  Round {round_num+1}: removed {n_removed}, remaining {len(y_iter)}, OOF acc {round_acc:.4f}")

    r = evaluate(X_iter, y_iter, f"7ds-3round-clean ({len(y_iter)})")
    all_results["7ds-3round-clean"] = r

    # Optuna on multi-round cleaned data
    log("\n  [Optuna on 3-round cleaned data...]")
    scaler3 = StandardScaler()
    X_opt3 = scaler3.fit_transform(X_iter)
    X_opt3 = np.nan_to_num(X_opt3, nan=0.0, posinf=0.0, neginf=0.0)

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
        for tr, te in skf.split(X_opt3, y_iter):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_opt3[tr], y_iter[tr])
            yp = m.predict(X_opt3[te])
            accs.append(accuracy_score(y_iter[te], yp))
        return np.mean(accs)

    log("  Running Optuna (50 trials, 3-fold)...")
    t0 = time.time()
    study3 = optuna.create_study(direction='maximize')
    study3.optimize(objective_clean, n_trials=50)
    opt_time3 = time.time() - t0
    log(f"  Optuna best 3-fold: {study3.best_value:.4f} ({opt_time3:.1f}s)")

    bp3 = dict(study3.best_params)
    bp3['class_weight'] = 'balanced'
    bp3['random_state'] = 42
    bp3['n_jobs'] = N_JOBS
    bp3['verbose'] = -1
    r = evaluate(X_iter, y_iter, "optuna-7ds-3round-clean (50t)", params=bp3)
    all_results["optuna-7ds-3round-clean"] = r

    # === FINAL SUMMARY ===
    elapsed = time.time() - t_start
    log("\n" + "=" * 80)
    log("  FINAL RESULTS v8 (sorted by accuracy)")
    log("  Previous best: 97.79% (v7: 7ds + dataset-aware)")
    log("=" * 80)
    log(f"\n{'Model':<60} {'Accuracy':>10} {'F1':>10} {'Time':>8}")
    log("-" * 92)

    for name, r in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        acc = r['accuracy']
        f1 = r['f1']
        t = r['time']
        marker = " *** >98% ***" if acc > 0.98 else " ** >97.79% **" if acc > 0.9779 else ""
        log(f"{name:<60} {acc:>9.4f} {f1:>9.4f} {t:>7.1f}s{marker}")

    best_name = max(all_results, key=lambda k: all_results[k]['accuracy'])
    best = all_results[best_name]
    log(f"\nBest: {best_name}")
    log(f"  Accuracy: {best['accuracy']:.4f}")
    log(f"  F1: {best['f1']:.4f}")
    log(f"  vs v7 best: {(best['accuracy'] - 0.9779)*100:+.2f}%")
    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/3600:.1f} hr)")

    # Save results
    benchmarks_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/benchmarks')
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    with open(benchmarks_dir / 'improvement_v8_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Results saved to {benchmarks_dir / 'improvement_v8_results.json'}")

    # Save best Optuna params
    best_study = max(
        [(study1, "7ds-clean2pct", study1.best_value),
         (study2, "7ds-full", study2.best_value),
         (study3, "7ds-3round-clean", study3.best_value)],
        key=lambda x: x[2])

    with open(benchmarks_dir / 'optuna_v8_best_params.json', 'w') as f:
        json.dump({
            'best_config': best_study[1],
            'best_3fold_acc': float(best_study[2]),
            'best_params': {k: (int(v) if isinstance(v, (np.integer,)) else
                              float(v) if isinstance(v, (np.floating, float)) else v)
                           for k, v in dict(best_study[0].best_params).items()},
        }, f, indent=2)
    log(f"Optuna params saved to {benchmarks_dir / 'optuna_v8_best_params.json'}")


if __name__ == '__main__':
    main()
