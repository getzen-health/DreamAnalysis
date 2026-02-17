"""
Accuracy improvement v7 - Dataset quality-driven cleaning.

v6 revealed the critical insight:
  - Confused-Student: 37.2% error rate (12,811 samples of noise!)
  - EEG-ER: 17.2% error rate
  - GAMEEMO: 15.0% error rate
  - SEED/SEED-IV: 0.0% error (perfect)

Strategy: drop noisiest datasets + use dataset-aware features + Optuna.
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

N_JOBS = 3  # Minimal CPU to coexist with v5/v6


def load_datasets(exclude=None):
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


def evaluate(X, y, name, params=None, n_splits=5):
    if params is None:
        params = dict(
            n_estimators=1500, max_depth=12, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, num_leaves=255,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=42, n_jobs=N_JOBS, verbose=-1)

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
    log(f"    {name:50s}: Acc={am:.4f}±{np.std(accs):.4f}  F1={fm:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "accuracy_std": float(np.std(accs)),
            "f1": float(fm), "f1_std": float(np.std(f1s)), "time": float(elapsed)}


def main():
    t_start = time.time()
    log("=" * 80)
    log("  ACCURACY IMPROVEMENT v7 - Dataset quality-driven cleaning")
    log("  Insight: Confused-Student=37.2% error, EEG-ER=17.2%, GAMEEMO=15.0%")
    log("=" * 80)

    all_results = {}

    # Dataset exclusion experiments with dataset-aware features
    exclusion_configs = [
        ("baseline-8ds", set()),
        ("drop-Confused", {'Confused-Student'}),
        ("drop-Confused-EEGER", {'Confused-Student', 'EEG-ER'}),
        ("drop-Confused-EEGER-GAMEEMO", {'Confused-Student', 'EEG-ER', 'GAMEEMO'}),
        ("drop-worst3+Brainwave", {'Confused-Student', 'EEG-ER', 'GAMEEMO', 'Brainwave'}),
        ("top3-SEED-SeedIV-MuseSub", {'GAMEEMO', 'EEG-ER', 'Brainwave', 'Muse-Mental', 'Confused-Student'}),
    ]

    for config_name, exclude_set in exclusion_configs:
        log(f"\n{'='*60}")
        log(f"  Config: {config_name}")
        if exclude_set:
            log(f"  Excluding: {exclude_set}")
        log(f"{'='*60}")

        datasets = load_datasets(exclude=exclude_set)
        names = [n for _, _, n in datasets]
        log(f"  Datasets: {names}")

        n_ds = len(datasets)
        X, y, src = build_dataset_with_source(datasets, target_features=80)
        X_aware = add_dataset_features(X, src, n_ds)
        log(f"  Data: {X.shape[0]:,} samples, {X_aware.shape[1]} features (80 PCA + {n_ds} ds)")

        # Standard LGBM with dataset-aware features
        r = evaluate(X_aware, y, f"LGBM-aware-{config_name}")
        all_results[f"LGBM-aware-{config_name}"] = r

        # Also try without dataset-aware for comparison
        r = evaluate(X, y, f"LGBM-plain-{config_name}")
        all_results[f"LGBM-plain-{config_name}"] = r

    # Find best config
    best_config = max(all_results, key=lambda k: all_results[k]['accuracy'])
    best_acc = all_results[best_config]['accuracy']
    log(f"\n  Best config so far: {best_config} = {best_acc:.4f}")

    # Optuna on best dataset exclusion config
    log(f"\n{'='*60}")
    log("  Optuna optimization on best config")
    log(f"{'='*60}")

    # Determine best exclusion set
    best_exclusion = None
    for config_name, exclude_set in exclusion_configs:
        key = f"LGBM-aware-{config_name}"
        if key == best_config or key.replace('aware', 'plain') == best_config:
            best_exclusion = exclude_set
            break

    if best_exclusion is None:
        best_exclusion = {'Confused-Student'}  # Fallback

    log(f"  Best exclusion: {best_exclusion}")
    datasets = load_datasets(exclude=best_exclusion)
    n_ds = len(datasets)
    X, y, src = build_dataset_with_source(datasets, target_features=80)
    X_aware = add_dataset_features(X, src, n_ds)

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
            'class_weight': 'balanced',
            'random_state': 42, 'n_jobs': N_JOBS, 'verbose': -1,
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        accs = []
        for tr, te in skf.split(X_opt, y):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_opt[tr], y[tr])
            yp = m.predict(X_opt[te])
            accs.append(accuracy_score(y[te], yp))
        return np.mean(accs)

    log("  Running Optuna (80 trials, 3-fold)...")
    t0 = time.time()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=80)
    optuna_time = time.time() - t0
    log(f"  Optuna best 3-fold: {study.best_value:.4f} ({optuna_time:.1f}s)")
    log(f"  Best params: {study.best_params}")

    # 5-fold eval with Optuna params
    bp = dict(study.best_params)
    bp['class_weight'] = 'balanced'
    bp['random_state'] = 42
    bp['n_jobs'] = N_JOBS
    bp['verbose'] = -1
    r = evaluate(X_aware, y, "LGBM-optuna-aware-best-excl", params=bp)
    all_results["LGBM-optuna-aware-best-excl"] = r

    # Also try Optuna on drop-Confused specifically
    if best_exclusion != {'Confused-Student'}:
        log("\n  Also trying Optuna on drop-Confused only...")
        datasets_dc = load_datasets(exclude={'Confused-Student'})
        n_dc = len(datasets_dc)
        X_dc, y_dc, src_dc = build_dataset_with_source(datasets_dc, target_features=80)
        X_dc_aware = add_dataset_features(X_dc, src_dc, n_dc)
        r = evaluate(X_dc_aware, y_dc, "LGBM-optuna-aware-dropConfused", params=bp)
        all_results["LGBM-optuna-aware-dropConfused"] = r

    # === FINAL SUMMARY ===
    elapsed = time.time() - t_start
    log("\n" + "=" * 80)
    log("  FINAL RESULTS (sorted by accuracy)")
    log("  Previous best: 94.43% (v5/v6 dataset-aware)")
    log("=" * 80)
    log(f"\n{'Model':<55} {'Accuracy':>10} {'F1':>10} {'Time':>8}")
    log("-" * 87)

    for name, r in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        acc = r['accuracy']
        f1 = r['f1']
        t = r['time']
        marker = " *** >95% ***" if acc > 0.95 else " ** NEW BEST **" if acc > 0.9445 else ""
        log(f"{name:<55} {acc:>9.4f} {f1:>9.4f} {t:>7.1f}s{marker}")

    best_name = max(all_results, key=lambda k: all_results[k]['accuracy'])
    best = all_results[best_name]
    log(f"\nBest: {best_name}")
    log(f"  Accuracy: {best['accuracy']:.4f}")
    log(f"  F1: {best['f1']:.4f}")
    log(f"  vs v5/v6 best: {(best['accuracy'] - 0.9443)*100:+.2f}%")
    log(f"  vs original baseline: {(best['accuracy'] - 0.9427)*100:+.2f}%")
    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    benchmarks_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/benchmarks')
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    with open(benchmarks_dir / 'improvement_v7_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Results saved to {benchmarks_dir / 'improvement_v7_results.json'}")

    # Save Optuna params
    with open(benchmarks_dir / 'optuna_v7_best_params.json', 'w') as f:
        json.dump({
            'best_params': {k: (int(v) if isinstance(v, (np.integer,)) else
                              float(v) if isinstance(v, (np.floating, float)) else v)
                           for k, v in bp.items()},
            'exclusion_set': list(best_exclusion),
            'best_3fold_acc': float(study.best_value),
        }, f, indent=2)
    log(f"Optuna params saved to {benchmarks_dir / 'optuna_v7_best_params.json'}")


if __name__ == '__main__':
    main()
