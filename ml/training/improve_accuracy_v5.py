"""
Accuracy improvement v5 - Breaking the 94.27% plateau.
Current best: LGBM-1500-ultraclean = 94.27% (v4 confirmed plateau)

v4 showed: standard LGBM hyperparameter tuning is saturated.
v5 uses fundamentally different strategies:

1. Dataset-aware features (add dataset source as features)
2. Stacking ensemble (LGBM + XGB + RF meta-learner)
3. Feature augmentation (interactions, squared features)
4. Optuna Bayesian hyperparameter optimization
5. Combined: best features + best model + stacking
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
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

N_JOBS = 4  # Leave cores for v4 still running


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
    """PCA-align and combine datasets, adding dataset source ID as features."""
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

    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    source = np.concatenate(all_source)
    return X_combined, y_combined, source


def add_dataset_features(X, source, n_datasets):
    """Add one-hot encoded dataset source + dataset-specific stats."""
    # One-hot encoding of dataset source
    one_hot = np.zeros((len(X), n_datasets))
    for i, s in enumerate(source):
        one_hot[i, int(s)] = 1.0
    return np.hstack([X, one_hot])


def add_interaction_features(X, top_k=20):
    """Add squared and interaction features for top-k PCA components."""
    X_top = X[:, :top_k]
    # Squared features
    X_sq = X_top ** 2
    # Pairwise interactions for top 10
    interactions = []
    for i in range(min(10, top_k)):
        for j in range(i + 1, min(10, top_k)):
            interactions.append(X[:, i] * X[:, j])
    X_inter = np.column_stack(interactions) if interactions else np.zeros((len(X), 0))
    return np.hstack([X, X_sq, X_inter])


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
    log(f"    {name:40s}: Acc={am:.4f}±{np.std(accs):.4f}  F1={fm:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "accuracy_std": float(np.std(accs)),
            "f1": float(fm), "f1_std": float(np.std(f1s)), "time": float(elapsed)}


def stacking_evaluate(X, y, name, n_splits=5):
    """Stacking ensemble: LGBM + XGB + RF base → LGBM meta-learner."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    n_classes = len(np.unique(y))

    base_models = [
        ("lgbm", lgb.LGBMClassifier(
            n_estimators=1500, max_depth=12, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, num_leaves=255,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            class_weight='balanced', random_state=42, n_jobs=N_JOBS, verbose=-1)),
        ("xgb", xgb.XGBClassifier(
            n_estimators=1500, max_depth=12, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.7, min_child_weight=3,
            tree_method='hist', random_state=42, n_jobs=N_JOBS, verbosity=0)),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_depth=20, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=N_JOBS)),
    ]

    outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    t0 = time.time()

    for outer_tr, outer_te in outer_skf.split(X_s, y):
        X_train, y_train = X_s[outer_tr], y[outer_tr]
        X_test, y_test = X_s[outer_te], y[outer_te]

        # Generate OOF predictions from base models using inner CV
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_train = np.zeros((len(X_train), n_classes * len(base_models)))
        test_preds = np.zeros((len(X_test), n_classes * len(base_models)))

        for midx, (mname, model) in enumerate(base_models):
            col_start = midx * n_classes
            col_end = (midx + 1) * n_classes

            fold_test_preds = []
            for inn_tr, inn_val in inner_skf.split(X_train, y_train):
                m = type(model)(**model.get_params())
                m.fit(X_train[inn_tr], y_train[inn_tr])
                oof_train[inn_val, col_start:col_end] = m.predict_proba(X_train[inn_val])
                fold_test_preds.append(m.predict_proba(X_test))

            # Average test predictions across inner folds
            test_preds[:, col_start:col_end] = np.mean(fold_test_preds, axis=0)

        # Stack: combine OOF predictions with original features
        meta_train = np.hstack([X_train, oof_train])
        meta_test = np.hstack([X_test, test_preds])

        # Train meta-learner
        meta_model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=63, class_weight='balanced',
            random_state=42, n_jobs=N_JOBS, verbose=-1)
        meta_model.fit(meta_train, y_train)
        yp = meta_model.predict(meta_test)

        accs.append(accuracy_score(y_test, yp))
        f1s.append(f1_score(y_test, yp, average='weighted'))

    elapsed = time.time() - t0
    am, fm = np.mean(accs), np.mean(f1s)
    log(f"    {name:40s}: Acc={am:.4f}±{np.std(accs):.4f}  F1={fm:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "accuracy_std": float(np.std(accs)),
            "f1": float(fm), "f1_std": float(np.std(f1s)), "time": float(elapsed)}


def optuna_lgbm_search(X, y, n_trials=50, n_splits=3):
    """Bayesian hyperparameter optimization with Optuna."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
            'max_depth': trial.suggest_int('max_depth', 8, 25),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 63, 1024),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 30),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': N_JOBS,
            'verbose': -1,
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = []
        for tr, te in skf.split(X_s, y):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_s[tr], y[tr])
            yp = m.predict(X_s[te])
            accs.append(accuracy_score(y[te], yp))
        return np.mean(accs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log(f"    Optuna best: {study.best_value:.4f} in {n_trials} trials")
    log(f"    Best params: {study.best_params}")
    return study.best_params, study.best_value


def evaluate_with_params(X, y, name, params, n_splits=5):
    """Evaluate LGBM with specific params using 5-fold CV."""
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
    log(f"    {name:40s}: Acc={am:.4f}±{np.std(accs):.4f}  F1={fm:.4f}±{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "accuracy_std": float(np.std(accs)),
            "f1": float(fm), "f1_std": float(np.std(f1s)), "time": float(elapsed)}


def main():
    t_start = time.time()
    log("=" * 80)
    log("  ACCURACY IMPROVEMENT v5 - Breaking the 94.27% plateau")
    log("  v4 confirmed: LGBM hyperparameter tuning saturated at 94.27%")
    log("  Strategy: fundamentally different approaches")
    log("=" * 80)

    log("\n[Loading ultra-clean datasets...]")
    datasets = load_datasets()
    n_datasets = len(datasets)
    for X, y, name in datasets:
        log(f"  {name}: {len(X)} samples, {X.shape[1]} features")
    total = sum(len(d[1]) for d in datasets)
    log(f"  Total: {n_datasets} datasets, {total:,} samples")

    # Build base dataset
    X80, y80, source80 = build_dataset_with_source(datasets, target_features=80)
    log(f"\n  Base data: {X80.shape[0]} x {X80.shape[1]}")

    all_results = {}
    best_lgbm = lgb.LGBMClassifier(
        n_estimators=1500, max_depth=12, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.7, num_leaves=255,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
        class_weight='balanced', random_state=42, n_jobs=N_JOBS, verbose=-1)

    # === EXP 1: Baseline on PCA=80 ===
    log("\n" + "=" * 60)
    log("  EXP 1: Baseline (PCA=80, LGBM-1500)")
    log("=" * 60)
    r = evaluate(X80, y80, "LGBM-baseline-pca80", best_lgbm)
    all_results["LGBM-baseline-pca80"] = r

    # === EXP 2: Dataset-aware features ===
    log("\n" + "=" * 60)
    log("  EXP 2: Dataset-aware features (one-hot dataset source)")
    log("=" * 60)
    X_aware = add_dataset_features(X80, source80, n_datasets)
    log(f"  Features: {X80.shape[1]} + {n_datasets} dataset IDs = {X_aware.shape[1]}")
    r = evaluate(X_aware, y80, "LGBM-dataset-aware", best_lgbm)
    all_results["LGBM-dataset-aware"] = r

    # === EXP 3: Feature augmentation ===
    log("\n" + "=" * 60)
    log("  EXP 3: Feature augmentation (interactions + squared)")
    log("=" * 60)

    # 3a: Interaction features only
    X_inter = add_interaction_features(X80, top_k=20)
    log(f"  Interactions: {X80.shape[1]} → {X_inter.shape[1]} features")
    r = evaluate(X_inter, y80, "LGBM-interactions-top20", best_lgbm)
    all_results["LGBM-interactions-top20"] = r

    # 3b: Dataset-aware + interactions
    X_all_feat = add_interaction_features(X_aware, top_k=20)
    log(f"  All features: {X_all_feat.shape[1]} features")
    r = evaluate(X_all_feat, y80, "LGBM-aware+interactions", best_lgbm)
    all_results["LGBM-aware+interactions"] = r

    # === EXP 4: Higher PCA with feature selection ===
    log("\n" + "=" * 60)
    log("  EXP 4: PCA=150 + LGBM feature importance selection")
    log("=" * 60)

    X150, y150, src150 = build_dataset_with_source(datasets, target_features=150)
    log(f"  PCA=150 data: {X150.shape}")

    # First pass: train LGBM on all 150 features, get importance
    scaler_tmp = StandardScaler()
    X_tmp = scaler_tmp.fit_transform(X150)
    X_tmp = np.nan_to_num(X_tmp, nan=0.0, posinf=0.0, neginf=0.0)
    tmp_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=12, learning_rate=0.05,
        num_leaves=255, class_weight='balanced', random_state=42,
        n_jobs=N_JOBS, verbose=-1)
    tmp_model.fit(X_tmp, y150)
    importance = tmp_model.feature_importances_
    top_idx = np.argsort(importance)[::-1]

    for n_select in [80, 100, 120]:
        selected = top_idx[:n_select]
        X_sel = X150[:, selected]
        r = evaluate(X_sel, y150, f"LGBM-pca150-select{n_select}", best_lgbm)
        all_results[f"LGBM-pca150-select{n_select}"] = r

    # Also try PCA=150 + dataset-aware
    X150_aware = add_dataset_features(X150, src150, n_datasets)
    log(f"  PCA=150 + dataset-aware: {X150_aware.shape[1]} features")
    r = evaluate(X150_aware, y150, "LGBM-pca150-dataset-aware", best_lgbm)
    all_results["LGBM-pca150-dataset-aware"] = r

    # === EXP 5: Stacking ensemble ===
    log("\n" + "=" * 60)
    log("  EXP 5: Stacking ensemble (LGBM + XGB + RF → meta-LGBM)")
    log("=" * 60)

    # 5a: Stack on base PCA=80
    r = stacking_evaluate(X80, y80, "Stack-base-pca80")
    all_results["Stack-base-pca80"] = r

    # 5b: Stack on dataset-aware features
    r = stacking_evaluate(X_aware, y80, "Stack-dataset-aware")
    all_results["Stack-dataset-aware"] = r

    # === EXP 6: Optuna optimization ===
    log("\n" + "=" * 60)
    log("  EXP 6: Optuna Bayesian hyperparameter optimization")
    log("=" * 60)

    # Find best feature set so far
    best_exp_name = max(all_results, key=lambda k: all_results[k]['accuracy'])
    best_exp_acc = all_results[best_exp_name]['accuracy']
    log(f"  Best so far: {best_exp_name} = {best_exp_acc:.4f}")

    # 6a: Optuna on base PCA=80 (50 trials, 3-fold for speed)
    log(f"\n  Running Optuna on PCA=80 (50 trials, 3-fold)...")
    t0 = time.time()
    best_params, best_val = optuna_lgbm_search(X80, y80, n_trials=50, n_splits=3)
    optuna_time = time.time() - t0
    log(f"  Optuna search took {optuna_time:.1f}s")

    # Evaluate Optuna best with 5-fold
    optuna_params = dict(best_params)
    optuna_params['class_weight'] = 'balanced'
    optuna_params['random_state'] = 42
    optuna_params['n_jobs'] = N_JOBS
    optuna_params['verbose'] = -1
    r = evaluate_with_params(X80, y80, "LGBM-optuna-pca80", optuna_params)
    r['optuna_time'] = optuna_time
    all_results["LGBM-optuna-pca80"] = r

    # 6b: Optuna on dataset-aware features
    log(f"\n  Running Optuna on dataset-aware features (50 trials)...")
    t0 = time.time()
    best_params_aware, best_val_aware = optuna_lgbm_search(X_aware, y80, n_trials=50, n_splits=3)
    optuna_time2 = time.time() - t0
    log(f"  Optuna search took {optuna_time2:.1f}s")

    optuna_params_aware = dict(best_params_aware)
    optuna_params_aware['class_weight'] = 'balanced'
    optuna_params_aware['random_state'] = 42
    optuna_params_aware['n_jobs'] = N_JOBS
    optuna_params_aware['verbose'] = -1
    r = evaluate_with_params(X_aware, y80, "LGBM-optuna-aware", optuna_params_aware)
    r['optuna_time'] = optuna_time2
    all_results["LGBM-optuna-aware"] = r

    # === EXP 7: Combined best ===
    log("\n" + "=" * 60)
    log("  EXP 7: Combined - best features + stacking + Optuna params")
    log("=" * 60)

    # Use best Optuna params on best feature set
    # Try stacking on interaction features
    log("  Stacking on interaction features...")
    r = stacking_evaluate(X_inter, y80, "Stack-interactions")
    all_results["Stack-interactions"] = r

    # Try stacking on PCA=150 + dataset-aware
    log("  Stacking on PCA=150 + dataset-aware...")
    r = stacking_evaluate(X150_aware, y150, "Stack-pca150-aware")
    all_results["Stack-pca150-aware"] = r

    # Optuna-optimized model on interactions + dataset-aware
    X_best_feat = add_interaction_features(X_aware, top_k=15)
    log(f"  Best feature combo: {X_best_feat.shape[1]} features")
    r = evaluate_with_params(X_best_feat, y80, "LGBM-optuna-best-features", optuna_params_aware)
    all_results["LGBM-optuna-best-features"] = r

    # === FINAL SUMMARY ===
    elapsed = time.time() - t_start
    log("\n" + "=" * 80)
    log("  FINAL RESULTS (sorted by accuracy)")
    log("  Baseline: LGBM-1500-ultraclean = 94.27%")
    log("=" * 80)
    log(f"\n{'Model':<43} {'Accuracy':>10} {'F1':>10} {'Time':>8}")
    log("-" * 76)

    for name, r in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        acc = r['accuracy']
        f1 = r['f1']
        t = r['time']
        marker = " *** NEW BEST ***" if acc > 0.9430 else " ++" if acc > 0.9427 else ""
        log(f"{name:<43} {acc:>9.4f} {f1:>9.4f} {t:>7.1f}s{marker}")

    best_name = max(all_results, key=lambda k: all_results[k]['accuracy'])
    best = all_results[best_name]
    log(f"\nBest: {best_name}")
    log(f"  Accuracy: {best['accuracy']:.4f}")
    log(f"  F1: {best['f1']:.4f}")
    log(f"  vs baseline: {(best['accuracy'] - 0.9427)*100:+.2f}%")
    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    benchmarks_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/benchmarks')
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    results_file = benchmarks_dir / 'improvement_v5_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Results saved to {results_file}")

    # Save Optuna best params
    optuna_file = benchmarks_dir / 'optuna_best_params.json'
    with open(optuna_file, 'w') as f:
        json.dump({
            'pca80_params': {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating, float)) else v) for k, v in optuna_params.items()},
            'aware_params': {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating, float)) else v) for k, v in optuna_params_aware.items()},
            'pca80_3fold_acc': float(best_val),
            'aware_3fold_acc': float(best_val_aware),
        }, f, indent=2)
    log(f"Optuna params saved to {optuna_file}")


if __name__ == '__main__':
    main()
