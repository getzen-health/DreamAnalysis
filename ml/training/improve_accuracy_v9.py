"""
Accuracy improvement v9 - Robust model that learns from ALL data (including noise).

Philosophy: Don't remove noisy data. The model should learn to handle outliers.
Use noise-robust techniques: label smoothing, mixup, dropout, sample weighting.

Also: Build a much more EFFICIENT model:
  - PyTorch MLP (~1MB) vs LightGBM (102MB)
  - Fast inference
  - Good generalization to unseen noisy data

Includes ALL 8+ datasets (no exclusions except truly incompatible ones).
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from mega_trainer import (
    load_gameemo_enhanced, load_eeg_er_enhanced,
    load_seed_enhanced, load_brainwave_emotions,
    load_muse_mental_state, load_confused_student,
    load_muse_subconscious, load_seed_iv,
    load_dreamer,
    load_stew,
    log
)

# ALL datasets (no exclusions - the model must learn from noisy data)
ALL_LOADERS = {
    'GAMEEMO': load_gameemo_enhanced,
    'EEG-ER': load_eeg_er_enhanced,
    'SEED': load_seed_enhanced,
    'Brainwave': load_brainwave_emotions,
    'Muse-Mental': load_muse_mental_state,
    'Confused-Student': load_confused_student,
    'Muse-Subconscious': load_muse_subconscious,
    'SEED-IV': load_seed_iv,
    'DREAMER': load_dreamer,
    'STEW': load_stew,
}

N_JOBS = 8
DEVICE = 'cpu'  # MPS can hang on some operations; CPU is more reliable


def load_datasets():
    datasets = []
    for name, loader in ALL_LOADERS.items():
        try:
            X, y, dname = loader()
            if len(X) > 0 and len(np.unique(y)) >= 2:
                datasets.append((X, y, dname))
                log(f"  {dname}: {len(X)} samples, {X.shape[1]} features, classes={np.unique(y).tolist()}")
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


# ========================================================================
# Efficient PyTorch MLP with noise-robust training
# ========================================================================
class EfficientEEGNet(nn.Module):
    """Compact MLP for EEG emotion classification.
    ~200KB model size vs 102MB LightGBM.
    """
    def __init__(self, n_features, n_classes=3, hidden_dims=(256, 128, 64)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.3),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LargerEEGNet(nn.Module):
    """Larger MLP with residual connections for better learning from noisy data."""
    def __init__(self, n_features, n_classes=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.block1 = self._make_block(512, 512)
        self.block2 = self._make_block(512, 256)
        self.block3 = self._make_block(256, 128)
        self.head = nn.Linear(128, n_classes)

    def _make_block(self, in_dim, out_dim):
        return nn.ModuleDict({
            'main': nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(0.25),
            ),
            'skip': nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity(),
        })

    def _forward_block(self, block, x):
        return block['main'](x) + block['skip'](x)

    def forward(self, x):
        x = self.input_proj(x)
        x = self._forward_block(self.block1, x)
        x = self._forward_block(self.block2, x)
        x = self._forward_block(self.block3, x)
        return self.head(x)


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for noise robustness."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def train_pytorch_model(X_train, y_train, X_val, y_val, model_class, n_features, n_classes=3,
                        epochs=100, lr=0.001, batch_size=512, label_smoothing=0.1, use_mixup=True):
    """Train PyTorch model with noise-robust techniques."""
    model = model_class(n_features, n_classes).to(DEVICE)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train.astype(int), minlength=n_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    weight_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.LongTensor(y_train.astype(int)).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    best_val_acc = 0
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            if use_mixup and np.random.random() < 0.5:
                xb_mix, ya, yb_mix, lam = mixup_data(xb, yb)
                out = model(xb_mix)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb_mix)
            else:
                out = model(xb)
                loss = criterion(out, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(X_v)
            val_pred = val_out.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_out = model(X_v)
        val_pred = val_out.argmax(dim=1).cpu().numpy()

    return model, val_pred


def evaluate_pytorch(X, y, name, model_class, n_splits=5, **kwargs):
    """5-fold CV with PyTorch model."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    t0 = time.time()

    n_features = X_s.shape[1]
    n_classes = len(np.unique(y))

    for fold, (tr, te) in enumerate(skf.split(X_s, y)):
        model, y_pred = train_pytorch_model(
            X_s[tr], y[tr], X_s[te], y[te],
            model_class, n_features, n_classes, **kwargs)
        accs.append(accuracy_score(y[te], y_pred))
        f1s.append(f1_score(y[te], y_pred, average='weighted'))

    elapsed = time.time() - t0
    am, fm = np.mean(accs), np.mean(f1s)
    log(f"    {name:55s}: Acc={am:.4f}+/-{np.std(accs):.4f}  F1={fm:.4f}+/-{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "accuracy_std": float(np.std(accs)),
            "f1": float(fm), "f1_std": float(np.std(f1s)), "time": float(elapsed)}


def evaluate_lgbm(X, y, name, n_splits=5, params=None):
    """5-fold CV with LightGBM."""
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
    log(f"    {name:55s}: Acc={am:.4f}+/-{np.std(accs):.4f}  F1={fm:.4f}+/-{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "accuracy_std": float(np.std(accs)),
            "f1": float(fm), "f1_std": float(np.std(f1s)), "time": float(elapsed)}


def main():
    t_start = time.time()
    log("=" * 80)
    log("  ACCURACY IMPROVEMENT v9 - Robust + Efficient (ALL data, no cleaning)")
    log(f"  Device: {DEVICE}")
    log("=" * 80)

    all_results = {}

    # Load ALL datasets
    log("\n[Loading ALL datasets (no exclusions)...]")
    datasets = load_datasets()
    n_datasets = len(datasets)
    total = sum(len(d[1]) for d in datasets)
    log(f"\n  Total: {n_datasets} datasets, {total:,} samples")

    # Build combined dataset
    X80, y80, source80 = build_dataset_with_source(datasets, target_features=80)
    X_aware = add_dataset_features(X80, source80, n_datasets)
    log(f"  Features: {X_aware.shape[1]} ({80} PCA + {n_datasets} dataset IDs)")
    log(f"  Labels: { {int(k): int(v) for k, v in zip(*np.unique(y80, return_counts=True))} }")

    n_features = X_aware.shape[1]

    # ====================================================================
    # EXP 1: LightGBM baselines (ALL data)
    # ====================================================================
    log("\n" + "=" * 70)
    log("  EXP 1: LightGBM on ALL 8 datasets (reference)")
    log("=" * 70)

    r = evaluate_lgbm(X_aware, y80, "LGBM-1500-aware-ALL8")
    all_results["LGBM-1500-aware-ALL8"] = r

    # Smaller/faster LGBM
    fast_params = dict(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, num_leaves=127,
        min_child_samples=20, reg_alpha=0.5, reg_lambda=2.0,
        class_weight='balanced', random_state=42, n_jobs=N_JOBS, verbose=-1)
    r = evaluate_lgbm(X_aware, y80, "LGBM-500-fast-ALL8", params=fast_params)
    all_results["LGBM-500-fast-ALL8"] = r

    # ====================================================================
    # EXP 2: PyTorch MLP (efficient, noise-robust)
    # ====================================================================
    log("\n" + "=" * 70)
    log("  EXP 2: PyTorch MLP models (noise-robust training)")
    log("=" * 70)

    # Compact model with label smoothing + mixup
    log("\n  [EfficientEEGNet - compact MLP ~200KB]")
    r = evaluate_pytorch(X_aware, y80, "MLP-compact-smooth0.1-mixup",
                         EfficientEEGNet, epochs=50, lr=0.001,
                         label_smoothing=0.1, use_mixup=True, batch_size=1024)
    all_results["MLP-compact-smooth0.1-mixup"] = r

    # Without mixup (comparison)
    r = evaluate_pytorch(X_aware, y80, "MLP-compact-smooth0.1-nomixup",
                         EfficientEEGNet, epochs=50, lr=0.001,
                         label_smoothing=0.1, use_mixup=False, batch_size=1024)
    all_results["MLP-compact-smooth0.1-nomixup"] = r

    # No label smoothing
    r = evaluate_pytorch(X_aware, y80, "MLP-compact-nosmooth-mixup",
                         EfficientEEGNet, epochs=50, lr=0.001,
                         label_smoothing=0.0, use_mixup=True, batch_size=1024)
    all_results["MLP-compact-nosmooth-mixup"] = r

    # Larger residual model
    log("\n  [LargerEEGNet - residual MLP ~1MB]")
    r = evaluate_pytorch(X_aware, y80, "MLP-large-residual-smooth-mixup",
                         LargerEEGNet, epochs=60, lr=0.001,
                         label_smoothing=0.1, use_mixup=True, batch_size=1024)
    all_results["MLP-large-residual-smooth-mixup"] = r

    r = evaluate_pytorch(X_aware, y80, "MLP-large-residual-smooth-nomixup",
                         LargerEEGNet, epochs=60, lr=0.001,
                         label_smoothing=0.1, use_mixup=False, batch_size=1024)
    all_results["MLP-large-residual-smooth-nomixup"] = r

    # Higher label smoothing for more noise tolerance
    r = evaluate_pytorch(X_aware, y80, "MLP-large-residual-smooth0.2-mixup",
                         LargerEEGNet, epochs=60, lr=0.001,
                         label_smoothing=0.2, use_mixup=True, batch_size=1024)
    all_results["MLP-large-residual-smooth0.2-mixup"] = r

    # ====================================================================
    # EXP 3: Model size comparison
    # ====================================================================
    log("\n" + "=" * 70)
    log("  EXP 3: Model size comparison")
    log("=" * 70)

    # Get model sizes
    compact = EfficientEEGNet(n_features, 3)
    large = LargerEEGNet(n_features, 3)
    compact_params = sum(p.numel() for p in compact.parameters())
    large_params = sum(p.numel() for p in large.parameters())
    compact_size_kb = compact_params * 4 / 1024  # float32
    large_size_kb = large_params * 4 / 1024

    log(f"  EfficientEEGNet: {compact_params:,} params, ~{compact_size_kb:.0f} KB")
    log(f"  LargerEEGNet:    {large_params:,} params, ~{large_size_kb:.0f} KB")
    log("  LightGBM-1500:   ~102 MB (saved model)")
    log("  LightGBM-500:    ~35 MB (estimated)")

    # ====================================================================
    # EXP 4: LightGBM with noise-robust settings
    # ====================================================================
    log("\n" + "=" * 70)
    log("  EXP 4: LightGBM noise-robust variants")
    log("=" * 70)

    # Higher regularization to prevent overfitting to noisy labels
    robust_params = dict(
        n_estimators=1500, max_depth=8, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, num_leaves=127,
        min_child_samples=30, reg_alpha=1.0, reg_lambda=5.0,
        class_weight='balanced', random_state=42, n_jobs=N_JOBS, verbose=-1)
    r = evaluate_lgbm(X_aware, y80, "LGBM-1500-robust-reg-ALL8", params=robust_params)
    all_results["LGBM-1500-robust-reg-ALL8"] = r

    # Very conservative (strong regularization)
    conservative_params = dict(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.6, colsample_bytree=0.5, num_leaves=63,
        min_child_samples=50, reg_alpha=3.0, reg_lambda=10.0,
        class_weight='balanced', random_state=42, n_jobs=N_JOBS, verbose=-1)
    r = evaluate_lgbm(X_aware, y80, "LGBM-800-conservative-ALL8", params=conservative_params)
    all_results["LGBM-800-conservative-ALL8"] = r

    # ====================================================================
    # EXP 5: Different learning rates for PyTorch
    # ====================================================================
    log("\n" + "=" * 70)
    log("  EXP 5: PyTorch learning rate sweep")
    log("=" * 70)

    for lr in [0.003, 0.0003]:
        r = evaluate_pytorch(X_aware, y80, f"MLP-large-lr{lr}-smooth-mixup",
                             LargerEEGNet, epochs=60, lr=lr,
                             label_smoothing=0.1, use_mixup=True, batch_size=1024)
        all_results[f"MLP-large-lr{lr}"] = r

    # === FINAL SUMMARY ===
    elapsed = time.time() - t_start
    log("\n" + "=" * 80)
    log("  FINAL RESULTS v9 - Robust + Efficient (ALL 8 datasets)")
    log("  Previous best (all data): 94.46% (v7 LGBM-aware-8ds)")
    log("=" * 80)
    log(f"\n{'Model':<60} {'Accuracy':>10} {'F1':>10} {'Time':>8}")
    log("-" * 92)

    for name, r in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        acc = r['accuracy']
        f1 = r['f1']
        t = r['time']
        model_type = "NN" if "MLP" in name else "LGBM"
        log(f"{name:<60} {acc:>9.4f} {f1:>9.4f} {t:>7.1f}s  [{model_type}]")

    best_name = max(all_results, key=lambda k: all_results[k]['accuracy'])
    best = all_results[best_name]
    log(f"\nBest: {best_name}")
    log(f"  Accuracy: {best['accuracy']:.4f}")
    log(f"  F1: {best['f1']:.4f}")
    log(f"  vs v7-all-data: {(best['accuracy'] - 0.9446)*100:+.2f}%")

    best_nn = max([k for k in all_results if 'MLP' in k],
                  key=lambda k: all_results[k]['accuracy'], default=None)
    if best_nn:
        log(f"\nBest neural net: {best_nn}")
        log(f"  Accuracy: {all_results[best_nn]['accuracy']:.4f}")
        log(f"  Model size: ~{large_size_kb:.0f} KB (vs 102 MB LightGBM)")

    log(f"\nTotal time: {elapsed:.1f}s ({elapsed/3600:.1f} hr)")

    # Save results
    benchmarks_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/benchmarks')
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    with open(benchmarks_dir / 'improvement_v9_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Results saved to {benchmarks_dir / 'improvement_v9_results.json'}")


if __name__ == '__main__':
    main()
