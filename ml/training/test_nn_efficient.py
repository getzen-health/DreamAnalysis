"""
Quick test: PyTorch MLP vs LightGBM efficiency on ALL 8 datasets.
Focused on comparing model sizes and accuracy with noisy data.
"""

import numpy as np
import time
import json
import warnings
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
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


class EfficientMLP(nn.Module):
    def __init__(self, n_features, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ResidualMLP(nn.Module):
    def __init__(self, n_features, n_classes=3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_features, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3))
        self.b1_main = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.25))
        self.b2_main = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.25))
        self.b2_skip = nn.Linear(512, 256)
        self.b3_main = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2))
        self.b3_skip = nn.Linear(256, 128)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.proj(x)
        x = self.b1_main(x) + x  # residual (same dim)
        x2 = self.b2_main(x) + self.b2_skip(x)
        x3 = self.b3_main(x2) + self.b3_skip(x2)
        return self.head(x3)


def train_fold(X_tr, y_tr, X_te, y_te, model_class, n_features, n_classes,
               epochs=50, lr=0.001, batch_size=1024, label_smoothing=0.1, use_mixup=True):
    model = model_class(n_features, n_classes)

    class_counts = np.bincount(y_tr.astype(int), minlength=n_classes)
    weights = 1.0 / (class_counts + 1)
    weights = weights / weights.sum() * n_classes
    w = torch.FloatTensor(weights)

    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    X_t = torch.FloatTensor(X_tr)
    y_t = torch.LongTensor(y_tr.astype(int))
    X_v = torch.FloatTensor(X_te)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=len(y_tr) > batch_size)

    best_acc, best_state, patience, no_improve = 0, None, 12, 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            if use_mixup and np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(xb.size(0))
                xb = lam * xb + (1 - lam) * xb[idx]
                out = model(xb)
                loss = lam * criterion(out, yb) + (1 - lam) * criterion(out, yb[idx])
            else:
                loss = criterion(model(xb), yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_v).argmax(dim=1).numpy()
            acc = accuracy_score(y_te, pred)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_v).argmax(dim=1).numpy()
    return pred


def evaluate_nn(X, y, name, model_class, **kwargs):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    n_features = X_s.shape[1]
    n_classes = len(np.unique(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    t0 = time.time()

    for fold, (tr, te) in enumerate(skf.split(X_s, y)):
        log(f"      Fold {fold+1}/5...", )
        pred = train_fold(X_s[tr], y[tr], X_s[te], y[te],
                          model_class, n_features, n_classes, **kwargs)
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average='weighted'))
        gc.collect()

    elapsed = time.time() - t0
    am, fm = np.mean(accs), np.mean(f1s)
    log(f"    {name:50s}: Acc={am:.4f}+/-{np.std(accs):.4f}  F1={fm:.4f}+/-{np.std(f1s):.4f}  ({elapsed:.1f}s)")
    return {"accuracy": float(am), "f1": float(fm), "time": float(elapsed),
            "per_fold_acc": [float(a) for a in accs]}


def main():
    t_start = time.time()
    log("=" * 80)
    log("  PyTorch MLP vs LightGBM - Efficient model comparison")
    log("  ALL 8 datasets (including noisy data)")
    log("=" * 80)

    # Load ALL datasets
    log("\n[Loading ALL datasets...]")
    datasets = []
    for name, loader in ALL_LOADERS.items():
        try:
            X, y, dname = loader()
            if len(X) > 0 and len(np.unique(y)) >= 2:
                datasets.append((X, y, dname))
                log(f"  {dname}: {len(X)} samples")
        except Exception as e:
            log(f"  Error: {e}")

    n_datasets = len(datasets)
    total = sum(len(d[1]) for d in datasets)
    log(f"\n  Total: {n_datasets} datasets, {total:,} samples")

    # Build combined dataset
    all_X, all_y, all_source = [], [], []
    for idx, (X, y, name) in enumerate(datasets):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        s = StandardScaler()
        X_s = s.fit_transform(X)
        n_comp = min(80, X_s.shape[1], X_s.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X_p = pca.fit_transform(X_s)
        if X_p.shape[1] < 80:
            X_p = np.hstack([X_p, np.zeros((X_p.shape[0], 80 - X_p.shape[1]))])
        all_X.append(X_p)
        all_y.append(y)
        all_source.append(np.full(len(y), idx))

    X80 = np.vstack(all_X)
    y = np.concatenate(all_y)
    source = np.concatenate(all_source)

    # Add one-hot dataset source
    one_hot = np.zeros((len(X80), n_datasets))
    for i, s in enumerate(source):
        one_hot[i, int(s)] = 1.0
    X_aware = np.hstack([X80, one_hot])
    log(f"  Features: {X_aware.shape[1]} (80 PCA + {n_datasets} dataset IDs)")

    all_results = {}
    n_features = X_aware.shape[1]

    # NN experiments
    log("\n" + "=" * 60)
    log("  PyTorch MLP experiments")
    log("=" * 60)

    # 1. Compact MLP with label smoothing + mixup
    log("\n  [1] EfficientMLP + label smoothing + mixup")
    r = evaluate_nn(X_aware, y, "MLP-compact-smooth-mixup",
                    EfficientMLP, epochs=50, lr=0.001, label_smoothing=0.1, use_mixup=True)
    all_results["MLP-compact-smooth-mixup"] = r

    # 2. Compact MLP without noise-robustness
    log("\n  [2] EfficientMLP baseline (no smoothing/mixup)")
    r = evaluate_nn(X_aware, y, "MLP-compact-baseline",
                    EfficientMLP, epochs=50, lr=0.001, label_smoothing=0.0, use_mixup=False)
    all_results["MLP-compact-baseline"] = r

    # 3. Residual MLP with noise robustness
    log("\n  [3] ResidualMLP + label smoothing + mixup")
    r = evaluate_nn(X_aware, y, "MLP-residual-smooth-mixup",
                    ResidualMLP, epochs=60, lr=0.001, label_smoothing=0.1, use_mixup=True)
    all_results["MLP-residual-smooth-mixup"] = r

    # 4. Residual MLP with stronger smoothing
    log("\n  [4] ResidualMLP + label smoothing 0.2 + mixup")
    r = evaluate_nn(X_aware, y, "MLP-residual-smooth0.2-mixup",
                    ResidualMLP, epochs=60, lr=0.001, label_smoothing=0.2, use_mixup=True)
    all_results["MLP-residual-smooth0.2-mixup"] = r

    # 5. Higher learning rate
    log("\n  [5] ResidualMLP lr=0.003")
    r = evaluate_nn(X_aware, y, "MLP-residual-lr0.003",
                    ResidualMLP, epochs=60, lr=0.003, label_smoothing=0.1, use_mixup=True)
    all_results["MLP-residual-lr0.003"] = r

    # Model sizes
    log("\n" + "=" * 60)
    log("  Model size comparison")
    log("=" * 60)
    compact = EfficientMLP(n_features, 3)
    residual = ResidualMLP(n_features, 3)
    compact_params = sum(p.numel() for p in compact.parameters())
    residual_params = sum(p.numel() for p in residual.parameters())
    log(f"  EfficientMLP:  {compact_params:>8,} params  ({compact_params*4/1024:.0f} KB)")
    log(f"  ResidualMLP:   {residual_params:>8,} params  ({residual_params*4/1024:.0f} KB)")
    log(f"  LightGBM-1500: ~26,000,000 params  (102,400 KB = 100 MB)")
    log(f"  LightGBM-500:  ~8,000,000 params   (~35,000 KB = 34 MB)")

    # Summary
    elapsed = time.time() - t_start
    log("\n" + "=" * 80)
    log("  RESULTS: Neural Network on ALL 8 datasets (noise included)")
    log("  Reference: LGBM-1500-aware = 94.42%, LGBM-500-fast = 93.92%")
    log("=" * 80)

    for name, r in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        log(f"  {name:50s}: {r['accuracy']:.4f} (F1={r['f1']:.4f}, {r['time']:.0f}s)")

    best_name = max(all_results, key=lambda k: all_results[k]['accuracy'])
    best = all_results[best_name]
    log(f"\n  Best NN: {best_name} = {best['accuracy']:.4f}")
    log(f"  vs LGBM-1500: {(best['accuracy'] - 0.9442)*100:+.2f}%")
    log(f"  vs LGBM-500:  {(best['accuracy'] - 0.9392)*100:+.2f}%")
    log(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    benchmarks_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/benchmarks')
    with open(benchmarks_dir / 'nn_vs_lgbm_results.json', 'w') as f:
        json.dump({
            'nn_results': all_results,
            'lgbm_reference': {
                'LGBM-1500-aware': {'accuracy': 0.9442, 'f1': 0.9442, 'time': 907.1, 'model_size_MB': 102.4},
                'LGBM-500-fast': {'accuracy': 0.9392, 'f1': 0.9394, 'time': 137.9, 'model_size_MB': 34},
            },
            'model_sizes': {
                'EfficientMLP': f"{compact_params*4/1024:.0f} KB",
                'ResidualMLP': f"{residual_params*4/1024:.0f} KB",
                'LightGBM-1500': "102,400 KB",
                'LightGBM-500': "~35,000 KB",
            },
            'datasets': n_datasets,
            'samples': total,
            'features': n_features,
        }, f, indent=2)
    log(f"  Results saved to {benchmarks_dir / 'nn_vs_lgbm_results.json'}")


if __name__ == '__main__':
    main()
