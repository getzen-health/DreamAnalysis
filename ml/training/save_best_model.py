"""
Save the best trained emotion model for deployment.
Trains on ALL available data and saves the model + scaler + PCA.
"""

import numpy as np
import json
import time
import warnings
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# Import loaders from mega_trainer
import sys
sys.path.insert(0, str(Path(__file__).parent))
from mega_trainer import (
    load_deap_enhanced, load_gameemo_enhanced, load_eeg_er_enhanced,
    load_dens, load_seed_enhanced, load_brainwave_emotions,
    load_muse_mental_state, load_muse2_motor_imagery, load_confused_student,
    load_muse_subconscious, load_emokey_moments, load_seed_iv
)


def main():
    log("=" * 70)
    log("  SAVING BEST EMOTION MODEL")
    log("=" * 70)

    # Load all datasets
    log("\n[Loading all datasets...]")
    datasets = []
    for loader in [load_deap_enhanced, load_gameemo_enhanced, load_eeg_er_enhanced,
                   load_dens, load_seed_enhanced, load_brainwave_emotions,
                   load_muse_mental_state, load_muse2_motor_imagery, load_confused_student,
                   load_muse_subconscious, load_emokey_moments, load_seed_iv]:
        try:
            X, y, name = loader()
            if len(X) > 0 and len(np.unique(y)) >= 2:
                datasets.append((X, y, name))
                log(f"  {name}: {len(X)} samples, {X.shape[1]} features")
            elif len(X) > 0:
                log(f"  {name}: Skipped (single class)")
        except Exception as e:
            log(f"  Error: {e}")

    total = sum(len(d[1]) for d in datasets)
    log(f"\n  Total: {len(datasets)} datasets, {total:,} samples")

    # PCA-align all datasets to 80 dims
    log("\n[PCA-aligning to 80 dimensions...]")
    TARGET_FEATURES = 80
    all_X, all_y = [], []

    for X, y, name in datasets:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_comp = min(TARGET_FEATURES, X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_scaled)

        if X_pca.shape[1] < TARGET_FEATURES:
            X_pca = np.hstack([X_pca, np.zeros((X_pca.shape[0], TARGET_FEATURES - X_pca.shape[1]))])

        all_X.append(X_pca)
        all_y.append(y)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    log(f"  Combined: {X_all.shape[0]} samples, {X_all.shape[1]} features")
    log(f"  Labels: { {int(k): int(v) for k, v in zip(*np.unique(y_all, return_counts=True))} }")

    # Final standardization
    log("\n[Training final model...]")
    final_scaler = StandardScaler()
    X_final = final_scaler.fit_transform(X_all)

    # Train best model (RF 500 trees)
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_final, y_all)
    train_time = time.time() - t0
    log(f"  Training time: {train_time:.1f}s")

    # Evaluate on training data (sanity check)
    y_pred = model.predict(X_final)
    train_acc = accuracy_score(y_all, y_pred)
    train_f1 = f1_score(y_all, y_pred, average='weighted')
    log(f"  Training accuracy: {train_acc:.4f}")
    log(f"  Training F1: {train_f1:.4f}")
    log(f"\n{classification_report(y_all, y_pred, target_names=['Positive', 'Neutral', 'Negative'])}")

    # Save model artifacts
    models_dir = Path('/Users/sravyalu/NeuralDreamWorkshop/ml/models')
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / 'emotion_classifier_rf.joblib'
    scaler_path = models_dir / 'emotion_scaler.joblib'
    metadata_path = models_dir / 'model_metadata.json'

    joblib.dump(model, model_path)
    joblib.dump(final_scaler, scaler_path)

    metadata = {
        'model_type': 'RandomForestClassifier',
        'n_estimators': 500,
        'max_depth': 15,
        'n_features': TARGET_FEATURES,
        'n_classes': 3,
        'class_names': ['positive', 'neutral', 'negative'],
        'training_samples': int(X_all.shape[0]),
        'n_datasets': len(datasets),
        'datasets': [name for _, _, name in datasets],
        'cv_accuracy': 0.8480,  # From mega_trainer 12-dataset results (RF on 139K combined)
        'cv_f1': 0.8482,
        'train_accuracy': float(train_acc),
        'train_f1': float(train_f1),
        'train_time_seconds': float(train_time),
        'feature_extraction': 'PCA-aligned 80-dim from multi-source EEG features',
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    log(f"\n[Model saved]")
    log(f"  Model: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    log(f"  Scaler: {scaler_path}")
    log(f"  Metadata: {metadata_path}")
    log(f"\n  CV Accuracy: 84.80% (MEGA-Combined RF on 139K+ samples)")
    log(f"  CV F1 Score: 0.8482")
    log(f"  Training samples: {X_all.shape[0]:,}")
    log(f"  Datasets: {len(datasets)}")


if __name__ == '__main__':
    main()
