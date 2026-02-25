# Training

Scripts for training ML models on EEG datasets. These run standalone — they are not imported by the API at runtime.

## Production Training Scripts

| Script | What It Trains |
|--------|---------------|
| `mega_trainer.py` | Legacy multi-algorithm trainer (LightGBM, XGBoost, MLP, RF). Superseded by `train_mega_lgbm_unified.py` |
| `train_mega_lgbm_unified.py` | Active unified trainer — global PCA 85→80 + LightGBM on 9 datasets. Produces 74.21% CV (163 534 samples) |
| `train_emotion.py` | Emotion classifier specifically |
| `train_sleep.py` | Sleep staging model |
| `train_dream.py` | Dream detector |
| `train_deap_muse.py` | Emotion model on DEAP + Muse datasets |
| `train_all_models.py` | Batch trains all model types |
| `train_all_new_models.py` | Trains newer cognitive models (drowsiness, attention, etc.) |
| `train_all_datasets.py` | Runs training across all 8 datasets |
| `train_artifact_classifier.py` | Artifact type classifier |
| `train_denoiser.py` | Denoising autoencoder |

## Supporting Scripts

| Script | Purpose |
|--------|---------|
| `data_loaders.py` | Loaders for 8 datasets: DEAP, SEED, GAMEEMO, Muse, AMIGOS, DREAMER, synthetic, combined |
| `benchmark.py` | Evaluates trained models, saves results to `../benchmarks/` |
| `save_best_model.py` | Picks the best model variant and saves to `../models/saved/` |
| `save_efficient_model.py` | Exports optimized model for deployment |
| `export_web_models.py` | Converts models to ONNX for browser inference |

## How to Train a Model

```bash
cd ml

# 1. Ensure dataset is in ml/data/ (gitignored)
# 2. Run training
python -m training.train_emotion

# 3. Check results
cat benchmarks/emotion_classifier_benchmark.json

# 4. Export for web (optional)
python -m training.export_web_models
```

Trained models are saved to `models/saved/`. The API auto-discovers them on next restart.
