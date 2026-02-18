# Agent Guide — ML Backend

## Code Patterns

### Model Class Pattern
```python
class MyModel:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)  # or onnx, torch

    def predict(self, features: np.ndarray) -> dict:
        if self.model:
            prediction = self.model.predict(features)
        else:
            prediction = self._feature_based_fallback(features)
        return {"label": ..., "confidence": ..., "details": ...}
```

### Endpoint Pattern
```python
@router.post("/predict-my-thing")
async def predict_my_thing(request: MyRequest):
    features = np.array(request.eeg_data)
    result = my_model.predict(features)
    return _numpy_safe(result)  # Always wrap with _numpy_safe()
```

### Processing Pipeline
```
Raw EEG → artifact_detector → bandpass filter (0.5-45 Hz)
       → notch filter (50/60 Hz) → feature extraction
       → 17-feature vector → model.predict()
```

## Conventions

- **Sampling rate**: 256 Hz (fs=256 everywhere)
- **Data format**: NumPy arrays, shape `(n_channels, n_samples)` or `(n_samples,)` for single-channel
- **Band powers**: Delta (0.5-4), Theta (4-8), Alpha (8-13), Beta (13-30), Gamma (30-45)
- **Feature vector**: 17 features — 5 band powers, 3 Hjorth parameters, spectral entropy, 4 band ratios, 2 asymmetry metrics, peak frequency, line noise ratio
- **JSON serialization**: Always use `_numpy_safe()` wrapper for any dict with numpy types

## Testing

```bash
cd ml && pytest tests/ -v
```

Test files are in `ml/tests/`. Tests mock EEG data and verify model outputs.

## File Organization

| Directory | What's In It |
|-----------|-------------|
| `api/` | FastAPI routes (routes.py) + WebSocket handler |
| `models/` | 15 model classes + `saved/` directory with weights |
| `processing/` | 11 signal processing modules (filter, extract, quality) |
| `training/` | Training scripts + data loaders for 8 datasets |
| `health/` | Apple Health + Google Fit + correlation engine |
| `hardware/` | BrainFlow device manager (Muse 2) |
| `simulation/` | EEG data simulator for testing |
| `storage/` | Session recording + analytics persistence |
| `neurofeedback/` | Neurofeedback protocol engine |
| `tools/` | CLI tools for demos + data import |
| `tests/` | pytest test files |
| `benchmarks/` | Training result JSON files |
| `data/` | Training datasets (gitignored) |

## Pitfalls

1. **routes.py is 2017 lines** — Find the right category section before adding endpoints. See line-number map in `api/README.md`.
2. **No training imports at runtime** — `training/` modules are not imported by the API. They run standalone.
3. **Large model files** — `.joblib`, `.pkl`, `.pt`, `.onnx` are all gitignored. The `emotion_classifier_rf.joblib` is 3.1GB.
4. **BrainFlow lazy-loads** — Hardware module only initializes when first device connection is requested (avoids import errors without BrainFlow installed).
5. **`_numpy_safe()` is mandatory** — FastAPI can't serialize numpy types. Every endpoint returning model output must wrap with `_numpy_safe()`.
6. **Model fallbacks** — All models work without saved weights by using feature-based heuristics. This is by design for demo mode.
