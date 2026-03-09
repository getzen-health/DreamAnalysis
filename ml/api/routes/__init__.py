"""ML API route package.

Assembles all sub-routers into a single FastAPI APIRouter.
main.py imports this as: from api.routes import router
"""

from fastapi import APIRouter

# Re-export shared singletons so websocket.py can do `from api.routes import …`
from ._shared import (
    _get_device_manager as _get_device_manager,
    _session_recorder as _session_recorder,
    sleep_model as sleep_model,
    emotion_model as emotion_model,
    dream_model as dream_model,
    flow_model as flow_model,
    creativity_model as creativity_model,
    memory_model as memory_model,
    drowsiness_model as drowsiness_model,
    cognitive_load_model as cognitive_load_model,
    attention_model as attention_model,
    stress_model as stress_model,
    lucid_dream_model as lucid_dream_model,
    meditation_model as meditation_model,
    food_emotion_model as food_emotion_model,
    fusion_model as fusion_model,
    get_biometric_snapshot as get_biometric_snapshot,
    update_biometric_snapshot as update_biometric_snapshot,
    predict_emotion as predict_emotion,
)

from .analysis import router as _analysis
from .models_status import router as _models_status
from .wavelet import router as _wavelet
from .neurofeedback import router as _neurofeedback
from .sessions import router as _sessions
from .data_collection import router as _data_collection
from .calibration import router as _calibration
from .connectivity import router as _connectivity
from .devices import router as _devices
from .datasets import router as _datasets
from .health import router as _health
from .accuracy import router as _accuracy
from .brain_timeline import router as _brain_timeline
from .spiritual import router as _spiritual
from .emotion_shift import router as _emotion_shift
from .cognitive import router as _cognitive
from .denoising import router as _denoising
from .food_emotion import router as _food_emotion
from .multimodal import router as _multimodal
from .parquet import router as _parquet
from .hrv_fusion import router as _hrv_fusion
try:
    from .personal import router as _personal
    _personal_available = True
except ImportError:
    # torch not installed in this environment (cloud/inference-only deploy)
    _personal = None
    _personal_available = False
from .biometrics import router as _biometrics
from .interventions import router as _interventions
from .voice_watch import router as _voice_watch
from .voice_biomarkers import router as _voice_biomarkers
from .eeg_storage import router as _eeg_storage
from .trauma_resilience import router as _trauma_resilience
from .ppg import router as _ppg
from .binaural import router as _binaural
from .dreams import router as _dreams
from .music import router as _music
from .federated import router as _federated

router = APIRouter()

# Registration order matters for path-param routes (specific before catch-all)
router.include_router(_analysis)
router.include_router(_models_status)
router.include_router(_wavelet)
router.include_router(_neurofeedback)
router.include_router(_sessions)          # /sessions/trends etc BEFORE /sessions/{id}
router.include_router(_data_collection)
router.include_router(_calibration)
router.include_router(_accuracy)          # /calibration/{user_id} variants after simple /calibration/*
router.include_router(_connectivity)
router.include_router(_devices)
router.include_router(_datasets)
router.include_router(_health)
router.include_router(_brain_timeline)
router.include_router(_spiritual)
router.include_router(_emotion_shift)
router.include_router(_cognitive)
router.include_router(_denoising)
router.include_router(_food_emotion)
router.include_router(_multimodal)
router.include_router(_parquet)
router.include_router(_hrv_fusion)
if _personal_available:
    router.include_router(_personal)
router.include_router(_biometrics)
router.include_router(_interventions)
router.include_router(_voice_watch)
router.include_router(_voice_biomarkers)
router.include_router(_eeg_storage)
router.include_router(_trauma_resilience)
router.include_router(_ppg)
router.include_router(_binaural)
router.include_router(_dreams)
router.include_router(_music)
router.include_router(_federated)
