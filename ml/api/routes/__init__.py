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
from .voice_checkin import router as _voice_checkin
from .eeg_storage import router as _eeg_storage
from .trauma_resilience import router as _trauma_resilience
from .ppg import router as _ppg
from .binaural import router as _binaural
from .dreams import router as _dreams
from .music import router as _music
from .federated import router as _federated
from .fatigue import router as _fatigue
from .phenotyping import router as _phenotyping
from .auth_biometric import router as _auth_biometric
from .gru_sleep import router as _gru_sleep
from .seizure import router as _seizure
from .emo_adapt import router as _emo_adapt
from .cognitive_reserve import router as _cognitive_reserve
from .emotion_regulation import router as _emotion_regulation
from .lucid_induction import router as _lucid_induction
from .tinnitus import router as _tinnitus
from .pain import router as _pain
from .social_cognition import router as _social_cognition
from .music_emotion import router as _music_emotion
from .brain_health import router as _brain_health
from .brain_maturation import router as _brain_maturation
from .neuroadaptive import router as _neuroadaptive
from .domain_adapt import router as _domain_adapt
from .few_shot import router as _few_shot
from .preictal import router as _preictal
from .sleep_quality import router as _sleep_quality
from .motor_imagery import router as _motor_imagery
from .meditation_depth_route import router as _meditation_depth
from .neurogame import router as _neurogame
from .deception import router as _deception
from .engagement import router as _engagement
from .music_genre_eeg import router as _music_genre_eeg
from .ptsd_neurofeedback import router as _ptsd_neurofeedback
from .vr_workload import router as _vr_workload
from .hyperscanning import router as _hyperscanning
from .cscl import router as _cscl
from .contrastive_transfer import router as _contrastive_transfer
from .emotion2vec import router as _emotion2vec
from .lsteeg import router as _lsteeg
from .health_summary import router as _health_summary
from .mental_health_questionnaire import router as _mental_health_questionnaire
from .femba import router as _femba
from .tmnet import router as _tmnet
from .altered_consciousness import router as _altered_consciousness
from .mci_screener import router as _mci_screener
from .neurostim_guidance import router as _neurostim_guidance
from .autism_screener import router as _autism_screener
from .decision_confidence import router as _decision_confidence
from .imagined_speech import router as _imagined_speech
from .motor_intention import router as _motor_intention
from .parkinsons_screener import router as _parkinsons_screener
from .drowsiness_alertness import router as _drowsiness_alertness
from .craving_detector import router as _craving_detector
from .brain_age import router as _brain_age
from .long_covid_screener import router as _long_covid_screener
from .big_five_estimator import router as _big_five_estimator
from .neuroaesthetic import router as _neuroaesthetic
from .placebo_predictor import router as _placebo_predictor
from .spatial_navigation import router as _spatial_navigation
from .humor_detector import router as _humor_detector
from .emotional_granularity import router as _emotional_granularity
from .ied_detector import router as _ied_detector
from .interoceptive_awareness import router as _interoceptive_awareness
from .affect_labeling import router as _affect_labeling
from .emotional_synchrony import router as _emotional_synchrony
from .emotional_memory import router as _emotional_memory
from .ei_composite import router as _ei_composite
from .multimodal_ei import router as _multimodal_ei
from .supplement_tracker import router as _supplement_tracker

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
router.include_router(_voice_checkin)
router.include_router(_eeg_storage)
router.include_router(_trauma_resilience)
router.include_router(_ppg)
router.include_router(_binaural)
router.include_router(_dreams)
router.include_router(_music)
router.include_router(_federated)
router.include_router(_fatigue)
router.include_router(_phenotyping)
router.include_router(_auth_biometric)
router.include_router(_gru_sleep)
router.include_router(_seizure)
router.include_router(_emo_adapt)
router.include_router(_cognitive_reserve)
router.include_router(_emotion_regulation)
router.include_router(_lucid_induction)
router.include_router(_tinnitus)
router.include_router(_pain)
router.include_router(_social_cognition)
router.include_router(_music_emotion)
router.include_router(_brain_health)
router.include_router(_brain_maturation)
router.include_router(_neuroadaptive)
router.include_router(_domain_adapt)
router.include_router(_few_shot)
router.include_router(_preictal)
router.include_router(_sleep_quality)
router.include_router(_motor_imagery)
router.include_router(_meditation_depth)
router.include_router(_neurogame)
router.include_router(_deception)
router.include_router(_engagement)
router.include_router(_music_genre_eeg)
router.include_router(_ptsd_neurofeedback)
router.include_router(_vr_workload)
router.include_router(_hyperscanning)
router.include_router(_cscl)
router.include_router(_contrastive_transfer)
router.include_router(_emotion2vec)
router.include_router(_lsteeg)
router.include_router(_health_summary)
router.include_router(_mental_health_questionnaire)
router.include_router(_femba)
router.include_router(_tmnet)
router.include_router(_altered_consciousness)
router.include_router(_mci_screener)
router.include_router(_neurostim_guidance)
router.include_router(_autism_screener)
router.include_router(_decision_confidence)
router.include_router(_imagined_speech)
router.include_router(_motor_intention)
router.include_router(_parkinsons_screener)
router.include_router(_drowsiness_alertness)
router.include_router(_craving_detector)
router.include_router(_brain_age)
router.include_router(_long_covid_screener)
router.include_router(_big_five_estimator)
router.include_router(_neuroaesthetic)
router.include_router(_placebo_predictor)
router.include_router(_spatial_navigation)
router.include_router(_humor_detector)
router.include_router(_emotional_granularity)
router.include_router(_ied_detector)
router.include_router(_interoceptive_awareness)
router.include_router(_affect_labeling)
router.include_router(_emotional_synchrony)
router.include_router(_emotional_memory)
router.include_router(_ei_composite)
router.include_router(_multimodal_ei)
router.include_router(_supplement_tracker)
