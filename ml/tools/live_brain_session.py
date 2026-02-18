"""Live Brain Session — Real-time EEG → 6-Model Analysis Pipeline.

Connects to a Muse 2 (or synthetic board for testing) via BrainFlow,
streams EEG, runs all 6 models, stores brain sessions, and correlates
with Apple Health data.

Integrated accuracy pipeline:
  1. Signal Quality Gate — reject noisy epochs before analysis
  2. Per-User Calibration — z-score normalize to YOUR baseline
  3. State Transition Engine — smooth predictions, block impossible jumps
  4. Confidence Calibration — honest confidence scores
  5. User Personalization — blend global + personal models from feedback

Usage:
    # Test with synthetic EEG (no hardware needed):
    python3 ml/tools/live_brain_session.py --device synthetic

    # Real Muse 2 (needs BLED112 dongle):
    python3 ml/tools/live_brain_session.py --device muse_2

    # With calibration + personalization:
    python3 ml/tools/live_brain_session.py --device synthetic --user myname

    # Run calibration first:
    python3 ml/tools/live_brain_session.py --device synthetic --user myname --calibrate
"""

import sys
import time
import signal
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.eeg_processor import extract_features, extract_band_powers, preprocess
from processing.signal_quality import SignalQualityChecker
from processing.calibration import UserCalibration, CalibrationRunner, CALIBRATION_STEPS
from processing.state_transitions import BrainStateEngine
from processing.confidence_calibration import ConfidenceCalibrator, add_uncertainty_labels
from processing.user_feedback import PersonalizedPipeline
from models.sleep_staging import SleepStagingModel
from models.emotion_classifier import EmotionClassifier
from models.dream_detector import DreamDetector
from models.flow_state_detector import FlowStateDetector
from models.creativity_detector import CreativityDetector, MemoryEncodingPredictor
from models.drowsiness_detector import DrowsinessDetector
from models.cognitive_load_estimator import CognitiveLoadEstimator
from models.attention_classifier import AttentionClassifier
from models.stress_detector import StressDetector
from models.lucid_dream_detector import LucidDreamDetector
from models.meditation_classifier import MeditationClassifier
from health.correlation_engine import HealthBrainDB


# ANSI colors for terminal output
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def print_banner():
    print(f"""
{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════╗
║        Svapnastra — Live Brain Session           ║
║    12-Model Real-Time EEG + Accuracy Pipeline    ║
╚══════════════════════════════════════════════════╝{C.END}
""")


def format_bar(value, max_val=1.0, width=20):
    """Create a visual bar chart."""
    filled = int((value / max_val) * width) if max_val > 0 else 0
    filled = max(0, min(filled, width))
    bar = "█" * filled + "░" * (width - filled)
    return bar


def print_analysis(analysis: dict, elapsed: float, quality: dict,
                   smoothed: dict, confidence_summary: dict):
    """Pretty-print the full analysis with accuracy pipeline info."""
    print("\033[2J\033[H", end="")  # Clear screen

    print_banner()
    print(f"  {C.DIM}Session time: {elapsed:.0f}s | {datetime.now().strftime('%H:%M:%S')}{C.END}")

    # ── Signal Quality ──
    q_score = quality.get("quality_score", 0)
    q_usable = quality.get("is_usable", False)
    q_color = C.GREEN if q_score >= 0.6 else C.YELLOW if q_score >= 0.4 else C.RED
    q_label = "GOOD" if q_score >= 0.6 else "FAIR" if q_score >= 0.4 else "POOR"
    print(f"\n  {C.BOLD}SIGNAL{C.END}     {q_color}{q_label:>8}{C.END}  {format_bar(q_score)} {q_score:.0%}")
    reasons = quality.get("rejection_reasons", [])
    if reasons:
        print(f"  {C.DIM}           {'; '.join(reasons[:2])}{C.END}")

    if not q_usable:
        print(f"\n  {C.RED}{C.BOLD}  Signal too noisy — skipping analysis{C.END}")
        print(f"  {C.DIM}Press Ctrl+C to stop session{C.END}")
        return

    # ── Confidence Summary ──
    conf_summary = confidence_summary or {}
    overall_rel = conf_summary.get("overall_reliability", "?")
    mean_conf = conf_summary.get("mean_confidence", 0)
    n_uncertain = conf_summary.get("n_uncertain", 0)
    rel_color = C.GREEN if overall_rel == "good" else C.YELLOW if overall_rel == "fair" else C.RED
    print(f"  {C.BOLD}CONFIDENCE{C.END} {rel_color}{overall_rel:>8}{C.END}  {format_bar(mean_conf)} {mean_conf:.0%}  {C.DIM}({n_uncertain} uncertain){C.END}")

    print()

    # Helper: get smoothed state or fall back to raw
    def get_state(model_key, raw_key, raw_pred):
        sm = smoothed.get(model_key, {})
        state = sm.get("smoothed_state", raw_pred.get(raw_key, "?"))
        was_overridden = sm.get("was_overridden", False)
        override_marker = f" {C.DIM}~{C.END}" if was_overridden else ""
        return state, override_marker

    # Helper: get calibrated confidence or fall back to raw
    def get_conf(pred, fallback_key="confidence"):
        return pred.get("calibrated_confidence", pred.get(fallback_key, 0))

    # ── 1. Sleep Staging ──
    sleep = analysis.get("sleep_staging", {})
    stage, sm_mark = get_state("sleep", "stage", sleep)
    conf = get_conf(sleep)
    uncertain = " ?" if sleep.get("is_uncertain") else ""
    stage_colors = {"Wake": C.GREEN, "N1": C.YELLOW, "N2": C.BLUE, "N3": C.CYAN, "REM": C.RED}
    sc = stage_colors.get(stage, C.END)
    print(f"  {C.BOLD}SLEEP{C.END}      {sc}{stage:>8}{C.END}{sm_mark}  {format_bar(conf)} {conf:.0%}{uncertain}")

    # ── 2. Emotion ──
    emo = analysis.get("emotions", {})
    emotion, sm_mark = get_state("emotion", "emotion", emo)
    valence = emo.get("valence", 0)
    arousal = emo.get("arousal", 0)
    uncertain = " ?" if emo.get("is_uncertain") else ""
    print(f"  {C.BOLD}EMOTION{C.END}    {emotion:>8}{sm_mark}  V={valence:+.2f}  A={arousal:+.2f}{uncertain}")

    # ── 3. Dream Detection ──
    dream = analysis.get("dream_detection", {})
    dream_sm = smoothed.get("dream", {})
    dreaming = dream_sm.get("is_dreaming", dream.get("is_dreaming", False))
    dream_conf = dream_sm.get("smoothed_probability", dream.get("probability", 0))
    dream_overridden = dream_sm.get("was_overridden", False)
    dm_mark = f" {C.DIM}~{C.END}" if dream_overridden else ""
    dream_str = f"{C.RED}DREAMING{C.END}" if dreaming else f"{C.DIM}awake{C.END}"
    print(f"  {C.BOLD}DREAM{C.END}      {dream_str:>17}{dm_mark}  {format_bar(dream_conf)} {dream_conf:.0%}")

    # ── 4. Flow State ──
    flow = analysis.get("flow_state", {})
    flow_state, sm_mark = get_state("flow", "state", flow)
    flow_score = get_conf(flow, "flow_score")
    uncertain = " ?" if flow.get("is_uncertain") else ""
    components = flow.get("components", {})
    flow_colors = {"no_flow": C.DIM, "micro_flow": C.YELLOW, "flow": C.GREEN, "deep_flow": C.CYAN}
    fc = flow_colors.get(flow_state, C.END)
    print(f"  {C.BOLD}FLOW{C.END}       {fc}{flow_state:>8}{C.END}{sm_mark}  {format_bar(flow_score)} {flow_score:.0%}{uncertain}")
    if components:
        abs_v = components.get("absorption", 0)
        eff_v = components.get("effortlessness", 0)
        foc_v = components.get("focus_quality", 0)
        print(f"  {C.DIM}           absorb={abs_v:.2f}  effort={eff_v:.2f}  focus={foc_v:.2f}{C.END}")

    # ── 5. Creativity ──
    cre = analysis.get("creativity", {})
    cre_state, sm_mark = get_state("creativity", "state", cre)
    cre_score = get_conf(cre, "creativity_score")
    uncertain = " ?" if cre.get("is_uncertain") else ""
    cre_colors = {"analytical": C.BLUE, "transitional": C.YELLOW, "creative": C.GREEN, "insight": C.RED}
    cc = cre_colors.get(cre_state, C.END)
    print(f"  {C.BOLD}CREATIVE{C.END}   {cc}{cre_state:>8}{C.END}{sm_mark}  {format_bar(cre_score)} {cre_score:.0%}{uncertain}")

    # ── 6. Memory Encoding ──
    mem = analysis.get("memory_encoding", {})
    mem_state, sm_mark = get_state("memory", "state", mem)
    mem_prob = get_conf(mem, "will_remember_probability")
    uncertain = " ?" if mem.get("is_uncertain") else ""
    mem_colors = {"poor_encoding": C.RED, "weak_encoding": C.YELLOW, "active_encoding": C.GREEN, "deep_encoding": C.CYAN}
    mc = mem_colors.get(mem_state, C.END)
    print(f"  {C.BOLD}MEMORY{C.END}     {mc}{mem_state:>8}{C.END}{sm_mark}  {format_bar(mem_prob)} {mem_prob:.0%}{uncertain}")

    print()

    # ── 7. Drowsiness ──
    drow = analysis.get("drowsiness", {})
    drow_state = drow.get("state", "?")
    drow_score = drow.get("drowsiness_index", 0)
    drow_colors = {"alert": C.GREEN, "drowsy": C.YELLOW, "sleepy": C.RED}
    dc = drow_colors.get(drow_state, C.END)
    print(f"  {C.BOLD}DROWSY{C.END}     {dc}{drow_state:>8}{C.END}  {format_bar(drow_score)} {drow_score:.0%}")

    # ── 8. Attention ──
    att = analysis.get("attention", {})
    att_state = att.get("state", "?")
    att_score = att.get("attention_score", 0)
    att_colors = {"distracted": C.RED, "passive": C.YELLOW, "focused": C.GREEN, "hyperfocused": C.CYAN}
    ac = att_colors.get(att_state, C.END)
    print(f"  {C.BOLD}ATTENTION{C.END}  {ac}{att_state:>8}{C.END}  {format_bar(att_score)} {att_score:.0%}")

    # ── 9. Cognitive Load ──
    cog = analysis.get("cognitive_load", {})
    cog_level = cog.get("level", "?")
    cog_score = cog.get("load_index", 0)
    cog_colors = {"low": C.GREEN, "moderate": C.YELLOW, "high": C.RED}
    coc = cog_colors.get(cog_level, C.END)
    print(f"  {C.BOLD}COG LOAD{C.END}   {coc}{cog_level:>8}{C.END}  {format_bar(cog_score)} {cog_score:.0%}")

    # ── 10. Stress ──
    stress = analysis.get("stress", {})
    stress_level = stress.get("level", "?")
    stress_idx = stress.get("stress_index", 0)
    stress_colors = {"relaxed": C.GREEN, "mild": C.YELLOW, "moderate": C.RED, "high": C.RED}
    stc = stress_colors.get(stress_level, C.END)
    print(f"  {C.BOLD}STRESS{C.END}     {stc}{stress_level:>8}{C.END}  {format_bar(stress_idx)} {stress_idx:.0%}")

    # ── 11. Meditation ──
    med = analysis.get("meditation", {})
    med_depth = med.get("depth", "?")
    med_score = med.get("meditation_score", 0)
    med_colors = {"surface": C.DIM, "light": C.YELLOW, "moderate": C.GREEN, "deep": C.CYAN, "transcendent": C.RED}
    mdc = med_colors.get(med_depth, C.END)
    print(f"  {C.BOLD}MEDITATE{C.END}   {mdc}{med_depth:>8}{C.END}  {format_bar(med_score)} {med_score:.0%}")

    # ── 12. Lucid Dream (only during REM) ──
    lucid = analysis.get("lucid_dream", {})
    lucid_state = lucid.get("state", "?")
    lucid_score = lucid.get("lucidity_score", 0)
    if lucid_state != "?" and sleep.get("stage") == "REM":
        print(f"  {C.BOLD}LUCID{C.END}      {C.RED}{lucid_state:>8}{C.END}  {format_bar(lucid_score)} {lucid_score:.0%}")

    # ── Personalization indicator ──
    personalization = flow.get("personalization", "none")
    if personalization != "none":
        print(f"\n  {C.DIM}Personalization: {personalization} (weight={flow.get('personal_model_weight', 0):.0%}){C.END}")

    # ── Band Powers ──
    bands = emo.get("band_powers", {})
    if bands:
        print()
        print(f"  {C.DIM}Band Powers:{C.END}")
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            v = bands.get(band, 0)
            print(f"    {band:>6}: {format_bar(v, max_val=50, width=30)} {v:.1f}")

    # ── Legend ──
    print()
    print(f"  {C.DIM}~ = smoothed by transition engine  ? = low confidence{C.END}")
    print(f"  {C.DIM}Press Ctrl+C to stop session{C.END}")


def run_analysis(signal_data, fs, models, calibration, personalization, features_array):
    """Run all 12 models on EEG data, with optional calibration and personalization."""
    (sleep_model, emotion_model, dream_model, flow_model, creativity_model,
     memory_model, drowsiness_model, cognitive_load_model, attention_model,
     stress_model, lucid_dream_model, meditation_model) = models

    # Run all 12 models
    sleep_result = sleep_model.predict(signal_data, fs)
    emotion_result = emotion_model.predict(signal_data, fs)
    dream_result = dream_model.predict(signal_data, fs)
    flow_result = flow_model.predict(signal_data, fs)
    creativity_result = creativity_model.predict(signal_data, fs)
    memory_result = memory_model.predict(signal_data, fs)
    drowsiness_result = drowsiness_model.predict(signal_data, fs)
    cognitive_load_result = cognitive_load_model.predict(signal_data, fs)
    attention_result = attention_model.predict(signal_data, fs)
    stress_result = stress_model.predict(signal_data, fs)
    lucid_result = lucid_dream_model.predict(
        signal_data, fs,
        is_rem=(sleep_result.get("stage") == "REM"),
        sleep_stage=sleep_result.get("stage_index", 0),
    )
    meditation_result = meditation_model.predict(signal_data, fs)

    predictions = {
        "sleep_staging": sleep_result,
        "emotions": emotion_result,
        "dream_detection": dream_result,
        "flow_state": flow_result,
        "creativity": creativity_result,
        "memory_encoding": memory_result,
        "drowsiness": drowsiness_result,
        "cognitive_load": cognitive_load_result,
        "attention": attention_result,
        "stress": stress_result,
        "lucid_dream": lucid_result,
        "meditation": meditation_result,
    }

    # Apply personalization if available (blends personal + global models)
    if personalization is not None and features_array is not None:
        for pred_key, model_key in [
            ("flow_state", "flow_state"),
            ("creativity", "creativity"),
            ("memory_encoding", "memory_encoding"),
            ("sleep_staging", "sleep_staging"),
            ("emotions", "emotion"),
        ]:
            if pred_key in predictions:
                predictions[pred_key] = personalization.blend(
                    model_key, predictions[pred_key], features_array
                )

    return predictions


def run_calibration(device_type: str, user_id: str):
    """Run the 2-minute calibration protocol."""
    from hardware.brainflow_manager import BrainFlowManager, BRAINFLOW_AVAILABLE

    if not BRAINFLOW_AVAILABLE:
        print(f"{C.RED}Error: BrainFlow not installed.{C.END}")
        sys.exit(1)

    print_banner()
    print(f"{C.BOLD}Starting Calibration for user: {user_id}{C.END}\n")
    print("This takes 2 minutes. Follow the instructions for each step.\n")

    # Connect
    manager = BrainFlowManager()
    result = manager.connect(device_type)
    fs = result["sample_rate"]
    window_samples = fs * 4
    print(f"  {C.GREEN}Connected: {result['device']}{C.END}\n")

    manager.start_streaming()
    time.sleep(2)  # Let stream stabilize

    runner = CalibrationRunner(fs=fs)
    quality_checker = SignalQualityChecker(fs=fs)

    for step in CALIBRATION_STEPS:
        name = step["name"]
        instruction = step["instruction"]
        duration = step["duration"]

        print(f"\n{C.BOLD}{C.CYAN}Step: {name}{C.END}")
        print(f"  {instruction}")
        print(f"  Duration: {duration}s")

        # Countdown
        for i in range(3, 0, -1):
            print(f"  Starting in {i}...", end="\r")
            time.sleep(1)
        print(f"  {C.GREEN}GO!{C.END}              ")

        start = time.time()
        epochs_collected = 0

        while time.time() - start < duration:
            data = manager.get_current_data(window_samples)
            if data is None or len(data["signals"]) == 0:
                time.sleep(0.5)
                continue

            signals = np.array(data["signals"])
            if signals.shape[0] > 0:
                channel_data = signals[0]
                if len(channel_data) >= window_samples:
                    # Check quality before using for calibration
                    quality = quality_checker.check_quality(channel_data)
                    if quality["is_usable"]:
                        runner.add_epoch(name, channel_data)
                        epochs_collected += 1
                        remaining = duration - (time.time() - start)
                        print(f"  Epochs: {epochs_collected}  |  Quality: {quality['quality_score']:.0%}  |  {remaining:.0f}s remaining   ", end="\r")

            time.sleep(2)

        print(f"\n  {C.GREEN}Collected {epochs_collected} good epochs{C.END}")

    # Compute and save calibration
    manager.stop_streaming()
    manager.disconnect()

    progress = runner.get_progress()
    if progress["is_complete"]:
        cal = runner.compute_calibration(user_id)
        print(f"\n{C.BOLD}{C.GREEN}Calibration complete!{C.END}")
        print(f"  Alpha reactivity: {cal.alpha_reactivity:.2f}")
        print(f"  Beta reactivity:  {cal.beta_reactivity:.2f}")
        print(f"  Theta/alpha rest: {cal.theta_alpha_ratio_rest:.2f}")
        print(f"\n  Saved to: data/calibrations/{user_id}.json")
        print("  Future sessions will use your personal baseline.")
    else:
        print(f"\n{C.YELLOW}Calibration incomplete — not enough clean epochs.{C.END}")
        print("  Try again in a quieter environment with better electrode contact.")


def run_session(device_type: str, user_id: str, duration: int):
    """Run a live brain session with full accuracy pipeline."""
    from hardware.brainflow_manager import BrainFlowManager, BRAINFLOW_AVAILABLE

    if not BRAINFLOW_AVAILABLE:
        print(f"{C.RED}Error: BrainFlow not installed.{C.END}")
        print("Install with: pip install brainflow")
        sys.exit(1)

    print_banner()

    # ── Initialize accuracy pipeline ──
    print("Loading accuracy pipeline...")

    # 1. Signal quality checker
    quality_checker = SignalQualityChecker(fs=256)  # Will update fs after connect
    print(f"  {C.GREEN}Signal quality gate{C.END}")

    # 2. Per-user calibration
    calibration = UserCalibration.load(user_id)
    if calibration.is_calibrated:
        cal_time = datetime.fromtimestamp(calibration.calibrated_at).strftime("%Y-%m-%d %H:%M")
        print(f"  {C.GREEN}User calibration loaded{C.END} (from {cal_time})")
    else:
        print(f"  {C.YELLOW}No calibration found{C.END} — run with --calibrate first for better accuracy")

    # 3. State transition engine
    state_engine = BrainStateEngine()
    print(f"  {C.GREEN}State transition engine{C.END}")

    # 4. Confidence calibrator
    confidence_cal = ConfidenceCalibrator()
    print(f"  {C.GREEN}Confidence calibration{C.END}")

    # 5. User personalization
    personalization = PersonalizedPipeline(user_id)
    personalization.update_from_feedback()
    p_status = personalization.get_personalization_status()
    n_feedback = p_status["total_feedback"]
    n_personalized = sum(1 for m in p_status["models"].values() if m["is_personalized"])
    if n_feedback > 0:
        print(f"  {C.GREEN}Personalization{C.END}: {n_feedback} feedback entries, {n_personalized} models personalized")
    else:
        print(f"  {C.DIM}No user feedback yet{C.END} — predictions use global models only")

    # ── Initialize 12 ML models ──
    print("\nLoading 12 analysis models...")
    sleep_model = SleepStagingModel()
    emotion_model = EmotionClassifier()
    dream_model = DreamDetector()
    flow_model = FlowStateDetector()
    creativity_model = CreativityDetector()
    memory_model = MemoryEncodingPredictor()
    drowsiness_model = DrowsinessDetector()
    cognitive_load_model = CognitiveLoadEstimator()
    attention_model = AttentionClassifier()
    stress_model = StressDetector()
    lucid_dream_model = LucidDreamDetector()
    meditation_model = MeditationClassifier()
    models = (sleep_model, emotion_model, dream_model, flow_model,
              creativity_model, memory_model, drowsiness_model,
              cognitive_load_model, attention_model, stress_model,
              lucid_dream_model, meditation_model)
    print(f"  {C.GREEN}All 12 models loaded{C.END}")

    # ── Initialize health DB ──
    db = HealthBrainDB()

    # ── Connect to device ──
    manager = BrainFlowManager()
    print(f"\nConnecting to {device_type}...")

    try:
        result = manager.connect(device_type)
        print(f"  {C.GREEN}Connected: {result['device']}{C.END}")
        print(f"  Channels: {result['channels']}, Sample rate: {result['sample_rate']}Hz")
    except Exception as e:
        print(f"  {C.RED}Connection failed: {e}{C.END}")
        if device_type == "muse_2":
            print("\n  Tips for Muse 2:")
            print("  1. Turn on Muse 2 (hold button until LED flashes)")
            print("  2. Make sure Muse is NOT connected to phone app")
            print("  3. macOS: uses native Bluetooth — no dongle needed")
            print("  4. If still fails, try 'muse_2_bled' with BLED112 USB dongle")
        sys.exit(1)

    fs = result["sample_rate"]
    window_samples = fs * 4  # 4-second analysis windows
    quality_checker = SignalQualityChecker(fs=fs, line_freq=60.0)

    # ── Start streaming ──
    print("\nStarting EEG stream...")
    manager.start_streaming()
    print(f"  {C.GREEN}Streaming!{C.END} Waiting 4 seconds for first window...\n")

    # Handle Ctrl+C
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    session_start = time.time()
    session_id = f"session_{int(session_start)}"
    analyses = []
    quality_scores = []
    rejected_epochs = 0

    time.sleep(4)

    try:
        while running:
            if 0 < duration <= (time.time() - session_start):
                break

            # Get latest EEG data
            data = manager.get_current_data(window_samples)
            if data is None or len(data["signals"]) == 0:
                time.sleep(0.5)
                continue

            signals = np.array(data["signals"])

            if signals.shape[0] > 0:
                channel_data = signals[0]

                if len(channel_data) >= window_samples:
                    # ═══ STEP 1: Signal Quality Gate ═══
                    quality = quality_checker.check_quality(channel_data)
                    quality_scores.append(quality["quality_score"])

                    if not quality["is_usable"]:
                        rejected_epochs += 1
                        elapsed = time.time() - session_start
                        print_analysis({}, elapsed, quality, {}, {})
                        time.sleep(2)
                        continue

                    # ═══ STEP 2: Extract & Calibrate Features ═══
                    processed = preprocess(channel_data, fs)
                    features_dict = extract_features(processed, fs)
                    features_array = np.array([v for _, v in sorted(features_dict.items())])
                    band_powers = extract_band_powers(processed, fs)

                    # Apply per-user calibration
                    if calibration.is_calibrated:
                        band_powers = calibration.normalize_band_powers(band_powers)
                        features_array = calibration.normalize_features(features_array)

                    # ═══ STEP 3: Run 6 Models (+ Personalization) ═══
                    analysis = run_analysis(
                        channel_data, fs, models,
                        calibration=calibration,
                        personalization=personalization if n_feedback > 0 else None,
                        features_array=features_array,
                    )

                    # ═══ STEP 4: Confidence Calibration ═══
                    add_uncertainty_labels(analysis, confidence_cal)
                    conf_summary = analysis.pop("_confidence_summary", {})

                    # ═══ STEP 5: State Transition Smoothing ═══
                    smoothed = state_engine.update({
                        "sleep": analysis.get("sleep_staging", {}),
                        "flow": analysis.get("flow_state", {}),
                        "emotion": analysis.get("emotions", {}),
                        "creativity": analysis.get("creativity", {}),
                        "memory": analysis.get("memory_encoding", {}),
                        "dream": analysis.get("dream_detection", {}),
                    })

                    # Check cross-state coherence
                    coherence = state_engine.get_cross_state_coherence()
                    if not coherence["is_coherent"]:
                        analysis["_coherence_warnings"] = coherence["warnings"]

                    analyses.append(analysis)

                    elapsed = time.time() - session_start
                    print_analysis(analysis, elapsed, quality, smoothed, conf_summary)

            time.sleep(2)

    finally:
        # ── Stop and disconnect ──
        print(f"\n\n{C.YELLOW}Stopping session...{C.END}")
        manager.stop_streaming()
        manager.disconnect()

        session_end = time.time()
        session_duration = session_end - session_start

        # ── Store session in health DB ──
        if analyses:
            avg_analysis = {
                "flow_state": {
                    "state": analyses[-1].get("flow_state", {}).get("state", "no_flow"),
                    "flow_score": float(np.mean([a.get("flow_state", {}).get("flow_score", 0) for a in analyses])),
                    "components": analyses[-1].get("flow_state", {}).get("components", {}),
                },
                "creativity": {
                    "state": analyses[-1].get("creativity", {}).get("state", "analytical"),
                    "creativity_score": float(np.mean([a.get("creativity", {}).get("creativity_score", 0) for a in analyses])),
                },
                "memory_encoding": {
                    "state": analyses[-1].get("memory_encoding", {}).get("state", "weak_encoding"),
                    "will_remember_probability": float(np.mean([a.get("memory_encoding", {}).get("will_remember_probability", 0) for a in analyses])),
                },
                "emotions": analyses[-1].get("emotions", {}),
                "sleep_stage": analyses[-1].get("sleep_staging", {}),
                "dream_detection": analyses[-1].get("dream_detection", {}),
            }

            session_data = {
                "session_id": session_id,
                "start_time": session_start,
                "end_time": session_end,
                "duration_seconds": session_duration,
                **avg_analysis,
            }

            db.store_brain_session(user_id, session_data)
            print(f"  {C.GREEN}Session stored in health DB{C.END}")

            # Check for correlations
            insights = db.generate_insights(user_id, days=30)
            if insights and insights[0].get("insight_type") != "info":
                print(f"\n{C.BOLD}Brain-Health Insights:{C.END}")
                for insight in insights:
                    print(f"  - {insight['title']}: {insight['description']}")

        # ── Session Summary ──
        print(f"\n{C.BOLD}Session Summary:{C.END}")
        print(f"  Duration: {session_duration:.0f}s ({session_duration / 60:.1f} min)")
        print(f"  Epochs analyzed: {len(analyses)}")
        print(f"  Epochs rejected (noisy): {rejected_epochs}")
        if quality_scores:
            print(f"  Avg signal quality: {np.mean(quality_scores):.0%}")

        if analyses:
            flow_scores = [a.get("flow_state", {}).get("flow_score", 0) for a in analyses]
            cre_scores = [a.get("creativity", {}).get("creativity_score", 0) for a in analyses]
            mem_scores = [a.get("memory_encoding", {}).get("will_remember_probability", 0) for a in analyses]
            print(f"  Avg flow:       {np.mean(flow_scores):.0%}")
            print(f"  Avg creativity: {np.mean(cre_scores):.0%}")
            print(f"  Avg memory:     {np.mean(mem_scores):.0%}")
            print(f"  Peak flow:      {max(flow_scores):.0%}")

        # ── State Transition Summary ──
        summary = state_engine.get_summary()
        override_info = []
        for name, tracker_summary in summary.items():
            pcts = tracker_summary.get("state_percentages", {})
            dominant = max(pcts, key=pcts.get) if pcts else "?"
            override_info.append(f"{name}: {dominant}")
        if override_info:
            print(f"\n{C.BOLD}Dominant States:{C.END}")
            for info in override_info:
                print(f"  {info}")

        # ── Personalization Status ──
        if n_feedback > 0:
            print(f"\n{C.BOLD}Personalization:{C.END}")
            for model, info in p_status["models"].items():
                if info["feedback_count"] > 0:
                    status = f"{C.GREEN}active{C.END}" if info["is_personalized"] else f"{C.YELLOW}{info['samples_until_personalization']} more needed{C.END}"
                    print(f"  {model}: {info['feedback_count']} feedback → {status}")

        print(f"\n  {C.DIM}Session ID: {session_id}{C.END}")
        print(f"  {C.DIM}Database: {db.db_path}{C.END}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralDreamWorkshop Live Brain Session")
    parser.add_argument("--device", default="synthetic", choices=[
        "synthetic", "muse_2", "muse_2_bled", "muse_s", "muse_s_bled",
        "openbci_cyton", "openbci_ganglion", "openbci_cyton_daisy",
    ], help="EEG device to connect to (default: synthetic)")
    parser.add_argument("--user", default="default", help="User ID for health correlation")
    parser.add_argument("--duration", type=int, default=0, help="Session duration in seconds (0 = unlimited)")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration protocol instead of session")

    args = parser.parse_args()

    if args.calibrate:
        run_calibration(args.device, args.user)
    else:
        run_session(args.device, args.user, args.duration)
