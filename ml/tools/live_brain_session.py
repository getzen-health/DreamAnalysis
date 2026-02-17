"""Live Brain Session — Real-time EEG → 6-Model Analysis Pipeline.

Connects to a Muse 2 (or synthetic board for testing) via BrainFlow,
streams EEG, runs all 6 models, stores brain sessions, and correlates
with Apple Health data.

Usage:
    # Test with synthetic EEG (no hardware needed):
    python3 ml/tools/live_brain_session.py --device synthetic

    # Real Muse 2 (needs BLED112 dongle):
    python3 ml/tools/live_brain_session.py --device muse_2

    # With Apple Health data already imported:
    python3 ml/tools/live_brain_session.py --device synthetic --user myname
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
from models.sleep_staging import SleepStagingModel
from models.emotion_classifier import EmotionClassifier
from models.dream_detector import DreamDetector
from models.flow_state_detector import FlowStateDetector
from models.creativity_detector import CreativityDetector, MemoryEncodingPredictor
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
║     NeuralDreamWorkshop — Live Brain Session     ║
║          6-Model Real-Time EEG Analysis          ║
╚══════════════════════════════════════════════════╝{C.END}
""")


def format_bar(value, max_val=1.0, width=20):
    """Create a visual bar chart."""
    filled = int((value / max_val) * width) if max_val > 0 else 0
    filled = min(filled, width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def print_analysis(analysis: dict, elapsed: float):
    """Pretty-print the 6-model analysis to terminal."""
    # Clear previous output (move cursor up)
    print("\033[2J\033[H", end="")  # Clear screen

    print_banner()
    print(f"  {C.DIM}Session time: {elapsed:.0f}s | {datetime.now().strftime('%H:%M:%S')}{C.END}")
    print()

    # 1. Sleep Staging
    sleep = analysis.get("sleep_staging", {})
    stage = sleep.get("stage", "?")
    conf = sleep.get("confidence", 0)
    stage_colors = {"Wake": C.GREEN, "N1": C.YELLOW, "N2": C.BLUE, "N3": C.CYAN, "REM": C.RED}
    sc = stage_colors.get(stage, C.END)
    print(f"  {C.BOLD}SLEEP{C.END}      {sc}{stage:>8}{C.END}  {format_bar(conf)} {conf:.0%}")

    # 2. Emotion
    emo = analysis.get("emotions", {})
    emotion = emo.get("emotion", "?")
    valence = emo.get("valence", 0)
    arousal = emo.get("arousal", 0)
    print(f"  {C.BOLD}EMOTION{C.END}    {emotion:>8}  V={valence:+.2f}  A={arousal:+.2f}")

    # 3. Dream Detection
    dream = analysis.get("dream_detection", {})
    dreaming = dream.get("is_dreaming", False)
    dream_conf = dream.get("confidence", 0)
    dream_str = f"{C.RED}DREAMING{C.END}" if dreaming else f"{C.DIM}awake{C.END}"
    print(f"  {C.BOLD}DREAM{C.END}      {dream_str:>17}  {format_bar(dream_conf)} {dream_conf:.0%}")

    # 4. Flow State
    flow = analysis.get("flow_state", {})
    flow_state = flow.get("state", "?")
    flow_score = flow.get("flow_score", 0)
    components = flow.get("components", {})
    flow_colors = {"no_flow": C.DIM, "micro_flow": C.YELLOW, "flow": C.GREEN, "deep_flow": C.CYAN}
    fc = flow_colors.get(flow_state, C.END)
    print(f"  {C.BOLD}FLOW{C.END}       {fc}{flow_state:>8}{C.END}  {format_bar(flow_score)} {flow_score:.0%}")
    if components:
        abs_v = components.get("absorption", 0)
        eff_v = components.get("effortlessness", 0)
        foc_v = components.get("focus_quality", 0)
        print(f"  {C.DIM}           absorb={abs_v:.2f}  effort={eff_v:.2f}  focus={foc_v:.2f}{C.END}")

    # 5. Creativity
    cre = analysis.get("creativity", {})
    cre_state = cre.get("state", "?")
    cre_score = cre.get("creativity_score", 0)
    cre_colors = {"analytical": C.BLUE, "transitional": C.YELLOW, "creative": C.GREEN, "insight": C.RED}
    cc = cre_colors.get(cre_state, C.END)
    print(f"  {C.BOLD}CREATIVE{C.END}   {cc}{cre_state:>8}{C.END}  {format_bar(cre_score)} {cre_score:.0%}")

    # 6. Memory Encoding
    mem = analysis.get("memory_encoding", {})
    mem_state = mem.get("state", "?")
    mem_prob = mem.get("will_remember_probability", 0)
    mem_colors = {"poor_encoding": C.RED, "weak_encoding": C.YELLOW, "active_encoding": C.GREEN, "deep_encoding": C.CYAN}
    mc = mem_colors.get(mem_state, C.END)
    print(f"  {C.BOLD}MEMORY{C.END}     {mc}{mem_state:>8}{C.END}  {format_bar(mem_prob)} {mem_prob:.0%}")

    # Band powers
    bands = emo.get("band_powers", {})
    if bands:
        print()
        print(f"  {C.DIM}Band Powers:{C.END}")
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            v = bands.get(band, 0)
            print(f"    {band:>6}: {format_bar(v, max_val=50, width=30)} {v:.1f}")

    print()
    print(f"  {C.DIM}Press Ctrl+C to stop session{C.END}")


def run_analysis(signal_data, fs, models):
    """Run all 6 models on a chunk of EEG data."""
    sleep_model, emotion_model, dream_model, flow_model, creativity_model, memory_model = models

    # Preprocess
    processed = preprocess(signal_data, fs)
    features = extract_features(processed, fs)
    band_powers = extract_band_powers(processed, fs)

    # 1. Sleep staging
    sleep_result = sleep_model.predict(signal_data, fs)

    # 2. Emotion
    emotion_result = emotion_model.predict(signal_data, fs)

    # 3. Dream detection
    dream_result = dream_model.predict(signal_data, fs)

    # 4. Flow state
    flow_result = flow_model.predict(signal_data, fs)

    # 5. Creativity
    creativity_result = creativity_model.predict(signal_data, fs)

    # 6. Memory encoding
    memory_result = memory_model.predict(signal_data, fs)

    return {
        "sleep_staging": sleep_result,
        "emotions": emotion_result,
        "dream_detection": dream_result,
        "flow_state": flow_result,
        "creativity": creativity_result,
        "memory_encoding": memory_result,
    }


def run_session(device_type: str, user_id: str, duration: int):
    """Run a live brain session."""
    from hardware.brainflow_manager import BrainFlowManager, BRAINFLOW_AVAILABLE

    if not BRAINFLOW_AVAILABLE:
        print(f"{C.RED}Error: BrainFlow not installed.{C.END}")
        print("Install with: pip install brainflow")
        sys.exit(1)

    print_banner()

    # Initialize models
    print("Loading 6 analysis models...")
    sleep_model = SleepStagingModel()
    emotion_model = EmotionClassifier()
    dream_model = DreamDetector()
    flow_model = FlowStateDetector()
    creativity_model = CreativityDetector()
    memory_model = MemoryEncodingPredictor()
    models = (sleep_model, emotion_model, dream_model, flow_model, creativity_model, memory_model)
    print(f"  {C.GREEN}All 6 models loaded{C.END}")

    # Initialize health DB
    db = HealthBrainDB()

    # Connect to device
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
            print("  1. Make sure BLED112 dongle is plugged in")
            print("  2. Turn on Muse 2 (hold button until light)")
            print("  3. Make sure Muse is not connected to phone app")
        sys.exit(1)

    fs = result["sample_rate"]
    window_samples = fs * 4  # 4-second analysis windows

    # Start streaming
    print("\nStarting EEG stream...")
    manager.start_streaming()
    print(f"  {C.GREEN}Streaming!{C.END} Waiting 4 seconds for first window...\n")

    # Handle Ctrl+C gracefully
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    session_start = time.time()
    session_id = f"session_{int(session_start)}"
    analyses = []

    # Wait for initial data
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

            # Use first channel for single-channel models
            if signals.shape[0] > 0:
                channel_data = signals[0]

                if len(channel_data) >= window_samples:
                    analysis = run_analysis(channel_data, fs, models)
                    analyses.append(analysis)

                    elapsed = time.time() - session_start
                    print_analysis(analysis, elapsed)

            time.sleep(2)  # Analyze every 2 seconds

    finally:
        # Stop and disconnect
        print(f"\n\n{C.YELLOW}Stopping session...{C.END}")
        manager.stop_streaming()
        manager.disconnect()

        session_end = time.time()
        session_duration = session_end - session_start

        # Store brain session in health DB
        if analyses:
            # Compute session averages
            avg_analysis = {
                "flow_state": {
                    "state": analyses[-1].get("flow_state", {}).get("state", "no_flow"),
                    "flow_score": np.mean([a.get("flow_state", {}).get("flow_score", 0) for a in analyses]),
                    "components": analyses[-1].get("flow_state", {}).get("components", {}),
                },
                "creativity": {
                    "state": analyses[-1].get("creativity", {}).get("state", "analytical"),
                    "creativity_score": np.mean([a.get("creativity", {}).get("creativity_score", 0) for a in analyses]),
                },
                "memory_encoding": {
                    "state": analyses[-1].get("memory_encoding", {}).get("state", "weak_encoding"),
                    "will_remember_probability": np.mean([a.get("memory_encoding", {}).get("will_remember_probability", 0) for a in analyses]),
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

            # Check for correlations with health data
            insights = db.generate_insights(user_id, days=30)
            if insights and insights[0].get("type") != "info":
                print(f"\n{C.BOLD}Brain-Health Insights:{C.END}")
                for insight in insights:
                    print(f"  - {insight['title']}: {insight['description']}")

        # Session summary
        print(f"\n{C.BOLD}Session Summary:{C.END}")
        print(f"  Duration: {session_duration:.0f}s ({session_duration / 60:.1f} min)")
        print(f"  Analyses: {len(analyses)} windows")
        if analyses:
            flow_scores = [a.get("flow_state", {}).get("flow_score", 0) for a in analyses]
            cre_scores = [a.get("creativity", {}).get("creativity_score", 0) for a in analyses]
            mem_scores = [a.get("memory_encoding", {}).get("will_remember_probability", 0) for a in analyses]
            print(f"  Avg flow:       {np.mean(flow_scores):.0%}")
            print(f"  Avg creativity: {np.mean(cre_scores):.0%}")
            print(f"  Avg memory:     {np.mean(mem_scores):.0%}")
            print(f"  Peak flow:      {max(flow_scores):.0%}")

        print(f"\n  {C.DIM}Session ID: {session_id}{C.END}")
        print(f"  {C.DIM}Database: {db.db_path}{C.END}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralDreamWorkshop Live Brain Session")
    parser.add_argument("--device", default="synthetic", choices=[
        "synthetic", "muse_2", "muse_s", "openbci_cyton", "openbci_ganglion", "neurosky"
    ], help="EEG device to connect to (default: synthetic)")
    parser.add_argument("--user", default="default", help="User ID for health correlation")
    parser.add_argument("--duration", type=int, default=0, help="Session duration in seconds (0 = unlimited)")

    args = parser.parse_args()
    run_session(args.device, args.user, args.duration)
