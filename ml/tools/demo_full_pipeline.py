"""Full Pipeline Demo — Apple Health + Brain Analysis + Insights.

Demonstrates the entire NeuralDreamWorkshop pipeline working end-to-end:
1. Imports simulated Apple Health data (heart rate, steps, sleep, HRV)
2. Runs brain analysis sessions using simulated EEG
3. Generates brain-health correlation insights
4. Shows daily summaries and trends

Run: python3 ml/tools/demo_full_pipeline.py
"""

import sys
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from health.apple_health import HEALTHKIT_TYPE_MAP
from health.correlation_engine import HealthBrainDB
from simulation.eeg_simulator import simulate_eeg
from models.sleep_staging import SleepStagingModel
from models.emotion_classifier import EmotionClassifier
from models.dream_detector import DreamDetector
from models.flow_state_detector import FlowStateDetector
from models.creativity_detector import CreativityDetector, MemoryEncodingPredictor


C_BOLD = "\033[1m"
C_GREEN = "\033[92m"
C_CYAN = "\033[96m"
C_YELLOW = "\033[93m"
C_DIM = "\033[2m"
C_END = "\033[0m"

USER_ID = "demo_user"
DEMO_DAYS = 14


def log(msg):
    print(f"  {msg}")


def generate_health_data(db: HealthBrainDB, days: int):
    """Generate realistic Apple Health data for N days."""
    print(f"\n{C_BOLD}Step 1: Importing Apple Health Data ({days} days){C_END}")

    now = time.time()
    total_samples = 0

    for day_offset in range(days, 0, -1):
        day_start = now - (day_offset * 86400)
        date_str = datetime.fromtimestamp(day_start).strftime("%Y-%m-%d")

        # Heart rate: samples throughout the day (60-100 bpm range)
        base_hr = random.uniform(62, 78)
        for hour in range(7, 23):  # 7am to 11pm
            ts = day_start + hour * 3600 + random.randint(0, 3599)
            # Higher HR during exercise hours
            exercise_bump = 30 if hour in (8, 17, 18) and random.random() > 0.5 else 0
            hr = base_hr + random.gauss(0, 5) + exercise_bump
            db.store_health_samples(USER_ID, "heart_rate", [{
                "timestamp": ts, "value": max(50, min(180, hr)),
                "source": "Apple Watch",
            }])
            total_samples += 1

        # Steps: daily total (3000-15000)
        is_active_day = random.random() > 0.4
        steps = random.randint(7000, 15000) if is_active_day else random.randint(2000, 6000)
        db.store_health_samples(USER_ID, "steps", [{
            "timestamp": day_start + 12 * 3600, "value": steps,
            "source": "iPhone",
        }])
        total_samples += 1

        # HRV (SDNN): higher = better recovery
        hrv = random.uniform(25, 80)
        db.store_health_samples(USER_ID, "hrv_sdnn", [{
            "timestamp": day_start + 8 * 3600, "value": hrv,
            "source": "Apple Watch",
        }])
        total_samples += 1

        # Sleep analysis: 6-8 hours
        sleep_start = day_start - random.uniform(5, 8) * 3600  # Previous night
        sleep_hours = random.uniform(5.5, 8.5)
        db.store_health_samples(USER_ID, "sleep_analysis", [{
            "timestamp": sleep_start, "end_timestamp": sleep_start + sleep_hours * 3600,
            "value": sleep_hours, "source": "Apple Watch",
        }])
        total_samples += 1

        # Exercise minutes
        exercise_min = random.randint(30, 90) if is_active_day else random.randint(0, 20)
        db.store_health_samples(USER_ID, "exercise_minutes", [{
            "timestamp": day_start + 17 * 3600, "value": exercise_min,
            "source": "Apple Watch",
        }])
        total_samples += 1

        # Resting heart rate
        rhr = base_hr - random.uniform(5, 15)
        db.store_health_samples(USER_ID, "resting_heart_rate", [{
            "timestamp": day_start + 7 * 3600, "value": max(45, rhr),
            "source": "Apple Watch",
        }])
        total_samples += 1

    log(f"{C_GREEN}Imported {total_samples} health samples across {days} days{C_END}")
    log("Metrics: heart_rate, steps, hrv_sdnn, sleep, exercise_minutes, resting_hr")


def generate_brain_sessions(db: HealthBrainDB, days: int, models):
    """Generate brain analysis sessions for N days."""
    print(f"\n{C_BOLD}Step 2: Running Brain Analysis Sessions ({days} days){C_END}")

    sleep_model, emotion_model, dream_model, flow_model, creativity_model, memory_model = models

    now = time.time()
    total_sessions = 0
    states = ["rest", "focus", "meditation", "rem", "deep_sleep", "light_sleep", "stress"]

    for day_offset in range(days, 0, -1):
        day_start = now - (day_offset * 86400)
        date_str = datetime.fromtimestamp(day_start).strftime("%Y-%m-%d")

        # 2-4 sessions per day at different times
        n_sessions = random.randint(2, 4)
        session_hours = sorted(random.sample(range(8, 23), n_sessions))

        for hour in session_hours:
            ts = day_start + hour * 3600

            # Pick a brain state (weighted by time of day)
            if hour < 10:
                state = random.choice(["rest", "meditation", "focus"])
            elif hour < 14:
                state = random.choice(["focus", "focus", "stress", "rest"])
            elif hour < 18:
                state = random.choice(["focus", "rest", "meditation"])
            else:
                state = random.choice(["rest", "light_sleep", "rem", "deep_sleep"])

            # Simulate EEG for this state
            eeg_data = simulate_eeg(state=state, duration=4.0, fs=256, n_channels=1)
            signal = np.array(eeg_data["signals"][0])

            # Run all 6 models
            sleep_result = sleep_model.predict(signal, 256)
            emotion_result = emotion_model.predict(signal, 256)
            dream_result = dream_model.predict(signal, 256)
            flow_result = flow_model.predict(signal, 256)
            creativity_result = creativity_model.predict(signal, 256)
            memory_result = memory_model.predict(signal, 256)

            session_data = {
                "session_id": f"demo_{int(ts)}",
                "start_time": ts,
                "end_time": ts + random.uniform(900, 3600),
                "duration_seconds": random.uniform(900, 3600),
                "flow_state": flow_result,
                "creativity": creativity_result,
                "memory_encoding": memory_result,
                "emotions": emotion_result,
                "sleep_stage": sleep_result,
                "dream_detection": dream_result,
            }

            db.store_brain_session(USER_ID, session_data)
            total_sessions += 1

    log(f"{C_GREEN}Generated {total_sessions} brain sessions across {days} days{C_END}")


def show_results(db: HealthBrainDB):
    """Display the full pipeline results."""

    # Daily summary for today
    print(f"\n{C_BOLD}Step 3: Daily Summary (today){C_END}")
    summary = db.get_daily_summary(USER_ID)

    if summary["brain"]["total_sessions"] > 0:
        brain = summary["brain"]
        log(f"Brain sessions: {brain['total_sessions']}")
        log(f"Avg flow score: {brain['avg_flow_score'] or 'N/A'}")
        log(f"Avg creativity: {brain['avg_creativity_score'] or 'N/A'}")
        log(f"Avg memory:     {brain['avg_encoding_score'] or 'N/A'}")
        log(f"Time in flow:   {brain['time_in_flow_minutes']:.0f} min")
        log(f"Dreams detected: {brain['dreams_detected']}")
    else:
        log(f"{C_DIM}No brain sessions for today{C_END}")

    if summary["health"]:
        log("\nHealth metrics today:")
        for metric, stats in summary["health"].items():
            log(f"  {metric}: avg={stats['average']}, range=[{stats['min']}-{stats['max']}]")

    # Generate insights
    print(f"\n{C_BOLD}Step 4: Brain-Health Correlation Insights{C_END}")
    insights = db.generate_insights(USER_ID, days=DEMO_DAYS)

    if insights:
        for i, insight in enumerate(insights, 1):
            strength = insight.get("correlation_strength", 0)
            strength_bar = "●" * int(strength * 5) + "○" * (5 - int(strength * 5))
            print(f"\n  {C_CYAN}Insight #{i}: {insight['title']}{C_END}")
            print(f"  {insight.get('description', '')}")
            print(f"  {C_DIM}Strength: [{strength_bar}] {strength:.2f}  |  Evidence: {insight.get('evidence_count', 0)} data points{C_END}")
    else:
        log("No insights generated (need more data)")

    # Brain trends
    print(f"\n{C_BOLD}Step 5: Brain State Trends ({DEMO_DAYS} days){C_END}")
    trends = db.get_brain_trends(USER_ID, days=DEMO_DAYS)

    if trends["trends"]:
        log(f"{'Date':<12} {'Flow':>6} {'Create':>8} {'Memory':>8} {'Valence':>9} {'Sessions':>10}")
        log("-" * 60)
        for day in trends["trends"][-7:]:  # Last 7 days
            flow_v = f"{day['avg_flow']:.2f}" if day["avg_flow"] else "  -"
            cre_v = f"{day['avg_creativity']:.2f}" if day["avg_creativity"] else "  -"
            mem_v = f"{day['avg_encoding']:.2f}" if day["avg_encoding"] else "  -"
            val_v = f"{day['avg_valence']:.2f}" if day["avg_valence"] else "  -"
            log(f"{day['day']:<12} {flow_v:>6} {cre_v:>8} {mem_v:>8} {val_v:>9} {day['sessions']:>10}")
    else:
        log("No trend data available")

    # What metrics are supported
    print(f"\n{C_BOLD}Supported Metrics:{C_END}")
    log(f"Apple Health import: {len(HEALTHKIT_TYPE_MAP)} types")
    log("  Heart rate, HRV, sleep, steps, calories, blood oxygen, respiratory rate, etc.")
    log("Brain analysis: 6 models")
    log("  Sleep staging, emotion, dream, flow, creativity, memory encoding")
    log("Correlation engine: 7 insight types")
    log("  Exercise→flow, HRV→creativity, sleep→memory, HR→arousal, meditation→flow, peak hour, dreams")

    print(f"\n{C_GREEN}{C_BOLD}Pipeline demo complete!{C_END}")
    print(f"\n{C_DIM}To import YOUR Apple Health data:{C_END}")
    print("  1. iPhone → Settings → Health → Export All Health Data")
    print("  2. Send the ZIP to your Mac")
    print("  3. python3 ml/tools/import_apple_health.py ~/Downloads/export.zip")
    print(f"\n{C_DIM}To run a live brain session:{C_END}")
    print("  python3 ml/tools/live_brain_session.py --device synthetic")
    print("  python3 ml/tools/live_brain_session.py --device muse_2")
    print()


def main():
    # Use a temp DB for demo
    db_path = str(Path(__file__).parent.parent / "data" / "demo_health_brain.db")

    # Clean previous demo
    if Path(db_path).exists():
        Path(db_path).unlink()

    db = HealthBrainDB(db_path=db_path)

    print(f"\n{C_BOLD}{C_CYAN}══════════════════════════════════════════════════{C_END}")
    print(f"{C_BOLD}{C_CYAN}  NeuralDreamWorkshop — Full Pipeline Demo{C_END}")
    print(f"{C_BOLD}{C_CYAN}══════════════════════════════════════════════════{C_END}")
    print(f"\n  {C_DIM}Simulating {DEMO_DAYS} days of Apple Health + Brain data{C_END}")

    # Step 1: Health data
    generate_health_data(db, DEMO_DAYS)

    # Load models
    print("\n  Loading 6 analysis models...")
    models = (
        SleepStagingModel(),
        EmotionClassifier(),
        DreamDetector(),
        FlowStateDetector(),
        CreativityDetector(),
        MemoryEncodingPredictor(),
    )
    print(f"  {C_GREEN}All 6 models loaded{C_END}")

    # Step 2: Brain sessions
    generate_brain_sessions(db, DEMO_DAYS, models)

    # Steps 3-5: Results
    show_results(db)

    # Cleanup demo DB
    Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
