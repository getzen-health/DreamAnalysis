"""Voice journal analyzer — emotion trajectory, topics, and insights.

Processes 2-minute free-form voice recordings to extract:
  - Emotional trajectory: per-segment emotion, valence, arousal, confidence
  - Voice biomarkers: jitter, shimmer, HNR, RMS per segment
  - Transcript: via Whisper (tiny) if available, else empty string
  - Topics: keyword frequency from transcript (stopword-filtered)
  - Summary: dominant emotion, mean valence/arousal, peak stress time
  - Insights: rule-based strings derived from the data — never fabricated

No torch/transformers required at the model level. Whisper is optional.
All features degrade gracefully when dependencies are missing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Whisper setup (tiny model — ~39 MB, CPU-friendly)
# ---------------------------------------------------------------------------
_whisper_model = None
_whisper_available = False

try:
    import whisper as _whisper_lib  # type: ignore

    _whisper_model = _whisper_lib.load_model("tiny")
    _whisper_available = True
    log.info("Whisper tiny model loaded for voice journal transcription")
except Exception as _we:
    log.info("Whisper not available (%s) — transcription disabled", _we)

# ---------------------------------------------------------------------------
# Stopword list (~20 common English words — no NLTK needed)
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "the", "a", "an", "and", "or", "but", "in", "on",
    "at", "to", "of", "for", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "did", "will", "would", "can", "could",
    "that", "this", "with", "from", "not", "so", "if", "as", "just",
    "about", "up", "out", "what", "when", "there", "then", "into",
    "its", "also", "more", "some", "no", "get", "got", "like", "know",
    "think", "really", "very", "much", "well", "even", "back", "still",
    "how", "all", "by", "him", "her", "who", "which", "been", "being",
}

# Minimum word length to count as a topic keyword
_MIN_WORD_LEN = 3


def _extract_topics(transcript: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """Return top_n keywords by frequency after stopword filtering."""
    if not transcript:
        return []

    words = transcript.lower().split()
    counts: Dict[str, int] = {}
    for raw in words:
        # Strip punctuation
        word = "".join(c for c in raw if c.isalpha())
        if len(word) >= _MIN_WORD_LEN and word not in _STOPWORDS:
            counts[word] = counts.get(word, 0) + 1

    if not counts:
        return []

    sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [{"word": w, "count": c} for w, c in sorted_words[:top_n]]


# ---------------------------------------------------------------------------
# Per-segment biomarker extraction (librosa-based, graceful fallback)
# ---------------------------------------------------------------------------

def _extract_segment_biomarkers(
    segment: np.ndarray, sr: int
) -> Dict[str, float]:
    """Extract jitter (approx), shimmer (approx), HNR (approx), and RMS.

    Uses librosa.pyin for F0 estimation when librosa is available.
    Falls back to RMS-only when librosa is missing.
    """
    out: Dict[str, float] = {
        "rms": float(np.sqrt(np.mean(segment ** 2))),
        "jitter": 0.0,
        "shimmer": 0.0,
        "hnr": 0.0,
    }

    try:
        import librosa  # type: ignore

        # F0 via pyin — fmin/fmax roughly cover speech range
        fmin = float(librosa.note_to_hz("C2"))
        fmax = float(librosa.note_to_hz("C7"))
        f0, voiced_flag, _ = librosa.pyin(
            segment.astype(np.float32),
            fmin=fmin,
            fmax=fmax,
            sr=sr,
        )

        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if len(voiced_f0) > 1:
            periods = 1.0 / voiced_f0
            jitter = float(
                np.mean(np.abs(np.diff(periods))) / np.mean(periods)
            )
            out["jitter"] = round(jitter, 6)

        # Shimmer: frame-level RMS amplitude variation over voiced frames
        frame_length = int(sr * 0.025)  # 25 ms
        hop_length = int(sr * 0.010)    # 10 ms
        rms_frames = librosa.feature.rms(
            y=segment.astype(np.float32),
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]
        if len(rms_frames) > 1 and np.mean(rms_frames) > 1e-8:
            shimmer = float(
                np.mean(np.abs(np.diff(rms_frames))) / np.mean(rms_frames)
            )
            out["shimmer"] = round(shimmer, 6)

        # HNR approximation via autocorrelation
        if len(segment) > sr // 10:
            autocorr = np.correlate(segment, segment, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]
            if autocorr[0] > 1e-12:
                lag_min = int(sr / fmax)
                lag_max = int(sr / fmin)
                lag_max = min(lag_max, len(autocorr) - 1)
                if lag_min < lag_max:
                    peak = float(np.max(autocorr[lag_min:lag_max]))
                    noise = float(autocorr[0] - peak)
                    if noise > 1e-12:
                        hnr_ratio = peak / noise
                        out["hnr"] = round(
                            float(10 * np.log10(max(hnr_ratio, 1e-10))), 2
                        )

    except Exception as exc:
        log.debug("Biomarker extraction failed for segment: %s", exc)

    return out


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class VoiceJournalAnalyzer:
    """Analyze a free-form voice journal recording.

    Usage::

        analyzer = VoiceJournalAnalyzer()
        result = analyzer.analyze(audio_array, sr=16000)
    """

    # Segment length in seconds
    SEGMENT_SEC: float = 2.0

    def __init__(self) -> None:
        # Import lazily to avoid circular imports
        self._emotion_model = None

    def _get_emotion_model(self):
        if self._emotion_model is None:
            from models.voice_emotion_model import VoiceEmotionModel  # type: ignore
            self._emotion_model = VoiceEmotionModel()
        return self._emotion_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        include_transcript: bool = True,
    ) -> Dict[str, Any]:
        """Run full journal analysis pipeline.

        Args:
            audio: Mono float32 audio array.
            sr: Sample rate in Hz (default 16000).
            include_transcript: Whether to attempt Whisper transcription.

        Returns:
            Dict with keys: trajectory, biomarker_timeline, transcript,
            topics, summary, insights, duration_sec, segment_count.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duration_sec = len(audio) / sr
        seg_samples = int(self.SEGMENT_SEC * sr)

        # Split into segments
        segments = self._split_segments(audio, seg_samples)

        # Per-segment emotion + biomarkers
        trajectory: List[Dict[str, Any]] = []
        biomarker_timeline: List[Dict[str, Any]] = []

        model = self._get_emotion_model()

        for idx, seg in enumerate(segments):
            time_sec = round(idx * self.SEGMENT_SEC, 2)

            # Emotion
            emo_result = model.predict(seg, sample_rate=sr)
            if emo_result is not None:
                trajectory.append({
                    "time_sec": time_sec,
                    "emotion": emo_result.get("emotion", "neutral"),
                    "valence": round(float(emo_result.get("valence", 0.0)), 4),
                    "arousal": round(float(emo_result.get("arousal", 0.5)), 4),
                    "confidence": round(float(emo_result.get("confidence", 0.5)), 4),
                })
            else:
                # Segment too short or model returned None — skip silently
                pass

            # Biomarkers
            bm = _extract_segment_biomarkers(seg, sr)
            bm["time_sec"] = time_sec
            biomarker_timeline.append(bm)

        # Transcript
        transcript = ""
        if include_transcript:
            transcript = self._transcribe(audio, sr)

        # Topics
        topics = _extract_topics(transcript)

        # Summary
        summary = self._build_summary(trajectory, biomarker_timeline)

        # Insights
        insights = self._generate_insights(trajectory, biomarker_timeline, topics, summary)

        return {
            "duration_sec": round(duration_sec, 2),
            "segment_count": len(segments),
            "trajectory": trajectory,
            "biomarker_timeline": biomarker_timeline,
            "transcript": transcript,
            "topics": topics,
            "summary": summary,
            "insights": insights,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_segments(
        audio: np.ndarray, seg_samples: int
    ) -> List[np.ndarray]:
        """Split audio into non-overlapping segments of seg_samples length."""
        segments = []
        n = len(audio)
        start = 0
        while start + seg_samples <= n:
            segments.append(audio[start : start + seg_samples])
            start += seg_samples
        # Include a tail segment if it is at least half a full segment
        tail = audio[start:]
        if len(tail) >= seg_samples // 2:
            segments.append(tail)
        return segments

    @staticmethod
    def _transcribe(audio: np.ndarray, sr: int) -> str:
        """Transcribe audio using Whisper tiny if available."""
        if not _whisper_available or _whisper_model is None:
            return ""
        try:
            # Whisper expects 16 kHz float32
            audio_16k = audio
            if sr != 16000:
                # Simple linear interpolation resampling
                n_out = int(len(audio) * 16000 / sr)
                audio_16k = np.interp(
                    np.linspace(0, len(audio) - 1, n_out),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)

            result = _whisper_model.transcribe(
                audio_16k.astype(np.float32), fp16=False, language=None
            )
            return result.get("text", "").strip()
        except Exception as exc:
            log.debug("Whisper transcription failed: %s", exc)
            return ""

    @staticmethod
    def _build_summary(
        trajectory: List[Dict[str, Any]],
        biomarker_timeline: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute aggregate statistics from trajectory and biomarkers."""
        if not trajectory:
            return {
                "dominant_emotion": None,
                "mean_valence": None,
                "mean_arousal": None,
                "peak_stress_time_sec": None,
                "valence_trend": None,
            }

        emotions = [t["emotion"] for t in trajectory]
        dominant = max(set(emotions), key=emotions.count)

        valences = [t["valence"] for t in trajectory]
        arousals = [t["arousal"] for t in trajectory]
        mean_valence = round(float(np.mean(valences)), 4)
        mean_arousal = round(float(np.mean(arousals)), 4)

        # Valence trend: positive = improving, negative = declining
        if len(valences) >= 2:
            trend_val = round(float(valences[-1] - valences[0]), 4)
        else:
            trend_val = None

        # Peak stress time: segment with highest arousal + lowest valence
        stress_scores = [
            (t["arousal"] - t["valence"], t["time_sec"])
            for t in trajectory
        ]
        peak_stress_time = max(stress_scores, key=lambda x: x[0])[1] if stress_scores else None

        return {
            "dominant_emotion": dominant,
            "mean_valence": mean_valence,
            "mean_arousal": mean_arousal,
            "peak_stress_time_sec": peak_stress_time,
            "valence_trend": trend_val,
        }

    @staticmethod
    def _generate_insights(
        trajectory: List[Dict[str, Any]],
        biomarker_timeline: List[Dict[str, Any]],
        topics: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> List[str]:
        """Generate rule-based insight strings.

        Only generates a statement when data is actually present.
        Never fabricates values or fills in placeholders.
        """
        insights: List[str] = []

        if not trajectory:
            return insights

        # Dominant emotion insight
        dominant = summary.get("dominant_emotion")
        if dominant and dominant != "neutral":
            insights.append(
                f"Your voice expressed {dominant} as the predominant emotion throughout the journal."
            )

        # Peak stress insight
        peak_stress = summary.get("peak_stress_time_sec")
        if peak_stress is not None:
            insights.append(
                f"Your voice showed the highest stress signal around {peak_stress:.1f} seconds into the recording."
            )

        # Valence trend insight
        trend = summary.get("valence_trend")
        if trend is not None:
            if trend > 0.15:
                insights.append(
                    "Your emotional tone became more positive over the course of the journal."
                )
            elif trend < -0.15:
                insights.append(
                    "Your emotional tone became more negative as the journal progressed."
                )
            else:
                insights.append(
                    "Your emotional tone remained relatively stable throughout the journal."
                )

        # Topic + emotion linkage (only when we have both)
        if topics:
            top_topic = topics[0]["word"]
            mean_valence = summary.get("mean_valence")
            if mean_valence is not None and mean_valence > 0.2:
                insights.append(
                    f'Your voice was most positive when discussing topics related to "{top_topic}".'
                )
            elif mean_valence is not None and mean_valence < -0.2:
                insights.append(
                    f'Your voice reflected tension when discussing topics related to "{top_topic}".'
                )

        # Jitter insight — high jitter often correlates with emotional stress
        jtrs = [b["jitter"] for b in biomarker_timeline if b.get("jitter", 0) > 0]
        if jtrs:
            mean_jitter = float(np.mean(jtrs))
            if mean_jitter > 0.015:
                insights.append(
                    "Voice irregularity (jitter) was elevated, which can indicate emotional agitation or fatigue."
                )

        return insights


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_analyzer_instance: Optional[VoiceJournalAnalyzer] = None


def get_voice_journal_analyzer() -> VoiceJournalAnalyzer:
    """Return the module-level singleton analyzer (lazy init)."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = VoiceJournalAnalyzer()
    return _analyzer_instance
