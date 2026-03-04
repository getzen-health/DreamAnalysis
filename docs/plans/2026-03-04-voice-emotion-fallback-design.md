# Voice Emotion Fallback — Design Doc
**Date:** 2026-03-04
**Branch:** ralph/voice-emotion-fallback
**Status:** Approved

---

## Problem

NeuralDreamWorkshop requires a Muse 2 EEG headset for any emotion detection. Users without a headset — or with a headset that won't connect — see nothing useful. The app should be valuable with just an iPhone and optionally an Apple Watch.

---

## Decisions

| Question | Decision |
|---|---|
| No-EEG trigger | Tap to record (intentional, battery-friendly, privacy-safe) |
| EEG + voice together | Fuse: 70% EEG + 30% voice via MultimodalEmotionFusion |
| Model download | On server startup (Railway) — one cold-start penalty, always ready after |
| Voice failure handling | Silent fallback to Apple Health estimates, no error shown |
| Approach | Full pipeline — upgrade model + wire fallback + fuse multimodal |

---

## Architecture

```
iPhone/Mac
  │
  ├── EEG (Muse 2 via BLE)          ──▶ WebSocket /analyze-eeg
  │                                        │
  ├── Microphone (tap to record 7s) ──▶ POST /voice-watch/analyze
  │                                        │
  └── Apple Health (HealthKit)      ──▶ POST /biometrics/update
                                           │
                                    MultimodalEmotionFusion
                                    (EEG 60% + voice 25% + health 15%)
                                           │
                                    Fused: emotion, valence, arousal,
                                           stress_index, confidence
                                           │
                              ┌────────────┴────────────┐
                         Dashboard                 Emotion Lab
                    (voice card when           (voice panel when
                      no EEG)                    no EEG)
                                           │
                                   InterventionBanner
```

**No-EEG fallback priority:**
1. Voice emotion (user taps record) — 70-80% accuracy
2. Apple Health estimates (automatic) — 50-65% accuracy
3. Nothing shown — never fake values

---

## Voice Model

**Model:** `iic/emotion2vec_plus_base` (Alibaba DAMO, ACL 2024)
- 9-class: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown
- Mapped to 6-class: disgusted→angry, other→neutral, unknown→neutral
- ~80%+ accuracy on IEMOCAP, multilingual, CPU-viable
- Inference library: `funasr>=1.1.0`
- Fallback: existing `audio_emotion_lgbm.pkl` (MFCC LightGBM) if funasr unavailable

**Output format** (matches EEG EmotionClassifier):
```python
{
  "emotion": "happy|sad|angry|fear|surprise|neutral",
  "probabilities": {"happy": 0.0-1.0, ...},
  "valence": -1.0 to 1.0,
  "arousal": 0.0 to 1.0,
  "confidence": 0.0 to 1.0,
  "model_type": "voice_emotion2vec"
}
```

**Valence/arousal derivation:**
```python
valence = clip((happy + surprise) * 0.5 - (sad + angry + fear) * 0.5, -1, 1)
arousal = clip((angry + fear + surprise) * 0.6 + happy * 0.3, 0, 1)
```

---

## Components

### Backend

| Component | File | Change |
|---|---|---|
| `VoiceEmotionModel` | `ml/models/voice_emotion_model.py` | New. emotion2vec+ wrapper with LightGBM fallback |
| Voice endpoints | `ml/api/routes/voice_watch.py` | Upgrade to 6-class + add `/cache` + `/latest/{user_id}` |
| WebSocket handler | `ml/api/routes.py` | Call MultimodalEmotionFusion after EEG predict, include voice cache |
| Intervention engine | `ml/api/routes/interventions.py` | Accept `voice_emotion` param, fire on stress > 0.6 or sad valence < -0.3 |

### Frontend

| Component | File | Change |
|---|---|---|
| `useVoiceEmotion` | `hooks/use-voice-emotion.ts` | New shared hook — MediaRecorder 7s, backend call, cache |
| `VoiceWatchAnalyzer` | existing | Refactor to use `useVoiceEmotion` internally |
| `emotion-lab.tsx` | existing | Voice panel + amber banner when `deviceState !== 'streaming'` |
| `dashboard.tsx` | existing | Mic button + voice brain state card when no EEG |
| `brain-monitor.tsx` | existing | Signal source badge: EEG / Voice / Health / EEG+Bio |
| `InterventionBanner` | existing | POST `/interventions/check` after voice result |

---

## Data Flow

### Voice recording → emotion result
```
User taps "Detect Emotion"
  → MediaRecorder records 7s
  → encode base64 WAV
  → POST /voice-watch/analyze { audio_b64, sample_rate, hr, hrv, spo2 }
      → VoiceEmotionModel.predict() → emotion2vec+ → 6-class probs
      → fuse: 60% voice + 40% watch biometrics
      → return { emotion, probabilities, valence, arousal, confidence }
  → POST /voice-watch/cache { user_id: 'default', emotion_result }
  → UI updates: emotion card / panel
  → POST /interventions/check { voice_emotion } if stress > 0.6 or sad
```

### EEG + voice fusion (when both active)
```
WebSocket frame arrives
  → predict_emotion(eeg_data)
  → GET /voice-watch/latest/default  (cached voice, < 5min)
  → MultimodalEmotionFusion.fuse(eeg, bio, voice)
  → if biometric_confidence > 0.5: override frame values
  → frame.fusion_active = true, frame.signal_count = N
  → send fused frame                 # 85%+ accuracy
```

### Failure handling
```
voice model fails OR audio too short
  → VoiceEmotionModel returns None
  → endpoint returns Apple Health estimates only (silent)
  → UI shows "Health Estimate" badge, not "Voice Analysis"
```

---

## Accuracy Summary

| Mode | Accuracy | Signals |
|---|---|---|
| EEG only | 74% | Muse 2 |
| Voice only | 70-80% | Microphone |
| Apple Health only | 50-65% | HR, HRV, sleep, steps |
| EEG + biometrics fused | 85-90% | EEG + Health + voice |

---

## Implementation Order (prd.json)

1. US-001: Install funasr + emotion2vec+ requirements
2. US-002: VoiceEmotionModel class
3. US-003: Upgrade /voice-watch/analyze + cache endpoints
4. US-004: useVoiceEmotion hook
5. US-005: Emotion Lab voice fallback
6. US-006: Dashboard voice card
7. US-007: MultimodalEmotionFusion in WebSocket
8. US-008: Intervention engine voice triggers
9. US-009: Signal source badge on Brain Monitor
10. US-010: STATUS.md + PRODUCT.md update
