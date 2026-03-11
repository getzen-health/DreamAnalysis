# Competitive Analysis — AntarAI (2026)

## Market Landscape

Consumer brain-computer interface (BCI) and emotion-tracking apps are fragmenting into three lanes: **EEG meditation tools**, **HRV/wearable stress monitors**, and **manual mood-tracking apps**. AntarAI is the product positioned to fuse all three with an ML pipeline and longitudinal emotion tracking.

---

## Direct Competitors

### 1. Muse S Athena (~$400 headband + subscription)

| Dimension | Muse S Athena | NDW |
|---|---|---|
| Hardware | 4-ch EEG + fNIRS (SpO2 proxy) | Muse 2 (user-owned), no upsell |
| Primary use case | Meditation guidance during session | All-day emotion + sleep + food tracking |
| Emotion detection | Binary calm/active | 6-class emotion + valence/arousal |
| Sleep analysis | Sleep tracking (basic) | Sleep staging, dream detection, REM |
| Food-emotion link | None | Food-emotion correlation ML model |
| Supplement tracking | None | Full supplement-emotion correlation |
| Interventions | Guided meditation in-app | Breathing, neurofeedback, binaural |
| AI companion | None | GPT-powered wellness chat |
| Our edge | 🟢 Downstream effect tracking — not just what happens during meditation, but how meditation shifts emotion for the next 4 hours |

### 2. Emotiv Insight (~$300 headband)

| Dimension | Emotiv Insight | NDW |
|---|---|---|
| Hardware | 5-ch semi-dry EEG | Muse 2 (user-owned) |
| Channels | 5 (AF3, AF4, T7, T8, Pz) | 4 (TP9, AF7, AF8, TP10) |
| Target user | Developer / researcher | Consumer / wellness user |
| Key features | Attention, stress, engagement scores | Full emotion taxonomy + health fusion |
| EI scoring | None | 5-dimension EI composite |
| Food-emotion | None | Yes |
| Interpersonal EI | None | Dyadic voice analysis (social_emotion) |
| Price | $300 hardware | $0 hardware if user has Muse 2 |
| Our edge | 🟢 No EI scoring, no food-emotion, no health integration — positioned for devs, not wellness consumers |

### 3. FocusCalm (~$150 headband)

| Dimension | FocusCalm | NDW |
|---|---|---|
| Hardware | 4-ch dry EEG | Muse 2 (user-owned) |
| Primary use case | Eyes-open focus training (games) | Broad emotion + sleep + lifestyle |
| Emotion detection | Focus score only | 6-class + valence/arousal + 16 models |
| Sleep | None | Sleep staging, dream detection, REM |
| Voice emotion | None | Voice check-ins, prosodic biomarkers |
| Health fusion | None | HRV + sleep + activity + food |
| Training protocol | Neurofeedback games | Full neurofeedback protocol suite |
| Our edge | 🟢 Narrowly focused on one metric (focus); we track the full emotional life arc |

### 4. Neurosity Crown (~$800)

| Dimension | Neurosity Crown | NDW |
|---|---|---|
| Hardware | 8-ch research-grade | Muse 2 (user-owned) |
| Target user | Developer / BCI engineer | Consumer |
| SDK quality | Excellent (Node.js, Python) | FastAPI + React |
| Wellness features | None (raw data platform) | 17 models, full UX |
| Price | $800 hardware | $0 hardware |
| Our edge | 🟢 Developer tool only; no consumer product, no wellness UX, no interventions |

### 5. Bearable (Free / $5/mo)

| Dimension | Bearable | NDW |
|---|---|---|
| Hardware required | None (manual) | Muse 2 (optional) |
| Data collection | Manual logs | EEG + voice + wearable + manual |
| Emotion detection | Self-report only | Objective ML from brain/voice signals |
| Supplement tracking | Yes (manual) | Yes (+ correlation with EEG biomarkers) |
| Our edge | 🟢 Objective measurement vs. subjective self-report; Bearable can't detect unconscious states |

---

## Feature Matrix

| Feature | NDW | Muse Athena | Emotiv | FocusCalm | Bearable |
|---|:---:|:---:|:---:|:---:|:---:|
| 6-class emotion | ✅ | ❌ | ❌ | ❌ | ❌ |
| Valence / arousal | ✅ | ❌ | Partial | ❌ | ❌ |
| Sleep staging | ✅ | ❌ | ❌ | ❌ | ❌ |
| Dream detection | ✅ | ❌ | ❌ | ❌ | ❌ |
| Food-emotion link | ✅ | ❌ | ❌ | ❌ | ❌ |
| Supplement correlation | ✅ | ❌ | ❌ | ❌ | Partial |
| HRV fusion | ✅ | ❌ | ❌ | ❌ | ❌ |
| Voice emotion | ✅ | ❌ | ❌ | ❌ | ❌ |
| Neurofeedback | ✅ | ❌ | ❌ | ✅ | ❌ |
| EI composite score | ✅ | ❌ | ❌ | ❌ | ❌ |
| Interpersonal EI | ✅ | ❌ | ❌ | ❌ | ❌ |
| AI wellness chat | ✅ | ❌ | ❌ | ❌ | ❌ |
| No hardware required | ✅ (voice mode) | ❌ | ❌ | ❌ | ✅ |
| Open-source ML | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## Unique Value Propositions by Lane

### For the meditation user (vs. Muse Athena)
> "Muse tells you how calm you were during your 10-minute session. NDW tells you how that session changed your emotional state for the rest of the day — and whether it worked."

### For the developer / researcher (vs. Emotiv)
> "Emotiv gives you raw signals. NDW gives you 16 trained models, a FastAPI backend, and a full React dashboard — skip 6 months of ML pipeline work."

### For the focus optimizer (vs. FocusCalm)
> "FocusCalm trains one metric. NDW tracks the full emotional life cycle: sleep → morning state → stress triggers → interventions → recovery."

### For the mood tracker (vs. Bearable)
> "Bearable records what you tell it. NDW detects what your brain is actually doing — even when you don't notice it yourself."

---

## Gaps We Need to Close

| Gap | Priority | Timeline |
|---|---|---|
| Hardware friction (user must own Muse 2) | High | Explore loaner/rental program for pilot |
| No mobile app (web only) | High | See issue #201 (App Store) |
| No clinical validation | Medium | See issue #200 (10-person pilot) |
| Accuracy not yet peer-reviewed | Medium | Paper draft in progress |
| No community features (vs. Oura/Whoop social) | Low | Community endpoint shipped in #235 |

---

## Positioning Statement

**AntarAI** combines real-time EEG, voice biomarkers, and health data into a unified ML pipeline for full-spectrum emotional intelligence tracking — from brain wave to behavior.

*Not just "how calm are you" — but "why, what triggered it, what helps, and how you're trending over time."*
