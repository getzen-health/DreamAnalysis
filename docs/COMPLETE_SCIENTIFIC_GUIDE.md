# Neural Dream Workshop — Complete Technical & Scientific Guide

## Svapnastra: The Brain-Computer Interface Dream Platform

> *"Animals sense human emotional shifts before we consciously notice them. They read physiological signals that change seconds before conscious awareness. EEG does the same."*

---

# PART I — THE HUMAN BRAIN: FOUNDATIONS

## 1. Brain Waves: The Language of the Mind

Your brain is a storm of electrical activity. Billions of neurons fire in rhythmic patterns, producing waves at different speeds. A Muse 2 EEG headset reads these waves from your scalp at 256 samples per second.

| Wave | Frequency | Speed | What Your Brain Is Doing |
|------|-----------|-------|--------------------------|
| **Delta** | 0.5 – 4 Hz | Slowest | Deep dreamless sleep, unconscious healing, physical restoration |
| **Theta** | 4 – 8 Hz | Slow | Drowsiness, dreaming, deep relaxation, creativity, memory encoding |
| **Alpha** | 8 – 12 Hz | Medium | Calm wakefulness, eyes closed, relaxed focus, meditation |
| **Beta** | 12 – 30 Hz | Fast | Active thinking, problem-solving, stress, alertness, anxiety |
| **Gamma** | 30 – 100 Hz | Fastest | Peak cognition, "aha" moments, spiritual experiences, binding information |

**The fundamental principle**: Slower waves = more relaxed/unconscious. Faster waves = more alert/stressed. The *ratio* between waves reveals your mental state.

---

## 2. How EEG Signals Are Processed

### Raw Signal Preprocessing (`eeg_processor.py`)

Before any analysis, raw EEG goes through cleaning:

1. **Bandpass Filter** (1–50 Hz) — Keeps only brain-relevant frequencies. Uses 5th-order Butterworth filter with zero-phase distortion (filtfilt).
2. **Notch Filters** (50 Hz + 60 Hz) — Removes powerline electrical interference. Q-factor = 30 (sharp notch).
3. **Optional: ML Denoiser** — A trained 1D convolutional autoencoder removes artifacts that classical filters miss.
4. **Optional: Artifact Classifier** — Identifies and rejects specific artifact types (blinks, muscle, electrode pops).

### Feature Extraction

From cleaned EEG, the system extracts 17 features:

**Band Powers (5)**: Relative power in each frequency band via Welch's Power Spectral Density method. Each band's power is divided by total power for normalization (0–1 scale).

**Hjorth Parameters (3)**:
- *Activity*: Signal variance — how "active" the brain signal is
- *Mobility*: Rate of change — how quickly the signal is changing
- *Complexity*: How the signal deviates from a pure sine wave

**Spectral Entropy (1)**: Measures signal disorder. 0 = pure tone (focused brain). 1 = white noise (chaotic brain). Computed as normalized Shannon entropy of the PSD.

**Differential Entropy (5)**: Per-band entropy using the formula: `H = 0.5 * log(2 * pi * e * variance)`. Higher values = more information content in that band.

**Band Ratios (3)**:
- Alpha/Beta ratio — relaxation vs. alertness
- Theta/Beta ratio — drowsiness vs. focus (clinical ADHD marker)
- Alpha/Theta ratio — calm awareness vs. dreamy state

---

## 3. Signal Quality Gate (`signal_quality.py`)

Before running ANY model, the system checks if the signal is actually usable. Bad electrode contact produces garbage-in-garbage-out results.

**5 Quality Checks (weighted average):**

| Check | Weight | What It Detects | Threshold |
|-------|--------|-----------------|-----------|
| Amplitude | 25% | Flatline (<1 uV) or saturated (>200 uV) | RMS 5-50 uV ideal |
| Line Noise | 20% | 50/60 Hz powerline contamination | < 40% of total power |
| EMG | 20% | Muscle artifact (>40 Hz content) | < 50% of total power |
| Stationarity | 20% | Motion artifacts, sudden signal changes | Coefficient of variation < 1.0 |
| Eye Blinks | 15% | Excessive blinking (bad forehead contact) | < 30 blinks/min |

**Result**: quality_score (0.0 to 1.0). If below 0.4, analysis is rejected with specific rejection reasons.

---

# PART II — THE 12 MACHINE LEARNING MODELS

The system runs 12 specialized models simultaneously on every chunk of EEG data. Each model has three inference paths: ONNX (fastest) > sklearn .pkl > feature-based fallback.

---

## Model 1: Emotion Classifier

**File**: `emotion_classifier.py`
**Output**: 6 emotions + valence + arousal + stress/focus/relaxation indices
**Trained Model**: LightGBM mega (global PCA 85→80), 74.21% CV cross-subject, trained on 163,534 samples from 9 datasets (DEAP, DREAMER, GAMEEMO, DENS, FACED, SEED-IV, EEG-ER, STEW, Muse-Sub)

### The 6 Emotions

| Emotion | Key Brain Signature | Formula |
|---------|-------------------|---------|
| **Happy** | High alpha, positive valence, moderate arousal | `valence * 0.4 + arousal * 0.3 + alpha * 0.3` |
| **Sad** | High delta, negative valence, low energy | `-valence * 0.4 + (1-arousal) * 0.3 + delta * 0.3` |
| **Angry** | High beta, negative valence, high energy | `-valence * 0.3 + arousal * 0.4 + beta * 0.3` |
| **Fearful** | High beta+gamma, negative, very high energy | `-valence * 0.25 + arousal * 0.35 + gamma * 0.2 + beta * 0.2` |
| **Relaxed** | High alpha, positive valence, low arousal | `valence * 0.3 + (1-arousal) * 0.3 + alpha * 0.4` |
| **Focused** | High beta, low theta, neutral valence | `(1-abs(valence)) * 0.2 + beta * 0.4 + (1-theta) * 0.2 + gamma * 0.2` |

### Valence & Arousal (The 2D Emotion Space)

**Valence** (happy vs. sad, -1 to +1):
```
valence = tanh((alpha - beta) * 2 + (theta - gamma) * 0.5)
```
More alpha than beta = positive mood. More beta than alpha = negative mood.

**Arousal** (energized vs. calm, 0 to 1):
```
arousal = clip(beta + gamma, 0, 1)
```
High-frequency power = high energy.

### Dashboard Indices (0–100)

| Index | Formula | Meaning |
|-------|---------|---------|
| **Stress** | `beta / alpha * 25` | Beta dominance over alpha = stressed |
| **Focus** | `100 - (theta / beta * 50)` | Low theta relative to beta = focused |
| **Relaxation** | `alpha * 100` | Pure alpha power = relaxed |

---

## Model 2: Sleep Staging

**File**: `sleep_staging_model.py`
**Output**: 5 sleep stages per AASM clinical standards
**Architecture**: EEGNet (temporal + spatial convolutions) with ONNX export

### The 5 Sleep Stages

| Stage | Clinical Name | Brain Signature | What's Happening |
|-------|--------------|-----------------|------------------|
| **Wake** | W | High beta, low delta, high alpha if relaxed | Eyes open or closed but awake |
| **N1** | Light Sleep | Theta increase, alpha drops, slow eye movements | Transition to sleep, easily woken |
| **N2** | Sleep | Sleep spindles (12-16 Hz bursts), K-complexes | True sleep onset, body temperature drops |
| **N3** | Deep Sleep | Delta dominance (>50% of signal) | Restorative sleep, growth hormone release |
| **REM** | Rapid Eye Movement | Theta + low beta, low delta, rapid eye movements | Dreaming, memory consolidation, emotional processing |

### Sleep Spindle Detection
- Bandpass filter 11-16 Hz
- Compute wavelet envelope at 13.5 Hz center
- Threshold: mean + 2.0 * standard deviation
- Keep bursts between 0.3-3.0 seconds duration

### K-Complex Detection
- Bandpass filter 0.5-1.5 Hz (very slow waves)
- Find negative peaks > 75 uV amplitude
- Check for positive deflection within 1 second (> 30 uV)

---

## Model 3: Dream Detector

**File**: `dream_detector.py`
**Output**: Binary (dreaming / not dreaming) + REM likelihood + dream intensity + lucidity estimate

### Dream Detection Logic

Dreams occur primarily during REM sleep. The detector identifies REM-like patterns:
- High theta power (dream content generation)
- Low delta power (not in deep sleep)
- Moderate beta (some cognitive activity)
- Low muscle tone (REM atonia)

**REM Likelihood**: Probability of being in REM based on theta/delta ratio and beta presence.

**Dream Intensity**: Scaled from theta power — more theta = more vivid dream imagery.

**Lucidity Estimate**: Based on gamma and frontal beta presence during REM (metacognitive awareness while dreaming).

---

## Model 4: Flow State Detector

**File**: `flow_state_detector.py`
**Output**: 4 flow levels with component breakdown
**Science**: Katahira et al. (2018), Ulrich et al. (2014)

### The 4 Flow States

| State | Score Range | Experience |
|-------|------------|------------|
| **No Flow** | 0 – 0.3 | Distracted, bored, or anxious |
| **Micro Flow** | 0.3 – 0.5 | Brief moments of engagement |
| **Flow** | 0.5 – 0.7 | Sustained "in the zone" state |
| **Deep Flow** | 0.7 – 1.0 | Profound absorption, time distortion |

### 4 Components

**Absorption** (35% weight): Deep engagement — theta increase + beta + gamma activity. Are you pulled into the task?

**Effortlessness** (25% weight): Relaxed concentration — alpha/beta balance. Working hard but not straining.

**Focus Quality** (25% weight): Sustained stable attention — structured beta, low delta, moderate complexity. Steady, not scattered.

**Time Distortion** (15% weight): Altered time perception — theta/alpha ratio shift. When hours feel like minutes.

---

## Model 5: Creativity Detector

**File**: `creativity_detector.py`
**Output**: 4 creativity states
**Science**: Fink & Benedek (2014), Lustenberger et al. (2015)

### The 4 Creativity States

| State | What's Happening |
|-------|-----------------|
| **Analytical** | Logical, sequential, convergent thinking |
| **Transitional** | Between analytical and creative modes |
| **Creative** | Divergent thinking, associative, imaginative |
| **Insight** | "Aha!" moment — gamma burst with theta base |

### Key Components

**Divergent Thinking** (30%): High alpha + low beta = reduced inhibition, free association. The "shower thought" state.

**Insight Potential** (25%): Gamma bursts above baseline + theta presence. The neural signature of breakthrough moments.

**Internal Attention** (25%): High theta + alpha, low beta = mind wandering productively. The daydream state where creativity lives.

**Associative Richness** (20%): Theta-gamma coupling — cross-frequency interaction where distant ideas get linked together.

---

## Model 6: Memory Encoding Predictor

**File**: `creativity_detector.py` (same file, separate class)
**Output**: 4 encoding levels + "will remember" probability
**Science**: Hanslmayr & Staudigl (2014), Nyhus & Curran (2010)

### The 4 Memory States

| State | What's Happening |
|-------|-----------------|
| **Poor Encoding** | Distracted, nothing sticking |
| **Weak Encoding** | Partially attending, some retention |
| **Active Encoding** | Engaged, information being stored |
| **Deep Encoding** | Optimal hippocampal theta — strong memory formation |

### The Science

The "subsequent memory effect" — certain EEG patterns during learning predict whether you'll remember something later:
- **Hippocampal theta** increases during successful encoding
- **Alpha desynchronization** (alpha drops) when you're truly paying attention
- **Theta-gamma coupling** = active information binding in memory networks

**Will-Remember Probability**: A 0.15-0.85 estimate of whether the current moment will be recalled later.

---

## Model 7: Cognitive Load Estimator

**File**: `cognitive_load_estimator.py`
**Output**: 3 load levels with 6 component scores
**Science**: Gevins et al. (1997), Klimesch (1999), Holm et al. (2009)

### The 3 Load Levels

| Level | Index Range | Your Brain |
|-------|-----------|------------|
| **Low** | 0 – 0.3 | Coasting, minimal demand |
| **Moderate** | 0.3 – 0.6 | Engaged, manageable |
| **High** | 0.6 – 1.0 | Overloaded, near capacity |

### The 6 Components

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Working Memory Load | 25% | Theta increase vs personal baseline |
| Task Engagement | 20% | Alpha suppression (alpha drops when engaged) |
| Cognitive Demand | 20% | Theta/alpha ratio — higher = harder task |
| Processing Intensity | 15% | Beta level — active analytical thinking |
| Gamma Activity | 10% | High-demand processing bursts |
| Signal Complexity | 10% | Hjorth complexity — more parallel processes |

---

## Model 8: Attention Classifier

**File**: `attention_classifier.py`
**Output**: 4 attention states with theta/beta ratio tracking
**Science**: Monastra et al. (2005), Clarke et al. (2001) — ADHD clinical literature

### The 4 Attention States

| State | Score Range | Experience |
|-------|-----------|------------|
| **Distracted** | 0 – 0.25 | Mind-wandering, default mode network active |
| **Passive** | 0.25 – 0.50 | Receiving but not actively processing |
| **Focused** | 0.50 – 0.75 | Sustained concentration, task-engaged |
| **Hyperfocused** | 0.75 – 1.0 | Deep tunnel vision, flow-adjacent |

### The Gold Standard: Theta/Beta Ratio

The theta/beta ratio (TBR) is the clinically validated ADHD attention marker. TBR > 4.5 indicates attention deficit (Monastra et al., 2005).

```
TBR score = 1.0 - tanh(theta_beta_ratio / 3.0)
```
Low TBR = focused. High TBR = distracted.

**Attention Stability**: Tracked over time — std of last 10 scores. Stable attention = consistent focus.

---

## Model 9: Stress Detector

**File**: `stress_detector.py`
**Output**: 4 stress levels with 8 component scores + trend
**Science**: Davidson (1992), Al-Shargie et al. (2016), Giannakakis et al. (2019)

### The 4 Stress Levels

| Level | Index Range | Nervous System |
|-------|-----------|---------------|
| **Relaxed** | 0 – 0.20 | Parasympathetic dominance, rest-and-digest |
| **Mild** | 0.20 – 0.45 | Slight arousal, manageable |
| **Moderate** | 0.45 – 0.70 | Sympathetic activation, fight-or-flight beginning |
| **High** | 0.70 – 1.0 | Full fight-or-flight, cortisol surge, cognitive impairment |

### 8 Stress Components

1. **Anxiety Activation** (25%): High-beta (20-30 Hz) elevation above baseline
2. **Alpha Suppression** (20%): Alpha drops when stressed (lose calm)
3. **Arousal Index** (20%): Beta/alpha ratio — the classic stress marker
4. **Theta Stress** (10%): Theta elevates under acute stress
5. **Gamma Suppression** (10%): Gamma drops during chronic stress
6. **Neural Irregularity** (15%): Hjorth complexity — chaotic signals under stress
7. **Cortisol Proxy**: Combined anxiety + alpha suppression (hormonal stress estimate)
8. **Autonomic Index**: Combined arousal + alpha suppression + theta stress

**Stress Trend**: Compares last 5 readings vs. previous 5 readings to show whether stress is rising or falling.

---

## Model 10: Drowsiness Detector

**File**: `drowsiness_detector.py`
**Output**: 3 alertness states with trend tracking
**Science**: Borghini et al. (2014), Lin et al. (2013)

### The 3 Alertness States

| State | Drowsiness Index | What's Happening |
|-------|-----------------|------------------|
| **Alert** | 0 – 0.3 | Fully awake, high beta |
| **Drowsy** | 0.3 – 0.6 | Microsleep onset, theta rising |
| **Sleepy** | 0.6 – 1.0 | Near-sleep, high delta/theta |

### Key Markers

**Theta/Beta Ratio** (30%): Gold standard drowsiness marker. Theta climbs, beta falls as you get sleepy.

**Beta Suppression** (25%): Alertness drops = beta drops below personal baseline.

**Slow Wave Increase** (20%): (delta + theta) / (alpha + beta + gamma). When slow waves dominate, you're falling asleep.

**Theta Trend** (15%): Rate of theta increase over recent history — predicts WHEN you'll fall asleep.

---

## Model 11: Lucid Dream Detector

**File**: `lucid_dream_detector.py`
**Output**: 4 lucidity states (only active during REM sleep)
**Science**: Voss et al. (2009, 2014), Dresler et al. (2012), LaBerge (1990)

### The 4 Lucidity States

| State | Score Range | Experience |
|-------|-----------|------------|
| **Non-Lucid** | 0 – 0.25 | Standard dreaming, no awareness |
| **Pre-Lucid** | 0.25 – 0.50 | Approaching awareness, reality-testing patterns |
| **Lucid** | 0.50 – 0.70 | Aware you're dreaming — gamma bursts in frontal cortex |
| **Controlled** | 0.70 – 1.0 | Actively controlling dream content |

### The Voss Signature

The defining signature of lucid dreaming is a **40 Hz gamma surge** in frontal/temporal regions during REM. This is the brain becoming self-aware while dreaming.

**Key Components**:
- Frontal 40 Hz gamma surge (30%) — the primary lucidity marker
- Metacognition index (20%) — frontal beta during REM (self-awareness)
- Alpha/gamma coupling (15%) — Holzinger signature
- Gamma onset detection (15%) — tracking the *moment* lucidity begins
- Delta reduction (10%) — lighter sleep state
- Theta maintenance (10%) — dream content richness preserved

**Safety**: Non-REM signals automatically return non-lucid with 90% confidence. Lucid dreaming is only possible during REM.

---

## Model 12: Meditation Classifier

**File**: `meditation_classifier.py`
**Output**: 5 meditation depths + tradition matching + session stats
**Science**: Lutz et al. (2004), Cahn & Polich (2006), Travis & Shear (2010)

### The 5 Meditation Depths

| Depth | Score Range | EEG Signature | Experience |
|-------|-----------|---------------|------------|
| **Surface** | 0 – 0.20 | Minimal change from baseline | Basic relaxation, eyes closed |
| **Light** | 0.20 – 0.40 | Alpha increase, slight theta | Mind settling, fewer thoughts |
| **Moderate** | 0.40 – 0.60 | Alpha-theta crossover, reduced beta | Sustained calm, expanded awareness |
| **Deep** | 0.60 – 0.80 | Theta dominance, minimal beta | Inner silence, witness consciousness |
| **Transcendent** | 0.80 – 1.0 | Gamma bursts + theta base | Non-dual awareness, ego dissolution |

### The Lutz Finding

Advanced meditators (Tibetan monks with 10,000+ hours) show gamma power **25x higher** than novices. This is the strongest finding in meditation neuroscience.

### Meditation Tradition Matching

The system identifies which tradition your meditation most resembles:
- **Shamatha** (Focused Attention): High alpha + moderate theta + low beta
- **Vipassana** (Open Monitoring): High theta + wide alpha + calm
- **Dzogchen/Sahaj** (Non-Dual): Gamma bursts + theta-gamma coupling
- **Metta** (Loving-Kindness): Alpha + gamma + high coherence

### Session Progression

A **session bonus** rewards sustained practice — up to +10% score after 20 minutes. Meditation deepens with time.

---

## Supporting Model: Denoising Autoencoder

**File**: `denoising_autoencoder.py`
**Architecture**: 1D Convolutional Autoencoder with U-Net skip connections

### How It Cleans EEG

**Encoder** (compresses signal):
- Conv1D layers with stride-2 downsampling: 1 → 32 → 64 → 128 → 256 channels
- Each layer: BatchNorm + LeakyReLU(0.1)

**Bottleneck**: 256-channel compressed representation of clean signal features

**Decoder** (reconstructs clean version):
- ConvTranspose1D layers with stride-2 upsampling: 256 → 128 → 64 → 32 → 1
- Skip connections from encoder preserve fine temporal details

**Training Loss**: L1 reconstruction + spectral loss (FFT magnitude comparison) * 0.1. Gradient clipping at 1.0. CosineAnnealing learning rate schedule.

**Fallback**: When PyTorch isn't available, falls back to classical bandpass (0.5-50 Hz) + notch filters (50/60 Hz).

---

## Supporting Model: Artifact Classifier

**File**: `artifact_classifier.py`
**Output**: 6 artifact types with 24-feature vector

### The 6 Artifact Types

| Type | What Causes It | Detection Method |
|------|---------------|-----------------|
| **Clean** | Good signal | Low kurtosis, normal amplitude |
| **Eye Blink** | Blinking | High kurtosis (>5.0), frontal delta bursts |
| **Muscle EMG** | Jaw clench, facial tension | High-frequency ratio >35% |
| **Electrode Pop** | Loose electrode | Extreme first-derivative spikes |
| **Motion** | Head movement | High crest factor (>10) |
| **Powerline** | 50/60 Hz interference | 50/60 Hz spectral peaks >15% |

### 24 Feature Vector

Includes: band powers (5), line noise peaks (2), temporal stats (kurtosis, skewness, zero crossings, crest factor, asymmetry), first/second derivative stats (max, std), Hjorth parameters (3), spectral entropy, HF ratio.

---

# PART III — ADVANCED PROCESSING SYSTEMS

## 1. Emotion Shift Detector (`emotion_shift_detector.py`)

> *"The brain's electrical signature shifts BEFORE you realize your mood is changing."*

EEG changes precede conscious emotion awareness by **2-8 seconds**. This module catches those pre-conscious signatures.

### How It Works

1. Maintains a **30-second sliding window** of emotional indicators (updated ~4 times/second)
2. Splits into **baseline** (2-8 seconds ago) and **recent** (last 2 seconds)
3. Computes deltas and matches against 6 known patterns:

| Shift Pattern | EEG Trigger | Body Feeling | Guidance |
|---------------|------------|-------------|----------|
| **Approaching Anxiety** | Stress rises >12pts, beta up, alpha down | Chest tightness, shallow breathing | "Take 3 slow breaths" |
| **Approaching Sadness** | Valence drops >0.15, arousal drops, theta rises | Heaviness, slower thoughts | "Notice without judgment" |
| **Approaching Calm** | Calm ratio rises >0.3, alpha up, beta down | Shoulders relaxing, deeper breath | "Let it happen" |
| **Approaching Focus** | Entropy drops >0.05, beta structured, theta down | Narrowing attention | "Channel it toward what matters" |
| **Approaching Joy** | Valence rises >0.15, gamma bursts | Lightness, warmth, energy | "Savor this" |
| **Emotional Turbulence** | High variability in valence (>0.15) AND arousal (>0.08) | Unsettled, reactive | "Pause. Breathe. Give yourself space." |

### Cooldown & Awareness Score

- **10-second cooldown** between alerts (prevents spam)
- **Emotional Awareness Score**: Tracks how many shifts you observe over time. Like training a muscle — noticing emotions before they hit you is a learnable skill.
- Levels: Beginning → Awakening → Growing Awareness → Deep Awareness

---

## 2. State Transition Engine (`state_transitions.py`)

Prevents rapid "flip-flopping" between states. Real brain states don't change every 250ms — they have inertia.

### Markov Transition Matrices

Each model has a probability matrix governing how likely transitions are. Example (sleep):
- Wake → N3 directly: **impossible** (blocked)
- N3 → REM directly: **impossible** (blocked)
- Deep flow → no flow: **impossible** (must decay gradually)

### Minimum Dwell Times

| Model | Min Time in State |
|-------|------------------|
| Sleep | 30 seconds |
| Flow | 10 seconds |
| Creativity | 8 seconds |
| Memory | 8 seconds |
| Emotion | 6 seconds |

### Exponential Moving Average

Raw predictions are smoothed: `smoothed = 0.3 * new + 0.7 * previous`. This prevents a single noisy reading from flipping the displayed state.

### Cross-State Coherence

The engine checks if states are physiologically plausible together:
- Deep sleep + flow = **impossible** (warning)
- Deep sleep + focused emotion = **unlikely** (warning)
- REM + deep flow = **very unlikely** unless lucid dreaming
- Waking + high dream probability + flow = **artifact** (warning)

---

## 3. Confidence Calibration (`confidence_calibration.py`)

Raw model confidence is often miscalibrated — a model saying "90% confident" might only be right 60% of the time. This module fixes that.

### Platt Scaling

Transforms raw logits through a sigmoid to produce honest probabilities:
```
calibrated = 1 / (1 + exp(-(a * raw_confidence + b)))
```
Parameters a, b are learned from validation data.

### Temperature Scaling

Softens or sharpens probability distributions:
```
calibrated_logits = logits / temperature
```
Temperature > 1.0 = softer (less overconfident). Temperature < 1.0 = sharper.

### Uncertainty Labels

Every prediction gets an uncertainty estimate:
- **Very Confident**: calibrated > 0.85
- **Confident**: calibrated > 0.70
- **Moderate**: calibrated > 0.50
- **Low Confidence**: calibrated > 0.35
- **Very Uncertain**: calibrated <= 0.35

---

## 4. Personal Calibration (`calibration.py`)

Every brain is different. "High theta" for one person might be normal for another. The 4-step calibration protocol establishes YOUR personal baseline:

| Step | Duration | Instruction | What It Measures |
|------|----------|-------------|------------------|
| **Relaxed** | 30s | Close eyes, breathe naturally | Your baseline alpha peak |
| **Focused** | 30s | Count backwards from 100 by 7s | Your baseline beta under load |
| **Stressed** | 30s | Remember a stressful event vividly | Your stress response signature |
| **Recovery** | 30s | Return to calm breathing | How quickly you recover |

After calibration, all features are z-score normalized to YOUR baselines, making every model personalized.

---

## 5. User Feedback & Personalization (`user_feedback.py`)

The system learns from your corrections over time.

### How It Works

1. **You correct a prediction**: "I wasn't stressed, I was focused"
2. Correction is stored with the EEG features that produced it
3. After **15+ corrections per model**, a personal k-NN model is created
4. Personal model blends with global model using weighted averaging:
   - 15 samples: 30% personal, 70% global
   - 100 samples: 60% personal, 40% global
   - 500+ samples: 80% personal, 20% global (never fully replaces global)

### Agreement Boost

When personal and global models agree, confidence gets a 10% boost — they're both seeing the same thing.

---

## 6. Brain Connectivity Analysis (`connectivity.py`)

For multi-channel recordings, analyzes how different brain regions communicate.

### Methods

**Coherence**: How synchronized two brain regions are in a frequency band. High coherence = regions working together.

**Phase Locking Value (PLV)**: Consistency of phase relationship between channels. 1.0 = perfectly locked.

**Granger Causality**: Does activity in region A *predict* future activity in region B? Establishes directed information flow.

**Directed Transfer Function (DTF)**: Multi-channel extension of Granger causality using autoregressive modeling.

### Graph Theory Metrics

- **Clustering Coefficient**: How interconnected neighboring nodes are (local integration)
- **Average Path Length**: How many steps to reach any region from any other (global efficiency)
- **Small-World Index**: Clustering / path_length — small-world networks are both locally and globally efficient
- **Betweenness Centrality**: Which regions are critical "hubs" for information flow

---

# PART IV — SPIRITUAL ENERGY SYSTEM

## Energy Centers / Chakras (`spiritual_energy.py`)

The project maps EEG frequency bands to the traditional 7-chakra system. The mapping is grounded in neuroscience — each chakra corresponds to a specific frequency range.

### The 7 Energy Centers

| # | Chakra | Sanskrit | Frequency | Brain Wave | Element | Color | Qualities |
|---|--------|----------|-----------|------------|---------|-------|-----------|
| 1 | **Root** | Muladhara | 0.5–4 Hz | Delta | Earth | Red | Grounding, stability, security |
| 2 | **Sacral** | Svadhisthana | 4–8 Hz | Theta | Water | Orange | Creativity, emotion, flow |
| 3 | **Solar Plexus** | Manipura | 8–10 Hz | Low Alpha | Fire | Gold | Willpower, confidence, power |
| 4 | **Heart** | Anahata | 10–12 Hz | High Alpha | Air | Green | Love, compassion, connection |
| 5 | **Throat** | Vishuddha | 12–20 Hz | Low Beta | Ether | Cyan | Expression, communication, truth |
| 6 | **Third Eye** | Ajna | 20–40 Hz | High Beta/Gamma | Light | Indigo | Intuition, insight, wisdom |
| 7 | **Crown** | Sahasrara | 40–100 Hz | Gamma | Cosmic | Violet | Transcendence, unity, enlightenment |

### Activation Status
- **70+%**: Highly active
- **40-70%**: Balanced
- **15-40%**: Low activity
- **<15%**: Dormant

### Chakra Balance Analysis

**Harmony Score**: `max(0, 100 - std(activations) * 2)`. More uniform = more harmonious.

**Energy Flow Direction**:
- Upper > lower * 1.3 = **Ascending** (spiritual seeking)
- Lower > upper * 1.3 = **Descending** (physically grounded)
- Equal = **Balanced** (ideal)

### Kundalini Flow

Tracks progressive activation from root to crown:
- **Full Kundalini Flow**: Crown + root active, >70% flow continuity
- **Upper Chakra Activation**: Third eye + heart open
- **Heart-Centered Grounding**: Heart + root active
- **Grounded Foundation**: Root active
- **Gathering Energy**: Building up

### Prana Balance (Hemispheric)

Uses bilateral EEG (left + right hemisphere) to determine energy balance:

| Nadi | Hemisphere | Quality | Guidance |
|------|-----------|---------|----------|
| **Ida** | Right brain dominant | Yin/lunar, receptive, intuitive | Right-nostril breathing to energize |
| **Pingala** | Left brain dominant | Yang/solar, active, analytical | Left-nostril breathing to calm |
| **Sushumna** | Balanced (<5% asymmetry) | Central channel, meditative | Ideal for meditation |

### Aura Visualization

Blends all band powers into an RGB color:
- Delta → Deep Red (physical healing)
- Theta → Orange (creative energy)
- Alpha → Green (heart-centered balance)
- Beta → Royal Blue (mental activity)
- Gamma → Violet (spiritual awareness)

Three layers: inner (strongest band), middle (second), outer (third).

### Consciousness Level (0–1000)

| Score | Level | Dominant Wave |
|-------|-------|--------------|
| 0–100 | Deep Sleep | Delta |
| 100–200 | Drowsy / Hypnagogic | Theta |
| 200–350 | Relaxed Awareness | Alpha |
| 350–500 | Focused Attention | Low Beta |
| 500–650 | Heightened Perception | High Beta |
| 650–800 | Meditative Absorption | Alpha-Theta |
| 800–900 | Transcendent Awareness | Gamma |
| 900–1000 | Cosmic Consciousness | Gamma Bursts |

### Third Eye (Ajna) Activation

Specifically measures high-beta (20-40 Hz) and gamma (40-100 Hz) activity:
```
ajna_raw = high_beta_power * 0.4 + gamma_power * 0.6
```
If alpha baseline is strong (>15%), multiply by 1.3 — calm + gamma = spiritual insight.

---

# PART V — THE APPLICATION: HOW IT HELPS HUMANS

## 13 Application Pages

### 1. Dashboard
The central hub. Shows real-time:
- **Wellness Score** (0-100): 40% relaxation + 35% inverse stress + 25% focus
- **Sleep Score**: Sleep staging confidence
- **Brain Score**: 40% focus + 30% relaxation + 30% inverse arousal
- **Mood Timeline**: Rolling chart of mood and stress
- **Emotional Shift Alerts**: Pre-conscious warnings with guidance
- **AI Insights**: Dynamic brain-state-aware advice (throttled 10s)

### 2. Brain Monitor
Deep EEG analysis showing everything at once:
- Live alpha/beta waveforms (last 50 points)
- All 12 model cards updating in real-time (throttled 5s)
- Band power bar charts (delta through gamma)
- Wavelet spectrogram with sleep spindle/K-complex annotations
- Electrode status grid (Active/Weak/Error per channel)
- Anomaly/seizure risk indicators

### 3. Emotion Lab
Detailed emotion exploration:
- **Emotion Wheel**: SVG radial plot of 6 emotions with probabilities
- **Brain Bands**: 5 frequency bars with labels
- **Mental State**: Stress/Focus/Relaxation progress bars + valence/arousal numbers
- **Timeline**: Stress, focus, relaxation trends over last 30 samples
- **Valence-Arousal Space**: 2D scatter plot — see your emotional trajectory in real-time

### 4. Inner Energy
Spiritual dimension:
- 7 chakra activation bars with Sanskrit names and colors
- Meditation depth gauge (0-100) with stage label
- Consciousness level gauge (0-100) with level name
- Third eye activation gauge
- AI guidance based on dominant chakra + meditation state

### 5. Health Analytics
Composite wellness scoring:
- **Brain Health Score**: Focus + Relaxation + Low Stress + Flow (equally weighted)
- **Cognitive Score**: Focus 30% + Creativity 25% + Memory 25% + Low Drowsiness 20%
- **Wellbeing Score**: Relaxation 35% + Low Stress 35% + Flow 30%
- 8 vital stat cards: Stress, Focus, Flow, Relaxation, Creativity, Memory, Cog Load, Drowsiness
- Timeline charts: Stress vs Relaxation, Focus & Cognitive Load

### 6. Dream Patterns
Sleep architecture visualization:
- **Hypnogram**: Sleep stage timeline (Wake, REM, N1, N2, N3)
- **Dream Activity**: Bar chart of dream intensity
- **REM Likelihood**: Area chart of REM probability
- **REM Cycle Progression**: Intensity + lucidity tracking across cycles
- Session stats: dream frames, avg REM %, intensity, cycle count

### 7. Brain Connectivity
Graph theory analysis (multi-channel):
- Connectivity graph with directed/undirected edges
- Graph metrics: Clustering, Path Length, Small-World Index, Modularity
- Hub node identification
- Connectivity matrix heatmap
- Granger causality flow direction

### 8. Neurofeedback
Real-time brain training:
- **Protocols**: Alpha Enhancement, SMR, Theta/Beta Ratio, Alpha Asymmetry
- 30-second calibration phase
- Live circular score gauge (0-100)
- Reward system: audio tones (523.25 Hz) + visual flashes when hitting targets
- Session summary: rewards, rate, avg score, best streak

### 9. AI Companion
Brain-state-aware conversational AI:
- Chat interface with brain-context responses
- Keyword matching for stress, breathing, meditation, focus
- Quick actions: Breathing (4-7-8 technique), Meditation, Mood check, Stress management
- Live brain metric bars (Stress, Focus, Relaxation, Mood)
- Drowsiness detection with nap suggestions

### 10. Insights
Aggregated brain intelligence:
- AI-generated insight cards (4 dynamic insights, throttled 12s)
- Brain wave trend charts (theta, alpha, beta, delta over 30 samples)
- Brain profile radar: 6-point radar (Focus, Creativity, Relaxation, Memory, Flow, Meditation)
- Current brain state summary

### 11. Dream Journal (API-powered)
AI dream analysis:
- Jungian archetypal analysis + Freudian symbolism + modern neuroscience
- Symbol extraction and frequency tracking
- Emotion intensity mapping
- Waking life connections
- Recurring pattern detection across dreams

### 12. Session History
Recorded session management:
- Browse past EEG sessions with duration, channel count
- Analysis timeline replay
- CSV export for external analysis
- Session comparison and weekly reports

### 13. Settings
Full configuration:
- BCI settings: Electrode count (32/64/128), sampling rate (250/500/1000 Hz), stress threshold
- Theme and animation controls
- Edge inference toggle: Local ONNX vs. Server API
- Privacy: Local processing, encryption, analytics toggles
- Data export (CSV) and clear all data
- Model performance benchmarks table

---

# PART VI — TECHNICAL ARCHITECTURE

## System Architecture

```
 Muse 2 EEG Headset (4 channels, 256 Hz)
          |
          | BrainFlow SDK
          v
 Python ML Backend (FastAPI + WebSocket at ~4Hz)
     |
     |── Signal Quality Gate (reject bad signals)
     |── Preprocessing (bandpass + notch + optional ML denoise)
     |── Per-User Calibration (z-score to personal baseline)
     |── 12 Model Inference (parallel)
     |── Confidence Calibration (Platt scaling)
     |── State Smoothing (Markov transitions + EMA)
     |── Cross-State Coherence (physiological validation)
     |── User Feedback Personalization (k-NN blending)
     |── Emotion Shift Detection (pre-conscious alerts)
     |── Spiritual Energy Analysis (chakra/kundalini/consciousness)
     |
     v
 WebSocket Stream → React Frontend (TypeScript + Vite + Tailwind)
     |
     v
 PostgreSQL Database (Drizzle ORM via Neon Serverless)
     |
     v
 Vercel Deployment (18+ API endpoints + OpenAI GPT-5 integration)
```

## Database Tables

| Table | Purpose | Key Fields |
|-------|---------|------------|
| **users** | User accounts | username, password (scrypt), email |
| **health_metrics** | Physical health data | heartRate, stressLevel, sleepQuality, neuralActivity, dailySteps |
| **dream_analysis** | Dream journal entries | dreamText, symbols (jsonb), emotions (jsonb), aiAnalysis, lucidityScore |
| **dream_symbols** | Recurring dream symbols | symbol, meaning, frequency |
| **emotion_readings** | EEG emotion snapshots | stress, happiness, focus, energy, dominantEmotion, valence, arousal |
| **ai_chats** | AI companion conversations | message, isUser |
| **user_settings** | Preferences | theme, electrodeCount, samplingRate, alertThresholds |

## Health Integration

Supports three health platforms:
- **Apple Health** (HealthKit): Heart rate, HRV, sleep, steps, mindful minutes
- **Google Fit**: Activity, heart rate, sleep
- **Health Connect** (Android): Steps, heart rate, sleep, body metrics

The **Correlation Engine** (SQLite) links brain sessions with health data to discover patterns like: *"Your focus is 40% higher on days when you sleep >7 hours"*.

## Noise Augmentation (`noise_augmentation.py`)

Training data augmentation simulates real-world conditions at 3 difficulty levels:

| Noise Type | Easy (Lab) | Medium (Consumer) | Hard (Worst Case) |
|-----------|-----------|-------------------|-------------------|
| Gaussian SNR | 15-30 dB | 8-20 dB | 3-12 dB |
| Electrode Drift | 20 uV, 20% prob | 50 uV, 50% prob | 100 uV, 80% prob |
| Motion Artifacts | 1 per window | 3 per window | 5 per window |
| Powerline (50/60Hz) | 2 uV | 5 uV | 10 uV |
| Muscle EMG | 10% intensity | 30% intensity | 50% intensity |
| Eye Blinks | 1 blink | 2 blinks | 3 blinks |
| Channel Dropout | 0% | 5% | 10% |

## Model Evolution

The emotion classifier went through 6 iterations:

| Version | Model | Accuracy |
|---------|-------|----------|
| v1 | Random Forest (500 trees) | 85.18% |
| v1.1 | Random Forest (2000 trees) | 86.07% |
| v2 | XGBoost (1000, cleaned data) | 88.22% |
| v3 | LightGBM (1500, ultra-clean) | 94.23% |
| v5 | LightGBM (dataset-aware features) | 94.43% |
| v7 | LightGBM (premium, 8 datasets) | 97.79% (inflated — within-subject + hardcoded CV, deleted) |
| v8 | Mega LGBM (global PCA, 9 datasets) | **74.21% CV** (cross-subject, honest benchmark) |

The v8 unified trainer uses a single global PCA fitted on all data — deployable with a single `.pkl` file.

---

# PART VII — WHAT THIS MEANS FOR HUMANS

## The Vision

Neural Dream Workshop gives humans abilities that were previously invisible:

1. **See your emotions before you feel them** — The emotion shift detector catches mood changes 2-8 seconds before conscious awareness. Like having an emotional early warning system.

2. **Know when your brain is actually learning** — The memory encoding predictor tells you whether information is sticking. Study when your encoding score is high.

3. **Train your brain like a muscle** — Neurofeedback rewards specific brain states. Want more alpha? The system plays a tone every time your alpha rises. Over time, your brain learns.

4. **Understand your sleep architecture** — See exactly when you enter REM, how many cycles you complete, and whether you're getting enough deep sleep.

5. **Detect lucid dreams** — The system identifies the moment you become aware in a dream through the 40 Hz gamma signature.

6. **Monitor cognitive overload** — Know when you're approaching burnout before it happens. The cognitive load estimator tracks working memory strain in real-time.

7. **Find your flow state** — The flow detector shows when you're "in the zone" and helps you understand what conditions trigger it.

8. **Bridge science and spirituality** — Chakra activations are measured through real EEG frequency bands. Meditation depth is quantified. Consciousness level is tracked. All grounded in published neuroscience.

9. **Personalize over time** — The system learns YOUR brain. After calibration and feedback, predictions become increasingly accurate for you specifically.

10. **Correlate brain with body** — Connect Apple Health/Google Fit data with brain states to discover how sleep, exercise, and heart rate affect your mental performance.

---

*Neural Dream Workshop — Svapnastra*
*First-of-its-Kind BCI Dream Platform*
*5 Sleep Stages. 6 Emotions. 12 ML Models. 111+ API Endpoints.*
*Bridging neuroscience, machine learning, and human consciousness.*
