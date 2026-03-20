# Google Play Store Listing

## App Name
Neural Dream

## Short Description (80 chars)
EEG brain monitor + AI emotion tracking with Muse 2 headband

## Full Description

Neural Dream is a brain-computer interface app that connects to your Muse 2 EEG headband and uses 16 machine learning models to track your emotions, focus, stress, sleep stages, and more in real time.

**What it does:**
- Real-time EEG brain wave visualization (delta, theta, alpha, beta bands)
- 6-class emotion classification (happy, sad, angry, fear, surprise, neutral)
- Stress, focus, and relaxation indices derived from your brain signals
- Sleep staging (Wake, N1, N2, N3, REM) with dream detection
- Frontal alpha asymmetry (FAA) for emotional valence tracking
- Baseline calibration for personalized accuracy (+15-29% improvement)
- AI dream journal with GPT-powered dream analysis
- Health analytics dashboard with brain health trends
- Neurofeedback training protocols
- Voice emotion analysis for multimodal mood tracking
- Food-mood correlation tracking

**How it works:**
Connect your Muse 2 headband via Bluetooth Low Energy. The app reads EEG signals at 256 Hz from 4 channels (TP9, AF7, AF8, TP10), processes them through a signal pipeline (bandpass filtering, artifact rejection, mastoid re-referencing), extracts frequency-domain features, and runs them through trained ML models for classification.

**Science-backed:**
Built on peer-reviewed neuroscience research including Davidson's frontal alpha asymmetry model (1992), Russell's circumplex model of affect, and validated EEG preprocessing pipelines. The mega-LightGBM emotion model was trained on 163,534 samples across 9 public EEG datasets with 74.21% cross-validated accuracy.

**Privacy first:**
All EEG processing happens on your device and our secure ML backend. No raw brainwave data is stored long-term. You can export or delete your data at any time.

**Medical disclaimer:**
Neural Dream is NOT a medical device. It is designed for general wellness monitoring, personal research, and educational purposes only. It does not claim to detect, manage, or prevent any disease or medical condition. Always consult a qualified healthcare professional for medical advice.

## Category
Health & Fitness

## Content Rating
Everyone

## Privacy Policy URL
https://neural-dream.vercel.app/privacy

## Screenshots Needed
1. Dashboard with brain health scores
2. Brain Monitor with live EEG waveforms
3. Emotion Lab with emotion classification
4. Settings page showing device connection
5. Calibration page

## Feature Graphic
1024x500 PNG — dark theme with brain visualization and app name

## App Icon
512x512 PNG — existing PWA icon at client/public/icons/icon-512x512.png
