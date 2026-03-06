# NeuralDreamWorkshop — Master TODO

## Current Blockers (macOS Desktop Testing)

- [ ] Muse 2 auto-connects to macOS CoreBluetooth, blocking both Web Bluetooth and BrainFlow
- [ ] Factory reset Muse (15s power hold) to break auto-connect loop
- [ ] Test BrainFlow connection on localhost after Muse reset
- [ ] Test Web Bluetooth connection after Muse reset
- [ ] Verify full EEG streaming pipeline: Muse -> BrainFlow -> ML backend -> frontend

---

## User EEG Data -> Model Training Pipeline

### Goal
Every EEG session a user captures gets stored and used to improve models over time.
Three loops: (1) personal model adapts per-user, (2) RL agent fine-tunes per-user,
(3) aggregate data improves base models.

### What Already Exists
- `online_learner.py` — PersonalModelAdapter with SGDClassifier (calibrate + adapt via partial_fit)
- `rl_nf_agent.pt` — Trained PPO agent for neurofeedback threshold control
- `train_rl_agent.py` — Offline RL training script with synthetic environment
- Calibration API endpoints — baseline frame collection
- RL endpoints — `/api/neurofeedback/evaluate` uses PPO agent live

### Phase A: Session Data Collection (Wire Into Live Pipeline)

- [x] Add `POST /api/sessions/save-eeg` endpoint — accepts raw EEG epochs (4ch x N samples) + timestamp + user_id
- [x] Store session data: `user_data/{user_id}/sessions/{session_id}.npz` (numpy compressed)
- [x] Save alongside: predicted emotion, user correction (if any), band powers, FAA
- [x] Frontend: auto-send EEG frames to `/sessions/save-eeg` during active streaming
- [x] Add session metadata: device type, protocol, duration, signal quality stats

### Phase B: Personal Model Adaptation (Per-User Learning)

- [x] Wire `PersonalModelAdapter` into `/api/analyze-eeg` response path (already in predict_emotion())
- [x] After emotion prediction, check if user has a personal model → blend predictions (OnlineLearner + k-NN blend)
- [x] Add `POST /api/personalization/correct` — user says "I was actually feeling X" (POST /api/feedback)
  - Calls `PersonalModelAdapter.adapt()` with the EEG epoch + correct label
  - Incrementally updates SGDClassifier via `partial_fit`
- [x] Add correction UI: after each emotion reading, show "Was this right?" with emotion buttons
- [x] After 50+ corrections, personal model starts blending with base model predictions
- [x] Personal models saved per user: `user_models/{user_id}_personal.pkl`

### Phase C: RL Agent Per-User Fine-Tuning

- [ ] Store neurofeedback session trajectories: (observation, action, reward) tuples
- [ ] After 5+ neurofeedback sessions, fine-tune PPO agent on real user data
- [ ] Add `POST /api/neurofeedback/rl/fine-tune` — trains on stored trajectories
- [ ] Save per-user RL models: `user_models/{user_id}_rl_agent.pt`
- [ ] Load user-specific RL agent in `/api/neurofeedback/evaluate` if available

### Phase D: Aggregate Model Improvement (Cross-User)

- [ ] Periodically aggregate anonymized session data across users
- [ ] Retrain base emotion classifier (LightGBM) with Muse-specific data
- [ ] Add Muse-collected data as a new dataset in `training/data_loaders.py`
- [ ] Target: 1000+ labeled samples from real Muse users → meaningful accuracy boost
- [ ] Track model version and accuracy drift over time

---

## Mobile App (Android & iOS)

### Phase 1: Capacitor Setup

- [x] Initialize Capacitor: `npx cap init "Neural Dream" com.neuraldream.app`
- [x] Add platforms: `npx cap add android && npx cap add ios`
- [x] Install BLE plugin: `npm install @capacitor-community/bluetooth-le`
- [x] Configure `capacitor.config.ts` (app name, webDir: dist/public, server URL for dev)
- [x] Add Bluetooth permissions to `AndroidManifest.xml` (BLUETOOTH_SCAN, BLUETOOTH_CONNECT, ACCESS_FINE_LOCATION)
- [x] Add Bluetooth permission to `Info.plist` (NSBluetoothAlwaysUsageDescription)

### Phase 2: BLE Connection (Code Complete — Needs Physical Device Testing)

- [x] Verify `muse-ble.ts` Capacitor native path works (`isNative` branch) — code complete
- [x] `@capacitor-community/bluetooth-le` requestDevice with Muse service UUID — implemented
- [ ] Test GATT connect + EEG characteristic subscriptions on Android (needs physical device)
- [ ] Test GATT connect + EEG characteristic subscriptions on iOS (needs physical device)
- [ ] Verify 4-channel EEG streaming at 256 Hz over native BLE
- [x] Reconnection after Muse disconnects — implemented in muse-ble.ts

### Phase 3: ML Backend Connection

- [x] Configure app to point to Railway ML backend URL (VITE_ML_API_URL in Vercel env)
- [ ] Test REST API calls from mobile app to Railway backend
- [ ] Test WebSocket EEG streaming from app to Railway backend
- [ ] Handle offline mode: queue EEG data locally, sync when online
- [x] Session data collection (Phase A above) works over mobile network — saveEEGEpoch wired

### Phase 4: UI/UX Adjustments

- [ ] Test all 17 pages on mobile viewport (already responsive with Tailwind)
- [ ] Fix any touch interaction issues (hover states, small tap targets)
- [x] Add native status bar / safe area handling — configured in capacitor.config.ts
- [x] Add splash screen and app icon — configured in capacitor.config.ts
- [x] Add emotion correction UI (Phase B) — "Was this right?" in emotion-lab.tsx
- [ ] Test dark mode on both platforms

### Phase 5: Build & Deploy

#### Android
- [ ] Build: `npx cap sync && npx cap open android`
- [ ] Test on physical Android device with Muse 2
- [ ] Generate signed APK for distribution
- [ ] (Optional) Publish to Google Play Store

#### iOS
- [ ] Build: `npx cap sync && npx cap open ios`
- [ ] Test on physical iPhone/iPad with Muse 2
- [ ] Configure signing & provisioning profiles
- [ ] (Optional) Submit to App Store

### Phase 6: On-Device ML (Future)

- [ ] Export emotion classifier + sleep staging to ONNX
- [ ] Use onnxruntime-web (already in deps) for client-side inference
- [ ] Run emotion classification on-device for low latency
- [ ] Keep heavier models (dream detection, anomaly, RL) on Railway backend

---

## Architecture

```
User with Muse 2
    |
    +-- Mobile: Native BLE (Capacitor) — reliable, no GATT issues
    +-- Desktop: Web Bluetooth (Chrome) or BrainFlow (local)
    |
    v
muse-ble.ts — decode 12-bit packets, compute features, emit frames
    |
    +-----> Local UI: real-time charts, emotion display, neurofeedback
    |
    +-----> Railway ML Backend (REST + WebSocket)
                |
                +-- /analyze-eeg — emotion classification (16 models)
                |       |
                |       +-- PersonalModelAdapter blends with base model
                |       +-- User corrections -> partial_fit -> improves over time
                |
                +-- /neurofeedback/evaluate — PPO RL agent adjusts difficulty
                |       |
                |       +-- Per-user RL fine-tuning from real session data
                |
                +-- /sessions/save-eeg — stores raw EEG for future training
                |
                +-- Periodic retrain: aggregate user data -> better base models
```

## Priority Order

1. **Fix Muse connection** (factory reset, test locally)
2. **Wire session data collection** (Phase A) — foundation for everything else
3. **Add emotion correction UI + personal model** (Phase B)
4. **Capacitor mobile app setup** (Phase 1-2)
5. **Test on Android with Muse** (Phase 2-3)
6. **RL per-user fine-tuning** (Phase C)
7. **On-device ML** (Phase 6)
8. **Aggregate retraining** (Phase D)
