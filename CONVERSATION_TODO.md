# Conversation To-Do List

Items from ongoing conversation. Delete this file once all items are done.
Work through sections in order: ML Gaps → App Gaps → Mobile App.

---

## SECTION 1: ML / Model Gaps (fix first — affects accuracy)

- [ ] Wire MultimodalEmotionFusion into live WebSocket stream — currently built but not connected to real-time inference path. Every frame that comes from Muse should go through multimodal fusion before returning to dashboard.
- [ ] Wire PersonalModel into live inference path — expose `POST /personal/label-epoch`, `GET /personal/status`, `POST /personal/fine-tune` endpoints so dashboard can send labeled epochs and show personalisation progress. `ml/models/personal_model.py` is built, needs API route + wiring into `predict()` flow.
- [ ] Wire OnlineLearner into live inference path — `ml/models/online_learner.py` exists but never called. After each prediction, call `online_learner.update(features, user_feedback)` so model adapts per-user over time. +15–20% accuracy for returning users.
- [ ] Add channel_map config layer — `compute_frontal_asymmetry()` and `compute_dasm_rasm()` currently hardcode ch1=AF7, ch2=AF8 (Muse-specific). Create a `CHANNEL_MAPS` dict keyed by device name so OpenBCI Cyton, Neurosity Crown etc. automatically get correct left/right frontal indices.
- [ ] Download Emognition dataset (Harvard Dataverse doi:10.7910/DVN/R9WAF4) — Muse 2 exact hardware, 43 subjects, 9 emotions. Add to mega LGBM retraining pipeline. Expected: +5–8% cross-person accuracy.
- [ ] Download AMIGOS dataset (Queen Mary University) — has ECG channel → use for HRV-stress fusion training data. Register at their site first.
- [ ] Retrain emotion mega LGBM once Emognition data is downloaded. Run `ml/training/mega_trainer.py`.
- [x] EEGNet variable-channel model — DONE. `ml/models/eegnet.py` + `ml/training/train_eegnet.py`. Works on 4/8/16 channels, ONNX export for mobile, wired as top inference priority in `emotion_classifier.py`.
- [x] Personalized per-user model — DONE. `ml/models/personal_model.py`. Central EEGNet backbone frozen, personal classifier head fine-tunes on user's own labeled EEG data. Activates after 30 epochs, improves each session. Saves to `ml/models/saved/personal/{user_id}/`.
- [ ] Train EEGNet on synthetic data to verify pipeline works: `python -m training.train_eegnet --channels 4 --use-synthetic --epochs 50` — generates `eegnet_emotion_4ch.pt` so it activates in inference.
- [ ] Emotiv EPOC X adapter — 14 channels, proprietary SDK (not BrainFlow). Write thin adapter layer that reads Emotiv SDK and outputs same format as BrainFlow. Unlocks AMIGOS dataset hardware for real users.

---

## SECTION 2: Auth / Login (do before app features — blocks everything)

- [ ] Audit existing auth page (`client/src/pages/auth.tsx`) and server auth routes — understand current state before building anything new
- [ ] Login page: email + password, remember me, forgot password link
- [ ] Register page: name, email, password, age (for research), device selection (Muse 2 / OpenBCI / none yet)
- [ ] Auth guards: redirect unauthenticated users to /auth, redirect logged-in users away from /auth
- [ ] User profile stored in DB: user_id, name, email, age, device_type, created_at
- [ ] user_id flows into: personal model (ml/models/saved/personal/{user_id}/), Parquet storage, per-user ML state, all API calls
- [ ] Session persistence: JWT or session cookie so user stays logged in across browser closes
- [ ] "Who am I" shown in sidebar: user name + avatar initial at the bottom

## SECTION 3: App Feature Gaps (missing pages / features)

- [ ] Daily Brain Report page (`/brain-report`) — THE north star feature per PRODUCT.md. Show: last night sleep summary, today's focus forecast, yesterday's insight, recommended action. File: `client/src/pages/daily-brain-report.tsx`. Route and sidebar link already wired.
- [ ] Intervention engine — real-time closed loop. When stress crosses threshold → auto-trigger music recommendation OR breathing exercise OR food suggestion. Build as `ml/api/routes/interventions.py` + frontend notification banner.
- [ ] Music intervention — Spotify / Apple Music API: when stress HIGH, suggest calming playlist. When focus LOW, suggest focus music (binaural beats). Log which songs actually reduced stress.
- [ ] Breathing intervention — when stress HIGH for >60 seconds, auto-open biofeedback breathing screen with push notification. Currently biofeedback is manual-only.
- [ ] Food intervention — based on `minutes_since_last_meal` + stress: "You haven't eaten in 4 hours and your stress is rising — have a protein snack, not sugar." Novel feature, nobody else has this.
- [ ] Intervention outcome tracking — after triggering an intervention, measure stress 5 min later. Log: what worked, what didn't. Feeds back into personalization.
- [ ] Yesterday's Insight card on Daily Brain Report — "Focus was 23% higher after your 11am walk." Needs pattern detection across sessions.
- [ ] Personal records gamification — "New focus record: 47 min — beat it?" Show streaks on dashboard. Motivates daily use.
- [ ] Sleep session mode — tap to enter sleep mode: screen dims, Muse streams overnight, morning shows sleep report. Currently sleep data is simulated.
- [ ] Just-in-time push notifications — server-side trigger (not scheduled). Fire when brain state needs action, not on a timer.
- [ ] Weekly brain summary — shareable card showing this week vs last week for stress, focus, sleep. PNG export for sharing.
- [ ] Intervention library — 5–10 evidence-based exercises in biofeedback page with before/after EEG comparison to show they work.

---

## SECTION 3: Data Collection (pilot study — start today)

- [ ] Pilot study — collect food-emotion EEG data from yourself + 20 people. Timing per session:
    - 12:00 PM — pre-lunch (peak hunger, best food-craving EEG)
    - 3:00 PM — afternoon snack
    - 7:00 PM — dinner
    - Protocol each session: 2 min baseline → 3 min food cue viewing → 5 min eating → 5 min post-meal
- [ ] Label each session with: food type, hunger level (1–10), mood before/after (1–10), meal size
- [ ] Store pilot data in `ml/data/pilot/` as Parquet files via existing pipeline

---

## SECTION 4: Mobile App (iOS + Android) — do after app gaps are closed

### Strategy: Capacitor first (fastest), React Native later (best performance)

Capacitor wraps the existing React web app into a native iOS/Android shell in ~2 weeks.
Then add native features progressively. No full rewrite needed to ship v1.

### Phase 1 — Capacitor wrapper (2 weeks, ship to TestFlight + Play Store beta)
- [ ] Install Capacitor: `npm install @capacitor/core @capacitor/cli @capacitor/ios @capacitor/android`
- [ ] Init: `npx cap init`, `npx cap add ios`, `npx cap add android`
- [ ] Fix mobile layouts — add responsive Tailwind breakpoints to all 11 sidebar pages
- [ ] Touch targets — all tap areas minimum 44×44px (iOS HIG requirement)
- [ ] Safe area insets — handle notch / Dynamic Island / Android navbar padding
- [ ] Splash screen + app icon (brain/neural theme)
- [ ] Test on iOS simulator + Android emulator

### Phase 2 — Native features (1–2 months)
- [ ] Bluetooth BLE for Muse 2 — `@capacitor-community/bluetooth-le` plugin. Replaces BrainFlow on mobile (BrainFlow is desktop-only). Muse SDK has iOS/Android BLE support.
- [ ] Apple HealthKit integration — `@capacitor-community/health` plugin. Auto-pull: HRV SDNN, resting HR, respiratory rate, sleep stages, steps, SpO2, skin temperature. Feed into MultimodalEmotionFusion automatically.
- [ ] Google Health Connect integration — Android equivalent. Same data fields.
- [ ] Push notifications — `@capacitor/push-notifications`. Server triggers when stress is high → "Time to breathe" alert.
- [ ] Haptic feedback — `@capacitor/haptics`. Pulse on breathing inhale/exhale in biofeedback screen.
- [ ] Background processing — iOS BackgroundFetch + Android WorkManager. Keep EEG streaming during sleep without screen on.
- [ ] Local ML inference — convert emotion LGBM model to ONNX, run on-device via onnxruntime-web (already in the frontend stack). No server needed for basic inference.
- [ ] Offline mode — queue EEG data locally when no internet, sync when connected.
- [ ] Home screen widget — iOS WidgetKit + Android App Widget. Show today's brain state (stress level + recommended action) without opening app.
- [ ] Spotify integration — `spotify-web-api-node`. When stress HIGH → auto-queue calming playlist.
- [ ] Siri Shortcuts (iOS) / Google Assistant Actions (Android) — "Hey Siri, check my brain state."

### Phase 3 — App Store submission
- [ ] Privacy policy page (required for HealthKit)
- [ ] App Store description + screenshots (5 screenshots required, show Brain Report, Brain State, Sleep, Dreams, Food)
- [ ] Submit as "wellness app" not "medical device" — avoids FDA clearance requirement
- [ ] TestFlight beta with pilot study participants first
- [ ] Google Play internal testing track first

---

## SECTION 5: Publication path

- [ ] Write paper: "Food-Emotion EEG: A First Dataset and Real-Time Intervention System" — arXiv first, then IEEE EMBC 2026 or Frontiers in Neuroscience
- [ ] Include in paper: dataset stats, model accuracy table (Muse 2 vs OpenBCI), intervention outcomes, device compatibility section
- [ ] STATUS.md — update with all completed items from this conversation
- [ ] PRODUCT.md — update honest assessment percentages

---

## Completed

- [x] Sidebar simplified to 11 items across 4 sections
- [x] Dashboard: 4 score circles → 1 Brain State Now card (Stress/Focus/Flow bars with HIGH/MEDIUM/LOW badges)
- [x] HRVEmotionFusion class — `ml/models/hrv_emotion_fusion.py`
- [x] MultimodalEmotionFusion class — all 8 signal layers (EEG + HRV + sleep + breathing + activity + temp + food + circadian)
- [x] POST /hrv/analyze endpoint — `ml/api/routes/hrv_fusion.py`
- [x] Emognition dataset loader — `ml/training/download_emognition.py`
- [x] Apache Parquet pipeline — `ml/storage/parquet_writer.py` + endpoints
- [x] Per-user ML backend (5 global singletons → per-user dicts)
- [x] WebSocket MAX_CONNECTIONS 50 → 200
- [x] BrainFlow device registry — OpenBCI Cyton/Ganglion/Cyton+Daisy already registered
