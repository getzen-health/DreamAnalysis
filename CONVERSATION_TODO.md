# Conversation To-Do List

Items from ongoing conversation. Delete this file once all items are done.
Work through sections in order: ML Gaps → App Gaps → Mobile App.

---

## SECTION 1: ML / Model Gaps (fix first — affects accuracy)

- [x] Wire MultimodalEmotionFusion into live WebSocket stream — DONE. WebSocket and /analyze-eeg both call fusion_model.fuse(emotion_result, bio) after each prediction. BiometricSnapshot cache updated via POST /biometrics/update. Fusion failure is silenced so raw EEG result is always returned as fallback.
- [x] Wire PersonalModel into live inference path — DONE. predict_emotion() helper in _shared.py tries PersonalModel first (personal adapter → central EEGNet → mega LGBM fallback). Wired into /analyze-eeg executor and WebSocket 15s window. API endpoints (label-epoch, status, fine-tune, session-complete, predict, reset) already existed in personal.py.
- [x] Wire OnlineLearner into live inference path — DONE. predict_emotion() now calls PersonalModelAdapter.predict(features) after the base prediction. If calibrated and confidence >0.6, its emotion label overrides the base (stress/valence/arousal kept from base model). POST /feedback triggers pma.adapt() for incremental updates. Result includes online_learner_active flag.
- [x] Add channel_map config layer — DONE. `ml/processing/channel_maps.py` with `CHANNEL_MAPS` dict (11 devices) + `get_channel_map(device, n_channels)`. All 5 hardcoded `left_ch=1, right_ch=2` spots replaced in `emotion_classifier.py` (`_build_muse_result`, `_predict_mega_lgbm`, `_predict_multichannel`, `_predict_features`). Also fixed wrong `left_ch=0, right_ch=1` in `_predict_multichannel` (was a Muse 2 bug). `extract_features_multichannel()` in `eeg_processor.py` now accepts `left_ch`/`right_ch` params. `EEGInput` schema adds `device_type` field. `predict_emotion()` propagates `device_type` to `emotion_model.predict()`.
- [ ] Download Emognition dataset (Harvard Dataverse doi:10.7910/DVN/R9WAF4) — Muse 2 exact hardware, 43 subjects, 9 emotions. Add to mega LGBM retraining pipeline. Expected: +5–8% cross-person accuracy.
- [ ] Download AMIGOS dataset (Queen Mary University) — has ECG channel → use for HRV-stress fusion training data. Register at their site first.
- [ ] Retrain emotion mega LGBM once Emognition data is downloaded. Run `ml/training/mega_trainer.py`.
- [x] EEGNet variable-channel model — DONE. `ml/models/eegnet.py` + `ml/training/train_eegnet.py`. Works on 4/8/16 channels, ONNX export for mobile, wired as top inference priority in `emotion_classifier.py`.
- [x] Personalized per-user model — DONE. `ml/models/personal_model.py`. Central EEGNet backbone frozen, personal classifier head fine-tunes on user's own labeled EEG data. Activates after 30 epochs, improves each session. Saves to `ml/models/saved/personal/{user_id}/`.
- [x] Train EEGNet on synthetic data to verify pipeline works — DONE. 85% val_acc (synthetic), early stopping at epoch 43. `models/saved/eegnet_emotion_4ch.pt` (24 KB) + `eegnet_emotion_4ch_benchmark.txt` (0.85) saved. EEGNet now activates as top inference priority (above mega LGBM). ONNX export skipped (onnxscript not installed, non-critical). Retrain on real Muse 2 data once pilot data collected.
- [x] Emotiv EPOC X adapter — DONE. `ml/hardware/emotiv_adapter.py`. Two backends: (1) Cortex WebSocket API (wss://localhost:6789, JSON-RPC, no binary SDK needed — runs alongside EmotivPro), (2) EDF/CSV/NPZ file reader for AMIGOS dataset replay. Same interface as BrainFlowManager (connect/disconnect/start_streaming/stop_streaming/get_current_data). `devices.py` updated to dispatch Emotiv device_types to EmotivAdapter; `list_devices` merges both adapter outputs. Channel map for EPOC X (F3/F4 as left/right frontal) already in `channel_maps.py`.

---

## SECTION 2: Auth / Login (do before app features — blocks everything)

- [x] Audit existing auth page (`client/src/pages/auth.tsx`) and server auth routes — understand current state before building anything new
- [x] Login page: username + password form (existing auth.tsx had this)
- [x] Register page: username, email, password, age (for research), device selection (Muse 2 / OpenBCI / none yet)
- [x] Auth guards: ProtectedRoute in App.tsx redirects unauthenticated to /auth; auth.tsx redirects logged-in away
- [x] User profile stored in DB: age + deviceType columns added to users table (schema.ts)
- [x] Session persistence: express-session cookie (7-day maxAge) — stays logged in across browser closes
- [x] "Who am I" shown in sidebar: username + avatar initial + logout button at bottom of sidebar

## SECTION 3: App Feature Gaps (missing pages / features)

- [x] Daily Brain Report page (`/brain-report`) — DONE. Full page with sleep summary, focus forecast, yesterday's insight, recommended action, weekly brain summary, pattern engine, streak counter.
- [x] Intervention engine — real-time closed loop. When stress crosses threshold → auto-trigger music recommendation OR breathing exercise OR food suggestion. Build as `ml/api/routes/interventions.py` + frontend notification banner. DONE. 7 API endpoints, per-user state, 10-min cooldown, 5 intervention types (breathing/music_calm/music_focus/food/walk). InterventionBanner polls every 30s; slide-in card with action + snooze.
- [x] Music intervention — No Spotify API keys needed. 6 curated playlists (3 calm, 3 focus) in biofeedback Music tab. Auto-selected via ?tab=music&mood=calm|focus URL param. Each card links to Spotify + YouTube, shows BPM and science citation.
- [x] Breathing intervention — biofeedback page now accepts ?protocol=coherence&auto=true, auto-selects exercise, auto-starts session. Intervention banner deep-links directly.
- [x] Food intervention — backend logic in interventions.py (4-hr meal gap + stress ≥ 0.45 → food card). Banner navigates to /food?alert=protein_snack.
- [x] Intervention outcome tracking — biofeedback session stop schedules a 5-min delayed POST /interventions/outcome with stress_after. GET /interventions/effectiveness/{user_id} aggregates which types worked.
- [x] Yesterday's Insight card on Daily Brain Report — DONE. New GET /api/brain/yesterday-insights/:userId endpoint cross-correlates 48h of emotion readings with biofeedback session times. Generates "Focus was 31% higher after your 3pm breathing session" and day-vs-day comparisons. Card shows up to 3 ranked insights; falls back to client-side heuristic when DB has no data.
- [x] Personal records gamification — DONE. longestEverStreak, focusTrend, nextMilestone helpers; newFocusRecord celebration banner; beat-it challenge framing per record row; live streaming comparison; streak + milestone countdown.
- [x] Sleep session mode — DONE. Screen auto-dims 15s after session start; tap-to-peek reveals controls for 8s; real sleep stats from DB; explicit dim/undim controls.
- [x] Just-in-time push notifications — DONE. POST /api/notifications/brain-state-trigger with 15-min per-user cooldown. stress≥0.70 → breathing push, focus≤0.25 → focus music push. Wired into InterventionBanner check cycle.
- [x] Weekly brain summary — DONE. /weekly-summary page with this week vs last week stress/focus/sleep comparison, trend arrows, week-in-one-sentence, and Canvas 2D PNG export (800×450, no extra deps). Sidebar: 'Week in Review'.
- [x] Intervention library — DONE. 'Evidence' tab in biofeedback page: personal before/after stress bars from /interventions/effectiveness/:userId, science citations for all 7 exercises with phase breakdowns.

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
- [x] Install Capacitor: packages installed (@capacitor/core@8.1.0, cli, ios, android)
- [x] capacitor.config.ts created (appId: com.neuraldreamworkshop.app, webDir: dist/public, SplashScreen + StatusBar + PushNotifications plugin config)
- [ ] Init native projects: `npx cap add ios` (needs Xcode.app) + `npx cap add android` (needs JDK + Android Studio) — run manually
- [x] Fix mobile layouts — responsive breakpoints added to daily-brain-report, emotion-lab; overflow-x-hidden on layout
- [x] Touch targets — sidebar nav links + settings link raised to min-h-[44px]; hamburger button 44×44px; global CSS min-height 44px on all interactive elements
- [x] Safe area insets — viewport-fit=cover; env(safe-area-inset-*) CSS vars; header pl-14 on mobile; sidebar bottom padding; home indicator clearance
- [x] PWA manifest updated (name, theme, icon purposes split for maskable compliance)
- [ ] Splash screen + app icon assets — need 1024×1024 PNG designed (placeholder: favicon.png)
- [ ] Test on iOS simulator + Android emulator (blocked on cap add)

### Phase 2 — Native features (1–2 months)
- [ ] Bluetooth BLE for Muse 2 — `@capacitor-community/bluetooth-le` plugin. Replaces BrainFlow on mobile (BrainFlow is desktop-only). Muse SDK has iOS/Android BLE support.
- [ ] Apple HealthKit integration — `@capacitor-community/health` plugin. Auto-pull: HRV SDNN, resting HR, respiratory rate, sleep stages, steps, SpO2, skin temperature. Feed into MultimodalEmotionFusion automatically.
- [ ] Google Health Connect integration — Android equivalent. Same data fields.
- [ ] Push notifications — `@capacitor/push-notifications`. Server triggers when stress is high → "Time to breathe" alert.
- [x] Haptic feedback — `@capacitor/haptics`. Pulse on breathing inhale/exhale in biofeedback screen.
- [ ] Background processing — iOS BackgroundFetch + Android WorkManager. Keep EEG streaming during sleep without screen on.
- [x] Local ML inference — emotion_classifier_model.onnx (2.2MB) served from client/public/models/. JS heuristics for sleep/dream. use-inference.ts: local first, server fallback.
- [x] Offline mode — IndexedDB queues for EEG sessions + health metrics + dream drafts. OfflineSyncBanner auto-syncs on reconnect. offline-store.ts v2.
- [ ] Home screen widget — iOS WidgetKit + Android App Widget. Show today's brain state (stress level + recommended action) without opening app.
- [ ] Spotify integration — `spotify-web-api-node`. When stress HIGH → auto-queue calming playlist.
- [ ] Siri Shortcuts (iOS) / Google Assistant Actions (Android) — "Hey Siri, check my brain state."

### Phase 3 — App Store submission
- [x] Privacy policy page (required for HealthKit) — `/privacy` route, no auth guard
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
