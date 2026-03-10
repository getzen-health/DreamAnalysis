# Mobile Apps — Android & iOS Build + Play Store Publish

**Date:** 2026-03-10
**Status:** Approved

## Goal

Build and run the NeuralDreamWorkshop app on Android and iOS devices using the existing Capacitor 8.1.0 setup. Publish a signed build to Google Play Store. iOS runs locally via Xcode (no Apple Developer account).

## Architecture

The mobile app is a Capacitor-wrapped React SPA talking to two backends:

```
Mobile App (Capacitor + React)
    │
    ├── REST ──▶ Express.js (Vercel) ──▶ PostgreSQL (Neon)
    │               └── auth, dreams, AI chat
    │
    └── REST + WS ──▶ FastAPI (Railway prod / localhost dev)
                        └── 16 ML models, EEG analysis, voice emotion
```

Build flow:
1. `npm run build` → compiles React to `dist/public/`
2. `npx cap sync` → copies web assets + native plugins to `ios/` and `android/`
3. Open in Xcode / Android Studio → build + run on device

ML backend switching:
- Dev builds: ML URL points to local IP (e.g. `http://192.168.1.x:8080`)
- Production builds: ML URL points to Railway service URL
- Controlled via environment variable at build time

## Phased Approach

### Phase 1 — Get builds running on devices

1. Fix any Capacitor config issues (verify `webDir`, app ID, SDK versions)
2. Build the React web app with production ML URL
3. `npx cap sync` to push web assets + plugin configs to native projects
4. Android: Open in Android Studio, build debug APK, run on phone
5. iOS: Open in Xcode, set signing team (personal), run on iPhone

### Phase 2 — Mobile-specific fixes

Things that typically break when going web → mobile:
- API URLs (localhost doesn't work on a phone)
- WebSocket connections (need absolute URLs)
- BLE permissions flow (need runtime permission prompts)
- Safe area / notch handling (already partially done)
- Splash screen + app icon verification

### Phase 3 — Publish Android to Google Play

1. Generate a signed release AAB (Android App Bundle)
2. Create Play Store listing (screenshots, description, privacy policy)
3. Upload AAB to Play Console
4. Internal testing track first, then production

### Phase 4 — iOS local builds

1. Build with personal team signing (runs on your iPhone only, 7-day expiry)
2. No App Store submission until Apple Developer account is set up

## Play Store Listing

- Package: `com.neuraldreamworkshop.app`
- Name: "Neural Dream"
- Category: Health & Fitness
- Content rating: Everyone
- Short description: "EEG brain monitor + AI emotion tracking with Muse 2"

Required assets:
- App icon: 512x512 PNG
- Feature graphic: 1024x500 PNG
- Screenshots: minimum 2 phone screenshots
- Privacy policy: hosted URL
- Medical disclaimer: "Not a medical device. For wellness and research purposes only."

## Success Criteria

1. Android debug APK runs on a real phone — BLE scanning works, ML API calls succeed
2. iOS debug build runs on iPhone via Xcode
3. Signed AAB uploaded to Google Play internal testing track
4. Play Store listing has screenshots, description, privacy policy
5. ML backend URL switches between Railway (prod) and local IP (dev) based on build config

## Out of Scope

- Apple App Store submission (no developer account)
- Custom native UI (Capacitor web wrapper is sufficient)
- Push notification backend triggers
- Offline-first mode beyond existing service worker caching

## Existing Infrastructure

Already configured:
- Capacitor 8.1.0 with iOS 14+ and Android API 24+
- `@capacitor-community/bluetooth-le` for Muse 2 BLE
- `@perfood/capacitor-healthkit` for Apple Health
- `capacitor-health` for Google Health Connect
- `@capacitor/push-notifications` for APNs + FCM
- `@capacitor/background-runner` for 15-min EEG data flush
- Responsive dark-theme React UI with safe area insets
- Service worker for offline caching
- PWA manifest with app icons (192px, 512px, 1024px)
