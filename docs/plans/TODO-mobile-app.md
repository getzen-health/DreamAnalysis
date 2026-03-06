# NeuralDreamWorkshop — Mobile App TODO

## Current Blockers (macOS Desktop Testing)

- [ ] Muse 2 auto-connects to macOS CoreBluetooth, blocking both Web Bluetooth and BrainFlow
- [ ] Factory reset Muse (15s power hold) to break auto-connect loop
- [ ] Test BrainFlow connection on localhost after Muse reset
- [ ] Test Web Bluetooth connection after Muse reset
- [ ] Verify full EEG streaming pipeline: Muse -> BrainFlow -> ML backend -> frontend

## Phase 1: Capacitor Setup

- [ ] Initialize Capacitor in the project: `npx cap init "Neural Dream" com.neuraldream.app`
- [ ] Add platforms: `npx cap add android && npx cap add ios`
- [ ] Install BLE plugin: `npm install @capacitor-community/bluetooth-le`
- [ ] Configure `capacitor.config.ts` (app name, webDir: dist/public, server URL for dev)
- [ ] Add Bluetooth permissions to `AndroidManifest.xml` (BLUETOOTH_SCAN, BLUETOOTH_CONNECT, ACCESS_FINE_LOCATION)
- [ ] Add Bluetooth permission to `Info.plist` (NSBluetoothAlwaysUsageDescription)

## Phase 2: BLE Connection (Already Built)

- [ ] Verify `muse-ble.ts` Capacitor native path works (`isNative` branch)
- [ ] Test `@capacitor-community/bluetooth-le` requestDevice with Muse service UUID
- [ ] Test GATT connect + EEG characteristic subscriptions on Android
- [ ] Test GATT connect + EEG characteristic subscriptions on iOS
- [ ] Verify 4-channel EEG streaming at 256 Hz over native BLE
- [ ] Test reconnection after Muse disconnects

## Phase 3: ML Backend Connection

- [ ] Decide ML backend hosting: Railway (always-on) vs on-device inference
- [ ] Configure app to point to Railway ML backend URL
- [ ] Test REST API calls from mobile app to Railway backend
- [ ] Test WebSocket EEG streaming from app to Railway backend
- [ ] Handle offline mode: queue data locally, sync when online

## Phase 4: UI/UX Adjustments

- [ ] Test all 17 pages on mobile viewport (already responsive with Tailwind)
- [ ] Fix any touch interaction issues (hover states, small tap targets)
- [ ] Add native status bar / safe area handling
- [ ] Add splash screen and app icon
- [ ] Test dark mode on both platforms

## Phase 5: Build & Deploy

### Android
- [ ] Build: `npx cap sync && npx cap open android`
- [ ] Test on physical Android device with Muse 2
- [ ] Generate signed APK for distribution
- [ ] (Optional) Publish to Google Play Store

### iOS
- [ ] Build: `npx cap sync && npx cap open ios`
- [ ] Test on physical iPhone/iPad with Muse 2
- [ ] Configure signing & provisioning profiles
- [ ] (Optional) Submit to App Store

## Phase 6: On-Device ML (Future)

- [ ] Export key models to ONNX format (emotion classifier, sleep staging)
- [ ] Use onnxruntime-web (already in dependencies) for client-side inference
- [ ] Reduce latency by running emotion classification on-device
- [ ] Keep heavier models (dream detection, anomaly) on Railway backend

## Architecture

```
Mobile App (Capacitor)
    |
    +-- Native BLE (@capacitor-community/bluetooth-le)
    |       |
    |       +-- Muse 2 EEG (4 channels, 256 Hz)
    |
    +-- muse-ble.ts (isNative path) -- decodes packets, computes features
    |
    +-- REST/WebSocket --> Railway ML Backend
    |       |
    |       +-- 16 ML models (emotion, sleep, dreams, etc.)
    |
    +-- React UI (same codebase, responsive)
```

## Key Advantage

Native BLE on mobile has NONE of the macOS Web Bluetooth issues:
- No OS-level GATT lock conflicts
- No Chrome permission issues
- Direct access to BLE stack
- Reliable, fast connections
