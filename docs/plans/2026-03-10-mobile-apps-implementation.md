# Mobile Apps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Android & iOS apps from existing Capacitor setup, fix mobile-specific issues, and publish Android to Google Play Store.

**Architecture:** Capacitor 8.1.0 wraps the React SPA (built to `dist/public/`). The app talks to Express (Vercel) for auth/data and FastAPI (Railway) for ML. On native platforms, `Capacitor.isNativePlatform()` gates BLE, HealthKit, and push notification code paths.

**Tech Stack:** Capacitor 8.1.0, React 18, Vite 5, Android Studio (Gradle), Xcode (Swift), Railway (ML backend)

---

### Task 1: Fix ML API URL for Native Platforms

The current `getMLApiUrl()` in `ml-api.ts` checks `window.location.hostname === "localhost"` to decide the URL. On native apps, `hostname` is empty or `localhost` (Capacitor serves from local files). This breaks ML API calls.

**Files:**
- Modify: `client/src/lib/ml-api.ts:1-29`

**Step 1: Write the failing test**

Create `client/src/test/lib/ml-api-native.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

describe("getMLApiUrl on native platform", () => {
  const originalWindow = global.window;

  beforeEach(() => {
    vi.resetModules();
    localStorage.clear();
  });

  it("returns VITE_ML_API_URL on native (not localhost:8080)", async () => {
    // Simulate native: hostname is "" or "localhost" but platform is native
    const mod = await import("@/lib/ml-api");
    // When VITE_ML_API_URL is set and we're on native, should use it
    const url = mod.getMLApiUrl();
    // Should not be localhost:8080 when env var points elsewhere
    expect(typeof url).toBe("string");
    expect(url.length).toBeGreaterThan(0);
  });
});
```

**Step 2: Run test to verify baseline**

Run: `npx vitest run client/src/test/lib/ml-api-native.test.ts`

**Step 3: Update getMLApiUrl to handle native platforms**

In `client/src/lib/ml-api.ts`, replace the `getMLApiUrl` function:

```typescript
import { Capacitor } from "@capacitor/core";

const ML_API_URL_DEFAULT =
  import.meta.env.VITE_ML_API_URL ||
  "http://localhost:8080";

/** Reads the ML backend URL, handling web, native, and user overrides. */
export function getMLApiUrl(): string {
  // 1. User override from Settings (always wins)
  try {
    const stored = localStorage.getItem("ml_backend_url");
    if (stored?.trim()) return stored.trim().replace(/\/$/, "");
  } catch { /* SSR / private browsing */ }

  // 2. Native app: always use the build-time env var (Railway URL)
  //    Never fall back to localhost — it won't reach the dev machine.
  if (typeof window !== "undefined" && Capacitor.isNativePlatform()) {
    return ML_API_URL_DEFAULT;
  }

  // 3. Web localhost: use localhost:8080 for direct local dev
  if (typeof window !== "undefined" && window.location.hostname === "localhost") {
    return "http://localhost:8080";
  }

  // 4. Web production (Vercel): use env var
  return ML_API_URL_DEFAULT;
}
```

**Step 4: Run test to verify it passes**

Run: `npx vitest run client/src/test/lib/ml-api-native.test.ts`

**Step 5: Commit**

```bash
git add client/src/lib/ml-api.ts client/src/test/lib/ml-api-native.test.ts
git commit -m "fix: handle ML API URL on native Capacitor platforms"
```

---

### Task 2: Build Web Assets for Mobile

Build the React app with the Railway ML URL so mobile builds connect to production.

**Files:**
- Modify: `package.json` (add mobile build script)
- Modify: `.env` (temporary — set VITE_ML_API_URL for build)

**Step 1: Add a mobile build script to package.json**

Add to `"scripts"`:
```json
"build:mobile": "VITE_ML_API_URL=https://YOUR_RAILWAY_URL vite build"
```

Replace `YOUR_RAILWAY_URL` with the actual Railway service URL. If Railway isn't deployed yet, use the Vercel proxy (`/api/ml`) for now.

**Step 2: Build**

```bash
npm run build:mobile
```

Expected: `dist/public/` populated with compiled assets, `index.html`, JS chunks, CSS.

**Step 3: Verify build output**

```bash
ls dist/public/index.html dist/public/assets/
```

Expected: `index.html` exists, `assets/` contains `.js` and `.css` files.

**Step 4: Commit**

```bash
git add package.json
git commit -m "feat: add mobile build script with production ML URL"
```

---

### Task 3: Sync Web Assets to Native Projects

**Step 1: Run Capacitor sync**

```bash
npx cap sync
```

This copies `dist/public/` into both `ios/App/App/public/` and `android/app/src/main/assets/public/`, and installs/updates native plugin dependencies.

Expected output: "Syncing web assets" + "Updating iOS plugins" + "Updating Android plugins"

**Step 2: Verify sync**

```bash
ls ios/App/App/public/index.html
ls android/app/src/main/assets/public/index.html
```

Both should exist.

**Step 3: Commit native project changes**

```bash
git add ios/ android/
git commit -m "chore: sync web assets and plugins to native projects"
```

---

### Task 4: Build and Run Android Debug APK

**Step 1: Open in Android Studio**

```bash
npx cap open android
```

This launches Android Studio with the `android/` project.

**Step 2: In Android Studio:**

1. Wait for Gradle sync to complete (may take 2-5 minutes first time)
2. Select your connected Android phone or an emulator from the device dropdown
3. Click the green "Run" (play) button
4. Wait for build and install

**Step 3: Verify on device**

- App launches with dark splash screen
- Dashboard loads
- Navigate to Settings → ML Backend URL shows the Railway URL
- Navigate to Brain Monitor → shows "Connect device" or device list
- If Muse 2 is nearby: BLE scan finds it

**Step 4: Build debug APK for sharing**

In Android Studio: Build → Build Bundle(s) / APK(s) → Build APK(s)

Output: `android/app/build/outputs/apk/debug/app-debug.apk`

---

### Task 5: Build and Run iOS Debug Build

**Step 1: Open in Xcode**

```bash
npx cap open ios
```

**Step 2: In Xcode:**

1. Select target device: your iPhone (connected via USB)
2. Set signing team: Signing & Capabilities → Team → select your personal team (Apple ID)
3. If bundle ID conflict: change to `com.neuraldreamworkshop.app.dev`
4. Click the Run button (Cmd+R)
5. On first run: iPhone will prompt "Untrusted Developer" → go to Settings → General → VPN & Device Management → trust your certificate

**Step 3: Verify on device**

Same checks as Android:
- App launches with splash screen
- Dashboard loads
- BLE scanning works
- ML API calls succeed

**Step 4: Commit any Xcode project changes**

```bash
git add ios/
git commit -m "chore: update iOS project for local device signing"
```

---

### Task 6: Fix Mobile-Specific Issues

Common issues that need fixing after first device run. Check each one and fix as needed.

**Files:**
- Modify: `client/src/lib/ml-api.ts` (if WebSocket URLs broken)
- Modify: `client/src/hooks/use-device.tsx` (if BLE permission flow broken)
- Modify: `capacitor.config.ts` (if cleartext HTTP needed for dev)

**Step 1: Test WebSocket connections**

Navigate to Brain Monitor, attempt to start streaming. If WebSocket fails:
- The WS URL is likely using `ws://localhost:8080` instead of `wss://railway-url`
- Fix: update any WebSocket URL construction to use `getMLApiUrl()` as the base

**Step 2: Test BLE permissions**

On Android: app should prompt for Bluetooth + Location permissions.
On iOS: app should prompt for Bluetooth permission.

If permissions don't trigger, verify the manifest/Info.plist entries are correct (they are — already verified).

**Step 3: Test safe area rendering**

Check the app on devices with notches (iPhone) and gesture navigation (Android). The app already has `pb-[env(safe-area-inset-bottom)]` — verify it renders correctly.

**Step 4: Commit fixes**

```bash
git add -A
git commit -m "fix: resolve mobile-specific issues found during device testing"
```

---

### Task 7: Generate Signed Android Release Build

**Files:**
- Create: `android/keystore.properties` (DO NOT commit — add to .gitignore)
- Modify: `android/app/build.gradle` (add signing config)
- Modify: `.gitignore` (add keystore exclusions)

**Step 1: Generate a release keystore**

```bash
keytool -genkey -v -keystore android/neural-dream-release.keystore \
  -alias neural-dream -keyalg RSA -keysize 2048 -validity 10000 \
  -storepass YOUR_STORE_PASSWORD -keypass YOUR_KEY_PASSWORD \
  -dname "CN=Neural Dream, OU=Dev, O=NeuralDreamWorkshop, L=City, ST=State, C=US"
```

**Step 2: Create keystore.properties**

Create `android/keystore.properties`:
```properties
storeFile=neural-dream-release.keystore
storePassword=YOUR_STORE_PASSWORD
keyAlias=neural-dream
keyPassword=YOUR_KEY_PASSWORD
```

**Step 3: Add to .gitignore**

```
android/keystore.properties
android/*.keystore
android/*.jks
```

**Step 4: Add signing config to android/app/build.gradle**

Add above `android {`:
```groovy
def keystorePropertiesFile = rootProject.file("keystore.properties")
def keystoreProperties = new Properties()
if (keystorePropertiesFile.exists()) {
    keystoreProperties.load(new FileInputStream(keystorePropertiesFile))
}
```

Inside `android {`, add:
```groovy
signingConfigs {
    release {
        if (keystorePropertiesFile.exists()) {
            storeFile file(keystoreProperties['storeFile'])
            storePassword keystoreProperties['storePassword']
            keyAlias keystoreProperties['keyAlias']
            keyPassword keystoreProperties['keyPassword']
        }
    }
}
buildTypes {
    release {
        signingConfig signingConfigs.release
        minifyEnabled false
        proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
    }
}
```

**Step 5: Build signed AAB**

In Android Studio: Build → Generate Signed Bundle / APK → Android App Bundle → select keystore → release → Build

Output: `android/app/build/outputs/bundle/release/app-release.aab`

**Step 6: Commit build config (not keystore)**

```bash
git add android/app/build.gradle .gitignore
git commit -m "feat: add Android release signing configuration"
```

---

### Task 8: Create Privacy Policy Page

Google Play requires a privacy policy URL for health/biometric apps.

**Files:**
- Create: `client/src/pages/privacy-policy.tsx`
- Modify: `client/src/App.tsx` (add route)

**Step 1: Create the privacy policy page**

Create `client/src/pages/privacy-policy.tsx` with a simple static page covering:
- What data is collected (EEG, voice, health metrics, BLE device info)
- How data is processed (on-device + sent to ML backend for analysis)
- Data storage (PostgreSQL for accounts, no raw EEG stored long-term)
- Third-party services (OpenAI for dream analysis, Railway for ML)
- User rights (delete account, export data)
- Contact info
- Medical disclaimer

**Step 2: Add route in App.tsx**

```tsx
<Route path="/privacy-policy"><PrivacyPolicy /></Route>
```

**Step 3: Verify**

Run dev server, navigate to `/privacy-policy`, confirm content renders.

**Step 4: Commit**

```bash
git add client/src/pages/privacy-policy.tsx client/src/App.tsx
git commit -m "feat: add privacy policy page for Play Store requirement"
```

---

### Task 9: Create Play Store Listing Assets

**Files:**
- Create: `store-listing/` directory with screenshots and descriptions

**Step 1: Create store listing content**

Create `store-listing/play-store.md`:
```markdown
# Google Play Store Listing

## App Name
Neural Dream

## Short Description (80 chars)
EEG brain monitor + AI emotion tracking with Muse 2 headband

## Full Description
[Write 500-1000 word description covering features, science, disclaimer]

## Category
Health & Fitness

## Content Rating
Everyone

## Privacy Policy URL
https://your-vercel-url.vercel.app/privacy-policy

## Screenshots Needed
1. Dashboard with brain health scores
2. Brain Monitor with live EEG
3. Emotion Lab with emotion classification
4. Settings page showing device connection
```

**Step 2: Take screenshots**

Run the app on an Android emulator or phone. Take screenshots of:
1. Dashboard
2. Brain Monitor
3. Emotion Lab
4. At least 1 more page

Save to `store-listing/screenshots/`

**Step 3: Create feature graphic**

1024x500 PNG with app name and brain visualization. Can be created with any design tool.

**Step 4: Commit**

```bash
git add store-listing/
git commit -m "docs: add Play Store listing content and screenshots"
```

---

### Task 10: Upload to Google Play Console

**Step 1: Go to Google Play Console**

https://play.google.com/console

**Step 2: Create new app**

- App name: Neural Dream
- Default language: English (US)
- App type: App
- Free / Paid: Free
- Category: Health & Fitness

**Step 3: Complete store listing**

- Upload app icon (512x512)
- Upload feature graphic (1024x500)
- Upload screenshots (minimum 2)
- Fill in short + full description
- Add privacy policy URL

**Step 4: Complete content rating questionnaire**

Answer the IARC questionnaire — no violence, no gambling, health data collected.

**Step 5: Upload AAB to internal testing track**

- Go to Testing → Internal testing → Create new release
- Upload `app-release.aab`
- Add release notes: "Initial release — EEG brain monitor with Muse 2"
- Add your Google account as an internal tester
- Roll out to internal testing

**Step 6: Test internal release**

- Open the internal testing link on your Android phone
- Install from Play Store
- Verify app works: dashboard, ML API, BLE

**Step 7: Promote to production (when ready)**

Only after internal testing passes. Go to Production → Create new release → use same AAB.

---

### Task 11: Final Build, Sync, and Push

**Step 1: Rebuild with all fixes**

```bash
npm run build:mobile
npx cap sync
```

**Step 2: Run full test suite**

```bash
npx vitest run
source ml/.venv/bin/activate && python3 -m pytest ml/tests/ -q --tb=line -W ignore
```

**Step 3: Commit and push**

```bash
git add -A
git commit -m "feat: mobile apps ready — Android Play Store + iOS local builds"
git push origin main
```
