# iOS App Store Submission Checklist

Neural Dream Workshop (AntarAI) -- App Store submission steps.

Bundle ID: `com.neuraldreamworkshop.app`
Min iOS: 14.0 (capacitor.config.ts) / 16.0 (Podfile)

---

## Phase 1: Apple Developer Account

- [ ] Enroll in the Apple Developer Program ($99/year)
  - https://developer.apple.com/programs/enroll/
  - Requires Apple ID with two-factor authentication enabled
  - Individual enrollment is fine (no need for organization)
  - Approval takes 24-48 hours

- [ ] Accept all agreements in App Store Connect
  - https://appstoreconnect.apple.com/agreements
  - Paid Applications agreement requires banking + tax info

---

## Phase 2: Local Environment Setup

- [ ] Install Xcode 15+ from the Mac App Store
  - Requires macOS 14 (Sonoma) or later for Xcode 16
  - After install: `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`

- [ ] Install Xcode command-line tools
  ```bash
  xcode-select --install
  ```

- [ ] Install CocoaPods
  ```bash
  sudo gem install cocoapods
  # or: brew install cocoapods
  ```

- [ ] Verify prerequisites
  ```bash
  xcodebuild -version        # Xcode 15.0+
  pod --version               # 1.14+
  node --version              # 18+
  npx cap --version           # 8.1.0+
  ```

---

## Phase 3: Signing Configuration

- [ ] Create an App ID in the Apple Developer portal
  - https://developer.apple.com/account/resources/identifiers/add/bundleId
  - Bundle ID: `com.neuraldreamworkshop.app`
  - Enable capabilities: HealthKit, Push Notifications, Background Modes

- [ ] Enable automatic signing in Xcode
  - Open `ios/App/App.xcworkspace`
  - Select the App target > Signing & Capabilities
  - Check "Automatically manage signing"
  - Select your team

- [ ] Verify entitlements match `App.entitlements`
  - HealthKit (already configured)
  - Push Notifications / APS Environment (already configured)
  - Background Modes: bluetooth-central, fetch, processing, remote-notification

- [ ] Verify Info.plist usage descriptions are present
  - NSBluetoothAlwaysUsageDescription (Muse headband)
  - NSBluetoothPeripheralUsageDescription (Muse headband)
  - NSHealthShareUsageDescription (Apple Health read)
  - NSHealthUpdateUsageDescription (Apple Health write)
  - NSMicrophoneUsageDescription (voice emotion analysis)
  - NSSpeechRecognitionUsageDescription (voice journaling)
  - NSMotionUsageDescription (head movement during EEG)
  - All are already in `ios/App/App/Info.plist`

---

## Phase 4: Build and Test Locally

- [ ] Build the web app and sync to iOS
  ```bash
  ./scripts/build-ios.sh
  ```

- [ ] Generate app icons (if not already done)
  ```bash
  ./scripts/generate-app-icons.sh
  ```

- [ ] Run on iOS Simulator
  - Open `ios/App/App.xcworkspace` in Xcode
  - Select iPhone 15 Pro simulator
  - Press Cmd+R
  - Verify: app loads, navigation works, dark theme renders correctly

- [ ] Test on a physical device
  - Connect iPhone via USB
  - Select device in Xcode scheme selector
  - Trust the developer profile on device: Settings > General > Device Management
  - Verify: Bluetooth scanning, HealthKit permissions, push notification prompt

---

## Phase 5: App Store Connect Setup

- [ ] Create the app in App Store Connect
  - https://appstoreconnect.apple.com > My Apps > + New App
  - Platform: iOS
  - Name: "Neural Dream"
  - Primary language: English (US)
  - Bundle ID: select `com.neuraldreamworkshop.app`
  - SKU: `neuraldream-v1`

- [ ] Fill in app metadata (see `docs/app-store-listing.md` for copy)
  - App name: Neural Dream
  - Subtitle: Voice + Health Readiness Tracker
  - Category: Health & Fitness
  - Description: (from app-store-listing.md)
  - Keywords: (from app-store-listing.md)
  - Support URL: https://neuraldreamworkshop.com/support
  - Marketing URL: https://neuraldreamworkshop.com

- [ ] Privacy policy URL
  - https://dream-analysis.vercel.app/privacy
  - This is required before submission

- [ ] App Privacy (nutrition labels)
  - Data Not Used to Track You: mark as true
  - Data Linked to You: Health & Fitness (voice, EEG, sleep, HRV), Identifiers (user ID)
  - Data Not Linked to You: Crash logs
  - See `docs/app-store-listing.md` for details

- [ ] Age rating
  - 4+ (no objectionable content, no user-generated content sharing)

---

## Phase 6: Screenshots

Required sizes (portrait, all must be provided):

- [ ] 6.7" display (iPhone 15 Pro Max): 1290 x 2796 px
- [ ] 6.5" display (iPhone 14 Plus): 1284 x 2778 px
- [ ] 5.5" display (iPhone 8 Plus): 1242 x 2208 px
- [ ] iPad Pro 12.9" (6th gen): 2048 x 2732 px (if supporting iPad)

Recommended screenshot content (5 minimum):
1. Onboarding -- "Works Without Any Hardware"
2. Voice check-in recording screen
3. Daily Brain Report with sleep forecast
4. Supplement correlation results
5. Biofeedback breathing session

Tips:
- Use Xcode Simulator > File > Screenshot (Cmd+S) to capture
- Add device frame + marketing text with tools like Previewed or Screenshots Pro
- Dark backgrounds match the app theme (bg: #13111a)

---

## Phase 7: TestFlight (Internal Testing)

- [ ] Create an archive in Xcode
  - Select "Any iOS Device (arm64)" as destination
  - Product > Archive
  - Wait for build to complete (may take a few minutes)

- [ ] Upload to App Store Connect
  - In Xcode Organizer (Window > Organizer), select the archive
  - Click "Distribute App"
  - Select "App Store Connect" > "Upload"
  - Follow prompts (automatic signing handles certificates)

- [ ] Wait for processing
  - Apple processes the build (5-30 minutes)
  - You will get an email when ready

- [ ] Add internal testers in TestFlight
  - App Store Connect > TestFlight > Internal Testing
  - Add testers by Apple ID email
  - Each tester gets an invite to install via TestFlight app

- [ ] Test the TestFlight build
  - Verify all permissions prompt correctly
  - Verify Bluetooth scanning works
  - Verify HealthKit data reads correctly
  - Verify voice recording works
  - Test on multiple iOS versions (14, 16, 17, 18)
  - Check crash logs in Xcode Organizer

---

## Phase 8: App Store Review Submission

- [ ] Select the build in App Store Connect
  - App Store > iOS App > Build section > select the TestFlight build

- [ ] Complete App Review Information
  - Contact info (name, phone, email)
  - Demo account credentials (if login required)
  - Notes for reviewer:
    ```
    Neural Dream is a mental wellness tracker. Core features work with
    voice check-ins and Apple Health data. EEG features require a Muse 2
    headband (optional). The app is fully functional without EEG hardware.
    ```

- [ ] Submit for review
  - Click "Submit for Review"
  - Review typically takes 24-48 hours (first submission may take longer)

---

## Phase 9: Post-Submission

- [ ] Monitor review status in App Store Connect
- [ ] If rejected: read the rejection reason carefully, fix, resubmit
  - Common rejection reasons:
    - Missing privacy policy
    - Incomplete metadata
    - HealthKit usage not justified in review notes
    - Bluetooth usage without clear purpose
    - Crash on reviewer device
- [ ] After approval: set release date (manual or automatic)
- [ ] Verify the live listing looks correct

---

## Common Rejection Risks for This App

1. **HealthKit**: Apple scrutinizes Health apps. The review notes must explain
   clearly what health data is read and why. The privacy policy must list
   all HealthKit data types.

2. **Bluetooth**: Must justify why Bluetooth is needed. Include that it is
   optional for connecting a Muse EEG headband and the app works without it.

3. **Background Modes**: bluetooth-central and fetch must be justified.
   Bluetooth for ongoing EEG streaming, fetch for periodic data sync.

4. **Medical claims**: Do not claim the app diagnoses, treats, or prevents
   any condition. Use "wellness" and "readiness" language, never "medical"
   or "clinical" in the store listing.

5. **Privacy**: Voice recording, health data, and EEG data are sensitive.
   The privacy policy at /privacy must comprehensively cover all data types.

---

## Quick Reference Commands

```bash
# Full build pipeline
./scripts/build-ios.sh

# Release build info
./scripts/build-ios.sh --release

# Generate all icon sizes
./scripts/generate-app-icons.sh

# Sync web changes to iOS (without full rebuild)
npx cap sync ios

# Open Xcode directly
open ios/App/App.xcworkspace
```
