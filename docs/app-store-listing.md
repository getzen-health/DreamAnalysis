# App Store Listing — Neural Dream Workshop

## App Name
Neural Dream

## Subtitle (30 chars max)
Brain Performance Tracker

## Category
Primary: Health & Fitness
Secondary: Medical (if approved) / Productivity

## App Store Description (4000 chars max)

```
Understand your brain the way elite athletes understand their body.

Neural Dream reads your EEG headband (Muse 2, Muse S) in real time and tells you what your brain is actually doing — stress, focus, sleep stages, and emotional state — all day long.

──── THE MORNING REPORT ────
Every morning, one screen answers the question you already have:
"What should I do today, and when?"

• Peak focus window — protect this time for your hardest work
• Afternoon slump forecast — so you plan around it
• Yesterday's insight — "Focus was 31% higher after your 3pm walk"
• Recommended action — coherence breathing, music, or a 10-min walk

──── REAL-TIME BIOFEEDBACK ────
The moment that changes everything: you're stressed. The live reading shows it. You do a 4-minute breathing exercise and watch your own stress line drop in real time on screen.

That moment — seeing your brain respond to something you chose to do — is unlike anything else on your phone.

Seven evidence-based exercises:
• Box breathing, 4-7-8, coherence breathing
• Physiological sigh, cyclic sigh, power breath
• Custom-duration sessions with haptic guidance

──── SLEEP INTELLIGENCE ────
Sleep with your Muse 2 on and wake up to:
• Time in deep sleep (N3), REM, and light sleep
• Dreams detected (and logged for you)
• A 7-day brain recovery score

──── WHAT MAKES IT DIFFERENT ────
Other wearables show you a score after a session ends. Neural Dream shows the change during the action. That's the differentiator.

• Personalized model that improves every session
• Closed-loop interventions — stress threshold crossed → music or breathing triggered automatically
• Works offline — all analysis runs locally on device
• Connects to Apple Health for richer context (HRV, steps, SpO2)
• Weekly shareable brain summary card

──── HARDWARE REQUIRED ────
Muse 2 or Muse S EEG headband (~$250 USD, sold separately). No headband? Use simulation mode to explore all features before purchasing.

──── PRIVACY FIRST ────
Your brain data is yours. No data is sold to third parties. All EEG processing can run locally on device. Optional server sync for multi-device access.

See our full privacy policy at neuraldreamworkshop.com/privacy

── Note: This app is a wellness tool, not a medical device. EEG emotion readings are statistical patterns, not clinical diagnoses.
```

## Keywords (100 chars max)
eeg,brain,muse,biofeedback,stress,focus,sleep,emotion,meditation,neurofeedback,hrv,wellness

## What's New (first release)
Initial release of Neural Dream — real-time EEG brain performance tracker with daily brain report, sleep intelligence, and closed-loop stress interventions.

## Support URL
https://neuraldreamworkshop.com/support

## Privacy Policy URL
https://dream-analysis.vercel.app/privacy

---

## Screenshot Descriptions (5 required)

### Screenshot 1 — Daily Brain Report
**Title:** "Your Morning Intelligence Brief"
**Content:** /brain-report page showing:
  - Peak focus window (9:30am – 12:00pm)
  - Yesterday's insight card
  - Recommended action: coherence breathing
  - Sleep summary (6h 42m, 2 dreams)

### Screenshot 2 — Live Brain State
**Title:** "Watch Your Brain in Real Time"
**Content:** Dashboard showing:
  - Stress: HIGH (red bar), Focus: MEDIUM (amber), Flow: LOW (blue)
  - Live session timeline strip (green/orange blocks)
  - "Beat your focus record" challenge card

### Screenshot 3 — Biofeedback
**Title:** "See Your Stress Drop — Live"
**Content:** /biofeedback page showing:
  - Expanding circle animation (coherence breathing)
  - Live stress chart dropping during exercise
  - Before: 0.72 → After: 0.41
  - Haptic guide indicator

### Screenshot 4 — Sleep Session
**Title:** "Sleep Intelligence You'll Actually Use"
**Content:** /sleep-session recording view showing:
  - Current stage: REM (purple)
  - 4h 23m elapsed
  - Hypnogram bar (Wake → N1 → N2 → N3 → REM)
  - "2 dreams detected" spark

### Screenshot 5 — Weekly Brain Summary
**Title:** "Share Your Mental Fitness Progress"
**Content:** /weekly-summary page showing:
  - This week vs last week: Stress ↓18%, Focus ↑12%
  - Week-in-one-sentence: "Your most focused week in a month"
  - Shareable card with PNG export button

---

## App Store Connect Privacy Nutrition Label

### Data Used to Track You: NONE

### Data Linked to You:
| Category | Type | Use |
|----------|------|-----|
| Health & Fitness | EEG readings, sleep data, HRV | App functionality |
| Identifiers | User ID | App functionality |

### Data Not Linked to You:
| Category | Type | Use |
|----------|------|-----|
| Usage Data | Crash logs | Analytics |

---

## Age Rating
4+ (no objectionable content)

## Copyright
© 2026 Neural Dream Workshop

---

## HealthKit Purpose Strings (Info.plist)

```xml
<key>NSHealthShareUsageDescription</key>
<string>Neural Dream reads your Apple Health data (heart rate, HRV, sleep, SpO2, steps) to enrich your daily brain performance report with physical context.</string>

<key>NSHealthUpdateUsageDescription</key>
<string>Neural Dream does not write to Apple Health in this version.</string>
```

## Bluetooth Usage (Info.plist)

```xml
<key>NSBluetoothAlwaysUsageDescription</key>
<string>Neural Dream uses Bluetooth to connect to your Muse 2 EEG headband for real-time brain monitoring.</string>

<key>NSBluetoothPeripheralUsageDescription</key>
<string>Neural Dream connects to your Muse 2 EEG headband via Bluetooth.</string>
```

## Background Modes (Info.plist — required for sleep recording)

```xml
<key>UIBackgroundModes</key>
<array>
  <string>bluetooth-central</string>
  <string>fetch</string>
  <string>processing</string>
</array>

<key>BGTaskSchedulerPermittedIdentifiers</key>
<array>
  <string>com.neuraldreamworkshop.eeg-flush</string>
</array>
```
