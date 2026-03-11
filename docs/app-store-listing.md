# App Store Listing — AntarAI

## App Name
Neural Dream

## Subtitle (30 chars max)
Voice + Health Readiness Tracker

## Category
Primary: Health & Fitness
Secondary: Medical (if approved) / Productivity

## App Store Description (4000 chars max)

```
Mental readiness tracker from your voice and health data. EEG optional.

Neural Dream gives you a daily mental readiness score from your voice, your health data, and what you consume — no wearable hardware required. Add EEG later for deeper brain insights.

──── YOUR MORNING READINESS REPORT ────
Every morning, one screen answers the question you already have:
"What should I do today, and when?"

• Mental readiness score for today
• Sleep-to-mood morning forecast: know if today will be a positive or challenging day before it starts
• Peak focus window so you can protect your best work time
• Recovery and stress context from Apple Health
• Yesterday's insight: "Readiness was 24% higher after your 3pm walk"
• Recommended action: breathing, a reset walk, or a short recovery block

──── 10-SECOND VOICE CHECK-INS ────
Record a short voice note three times a day and Neural Dream estimates your emotional state, stress load, and readiness trend — no wearable, no hardware barrier.

• Morning, noon, and evening check-ins in under 10 seconds each
• Voice-based emotion and energy snapshots
• Trend tracking across days and weeks
• Optional multimodal fusion with health and EEG data

──── APPLE HEALTH INTEGRATION ────
Your mental state does not exist in isolation. Neural Dream connects the dots with data you already have.

• Sleep duration, HRV, and heart rate trends from Apple Health
• Sleep-to-mood forecast: predicts tomorrow's focus window and stress risk from last night's sleep
• Supplement tracking to see what actually helps your mood and readiness
• Food and routine correlations that explain good and bad days
• A Daily Brain Report that tells you when to push and when to rest

──── REAL-TIME REGULATION TOOLS ────
When stress rises, the app gives you something to do about it.

• Coherence breathing, box breathing, 4-7-8, and physiological sigh
• Biofeedback charts that show before/after changes
• A simple loop: measure, act, and see the result

──── WHAT MAKES IT DIFFERENT ────
Most wellness apps tell you how your body is doing. Neural Dream is built to tell you how your mind is doing — and it works without any hardware.

• Voice-first: works the moment you install it, no wearable required
• Apple Health integration for sleep, HRV, and recovery context
• Sleep-to-mood forecasting from your health data
• Optional EEG for lab-grade brain measurements
• Personalized recommendations from your own patterns
• Privacy-first design with optional local analysis paths

──── OPTIONAL EEG MODE ────
Have a Muse 2 or Muse S headband? Connect it to unlock deeper live brain metrics, sleep staging, dream tracking, and real-time neurofeedback.

No EEG hardware? The app is fully functional in voice + health mode — most users never need it.

──── PRIVACY FIRST ────
Your data is yours. No data is sold to third parties. Voice, health, and EEG features are built for personal insight, not advertising.

See our full privacy policy at neuraldreamworkshop.com/privacy

── Note: This app is a wellness tool, not a medical device. Voice and EEG insights are statistical wellness signals, not clinical diagnoses.
```

## Keywords (100 chars max)
mental readiness,voice mood tracker,stress,focus,recovery,sleep,health,biofeedback,eeg,wellness

## What's New (first release)
Voice-first launch of Neural Dream with Daily Brain Report, mental readiness scoring, health-context insights, supplement tracking, and optional EEG integration for deeper brain metrics.

## Support URL
https://neuraldreamworkshop.com/support

## Privacy Policy URL
https://dream-analysis.vercel.app/privacy

---

## Screenshot Descriptions (5 required)

### Screenshot 1 — Onboarding / No Hardware Required
**Title:** "Works Without Any Hardware"
**Content:** onboarding screen showing:
  - "Start with voice + Apple Health" primary CTA
  - "Add EEG later (optional)" secondary option
  - Short list of what you get immediately: Daily Brain Report, voice check-ins, sleep forecast
  - "No Muse headband needed" callout

### Screenshot 2 — Voice Check-In
**Title:** "10 Seconds. Better Self-Knowledge."
**Content:** voice check-in recording screen showing:
  - Large record button with morning/noon/evening selector
  - Prompt text for the daily voice reflection
  - Recent emotion result badge (e.g. "Calm · Valence +0.4")
  - "No wearable required" subtext

### Screenshot 3 — Daily Brain Report + Sleep Forecast
**Title:** "Your Mental Readiness, Every Morning"
**Content:** /brain-report page showing:
  - Sleep-to-mood forecast card ("Tonight's sleep predicts... Positive mood")
  - Mental readiness score
  - Peak focus window (9:30am – 12:00pm)
  - Yesterday's insight card
  - Recommended action card
  - "Based on: Voice + Health" source badge

### Screenshot 4 — Supplement Correlation
**Title:** "See What Actually Helps"
**Content:** supplement tracking results showing:
  - Positive/neutral effect cards
  - Example: "Vitamin D: positive effect"
  - Mood/readiness correlation summary
  - Time-of-day trend

### Screenshot 5 — Biofeedback Reset
**Title:** "Shift Your State In Real Time"
**Content:** /biofeedback page showing:
  - Expanding circle animation
  - Before/after stress chart
  - Guided breathing timer
  - Recovery message after session

---

## App Store Connect Privacy Nutrition Label

### Data Used to Track You: NONE

### Data Linked to You:
| Category | Type | Use |
|----------|------|-----|
| Health & Fitness | Voice wellness signals, EEG readings, sleep data, HRV | App functionality |
| Identifiers | User ID | App functionality |

### Data Not Linked to You:
| Category | Type | Use |
|----------|------|-----|
| Usage Data | Crash logs | Analytics |

---

## Age Rating
4+ (no objectionable content)

## Copyright
© 2026 AntarAI

---

## HealthKit Purpose Strings (Info.plist)

```xml
<key>NSHealthShareUsageDescription</key>
<string>Neural Dream reads your Apple Health data (heart rate, HRV, sleep, SpO2, steps) to enrich your daily mental readiness report with physical context.</string>

<key>NSHealthUpdateUsageDescription</key>
<string>Neural Dream does not write to Apple Health in this version.</string>
```

## Bluetooth Usage (Info.plist)

```xml
<key>NSBluetoothAlwaysUsageDescription</key>
<string>Neural Dream optionally connects to a Muse 2 or Muse S EEG headband via Bluetooth for live brain monitoring. Bluetooth is not required — the app works in voice + health mode without it.</string>

<key>NSBluetoothPeripheralUsageDescription</key>
<string>Neural Dream can connect to a Muse 2 or Muse S EEG headband via Bluetooth for optional deeper brain metrics. This feature is not required to use the app.</string>
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
