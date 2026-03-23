/**
 * Privacy Policy — required for HealthKit access and App Store submission.
 * Accessible without authentication at /privacy.
 */

import { Shield, Brain, Database, Lock, Eye, Mail, Mic, Heart, Utensils, Download, Trash2, Scale } from "lucide-react";
import { Card } from "@/components/ui/card";

interface PolicySection {
  icon: React.ElementType;
  title: string;
  content: string[];
}

const SECTIONS: PolicySection[] = [
  {
    icon: Brain,
    title: "What We Collect",
    content: [
      "EEG brainwave data recorded during sessions using a connected Muse 2, Muse S, or compatible BCI headset. Raw EEG signal buffers are processed in memory and are never stored to disk or sent to any server.",
      "Voice recordings used for emotion analysis. Voice data is processed on-device using on-device ML models. Raw audio is not sent to external servers and is not retained after analysis is complete.",
      "Emotion classification results (stress level, focus index, valence, arousal, mood labels) derived from voice analysis or EEG signals.",
      "Mood logs and emotion corrections you provide when adjusting detected emotions. These corrections are used to personalize your emotion model over time.",
      "Nutrition data you log: meals, snacks, supplements, calorie estimates, and macronutrient breakdowns.",
      "GLP-1 injection records: dose, injection site, date, and schedule. This data is stored locally on your device via localStorage.",
      "Health metrics synced from connected platforms — only when you grant explicit permission:",
      "  - Apple HealthKit: HRV, resting heart rate, respiratory rate, sleep analysis, step count, blood oxygen, skin temperature.",
      "  - Google Health Connect: steps, heart rate, active calories, workouts, mindfulness sessions.",
      "  - Oura Ring: readiness score, sleep stages, activity, heart rate.",
      "  - WHOOP: recovery score, strain, sleep, HRV.",
      "  - Garmin: steps, stress, body battery, workouts.",
      "Dream journal entries you write voluntarily.",
      "Account information: username, email address, and age (used for research cohort analysis).",
    ],
  },
  {
    icon: Mic,
    title: "Voice Analysis",
    content: [
      "Voice emotion analysis is processed on-device whenever possible, using on-device ML models (ONNX runtime). No raw audio is transmitted to external servers.",
      "When on-device processing is unavailable, voice features (not raw audio) may be sent to our ML backend for classification. Only extracted acoustic features — pitch, energy, spectral characteristics — are transmitted, never the spoken words.",
      "No speech-to-text transcription is performed. We do not know or store what you say — only how you sound.",
      "Voice analysis results (mood scores, emotion labels) are stored in your account to show trends over time.",
    ],
  },
  {
    icon: Database,
    title: "How We Use Your Data",
    content: [
      "To show you real-time brain state feedback, emotion trends, nutrition correlations, and wellness reports inside the app.",
      "To personalize the ML models to your brain and voice — your labeled emotion corrections are used to fine-tune a personal classifier that improves accuracy over time.",
      "For aggregated, anonymized research into EEG-based emotion recognition and voice-based wellness analysis. Individual-level data is never published.",
      "To send notifications and reminders (session reminders, streak updates, supplement/injection reminders, weekly summaries) — only if you enable them.",
      "We do not sell, rent, or share your personal data with third parties for advertising or marketing purposes.",
    ],
  },
  {
    icon: Lock,
    title: "Data Storage and Security",
    content: [
      "Server-side data is stored in a PostgreSQL database with encryption at rest (AES-256) and in transit (TLS 1.3).",
      "EEG session data and emotion readings are associated with your user account and not shared across users.",
      "Local inference runs entirely on your device — EEG features and voice features are processed on-device using onnxruntime-web before any network call is made.",
      "GLP-1 injection records, notification history, and certain preferences are stored locally on your device via localStorage and are never uploaded to our servers.",
      "We retain your server-side data for as long as your account is active. You may request deletion at any time.",
      "Raw EEG signal buffers are processed in memory and are never stored to disk or sent to any server.",
    ],
  },
  {
    icon: Eye,
    title: "Third-Party Integrations",
    content: [
      "Apple HealthKit: We request access to specific HealthKit data categories only when you explicitly connect via Connected Assets. HealthKit data is used solely to correlate physiological signals with your wellness metrics. We do not share HealthKit data with any third party. You can revoke access at any time in iOS Settings > Privacy > Health.",
      "Google Health Connect: On Android, we request specific Health Connect permissions (steps, heart rate, calories, workouts, mindfulness). Data is synced only when you grant permission and connect. You can revoke access in Android Settings > Health Connect.",
      "Oura, WHOOP, Garmin: These wearable connections use OAuth 2.0. We request only the data scopes needed for wellness tracking (sleep, activity, recovery, HRV). Previously synced data remains in your profile even after disconnecting. You can reconnect or delete the data at any time.",
      "No third-party integration data is shared with advertisers or sold to data brokers.",
    ],
  },
  {
    icon: Heart,
    title: "EEG Data",
    content: [
      "EEG data is processed locally on your device via ML models (EEGNet, LightGBM, or feature-based heuristics).",
      "Raw EEG signals from your headband are buffered in-browser memory for real-time processing and are never written to disk or transmitted to external servers.",
      "Processed results (emotion labels, focus/stress/relaxation indices, sleep staging) are stored on our server to display your session history and trends.",
      "Baseline calibration data (30-second resting-state recordings) is stored server-side to normalize your live EEG readings. You can reset calibration at any time from Settings.",
    ],
  },
  {
    icon: Utensils,
    title: "Nutrition and GLP-1 Data",
    content: [
      "Meal logs, supplement records, and nutrition data you enter are stored on our server to power food-emotion correlation analysis and trend reports.",
      "GLP-1 injection tracking data (dose, site, date) is stored locally on your device via localStorage. It is not uploaded to our servers unless you explicitly export it.",
      "Nutrition data may be correlated with your mood and voice analysis data to generate food-emotion insights. These correlations are computed for your account only and are not shared.",
    ],
  },
  {
    icon: Download,
    title: "Data Export",
    content: [
      "Under GDPR Article 20 (Right to Data Portability), you can download all your data at any time.",
      "Export options include: brain session data (CSV/JSON), health metrics, dream journals, emotion readings, and a full JSON archive of your entire account.",
      "Go to the Export page or Settings > Data & Privacy to download your data.",
      "Exported files are generated on-demand and delivered directly to your device. We do not retain copies of exported files.",
    ],
  },
  {
    icon: Trash2,
    title: "Data Deletion",
    content: [
      "Under GDPR Article 17 (Right to Erasure), you can request deletion of your account and all associated data.",
      "Go to Settings > Data & Privacy > Delete My Account to submit a deletion request.",
      "Deletion requests have a 30-day grace period. During this period, you can cancel by contacting support. After 30 days, all data is permanently and irreversibly removed.",
      "Deleted data includes: EEG sessions, voice analysis results, mood logs, dream journals, health metrics, nutrition data, AI chat history, and account information.",
      "You can also email privacy@neuraldreamworkshop.com with subject 'Data Deletion Request' to request deletion manually.",
    ],
  },
  {
    icon: Shield,
    title: "Research Participation",
    content: [
      "If you join the research study via the Research Hub, your data may be used in peer-reviewed publications. All published data is fully anonymized — no names, emails, or identifiers are included.",
      "Research participation is voluntary. You may withdraw at any time by contacting us, and your data will be excluded from all future analyses.",
      "The research protocol was designed to comply with IRB guidelines for non-invasive EEG research.",
    ],
  },
  {
    icon: Scale,
    title: "EU AI Act Notice",
    content: [
      "This app uses EEG-based emotion recognition, classified as high-risk AI under EU AI Act Annex III.",
      "This system is designed for personal wellness use only. It is not deployed in workplace or educational settings.",
      "Users maintain full control over their data and can disable emotion recognition at any time via Biometric Consent settings.",
      "For questions about our AI compliance, contact privacy@neuraldreamworkshop.com.",
    ],
  },
  {
    icon: Shield,
    title: "Medical Disclaimer",
    content: [
      "AntarAI is a wellness and self-tracking application. It is NOT a medical device and is not intended to diagnose, treat, cure, or prevent any disease or medical condition.",
      "EEG readings, emotion classifications, sleep staging, and all other outputs are for personal wellness tracking and informational purposes only.",
      "GLP-1 injection tracking is a personal logging tool. Always follow your healthcare provider's instructions for medication dosing and scheduling.",
      "If you have concerns about your mental health, sleep, or any medical condition, consult a qualified healthcare professional.",
    ],
  },
  {
    icon: Mail,
    title: "Contact and Data Requests",
    content: [
      "To request deletion of your account: use Settings > Data & Privacy > Delete My Account, or email privacy@neuraldreamworkshop.com with subject 'Data Deletion Request'.",
      "To request a copy of your data: use the Export page, or email privacy@neuraldreamworkshop.com with subject 'Data Export Request'. We will respond within 30 days.",
      "To withdraw from the research study: email research@neuraldreamworkshop.com.",
      "For any other privacy questions: contact privacy@neuraldreamworkshop.com.",
      "We comply with GDPR (EU), CCPA (California), and applicable data protection regulations.",
    ],
  },
];

export default function PrivacyPolicy() {
  return (
    <main className="p-4 md:p-6 space-y-6 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-start gap-4">
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0 mt-1"
          style={{ background: "hsl(152, 60%, 48%, 0.12)", border: "1px solid hsl(152, 60%, 48%, 0.3)" }}
        >
          <Shield className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-semibold">Privacy Policy</h1>
          <p className="text-sm text-muted-foreground mt-1">
            AntarAI &mdash; effective March 2026
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            AntarAI is a wellness and research application. We take your data seriously —
            especially sensitive data like brainwave recordings, voice analysis, health metrics,
            and medication tracking. This policy explains what we collect, why, and how
            you can control it.
          </p>
        </div>
      </div>

      {/* Consent settings link */}
      <Card className="border-primary/30 bg-primary/5 p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Manage your biometric data consent</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              Control which types of data the app can collect
            </p>
          </div>
          <a
            href="/consent-settings"
            className="text-sm font-medium text-primary hover:underline shrink-0"
          >
            Consent Settings
          </a>
        </div>
      </Card>

      {/* Policy sections */}
      {SECTIONS.map(({ icon: Icon, title, content }) => (
        <Card key={title} className="glass-card p-5 space-y-3">
          <div className="flex items-center gap-2">
            <Icon className="h-4 w-4 text-primary shrink-0" />
            <h2 className="text-sm font-semibold">{title}</h2>
          </div>
          <ul className="space-y-2">
            {content.map((line, i) => (
              <li key={i} className="text-sm text-muted-foreground flex gap-2">
                <span className="text-primary mt-0.5 shrink-0">
                  {line.startsWith("  -") ? "" : "\u2022"}
                </span>
                <span>{line.startsWith("  -") ? line.trim() : line}</span>
              </li>
            ))}
          </ul>
        </Card>
      ))}

      {/* Last updated */}
      <p className="text-xs text-muted-foreground text-center pb-4">
        Last updated: March 2026. We will notify you of material changes via in-app
        notification before they take effect.
      </p>
    </main>
  );
}
