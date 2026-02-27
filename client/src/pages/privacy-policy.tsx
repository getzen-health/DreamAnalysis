/**
 * Privacy Policy — required for HealthKit access and App Store submission.
 * Accessible without authentication at /privacy.
 */

import { Shield, Brain, Database, Lock, Eye, Mail } from "lucide-react";
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
      "EEG brainwave data recorded during sessions using a connected Muse 2 or OpenBCI headset.",
      "Sleep staging and dream detection data generated from EEG analysis.",
      "Emotion classification results (stress level, focus index, valence, arousal) derived from your EEG signal.",
      "Health metrics you explicitly log: sleep duration, sleep quality, heart rate variability, meals, and activity.",
      "Apple HealthKit or Google Health Connect data — only when you grant permission. This includes HRV, resting heart rate, sleep stages, steps, SpO₂, and skin temperature.",
      "Dream journal entries you write voluntarily.",
      "Account information: username, email address, and age (used for research cohort analysis).",
    ],
  },
  {
    icon: Database,
    title: "How We Use Your Data",
    content: [
      "To show you real-time brain state feedback, emotion trends, and sleep reports inside the app.",
      "To personalize the ML models to your brain — your labeled EEG data is used to fine-tune a personal classifier head that runs locally on your device.",
      "For aggregated, anonymized research into EEG-based emotion recognition and sleep staging. Individual-level data is never published.",
      "To send just-in-time intervention alerts (breathing reminders, focus prompts) when your brain state crosses pre-set thresholds — only if you enable notifications.",
      "We do not sell, rent, or share your personal data with third parties.",
    ],
  },
  {
    icon: Lock,
    title: "Data Storage and Security",
    content: [
      "Your data is stored in a Neon PostgreSQL database (US-East-1 region) with encryption at rest (AES-256) and in transit (TLS 1.3).",
      "EEG session data and emotion readings are associated with your user account and not shared across users.",
      "Local inference runs entirely on your device — EEG features are processed on-device using onnxruntime-web before any network call is made.",
      "We retain your data for as long as your account is active. You may request deletion at any time.",
      "Raw EEG signal buffers are processed in memory and are never stored to disk or sent to any server.",
    ],
  },
  {
    icon: Eye,
    title: "Apple HealthKit",
    content: [
      "Neural Dream may request access to Apple HealthKit data categories: HRV SDNN, resting heart rate, respiratory rate, sleep analysis, step count, blood oxygen, and skin temperature.",
      "HealthKit data is used solely to improve the accuracy of the MultimodalEmotionFusion model — cross-correlating EEG states with physiological signals.",
      "We do not share HealthKit data with any third party, including advertisers.",
      "HealthKit data is never uploaded to our servers unless you explicitly export a session report.",
      "You can revoke HealthKit access at any time in iOS Settings → Privacy → Health.",
    ],
  },
  {
    icon: Shield,
    title: "Research Participation",
    content: [
      "If you join the research study via the Research Hub, your data may be used in peer-reviewed publications. All published data is fully anonymized — no names, emails, or identifiers are included.",
      "Research participation is voluntary. You may withdraw at any time by contacting us, and your data will be excluded from all future analyses.",
      "The research protocol was designed to comply with IRB guidelines for non-invasive EEG research.",
      "You may request a copy of any data attributed to your account at any time.",
    ],
  },
  {
    icon: Mail,
    title: "Contact and Data Requests",
    content: [
      "To request deletion of your account and all associated data, email privacy@neuraldreamworkshop.com with subject line \"Data Deletion Request\".",
      "To request a copy of your data, email privacy@neuraldreamworkshop.com with subject line \"Data Export Request\". We will respond within 30 days.",
      "To withdraw from the research study, email research@neuraldreamworkshop.com.",
      "For any other privacy questions, contact privacy@neuraldreamworkshop.com.",
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
            Neural Dream Workshop &mdash; effective February 2026
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            Neural Dream Workshop is a wellness and research application. We take your
            brainwave data seriously. This policy explains what we collect, why, and how
            you can control it.
          </p>
        </div>
      </div>

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
                <span className="text-primary mt-0.5 shrink-0">•</span>
                <span>{line}</span>
              </li>
            ))}
          </ul>
        </Card>
      ))}

      {/* Last updated */}
      <p className="text-xs text-muted-foreground text-center pb-4">
        Last updated: February 2026. We will notify you of material changes via in-app
        notification before they take effect.
      </p>
    </main>
  );
}
