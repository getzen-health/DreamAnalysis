/**
 * Regulatory compliance constants and documentation.
 *
 * This module centralizes all compliance-related text, disclaimers,
 * and regulatory classification metadata for AntarAI. Reference this
 * module from any UI component that needs to display compliance info.
 */

// ---------------------------------------------------------------------------
// FDA / General Wellness
// ---------------------------------------------------------------------------

export const FDA_DISCLAIMER =
  "AntarAI is a wellness and self-improvement app. It is NOT a medical device " +
  "and is not intended to diagnose, treat, cure, or prevent any disease. EEG " +
  "and voice analysis features provide general wellness insights only. Not FDA cleared.";

export const FDA_CLASSIFICATION = {
  status: "General Wellness — FDA exempt",
  guidance: "FDA General Wellness: Policy for Low Risk Devices (January 2026)",
  permittedClaims: [
    "Track your emotional patterns over time",
    "See how your brain responds to different activities",
    "Understand your stress and focus trends",
  ],
  prohibitedClaims: [
    "Diagnose, treat, prevent, or cure any condition",
    "Clinically validated (without IRB study + peer-reviewed paper)",
    "Clinical thresholds (e.g. 'your stress is clinically elevated')",
    "Treatment prompts (e.g. 'take this medication')",
  ],
} as const;

// ---------------------------------------------------------------------------
// EU AI Act (effective August 2025)
// ---------------------------------------------------------------------------

export const EU_AI_ACT_NOTICE =
  "This app uses EEG-based emotion recognition, classified as high-risk AI " +
  "under EU AI Act Annex III. This system is designed for personal wellness " +
  "use only. It is not deployed in workplace or educational settings. Users " +
  "maintain full control over their data and can disable emotion recognition " +
  "at any time via Biometric Consent settings.";

export const EU_AI_ACT_CLASSIFICATION = {
  annexCategory: "Annex III — High-risk AI systems",
  riskLevel: "high-risk",
  subCategory: "Emotion recognition systems",
  deploymentContext: "Personal wellness — not workplace or educational",
  mitigations: [
    "Users can disable emotion recognition via Biometric Consent settings",
    "All EEG processing can run on-device (local processing mode)",
    "Full data export available under GDPR Article 20",
    "Full data deletion available under GDPR Article 17",
    "No automated decision-making that affects users legally or similarly",
  ],
  transparencyObligations: [
    "Users are informed the system uses EEG-based emotion recognition",
    "Confidence levels are displayed alongside all classifications",
    "Model accuracy limitations are documented and surfaced in-app",
  ],
} as const;

// ---------------------------------------------------------------------------
// GDPR (EU General Data Protection Regulation)
// ---------------------------------------------------------------------------

export const GDPR_COMPLIANCE = {
  dataController: "AntarAI / Neural Dream Workshop",
  contactEmail: "privacy@neuraldreamworkshop.com",
  legalBasis: "Explicit consent (Article 9(2)(a)) for biometric data processing",
  dataSubjectRights: [
    "Right of access (Article 15)",
    "Right to rectification (Article 16)",
    "Right to erasure (Article 17)",
    "Right to data portability (Article 20)",
    "Right to object to processing (Article 21)",
    "Right to withdraw consent at any time",
  ],
  dataRetention: "Data retained while account is active; deleted within 30 days of deletion request",
  internationalTransfers: "Data processed within the user's region when possible",
} as const;

// ---------------------------------------------------------------------------
// CCPA (California Consumer Privacy Act)
// ---------------------------------------------------------------------------

export const CCPA_COMPLIANCE = {
  consumerRights: [
    "Right to know what personal information is collected",
    "Right to delete personal information",
    "Right to opt-out of the sale of personal information",
    "Right to non-discrimination for exercising CCPA rights",
  ],
  dataCategories: [
    "Biometric information (EEG data, voice features)",
    "Health information (mood, stress, sleep metrics)",
    "Internet activity (app usage, session history)",
    "Identifiers (username, email)",
  ],
  saleOfData: false,
  sharingWithThirdParties: false,
} as const;

// ---------------------------------------------------------------------------
// BIPA (Illinois Biometric Information Privacy Act)
// ---------------------------------------------------------------------------

export const BIPA_COMPLIANCE = {
  applicability: "EEG and voice biometric data may be covered under BIPA",
  obligations: [
    "Written informed consent obtained before collection (via Biometric Consent settings)",
    "Written policy on retention and destruction schedules",
    "Biometric data not sold, leased, traded, or otherwise profited from",
  ],
} as const;

// ---------------------------------------------------------------------------
// Google Play Health App Declaration
// ---------------------------------------------------------------------------

export const GOOGLE_PLAY_HEALTH_DECLARATION = {
  appType: "Health & Fitness",
  healthClaims: false,
  medicalDeviceClaim: false,
  disclaimer: FDA_DISCLAIMER,
  contentRating: "Everyone",
  dataHandling: {
    collectsHealthData: true,
    healthDataTypes: ["EEG brainwave data", "voice emotion features", "mood logs", "sleep data"],
    encryptedInTransit: true,
    encryptedAtRest: true,
    userCanRequestDeletion: true,
    userCanExportData: true,
  },
} as const;

// ---------------------------------------------------------------------------
// Apple App Store Health Guidelines
// ---------------------------------------------------------------------------

export const APPLE_HEALTH_GUIDELINES = {
  healthKitIntegration: true,
  healthKitDataTypes: [
    "HRV", "resting heart rate", "respiratory rate", "sleep analysis",
    "step count", "blood oxygen", "skin temperature",
  ],
  purposeStrings: {
    healthKit: "AntarAI reads HealthKit data to correlate physiological metrics with your EEG-based wellness insights.",
    microphone: "AntarAI uses the microphone for voice emotion analysis. Audio is processed on-device and never stored.",
    bluetooth: "AntarAI connects to your Muse 2 EEG headband via Bluetooth for brain wave monitoring.",
  },
  medicalDisclaimer: FDA_DISCLAIMER,
} as const;
