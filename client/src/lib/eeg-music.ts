/**
 * eeg-music.ts
 *
 * Maps current EEG state to music recommendations.
 * Uses dominant brainwave band, arousal, stress, focus, and emotion
 * to suggest a music category with optional binaural beat frequency.
 *
 * Research basis:
 * - Brain.fm + Muse: "Neural Phase Locking" verified by EEG
 * - FRENZ Brainband: real-time EEG drives personalized frequency music
 * - 40 Hz gamma entrainment improves attention in ADHD adults
 * - 10 Hz alpha entrainment promotes relaxation (Berger, 1929; Klimesch, 1999)
 * - 6 Hz theta entrainment deepens meditation (Lagopoulos et al., 2009)
 * - 3 Hz delta entrainment aids sleep onset (Marshall et al., 2006)
 */

export interface MusicRecommendation {
  category: "focus" | "calm" | "energy" | "sleep" | "meditation";
  reason: string;
  spotifyQuery: string;
  binauralFreq?: number;
  colorAccent: string;
}

export interface EegMusicState {
  dominantBand: string;
  arousal: number;
  stress: number;
  focus: number;
  emotion: string;
}

/**
 * Given a current EEG state, return a single music recommendation.
 *
 * Priority order (first matching rule wins):
 * 1. Pre-sleep (theta dominant + low arousal) -> sleep
 * 2. High stress (>0.6) + high arousal -> calm
 * 3. High alpha + relaxed emotion -> meditation
 * 4. High focus (>0.6) + low stress -> focus
 * 5. Low arousal (<0.3) -> energy
 * 6. Fallback -> calm
 */
export function recommendMusic(eegState: EegMusicState): MusicRecommendation {
  const { dominantBand, arousal, stress, focus, emotion } = eegState;

  // Rule 1: Pre-sleep — theta dominant with low arousal
  if (dominantBand === "theta" && arousal < 0.4) {
    return {
      category: "sleep",
      reason: "Your brain is winding down -- sleep sounds recommended.",
      spotifyQuery: "deep sleep sounds delta waves",
      binauralFreq: 3,
      colorAccent: "#7c3aed",
    };
  }

  // Rule 2: High stress + high arousal -> calm
  if (stress > 0.6 && arousal > 0.4) {
    return {
      category: "calm",
      reason: "Your stress is elevated -- try calming soundscapes.",
      spotifyQuery: "calming soundscapes ambient relaxation",
      binauralFreq: 10,
      colorAccent: "#0891b2",
    };
  }

  // Rule 3: Alpha dominant + relaxed emotion -> meditation
  const relaxedEmotions = ["neutral", "happy", "calm", "relaxed", "peaceful"];
  if (dominantBand === "alpha" && stress < 0.3 && relaxedEmotions.includes(emotion)) {
    return {
      category: "meditation",
      reason: "Deep relaxation detected -- meditation music will sustain this state.",
      spotifyQuery: "meditation theta waves tibetan bowls",
      binauralFreq: 6,
      colorAccent: "#a78bfa",
    };
  }

  // Rule 4: High focus + low stress -> focus
  if (focus > 0.6 && stress < 0.4) {
    return {
      category: "focus",
      reason: "You're in a focused state -- enhance it.",
      spotifyQuery: "deep focus instrumental concentration",
      binauralFreq: 40,
      colorAccent: "#6366f1",
    };
  }

  // Rule 5: Low arousal -> energy
  if (arousal < 0.3) {
    return {
      category: "energy",
      reason: "Low energy detected -- upbeat music can help.",
      spotifyQuery: "upbeat energy morning motivation",
      colorAccent: "#ea580c",
    };
  }

  // Fallback: calm (safe default)
  return {
    category: "calm",
    reason: "Take a moment to relax with soothing music.",
    spotifyQuery: "ambient calm peaceful music",
    binauralFreq: 10,
    colorAccent: "#0891b2",
  };
}
