import { getSupabase } from "@/lib/supabase-browser";

export type Quadrant = "ha_pos" | "ha_neg" | "la_pos" | "la_neg";

export interface EEGSnapshot {
  valence: number;
  arousal: number;
  stress_index: number | null;
  focus_index: number | null;
  alpha_power: number | null;
  beta_power: number | null;
  theta_power: number | null;
  frontal_asymmetry: number | null;
}

export interface EmotionFingerprint {
  id: string;
  userId: string;
  label: string;
  quadrant: Quadrant;
  centroid: EEGSnapshot;
  sampleCount: number;
  lastSeen: string;
  isPersonal: boolean;
}

const PRESETS: Record<Quadrant, string[]> = {
  ha_pos: ["excited", "inspired", "euphoric", "motivated", "awe", "flow", "electric", "energized",
           "playful", "confident", "fierce", "bold", "grateful", "proud", "radiant", "joyful"],
  ha_neg: ["overwhelmed", "anxious", "scattered", "wired", "dread", "rage", "panicked", "frantic",
           "irritated", "restless", "tense", "on-edge", "desperate", "trapped", "chaotic", "alarmed"],
  la_pos: ["calm", "content", "grateful", "nostalgic", "tender", "serene", "fulfilled", "peaceful",
           "cozy", "reflective", "safe", "gentle", "dreamy", "open", "grounded", "clear"],
  la_neg: ["sad", "hollow", "numb", "melancholy", "grief", "drained", "detached", "empty",
           "defeated", "hopeless", "foggy", "withdrawn", "invisible", "flat", "resigned", "heavy"],
};

const LOCAL_KEY = "ndw_emotion_fingerprints";
const MIN_SAMPLES_FOR_SUGGESTION = 3;
const EUCLIDEAN_THRESHOLD = 0.3;

function isPrivacyModeEnabled(): boolean {
  try { return localStorage.getItem("ndw_privacy_mode") === "true"; } catch { return false; }
}

function euclideanDistance(a: EEGSnapshot, b: EEGSnapshot): number {
  const fields: Array<keyof EEGSnapshot> = ["valence", "arousal", "stress_index", "focus_index"];
  let sum = 0;
  let count = 0;
  for (const field of fields) {
    const av = a[field] as number | null;
    const bv = b[field] as number | null;
    if (av !== null && bv !== null) {
      sum += (av - bv) ** 2;
      count++;
    }
  }
  return count > 0 ? Math.sqrt(sum / count) : Infinity;
}

function runningAverageCentroid(existing: EEGSnapshot, incoming: EEGSnapshot, n: number): EEGSnapshot {
  const avg = (a: number | null, b: number | null): number | null => {
    if (a === null && b === null) return null;
    if (a === null) return b;
    if (b === null) return a;
    return (a * (n - 1) + b) / n;
  };
  return {
    valence:           avg(existing.valence, incoming.valence) as number,
    arousal:           avg(existing.arousal, incoming.arousal) as number,
    stress_index:      avg(existing.stress_index, incoming.stress_index),
    focus_index:       avg(existing.focus_index, incoming.focus_index),
    alpha_power:       avg(existing.alpha_power, incoming.alpha_power),
    beta_power:        avg(existing.beta_power, incoming.beta_power),
    theta_power:       avg(existing.theta_power, incoming.theta_power),
    frontal_asymmetry: avg(existing.frontal_asymmetry, incoming.frontal_asymmetry),
  };
}

export class EmotionTaxonomy {
  private fingerprints: EmotionFingerprint[];

  constructor(private userId: string) {
    try {
      this.fingerprints = JSON.parse(localStorage.getItem(LOCAL_KEY) || "[]");
    } catch {
      this.fingerprints = [];
    }
  }

  getQuadrant(valence: number, arousal: number): Quadrant {
    const highArousal = arousal >= 0.5;
    const positiveValence = valence >= 0.5;
    if (highArousal && positiveValence)  return "ha_pos";
    if (highArousal && !positiveValence) return "ha_neg";
    if (!highArousal && positiveValence) return "la_pos";
    return "la_neg";
  }

  getPresetsForQuadrant(quadrant: Quadrant): string[] {
    return PRESETS[quadrant];
  }

  getFingerprints(): EmotionFingerprint[] {
    return this.fingerprints;
  }

  async labelEmotion(label: string, snapshot: EEGSnapshot): Promise<EmotionFingerprint> {
    const quadrant = this.getQuadrant(snapshot.valence, snapshot.arousal);
    const now = new Date().toISOString();
    const existing = this.fingerprints.find(f => f.label === label);

    let fp: EmotionFingerprint;
    if (existing) {
      const newCount = existing.sampleCount + 1;
      fp = {
        ...existing,
        centroid: runningAverageCentroid(existing.centroid, snapshot, newCount),
        sampleCount: newCount,
        lastSeen: now,
      };
      const idx = this.fingerprints.indexOf(existing);
      this.fingerprints[idx] = fp;
    } else {
      fp = {
        id: crypto.randomUUID(),
        userId: this.userId,
        label,
        quadrant,
        centroid: snapshot,
        sampleCount: 1,
        lastSeen: now,
        isPersonal: true,
      };
      this.fingerprints.push(fp);
    }

    this.persist();
    await this.syncToSupabase(fp);
    return fp;
  }

  suggestFromEEG(snapshot: EEGSnapshot): string | null {
    const eligible = this.fingerprints.filter(f => f.sampleCount >= MIN_SAMPLES_FOR_SUGGESTION);
    let closest: { label: string; dist: number } | null = null;
    for (const fp of eligible) {
      const dist = euclideanDistance(fp.centroid, snapshot);
      if (!closest || dist < closest.dist) {
        closest = { label: fp.label, dist };
      }
    }
    if (closest && closest.dist < EUCLIDEAN_THRESHOLD) return closest.label;
    return null;
  }

  private persist(): void {
    try { localStorage.setItem(LOCAL_KEY, JSON.stringify(this.fingerprints)); } catch {}
  }

  private async syncToSupabase(fp: EmotionFingerprint): Promise<void> {
    if (isPrivacyModeEnabled()) return;
    const supabase = await getSupabase();
    if (!supabase) return;
    await supabase.from("emotion_fingerprints").upsert({
      id: fp.id,
      user_id: fp.userId,
      label: fp.label,
      quadrant: fp.quadrant,
      centroid: fp.centroid,
      sample_count: fp.sampleCount,
      last_seen: fp.lastSeen,
      is_personal: fp.isPersonal,
    }, { onConflict: "user_id,label" });
  }
}
