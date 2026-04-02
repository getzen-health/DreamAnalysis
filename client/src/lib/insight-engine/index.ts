// client/src/lib/insight-engine/index.ts
export { type DeviationMetric, type NormalizedReading, type BaselineCell } from "./baseline-store";
export { type DeviationEvent } from "./deviation-detector";
export { type StoredInsight, type PassType } from "./pattern-discovery";
export { type EmotionFingerprint, type EEGSnapshot, type Quadrant } from "./emotion-taxonomy";

import { BaselineStore, type NormalizedReading, type DeviationMetric } from "./baseline-store";
import { DeviationDetector, type DeviationEvent } from "./deviation-detector";
import { PatternDiscovery, type StoredInsight } from "./pattern-discovery";
import { EmotionTaxonomy, type EmotionFingerprint, type EEGSnapshot } from "./emotion-taxonomy";
import { InterventionLibrary } from "./intervention-library";
import { apiRequest } from "@/lib/queryClient";

export interface BriefingRequest {
  sleepData: {
    totalHours: number | null;
    deepHours: number | null;
    remHours: number | null;
    efficiency: number | null;
    dataAvailability: "full" | "total_only" | "none";
  };
  morningHrv: number | null;
  hrvRange: { min: number; max: number } | null;
  emotionSummary: {
    readingCount: number;
    avgStress: number;
    avgFocus: number;
    avgValence: number;
    dominantLabel: string;
    dominantMinutes: number;
  };
  patternSummaries: string[];
  yesterdaySummary: string;
  dreamContext?: {
    keyInsight: string | null;
    themes: string[];
    emotionalArc: string | null;
    isSleepDistress: boolean;
  } | null;
}

export interface BriefingResponse {
  stateSummary: string;
  actions: [string, string, string];
  forecast: { label: string; probability: number; reason: string };
}

interface BriefingCache {
  date: string; // UTC YYYY-MM-DD
  content: BriefingResponse;
}

const BRIEFING_CACHE_KEY = (userId: string) => `ndw_morning_briefing_${userId}`;
const BANNER_COOLDOWN_KEY = "ndw_banner_cooldown";
const BANNER_COOLDOWN_MS = 15 * 60 * 1000;

export class InsightEngine {
  private baseline: BaselineStore;
  private detector: DeviationDetector;
  private discovery: PatternDiscovery;
  private taxonomy: EmotionTaxonomy;
  private interventions: InterventionLibrary;
  private lastEvents: DeviationEvent[] = [];
  private lastReading: NormalizedReading | undefined = undefined;

  constructor(private userId: string) {
    this.baseline     = new BaselineStore();
    this.detector     = new DeviationDetector(this.baseline);
    this.discovery    = new PatternDiscovery(userId);
    this.taxonomy     = new EmotionTaxonomy(userId);
    this.interventions = new InterventionLibrary();
  }

  ingest(reading: NormalizedReading): void {
    const ts = reading.timestamp || new Date().toISOString();
    this.lastReading = reading;
    this.baseline.update(reading, ts);
    this.lastEvents = this.detector.detect(reading, ts);
    // Check intervention effectiveness for any recovered metrics
    for (const event of this.lastEvents) {
      this.interventions.checkEffectiveness(event.metric, event.zScore);
    }
  }

  getRealTimeInsights(): DeviationEvent[] {
    return this.lastEvents;
  }

  isBannerAllowed(): boolean {
    try {
      const last = Number(localStorage.getItem(BANNER_COOLDOWN_KEY) || "0");
      return Date.now() - last > BANNER_COOLDOWN_MS;
    } catch { return true; }
  }

  recordBannerShown(): void {
    try { localStorage.setItem(BANNER_COOLDOWN_KEY, String(Date.now())); } catch {}
  }

  async getStoredInsights(nowIso?: string): Promise<StoredInsight[]> {
    // Pass lastReading as current so timeBucketPass can compare against live state.
    // If ingest() has not been called yet, timeBucketPass gracefully returns [].
    return this.discovery.run(nowIso || new Date().toISOString(), this.lastReading);
  }

  getMorningBriefing(): BriefingResponse | null {
    try {
      const cached = JSON.parse(localStorage.getItem(BRIEFING_CACHE_KEY(this.userId)) || "null") as BriefingCache | null;
      if (!cached) return null;
      const today = new Date().toISOString().slice(0, 10);
      if (cached.date !== today) return null;
      return cached.content;
    } catch { return null; }
  }

  async generateMorningBriefing(request: BriefingRequest): Promise<BriefingResponse> {
    const resp = await apiRequest("POST", "/api/morning-briefing", request);
    if (!resp.ok) throw new Error(`Morning briefing failed: ${resp.status}`);
    const content = await resp.json() as BriefingResponse;
    const today = new Date().toISOString().slice(0, 10);
    try { localStorage.setItem(BRIEFING_CACHE_KEY(this.userId), JSON.stringify({ date: today, content })); } catch {}
    return content;
  }

  async labelEmotion(label: string, eegSnapshot: EEGSnapshot): Promise<EmotionFingerprint> {
    return this.taxonomy.labelEmotion(label, eegSnapshot);
  }

  getFingerprints(): EmotionFingerprint[] {
    return this.taxonomy.getFingerprints();
  }

  suggestEmotionFromEEG(snapshot: EEGSnapshot): string | null {
    return this.taxonomy.suggestFromEEG(snapshot);
  }

  recordInterventionTap(interventionId: string, metric: DeviationMetric): void {
    const lastEvent = this.lastEvents.find(e => e.metric === metric);
    // Only record when an active deviation exists — avoids storing a fake baseline zScore.
    if (!lastEvent) return;
    this.interventions.recordTap(interventionId, metric, lastEvent.zScore);
  }
}
