import { sbGetSetting, sbSaveSetting } from "./supabase-store";
/**
 * adaptive-sampling.ts — Adaptive EEG sampling rate for battery optimization.
 *
 * Reduces processing frequency when the user is idle (low signal variance)
 * and restores full rate during active sessions. Also provides a Low Power
 * Mode toggle that reduces sync frequency and disables background processing.
 *
 * Usage:
 *   import { AdaptiveSampler, LowPowerManager } from "@/lib/adaptive-sampling";
 *
 *   const sampler = new AdaptiveSampler({ baseSampleRateHz: 256 });
 *   // In your EEG loop:
 *   sampler.reportVariance(computedVariance, Date.now());
 *   if (sampler.shouldProcess(sampleIndex)) {
 *     // process this sample
 *   }
 *
 *   const lowPower = new LowPowerManager();
 *   lowPower.setEnabled(true);
 *   const syncMs = lowPower.getSyncIntervalMs(); // 300000 (5 min)
 */

// ── Constants ───────────────────────────────────────────────────────────────

export const LOW_POWER_STORAGE_KEY = "ndw_low_power_mode";

const LOW_POWER_SYNC_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

// ── AdaptiveConfig ──────────────────────────────────────────────────────────

export interface AdaptiveConfig {
  /** Base sampling rate in Hz (default 256). */
  baseSampleRateHz: number;
  /** Signal variance below this = idle (default 1.0 uV^2). */
  idleVarianceThreshold?: number;
  /** How long variance must stay below threshold before reducing rate (ms). */
  idleDurationMs?: number;
  /** Reduced sampling rate when idle (Hz). */
  reducedRateHz?: number;
}

// ── AdaptiveSampler ─────────────────────────────────────────────────────────

export class AdaptiveSampler {
  private readonly _baseRate: number;
  private readonly _reducedRate: number;
  private readonly _idleThreshold: number;
  private readonly _idleDurationMs: number;

  private _currentRate: number;
  private _idleSince: number | null = null; // timestamp when idle started
  private _idle: boolean = false;

  constructor(config: AdaptiveConfig) {
    this._baseRate = config.baseSampleRateHz;
    this._reducedRate = config.reducedRateHz ?? 64;
    this._idleThreshold = config.idleVarianceThreshold ?? 1.0;
    this._idleDurationMs = config.idleDurationMs ?? 5000;
    this._currentRate = this._baseRate;
  }

  /**
   * Report the current signal variance. Called periodically (e.g., every 200ms).
   * The sampler tracks how long variance stays below the idle threshold.
   */
  reportVariance(variance: number, timestampMs: number): void {
    if (variance >= this._idleThreshold) {
      // Active signal — restore full rate immediately
      this._idleSince = null;
      this._idle = false;
      this._currentRate = this._baseRate;
      return;
    }

    // Variance below threshold
    if (this._idleSince === null) {
      this._idleSince = timestampMs;
    }

    const idleDuration = timestampMs - this._idleSince;
    if (idleDuration >= this._idleDurationMs) {
      this._idle = true;
      this._currentRate = this._reducedRate;
    }
  }

  /** Current effective sampling rate. */
  getCurrentRate(): number {
    return this._currentRate;
  }

  /** Whether the sampler has entered idle mode. */
  isIdle(): boolean {
    return this._idle;
  }

  /**
   * Given a sample index within a second of data (0..baseSampleRate-1),
   * returns whether this sample should be processed at the current rate.
   *
   * At full rate, all samples are processed. At reduced rate, only every Nth.
   */
  shouldProcess(sampleIndex: number): boolean {
    if (this._currentRate >= this._baseRate) return true;
    const decimationFactor = Math.round(this._baseRate / this._currentRate);
    return sampleIndex % decimationFactor === 0;
  }
}

// ── LowPowerManager ────────────────────────────────────────────────────────

export class LowPowerManager {
  private _enabled: boolean;

  constructor() {
    this._enabled = this._loadState();
  }

  private _loadState(): boolean {
    try {
      return sbGetSetting(LOW_POWER_STORAGE_KEY) === "true";
    } catch {
      return false;
    }
  }

  private _saveState(): void {
    try {
      sbSaveSetting(LOW_POWER_STORAGE_KEY, String(this._enabled));
    } catch {
      // localStorage unavailable
    }
  }

  /** Whether Low Power Mode is currently enabled. */
  isEnabled(): boolean {
    return this._enabled;
  }

  /** Enable or disable Low Power Mode. Persists to localStorage. */
  setEnabled(enabled: boolean): void {
    this._enabled = enabled;
    this._saveState();
  }

  /**
   * Sync interval in milliseconds.
   * 0 = real-time (no artificial delay).
   * 300000 = 5 minutes when Low Power Mode is on.
   */
  getSyncIntervalMs(): number {
    return this._enabled ? LOW_POWER_SYNC_INTERVAL_MS : 0;
  }

  /**
   * Whether background processing (e.g., background EEG analysis) should run.
   * Disabled in Low Power Mode to save battery.
   */
  shouldProcessInBackground(): boolean {
    return !this._enabled;
  }
}
