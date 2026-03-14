/**
 * audio-fade.ts
 *
 * Smooth exponential audio fade-out utility.
 * Supports both HTMLAudioElement (legacy) and Web Audio API GainNode targets.
 *
 * Used by SleepStoryPlayer to fade audio on EEG-detected sleep onset
 * instead of cutting out abruptly on a dumb timer.
 */

// ─── Fade target abstraction ──────────────────────────────────────────────────

export type FadeTarget = HTMLAudioElement | GainNode;

// Track active fades — WeakMap keyed by target so multiple callers don't conflict
const fadeTimers = new WeakMap<FadeTarget, ReturnType<typeof setInterval>>();

// ─── HTMLAudioElement fade (legacy) ───────────────────────────────────────────

/**
 * Fade an HTMLAudioElement to silence using interval-based exponential decay.
 */
function fadeOutHTMLAudio(
  audioElement: HTMLAudioElement,
  durationMs: number,
): Promise<void> {
  return new Promise((resolve) => {
    cancelFade(audioElement);

    const initialVolume = audioElement.volume > 0 ? audioElement.volume : 1;
    const startTime = Date.now();
    const tickMs = 250;
    const k = Math.log(100) / durationMs;

    const timer = setInterval(() => {
      const elapsed = Date.now() - startTime;

      if (elapsed >= durationMs) {
        audioElement.volume = 0;
        audioElement.pause();
        clearInterval(timer);
        fadeTimers.delete(audioElement);
        resolve();
        return;
      }

      const newVolume = initialVolume * Math.exp(-k * elapsed);
      audioElement.volume = Math.max(0, newVolume);
    }, tickMs);

    fadeTimers.set(audioElement, timer);
  });
}

// ─── GainNode fade (Web Audio API) ────────────────────────────────────────────

/**
 * Fade a GainNode to silence using exponentialRampToValueAtTime.
 *
 * This uses the Web Audio API's built-in scheduling — smoother and more
 * accurate than setInterval, since the audio thread handles the ramp.
 *
 * @param gainNode     The GainNode to fade
 * @param durationMs   Total fade duration in milliseconds (default: 90 000 = 90 s)
 * @param onComplete   Optional callback fired when the fade finishes
 */
function fadeOutGainNode(
  gainNode: GainNode,
  durationMs: number,
  onComplete?: () => void,
): Promise<void> {
  return new Promise((resolve) => {
    cancelFade(gainNode);

    const ctx = gainNode.context;
    const now = ctx.currentTime;
    const endTime = now + durationMs / 1000;

    // Ensure current value is set explicitly before ramping
    const currentValue = gainNode.gain.value;
    gainNode.gain.cancelScheduledValues(now);
    gainNode.gain.setValueAtTime(Math.max(currentValue, 0.001), now);

    // Exponential ramp to near-zero (cannot ramp to exactly 0)
    gainNode.gain.exponentialRampToValueAtTime(0.001, endTime);

    // Schedule a check to resolve the promise and set gain to true zero
    const timer = setInterval(() => {
      if (ctx.currentTime >= endTime) {
        gainNode.gain.cancelScheduledValues(ctx.currentTime);
        gainNode.gain.setValueAtTime(0, ctx.currentTime);
        clearInterval(timer);
        fadeTimers.delete(gainNode);
        onComplete?.();
        resolve();
      }
    }, 500);

    fadeTimers.set(gainNode, timer);
  });
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Fade an audio target to silence over durationMs using exponential decay.
 *
 * The volume curve follows:
 *   v(t) = initialVolume * exp(-k * t)
 * where k is chosen so the volume reaches ~1% of the initial value at durationMs.
 *
 * Accepts either an HTMLAudioElement or a GainNode.
 *
 * @param target       The HTMLAudioElement or GainNode to fade
 * @param durationMs   Total fade duration in milliseconds (default: 90 000 = 90 s)
 * @param onComplete   Optional callback when fade finishes (GainNode only)
 * @returns            A promise that resolves when the fade is complete (volume == 0)
 */
export function fadeOutAudio(
  target: FadeTarget,
  durationMs: number = 90_000,
  onComplete?: () => void,
): Promise<void> {
  if (target instanceof HTMLAudioElement) {
    return fadeOutHTMLAudio(target, durationMs);
  }
  return fadeOutGainNode(target, durationMs, onComplete);
}

/**
 * Cancel any in-progress fade on a target.
 * The volume is left wherever it was when cancelled.
 *
 * @param target  The HTMLAudioElement or GainNode whose fade should be cancelled
 */
export function cancelFade(target: FadeTarget): void {
  const timer = fadeTimers.get(target);
  if (timer !== undefined) {
    clearInterval(timer);
    fadeTimers.delete(target);
  }

  // For GainNode, also cancel any scheduled ramps
  if (target instanceof GainNode) {
    target.gain.cancelScheduledValues(target.context.currentTime);
  }
}
