/**
 * audio-fade.ts
 *
 * Smooth exponential audio fade-out utility.
 * Used by SleepStoryPlayer to fade audio on EEG-detected sleep onset
 * instead of cutting out abruptly on a dumb timer.
 */

// WeakMap keyed by audio element so multiple callers don't conflict
const fadeTimers = new WeakMap<HTMLAudioElement, ReturnType<typeof setInterval>>();

/**
 * Fade an audio element to silence over durationMs using exponential decay.
 *
 * The volume curve follows:
 *   v(t) = initialVolume * exp(-k * t)
 * where k is chosen so the volume reaches ~1% of the initial value at durationMs.
 *
 * @param audioElement  The HTMLAudioElement to fade
 * @param durationMs    Total fade duration in milliseconds (default: 90 000 = 90 s)
 * @returns             A promise that resolves when the fade is complete (volume == 0)
 */
export function fadeOutAudio(
  audioElement: HTMLAudioElement,
  durationMs: number = 90_000,
): Promise<void> {
  return new Promise((resolve) => {
    // Cancel any in-progress fade on this element first
    cancelFade(audioElement);

    const initialVolume = audioElement.volume > 0 ? audioElement.volume : 1;
    const startTime = Date.now();
    const tickMs = 250; // resolution: update volume 4× per second

    // k chosen so that v(durationMs) ≈ 0.01 * initialVolume
    //   initialVolume * exp(-k * durationMs) = 0.01 * initialVolume
    //   -k * durationMs = ln(0.01)
    //   k = -ln(0.01) / durationMs = ln(100) / durationMs
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

/**
 * Cancel any in-progress fade on an audio element.
 * The volume is left wherever it was when cancelled.
 *
 * @param audioElement  The HTMLAudioElement whose fade should be cancelled
 */
export function cancelFade(audioElement: HTMLAudioElement): void {
  const timer = fadeTimers.get(audioElement);
  if (timer !== undefined) {
    clearInterval(timer);
    fadeTimers.delete(audioElement);
  }
}
