/**
 * BLE auto-reconnection with exponential backoff (Issue #512).
 *
 * Exported constants, types, and utility functions for managing BLE
 * reconnection state in use-device.tsx. The actual reconnection logic
 * lives in the hook; this module provides the pure computation layer.
 *
 * Backoff schedule: 1s, 2s, 4s, 8s, 16s (max 5 attempts)
 * After 5 failed attempts, give up and show a manual reconnect prompt.
 * Counter resets to 0 on successful reconnect.
 */

/** Base delay for the first reconnection attempt (milliseconds). */
export const BLE_RECONNECT_BASE_MS = 1000;

/** Maximum delay cap for exponential backoff (milliseconds). */
export const BLE_RECONNECT_MAX_DELAY_MS = 16000;

/** Maximum number of reconnection attempts before giving up. */
export const BLE_RECONNECT_MAX_ATTEMPTS = 5;

/**
 * State of the BLE reconnection process.
 * Exposed to the UI so reconnection status can be shown to the user.
 */
export interface BleReconnectState {
  /** Current attempt number (0-based). 0 = first attempt. */
  attempt: number;
  /** True while a reconnection attempt is in progress. */
  isReconnecting: boolean;
  /** Last error message, or null if no error. */
  lastError: string | null;
  /** True when max attempts exhausted and reconnection has stopped. */
  gaveUp: boolean;
}

/**
 * Compute the backoff delay for a given attempt number.
 *
 * Uses exponential backoff: delay = base * 2^attempt, capped at max.
 *
 * @param attempt - Zero-based attempt number (0 = first attempt)
 * @returns Delay in milliseconds before the next reconnection attempt
 */
export function computeBleBackoffDelay(attempt: number): number {
  const delay = BLE_RECONNECT_BASE_MS * Math.pow(2, attempt);
  return Math.min(delay, BLE_RECONNECT_MAX_DELAY_MS);
}

/**
 * Create the initial (clean) reconnection state.
 */
export function createInitialBleReconnectState(): BleReconnectState {
  return {
    attempt: 0,
    isReconnecting: false,
    lastError: null,
    gaveUp: false,
  };
}

/**
 * Determine if another reconnection attempt should be made.
 *
 * @param state - Current reconnection state
 * @returns true if attempt < BLE_RECONNECT_MAX_ATTEMPTS
 */
export function shouldRetryBleReconnect(state: BleReconnectState): boolean {
  return state.attempt < BLE_RECONNECT_MAX_ATTEMPTS && !state.gaveUp;
}
