import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

/**
 * Tests for BLE auto-reconnection with exponential backoff (Issue #512).
 *
 * The reconnection logic lives in use-device.tsx's BLE status callback.
 * We test the backoff timing, max attempts, and state transitions here
 * by verifying the exported constants and the reconnection algorithm.
 */

// Import the constants and types we'll use
import {
  BLE_RECONNECT_MAX_ATTEMPTS,
  BLE_RECONNECT_BASE_MS,
  BLE_RECONNECT_MAX_DELAY_MS,
  computeBleBackoffDelay,
  type BleReconnectState,
} from "@/lib/ble-reconnect";

describe("BLE auto-reconnection with exponential backoff", () => {
  describe("constants", () => {
    it("has a base delay of 1000ms", () => {
      expect(BLE_RECONNECT_BASE_MS).toBe(1000);
    });

    it("caps delay at 16000ms", () => {
      expect(BLE_RECONNECT_MAX_DELAY_MS).toBe(16000);
    });

    it("allows a maximum of 5 reconnect attempts", () => {
      expect(BLE_RECONNECT_MAX_ATTEMPTS).toBe(5);
    });
  });

  describe("computeBleBackoffDelay", () => {
    it("returns 1s for first attempt (attempt=0)", () => {
      expect(computeBleBackoffDelay(0)).toBe(1000);
    });

    it("returns 2s for second attempt", () => {
      expect(computeBleBackoffDelay(1)).toBe(2000);
    });

    it("returns 4s for third attempt", () => {
      expect(computeBleBackoffDelay(2)).toBe(4000);
    });

    it("returns 8s for fourth attempt", () => {
      expect(computeBleBackoffDelay(3)).toBe(8000);
    });

    it("returns 16s for fifth attempt", () => {
      expect(computeBleBackoffDelay(4)).toBe(16000);
    });

    it("caps at max delay even for high attempt numbers", () => {
      expect(computeBleBackoffDelay(10)).toBe(16000);
      expect(computeBleBackoffDelay(100)).toBe(16000);
    });

    it("never returns a negative value", () => {
      for (let i = 0; i < 20; i++) {
        expect(computeBleBackoffDelay(i)).toBeGreaterThan(0);
      }
    });

    it("follows exponential growth: 1s, 2s, 4s, 8s, 16s", () => {
      const delays = [0, 1, 2, 3, 4].map(computeBleBackoffDelay);
      expect(delays).toEqual([1000, 2000, 4000, 8000, 16000]);
    });
  });

  describe("BleReconnectState type", () => {
    it("has correct shape for initial state", () => {
      const state: BleReconnectState = {
        attempt: 0,
        isReconnecting: false,
        lastError: null,
        gaveUp: false,
      };
      expect(state.attempt).toBe(0);
      expect(state.isReconnecting).toBe(false);
      expect(state.lastError).toBeNull();
      expect(state.gaveUp).toBe(false);
    });

    it("can represent an active reconnection attempt", () => {
      const state: BleReconnectState = {
        attempt: 3,
        isReconnecting: true,
        lastError: "Connection lost",
        gaveUp: false,
      };
      expect(state.attempt).toBe(3);
      expect(state.isReconnecting).toBe(true);
      expect(state.lastError).toBe("Connection lost");
    });

    it("can represent exhausted attempts (gave up)", () => {
      const state: BleReconnectState = {
        attempt: 5,
        isReconnecting: false,
        lastError: "Max attempts reached",
        gaveUp: true,
      };
      expect(state.gaveUp).toBe(true);
      expect(state.attempt).toBe(5);
    });
  });
});
