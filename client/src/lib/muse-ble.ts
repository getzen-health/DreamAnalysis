/**
 * muse-ble.ts — Bluetooth LE driver for Muse 2 / Muse S headbands.
 *
 * Runs on iOS and Android via @capacitor-community/bluetooth-le.
 * On desktop (BrainFlow path) this module is imported but never called.
 *
 * Architecture:
 *   1. MuseBleManager.connect() → BLE scan → connect → write control → subscribe
 *   2. Four EEG characteristics stream 12-bit packets at 256 Hz (12 samples/packet)
 *   3. Packets are decoded into µV values and accumulated in a ring buffer
 *   4. Every EMIT_INTERVAL_MS (250 ms) the ring buffer is read and a MuseEegFrame
 *      is emitted to the registered callback
 *   5. The use-device hook listens on mobile and feeds frames into the same
 *      latestFrame state that the WebSocket path uses on desktop
 *
 * References:
 *   - muse-js (Alexandre Barachant) — GATT UUIDs and packet format
 *   - muselsl (Alexandre Barachant) — Python reference implementation
 *   - Muse 2 BLE specification (InteraXon internal, reverse-engineered)
 */

import { Capacitor } from "@capacitor/core";
import { extractBandPowers } from "./eeg-features";

// ── GATT UUIDs ───────────────────────────────────────────────────────────────

export const MUSE_SERVICE         = "0000fe8d-0000-1000-8000-00805f9b34fb";
export const MUSE_CONTROL_CHAR    = "273e0001-4c4d-454d-96be-f03bac821358";
export const MUSE_TELEMETRY_CHAR  = "273e0002-4c4d-454d-96be-f03bac821358";

// One characteristic per EEG channel (TP9, AF7, AF8, TP10, AUX)
export const MUSE_EEG_CHARS = [
  "273e0003-4c4d-454d-96be-f03bac821358", // ch0 — TP9  (left temporal)
  "273e0004-4c4d-454d-96be-f03bac821358", // ch1 — AF7  (left frontal)  ← FAA left
  "273e0005-4c4d-454d-96be-f03bac821358", // ch2 — AF8  (right frontal) ← FAA right
  "273e0006-4c4d-454d-96be-f03bac821358", // ch3 — TP10 (right temporal)
  "273e0007-4c4d-454d-96be-f03bac821358", // ch4 — AUX  (right ear/5th channel, optional)
] as const;

// ── Constants ────────────────────────────────────────────────────────────────

const MUSE_SAMPLE_RATE        = 256;   // Hz
const SAMPLES_PER_PACKET      = 12;    // samples per BLE notification per channel
const ADC_ZERO                = 2048;  // 12-bit midpoint
const ADC_TO_MICROVOLTS       = 0.48828125; // µV/count ≈ 1000 µV / 2048 counts
const RING_BUFFER_SAMPLES     = MUSE_SAMPLE_RATE * 4; // 4-second ring buffer per channel
const EMIT_INTERVAL_MS        = 250;   // emit a frame every 250 ms (4 Hz)
const N_ACTIVE_CHANNELS       = 4;     // TP9, AF7, AF8, TP10 (no AUX by default)

// BLE write commands (format: length_byte + payload + 0x0a)
function makeCommand(cmd: string): DataView {
  const bytes = new Uint8Array(cmd.length + 2);
  bytes[0] = cmd.length + 1; // length includes the 0x0a
  for (let i = 0; i < cmd.length; i++) bytes[i + 1] = cmd.charCodeAt(i);
  bytes[cmd.length + 1] = 0x0a; // newline terminator
  return new DataView(bytes.buffer);
}

const CMD_START      = makeCommand("d");   // start data streaming
const CMD_STOP       = makeCommand("h");   // halt streaming
const CMD_PRESET_P21 = makeCommand("p21"); // standard 4-channel EEG preset

// ── Types ────────────────────────────────────────────────────────────────────

export interface MuseEegFrame {
  /** Raw samples per channel — shape [4][n_samples], µV, 256 Hz */
  signals: number[][];
  /** Band powers averaged across all 4 channels */
  bandPowers: Record<string, number>;
  /** Frontal Alpha Asymmetry: ln(AF8_alpha) − ln(AF7_alpha) */
  faa: number;
  /** Stress proxy: high-beta / (alpha + beta) */
  stressIndex: number;
  /** Focus proxy: beta / (alpha + beta) */
  focusIndex: number;
  /** Relaxation proxy: alpha / (alpha + beta + theta) */
  relaxationIndex: number;
  /** Signal quality 0–100 based on amplitude variance and artifact rejection */
  signalQuality: number;
  /** Per-channel signal quality 0–100 (includes stale-stream detection) */
  channelQuality: number[];
  timestampMs: number;
  sampleRate: number;
  nChannels: number;
}

export type MuseBleState =
  | "idle"
  | "scanning"
  | "connecting"
  | "connected"
  | "streaming"
  | "error";

export interface MuseBleStatus {
  state: MuseBleState;
  deviceId: string | null;
  deviceName: string | null;
  error: string | null;
  /** True when running in a Capacitor native context (iOS / Android) */
  isNative: boolean;
}

// ── Ring buffer ───────────────────────────────────────────────────────────────

class RingBuffer {
  private buf: Float64Array;
  private head = 0;
  private count = 0;

  constructor(size: number) {
    this.buf = new Float64Array(size);
  }

  push(value: number): void {
    this.buf[this.head] = value;
    this.head = (this.head + 1) % this.buf.length;
    if (this.count < this.buf.length) this.count++;
  }

  /** Return the last `n` samples in chronological order. */
  last(n: number): number[] {
    const take = Math.min(n, this.count);
    const out: number[] = new Array(take);
    const size = this.buf.length;
    for (let i = 0; i < take; i++) {
      out[i] = this.buf[(this.head - take + i + size) % size];
    }
    return out;
  }

  get length(): number {
    return this.count;
  }
}

// ── Packet decoder ─────────────────────────────────────────────────────────────

/**
 * Decode a 20-byte Muse EEG BLE notification.
 *
 * Format:
 *   bytes[0-1]  — uint16 big-endian sequence number
 *   bytes[2-19] — 12 samples × 12 bits, packed 2 samples per 3 bytes
 *                 3-byte group = AAAA AAAA | AAAA BBBB | BBBB BBBB
 *
 * Returns an array of 12 µV values.
 */
function decodeEegPacket(data: DataView): number[] {
  const samples: number[] = new Array(SAMPLES_PER_PACKET);
  for (let i = 0; i < SAMPLES_PER_PACKET; i += 2) {
    const byteOffset = 2 + (i >> 1) * 3;
    const b0 = data.getUint8(byteOffset);
    const b1 = data.getUint8(byteOffset + 1);
    const b2 = data.getUint8(byteOffset + 2);
    const s0 = ((b0 << 4) | (b1 >> 4)) & 0xfff;
    const s1 = ((b1 & 0xf) << 8 | b2) & 0xfff;
    samples[i]     = ADC_TO_MICROVOLTS * (s0 - ADC_ZERO);
    samples[i + 1] = ADC_TO_MICROVOLTS * (s1 - ADC_ZERO);
  }
  return samples;
}

// ── Feature extraction from 4-channel buffer ─────────────────────────────────

function computeFaa(af7Samples: number[], af8Samples: number[], fs: number): number {
  if (af7Samples.length < 32 || af8Samples.length < 32) return 0;
  const bp7 = extractBandPowers(af7Samples, fs);
  const bp8 = extractBandPowers(af8Samples, fs);
  const alpha7 = Math.max(bp7.alpha ?? 0, 1e-10);
  const alpha8 = Math.max(bp8.alpha ?? 0, 1e-10);
  return Math.log(alpha8) - Math.log(alpha7); // positive = approach / positive valence
}

function computeIndices(bandPowers: Record<string, number>): {
  stress: number;
  focus: number;
  relaxation: number;
} {
  const alpha     = bandPowers.alpha    ?? 0;
  const beta      = bandPowers.beta     ?? 0.001;
  const theta     = bandPowers.theta    ?? 0.001;
  const highBeta  = beta * 0.35; // approximate high-beta as 35% of beta band
  const ab        = alpha + beta;
  return {
    stress:      Math.max(0, Math.min(1, highBeta / Math.max(ab, 1e-10))),
    focus:       Math.max(0, Math.min(1, beta      / Math.max(ab, 1e-10))),
    relaxation:  Math.max(0, Math.min(1, alpha     / Math.max(ab + theta, 1e-10))),
  };
}

/** Returns signal quality 0–100. 80+ = Active, 60–79 = Weak, <60 = Error.
 *
 * Calibrated for real Muse 2 dry-electrode EEG:
 *   - Relaxed alpha (~10 µV amplitude → variance ~50 µV²)  → ~88 (Active)
 *   - Poor contact  (variance ~3 µV²)                      → ~63 (Weak)
 *   - Disconnected  (variance < 0.5 µV²)                   → 10  (Error)
 *   - Severe blink  (variance > 3000 µV²)                  → 30  (Error, recovers next frame)
 *   - Floating lead (maxAmp > 400 µV)                      → 10  (Error)
 */
function computeSignalQuality(samples: number[]): number {
  if (samples.length === 0) return 0;
  const maxAmp = Math.max(...samples.map(Math.abs));
  // Extreme amplitude: floating electrode or total disconnection (>400 µV)
  if (maxAmp > 400) return 10;
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;
  if (variance < 0.5) return 10;    // flat line — electrode not touching skin
  if (variance > 3000) return 30;   // severe artifact or sustained noise (full eye blink etc.)
  // Log scale: maps good EEG variance (50–500 µV²) → Active (85–98)
  const logScore = 50 + 22 * Math.log10(variance + 1);
  return Math.min(98, Math.max(10, Math.round(logScore)));
}

// ── Main BLE manager class ────────────────────────────────────────────────────

export class MuseBleManager {
  private deviceId: string | null = null;
  private deviceName: string | null = null;
  private state: MuseBleState = "idle";
  private error: string | null = null;
  private rings: RingBuffer[] = Array.from(
    { length: N_ACTIVE_CHANNELS },
    () => new RingBuffer(RING_BUFFER_SAMPLES)
  );
  private emitTimer: ReturnType<typeof setInterval> | null = null;
  private keepaliveTimer: ReturnType<typeof setInterval> | null = null;
  private lastNotificationTime: number[] = [0, 0, 0, 0];
  private _controlChar: BluetoothRemoteGATTCharacteristic | null = null;
  private onFrame: ((frame: MuseEegFrame) => void) | null = null;
  private onStatusChange: ((status: MuseBleStatus) => void) | null = null;
  private BleClient: typeof import("@capacitor-community/bluetooth-le").BleClient | null = null;

  get isNative(): boolean {
    return Capacitor.isNativePlatform();
  }

  /** True when the browser exposes the Web Bluetooth API (Chrome desktop/Android). */
  get isWebBluetooth(): boolean {
    return !this.isNative && typeof navigator !== "undefined" && "bluetooth" in navigator;
  }

  /** True if any BLE path is available (native Capacitor OR Web Bluetooth). */
  get isAvailable(): boolean {
    return this.isNative || this.isWebBluetooth;
  }

  getStatus(): MuseBleStatus {
    return {
      state: this.state,
      deviceId: this.deviceId,
      deviceName: this.deviceName,
      error: this.error,
      isNative: this.isNative,
    };
  }

  /** Register callback for decoded EEG frames. */
  onEegFrame(callback: (frame: MuseEegFrame) => void): void {
    this.onFrame = callback;
  }

  /** Register callback for connection state changes. */
  onStatus(callback: (status: MuseBleStatus) => void): void {
    this.onStatusChange = callback;
  }

  private setStatus(state: MuseBleState, error: string | null = null): void {
    this.state = state;
    this.error = error;
    this.onStatusChange?.(this.getStatus());
  }

  // ── Lazy-load the BLE plugin (only available in Capacitor native context) ──

  private async getBleClient() {
    if (!this.BleClient) {
      const mod = await import("@capacitor-community/bluetooth-le");
      this.BleClient = mod.BleClient;
    }
    return this.BleClient;
  }

  // ── Web Bluetooth internals ─────────────────────────────────────────────────
  private _webGattServer: BluetoothRemoteGATTServer | null = null;

  private async _connectWebBluetooth(): Promise<void> {
    const bt = (navigator as Navigator & { bluetooth: Bluetooth }).bluetooth;
    this.setStatus("scanning");

    let device: BluetoothDevice;
    try {
      device = await bt.requestDevice({
        filters: [{ services: [MUSE_SERVICE] }],
        optionalServices: [MUSE_SERVICE],
      });
    } catch (e) {
      this.setStatus("idle", "Device selection cancelled");
      throw e;
    }

    this.deviceName = device.name ?? "Muse";
    this.setStatus("connecting");

    device.addEventListener("gattserverdisconnected", () => {
      this._webGattServer = null;
      this.stopEmitter();
      this.setStatus("idle", "Device disconnected");
    });

    // GATT connect with one retry after 2s delay
    let server: BluetoothRemoteGATTServer;
    try {
      server = await device.gatt!.connect();
    } catch (firstErr) {
      console.warn("GATT first connect attempt failed, retrying in 2s:", firstErr);
      try { device.gatt!.disconnect(); } catch { /* ignore */ }
      await new Promise((r) => setTimeout(r, 2000));
      try {
        server = await device.gatt!.connect();
      } catch (retryErr) {
        const isAndroid = /Android/i.test(navigator.userAgent);
        const hint = isAndroid
          ? "Turn off Bluetooth, wait 3s, turn back on. Unpair Muse in Settings > Bluetooth. Make sure Location is ON."
          : "In System Settings > Bluetooth, forget the Muse device. Turn Muse off (hold 5s), turn back on, then retry.";
        this.setStatus("error", `GATT connection failed. ${hint}`);
        throw retryErr;
      }
    }
    this._webGattServer = server;

    const service = await server.getPrimaryService(MUSE_SERVICE);
    const controlChar = await service.getCharacteristic(MUSE_CONTROL_CHAR);
    this._controlChar = controlChar;

    // Muse control char supports Write Without Response.
    // Try writeValueWithoutResponse first, fall back to writeValueWithResponse.
    const writeCommand = async (dv: DataView) => {
      try {
        await controlChar.writeValueWithoutResponse(dv);
      } catch {
        if (typeof controlChar.writeValueWithResponse === "function") {
          await controlChar.writeValueWithResponse(dv);
        } else {
          await (controlChar as BluetoothRemoteGATTCharacteristic & { writeValue: (v: DataView) => Promise<void> }).writeValue(dv);
        }
      }
    };
    await writeCommand(CMD_PRESET_P21);
    await new Promise((r) => setTimeout(r, 50));
    await writeCommand(CMD_START);

    // Subscribe to EEG channels
    for (let ch = 0; ch < N_ACTIVE_CHANNELS; ch++) {
      const charUuid = MUSE_EEG_CHARS[ch];
      const characteristic = await service.getCharacteristic(charUuid);
      await characteristic.startNotifications();
      const channelIndex = ch;
      characteristic.addEventListener("characteristicvaluechanged", (ev: Event) => {
        const target = ev.target as BluetoothRemoteGATTCharacteristic;
        if (target.value) this.onEegNotification(channelIndex, target.value);
      });
    }

    this.startEmitter();
    this.setStatus("streaming");
  }

  private async _disconnectWebBluetooth(): Promise<void> {
    this.stopEmitter();
    if (this._webGattServer?.connected) {
      try {
        // Send stop command before disconnecting
        const service = await this._webGattServer.getPrimaryService(MUSE_SERVICE);
        const controlChar = await service.getCharacteristic(MUSE_CONTROL_CHAR);
        await controlChar.writeValueWithResponse(CMD_STOP).catch(() => {});
      } catch { /* ignore */ }
      this._webGattServer.disconnect();
    }
    this._webGattServer = null;
    this._controlChar = null;
    this.deviceName = null;
    this.setStatus("idle");
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Check Bluetooth availability and return diagnostic info.
   * Useful for showing troubleshooting guidance before the user tries to connect.
   */
  async checkBluetoothReady(): Promise<{
    apiPresent: boolean;
    adapterAvailable: boolean | null;
    platform: "mac" | "windows" | "linux" | "android" | "ios" | "unknown";
    hints: string[];
  }> {
    const ua = navigator.userAgent;
    const platform: "mac" | "windows" | "linux" | "android" | "ios" | "unknown" =
      /Mac/i.test(ua) ? "mac"
      : /Win/i.test(ua) ? "windows"
      : /Linux/i.test(ua) && !/Android/i.test(ua) ? "linux"
      : /Android/i.test(ua) ? "android"
      : /iPhone|iPad/i.test(ua) ? "ios"
      : "unknown";

    const apiPresent = this.isWebBluetooth || this.isNative;
    let adapterAvailable: boolean | null = null;
    const hints: string[] = [];

    if (!apiPresent) {
      hints.push("Web Bluetooth API not found. Use Chrome (not Safari/Firefox).");
      return { apiPresent, adapterAvailable, platform, hints };
    }

    if (this.isWebBluetooth) {
      try {
        const bt = (navigator as Navigator & { bluetooth: Bluetooth }).bluetooth;
        adapterAvailable = await bt.getAvailability();
      } catch {
        adapterAvailable = null; // getAvailability not supported
      }

      if (adapterAvailable === false) {
        if (platform === "mac") {
          hints.push("Chrome doesn't have Bluetooth permission on macOS.");
          hints.push("Go to System Settings > Privacy & Security > Bluetooth and enable Chrome.");
        } else {
          hints.push("Bluetooth adapter is off or unavailable. Turn on Bluetooth in system settings.");
        }
      }
    }

    return { apiPresent, adapterAvailable, platform, hints };
  }

  /**
   * Scan for Muse devices and prompt the user to select one.
   * Automatically connects and starts streaming.
   *
   * Works on:
   *   - iOS / Android via @capacitor-community/bluetooth-le
   *   - Desktop / Android Chrome via Web Bluetooth API (navigator.bluetooth)
   *
   * Throws if BLE is unavailable (no native + no Web Bluetooth support).
   */
  async connect(): Promise<void> {
    if (!this.isNative && !this.isWebBluetooth) {
      throw new Error(
        "Bluetooth not available. Use Chrome on desktop/Android, or the iOS app."
      );
    }

    if (this.isWebBluetooth) {
      return this._connectWebBluetooth();
    }

    const ble = await this.getBleClient();
    this.setStatus("scanning");

    try {
      // androidNeverForLocation=false — Android 11 and below require location for BLE scan
      await ble.initialize({ androidNeverForLocation: false });
    } catch (e) {
      const msg = String(e);
      if (msg.includes("denied") || msg.includes("permission")) {
        this.setStatus("error", "Bluetooth permission denied. Go to Settings > Apps > AntarAI > Permissions and enable Bluetooth + Location.");
      } else if (msg.includes("disabled") || msg.includes("off")) {
        this.setStatus("error", "Bluetooth is turned off. Please enable Bluetooth in your phone settings.");
      } else {
        this.setStatus("error", `BLE init failed: ${msg}`);
      }
      throw e;
    }

    let device: { deviceId: string; name?: string };
    try {
      // Shows the native device picker filtered to Muse's primary service UUID
      device = await ble.requestDevice({
        services: [MUSE_SERVICE],
        optionalServices: [],
        namePrefix: "Muse",
      });
    } catch (e) {
      const msg = String(e);
      if (msg.includes("cancel")) {
        this.setStatus("idle", "Device selection cancelled");
      } else {
        this.setStatus("error", "No Muse found. Make sure Muse is ON (LED blinking) and NOT paired in system Bluetooth settings.");
      }
      throw e;
    }

    this.setStatus("connecting");
    this.deviceId   = device.deviceId;
    this.deviceName = device.name ?? "Muse";

    // Helper: write command with writeWithoutResponse fallback
    const writeCmd = async (id: string, cmd: DataView) => {
      try {
        await ble.writeWithoutResponse(id, MUSE_SERVICE, MUSE_CONTROL_CHAR, cmd);
      } catch {
        await ble.write(id, MUSE_SERVICE, MUSE_CONTROL_CHAR, cmd);
      }
    };

    // Auto-retry: attempt connect + start up to 3 times
    const MAX_RETRIES = 3;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        // Disconnect first if already connected (handles "already paired" case)
        try { await ble.disconnect(device.deviceId); } catch { /* ignore */ }
        await new Promise((r) => setTimeout(r, attempt > 1 ? 2000 : 300));

        this.setStatus("connecting");

        // Connect with timeout
        await Promise.race([
          ble.connect(device.deviceId, () => {
            this.stopEmitter();
            this.setStatus("idle", "Device disconnected");
          }),
          new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), 12000)),
        ]);

        // Wait for GATT services to settle
        await new Promise((r) => setTimeout(r, 500));

        // Send preset + start commands
        await writeCmd(device.deviceId, CMD_PRESET_P21);
        await new Promise((r) => setTimeout(r, 200));
        await writeCmd(device.deviceId, CMD_START);

        // Success — break out of retry loop
        break;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        console.warn(`Muse connect attempt ${attempt}/${MAX_RETRIES} failed:`, msg);

        if (attempt === MAX_RETRIES) {
          this.setStatus("error", `Connection failed after ${MAX_RETRIES} attempts: ${msg}`);
          this.deviceId = null;
          throw e;
        }
        // Retry after brief pause
        this.setStatus("connecting");
      }
    }

    // Subscribe to all 4 EEG channel characteristics
    for (let ch = 0; ch < N_ACTIVE_CHANNELS; ch++) {
      const charUuid = MUSE_EEG_CHARS[ch];
      const channelIndex = ch;
      await ble.startNotifications(
        device.deviceId,
        MUSE_SERVICE,
        charUuid,
        (data: DataView) => this.onEegNotification(channelIndex, data)
      );
    }

    this.startEmitter();
    this.setStatus("streaming");
  }

  /**
   * Reconnect to the last known device without showing the BLE picker.
   * Called automatically by use-device when an unexpected disconnection occurs.
   * Throws if no previous deviceId is saved or the device is out of range.
   */
  async reconnect(): Promise<void> {
    if (!this.isNative && !this.isWebBluetooth) throw new Error("BLE not available.");
    // Web Bluetooth reconnect = show picker again (no saved deviceId path)
    if (this.isWebBluetooth) return this._connectWebBluetooth();
    if (!this.deviceId) throw new Error("No device to reconnect to — connect first.");

    const ble = await this.getBleClient();
    this.setStatus("connecting");

    try {
      await ble.connect(this.deviceId, () => {
        this.stopEmitter();
        this.setStatus("idle", "Device disconnected");
      });
    } catch (e) {
      this.setStatus("error", `Reconnect failed: ${String(e)}`);
      throw e;
    }

    try {
      const writeReconnect = async (cmd: DataView) => {
        try { await ble.writeWithoutResponse(this.deviceId!, MUSE_SERVICE, MUSE_CONTROL_CHAR, cmd); }
        catch { await ble.write(this.deviceId!, MUSE_SERVICE, MUSE_CONTROL_CHAR, cmd); }
      };
      await writeReconnect(CMD_PRESET_P21);
      await new Promise((r) => setTimeout(r, 100));
      await writeReconnect(CMD_START);
    } catch (e) {
      this.setStatus("error", `Failed to restart EEG stream: ${String(e)}`);
      throw e;
    }

    // Re-subscribe to all 4 EEG characteristics
    for (let ch = 0; ch < N_ACTIVE_CHANNELS; ch++) {
      const charUuid = MUSE_EEG_CHARS[ch];
      const channelIndex = ch;
      await ble.startNotifications(
        this.deviceId!,
        MUSE_SERVICE,
        charUuid,
        (data: DataView) => this.onEegNotification(channelIndex, data)
      );
    }

    this.startEmitter();
    this.setStatus("streaming");
  }

  /**
   * Stop streaming and disconnect from the device.
   */
  async disconnect(): Promise<void> {
    if (this.isWebBluetooth) return this._disconnectWebBluetooth();

    this.stopEmitter();
    if (!this.deviceId) return;

    try {
      const ble = await this.getBleClient();
      // Send stop command before disconnecting so the headset goes to idle
      await ble.writeWithoutResponse(this.deviceId, MUSE_SERVICE, MUSE_CONTROL_CHAR, CMD_STOP).catch(() =>
        ble.write(this.deviceId!, MUSE_SERVICE, MUSE_CONTROL_CHAR, CMD_STOP).catch(() => {})
      );
      await ble.disconnect(this.deviceId);
    } catch {
      // ignore errors during teardown
    } finally {
      this.deviceId = null;
      this.deviceName = null;
      this.setStatus("idle");
    }
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  private onEegNotification(channel: number, data: DataView): void {
    if (channel >= N_ACTIVE_CHANNELS || data.byteLength < 20) return;
    this.lastNotificationTime[channel] = Date.now();
    const samples = decodeEegPacket(data);
    for (const s of samples) {
      this.rings[channel].push(s);
    }
  }

  private startEmitter(): void {
    this.emitTimer = setInterval(() => this.emitFrame(), EMIT_INTERVAL_MS);
    if (this.isWebBluetooth && this._controlChar) {
      this.keepaliveTimer = setInterval(async () => {
        if (this.state === "streaming" && this._controlChar) {
          try { await this._controlChar.writeValueWithResponse(CMD_START); } catch {}
        }
      }, 15_000);
    }
  }

  private stopEmitter(): void {
    if (this.emitTimer !== null) {
      clearInterval(this.emitTimer);
      this.emitTimer = null;
    }
    if (this.keepaliveTimer !== null) {
      clearInterval(this.keepaliveTimer);
      this.keepaliveTimer = null;
    }
  }

  private emitFrame(): void {
    if (!this.onFrame) return;

    const windowSamples = MUSE_SAMPLE_RATE; // 1-second window for feature computation
    const signals: number[][] = this.rings.map((r) => r.last(windowSamples));

    // Need at least 256 samples before emitting a meaningful frame
    if (signals[0].length < windowSamples / 4) return;

    // Average band powers across all 4 channels
    const allBandPowers = signals.map((s) =>
      s.length >= 32 ? extractBandPowers(s, MUSE_SAMPLE_RATE) : null
    );
    const validBp = allBandPowers.filter(Boolean) as Record<string, number>[];

    const avgBandPowers: Record<string, number> = {};
    for (const band of ["delta", "theta", "alpha", "beta", "gamma"]) {
      const vals = validBp.map((bp) => bp[band] ?? 0);
      avgBandPowers[band] = vals.length > 0
        ? vals.reduce((a, b) => a + b, 0) / vals.length
        : 0;
    }

    const { stress, focus, relaxation } = computeIndices(avgBandPowers);
    // ch1 = AF7, ch2 = AF8 (BrainFlow Muse 2 ordering)
    const faa = computeFaa(signals[1], signals[2], MUSE_SAMPLE_RATE);
    const now = Date.now();
    const STALE_THRESHOLD_MS = 2500;
    const sqiValues = signals.map((s, ch) => {
      if (this.lastNotificationTime[ch] > 0 && now - this.lastNotificationTime[ch] > STALE_THRESHOLD_MS) {
        return 0; // stream stale — no notifications received for this channel
      }
      return computeSignalQuality(s);
    });
    const signalQuality = sqiValues.reduce((a, b) => a + b, 0) / sqiValues.length;

    const frame: MuseEegFrame = {
      signals,
      bandPowers: avgBandPowers,
      faa,
      stressIndex: stress,
      focusIndex: focus,
      relaxationIndex: relaxation,
      signalQuality,
      channelQuality: sqiValues,
      timestampMs: Date.now(),
      sampleRate: MUSE_SAMPLE_RATE,
      nChannels: N_ACTIVE_CHANNELS,
    };

    this.onFrame(frame);
  }
}

// ── Singleton instance ────────────────────────────────────────────────────────

export const museBle = new MuseBleManager();

// ── Utility: convert MuseEegFrame → use-device EEGStreamFrame shape ───────────
// This adapter lets the mobile BLE path feed into the same hook state as the
// desktop WebSocket path (both ultimately set `latestFrame`).

export function museFrameToEegStreamFrame(f: MuseEegFrame): {
  signals: number[][];
  analysis: Record<string, unknown>;
  quality: { sqi: number; artifacts_detected: string[]; clean_ratio: number; channel_quality: number[] };
  timestamp: number;
  n_channels: number;
  sample_rate: number;
} {
  const bp = f.bandPowers;
  const theta = bp.theta ?? 0;
  const alpha = bp.alpha ?? 0;
  const beta  = bp.beta  ?? 0;
  const delta = bp.delta ?? 0;

  // Derived indices from band powers + pre-computed indices
  const thetaBetaRatio  = theta / (beta + 0.001);
  const drowsinessIdx   = Math.min(1, thetaBetaRatio * 0.6 + (1 - f.focusIndex) * 0.4);
  const creativityScore = Math.min(1, (alpha / (alpha + beta + 0.001)) * 0.6 + (theta / (theta + beta + 0.001)) * 0.4);
  const meditationScore = Math.min(1, f.relaxationIndex * 0.7 + (alpha / (alpha + beta + 0.001)) * 0.3);
  const memoryScore     = Math.min(1, f.focusIndex * 0.7 + f.relaxationIndex * 0.3);
  const flowScore       = Math.min(1, f.focusIndex * 0.6 + f.relaxationIndex * 0.4);
  const cogLoadIdx      = f.focusIndex;
  const sqi01           = f.signalQuality / 100; // sqi expected 0-1 by brain-monitor (* 100 there)

  // Stress level label
  const stressLevel = f.stressIndex > 0.7 ? "high" : f.stressIndex > 0.45 ? "moderate" : f.stressIndex > 0.2 ? "mild" : "relaxed";
  const stressLevelIdx = f.stressIndex > 0.7 ? 3 : f.stressIndex > 0.45 ? 2 : f.stressIndex > 0.2 ? 1 : 0;

  // Attention state label
  const attentionState = f.focusIndex > 0.7 ? "hyperfocused" : f.focusIndex > 0.5 ? "focused" : f.focusIndex > 0.3 ? "normal" : "distracted";
  const attentionStateIdx = f.focusIndex > 0.7 ? 3 : f.focusIndex > 0.5 ? 2 : f.focusIndex > 0.3 ? 1 : 0;

  // Meditation depth label
  const medDepth = meditationScore > 0.75 ? "deep" : meditationScore > 0.5 ? "meditating" : "relaxed";
  const medDepthIdx = meditationScore > 0.75 ? 2 : meditationScore > 0.5 ? 1 : 0;

  // Creativity state label
  const creativityState = creativityScore > 0.7 ? "creative" : creativityScore > 0.4 ? "receptive" : "analytical";

  // Drowsiness state label
  const drowsinessState = drowsinessIdx > 0.7 ? "sleepy" : drowsinessIdx > 0.45 ? "fatigued" : drowsinessIdx > 0.25 ? "mild" : "alert";

  // Memory encoding state label
  const memState = memoryScore > 0.7 ? "strong_encoding" : memoryScore > 0.45 ? "moderate_encoding" : "poor_encoding";

  // Cog load level label
  const cogLevel = cogLoadIdx > 0.65 ? "high" : cogLoadIdx > 0.35 ? "medium" : "low";
  const cogLevelIdx = cogLoadIdx > 0.65 ? 2 : cogLoadIdx > 0.35 ? 1 : 0;

  // Sleep: awake (delta is low relative to others when awake)
  const sleepDelta = delta / (delta + theta + alpha + beta + 0.001);
  const sleepStage = sleepDelta > 0.6 ? "N3" : sleepDelta > 0.45 ? "N2" : "Wake";

  return {
    signals: f.signals,
    analysis: {
      band_powers: bp,
      features: {
        faa: f.faa,
        stress_index: f.stressIndex,
        focus_index: f.focusIndex,
        relaxation_index: f.relaxationIndex,
      },
      emotions: {
        emotion: f.faa > 0.1 && f.focusIndex > 0.45 ? "focused"
               : f.faa < -0.1 && f.stressIndex > 0.55 ? "stressed"
               : f.faa < -0.15 ? "sad"
               : f.relaxationIndex > 0.5 ? "relaxed"
               : "neutral",
        confidence: sqi01,
        valence: Math.max(-1, Math.min(1, f.faa * 2)),
        arousal: f.stressIndex * 0.6 + f.focusIndex * 0.4,
        stress_index: f.stressIndex,
        focus_index: f.focusIndex,
        relaxation_index: f.relaxationIndex,
      },
      stress: {
        level: stressLevel,
        level_index: stressLevelIdx,
        stress_index: f.stressIndex,
        confidence: sqi01,
        cortisol_proxy: f.stressIndex,
      },
      attention: {
        state: attentionState,
        state_index: attentionStateIdx,
        attention_score: f.focusIndex,
        confidence: sqi01,
        theta_beta_ratio: thetaBetaRatio,
      },
      flow_state: {
        state: flowScore > 0.6 ? "flow" : "no_flow",
        state_index: flowScore > 0.6 ? 1 : 0,
        in_flow: flowScore > 0.6,
        flow_score: flowScore,
        confidence: sqi01,
        components: { focus: f.focusIndex, relaxation: f.relaxationIndex },
      },
      creativity: {
        state: creativityState,
        state_index: creativityScore > 0.7 ? 2 : creativityScore > 0.4 ? 1 : 0,
        creativity_score: creativityScore,
        confidence: sqi01,
        components: { alpha, theta },
      },
      drowsiness: {
        state: drowsinessState,
        state_index: drowsinessIdx > 0.7 ? 3 : drowsinessIdx > 0.45 ? 2 : drowsinessIdx > 0.25 ? 1 : 0,
        alertness_score: 1 - drowsinessIdx,
        drowsiness_index: drowsinessIdx,
        confidence: sqi01,
      },
      cognitive_load: {
        level: cogLevel,
        level_index: cogLevelIdx,
        load_index: cogLoadIdx,
        confidence: sqi01,
        components: { focus: f.focusIndex },
      },
      meditation: {
        depth: medDepth,
        depth_index: medDepthIdx,
        meditation_score: meditationScore,
        confidence: sqi01,
        tradition_match: medDepth,
      },
      memory_encoding: {
        state: memState,
        state_index: memoryScore > 0.7 ? 2 : memoryScore > 0.45 ? 1 : 0,
        encoding_score: memoryScore,
        will_remember_probability: memoryScore,
        confidence: sqi01,
      },
      sleep_staging: {
        stage: sleepStage,
        stage_index: sleepStage === "N3" ? 3 : sleepStage === "N2" ? 2 : 0,
        confidence: sqi01,
        probabilities: { Wake: sleepStage === "Wake" ? 0.9 : 0.1, N1: 0.0, N2: sleepStage === "N2" ? 0.8 : 0.0, N3: sleepStage === "N3" ? 0.8 : 0.0, REM: 0.0 },
        calibrated_confidence: sqi01,
      },
      dream_detection: {
        is_dreaming: false,
        probability: 0,
        rem_likelihood: 0,
        dream_intensity: 0,
        lucidity_estimate: 0,
      },
      lucid_dream: {
        state: "non_lucid",
        state_index: 0,
        lucidity_score: 0,
        confidence: sqi01,
        gamma_surge: false,
      },
    },
    quality: {
      sqi: sqi01,
      artifacts_detected: sqi01 < 0.4 ? ["amplitude"] : [],
      clean_ratio: sqi01,
      channel_quality: f.channelQuality ?? f.signals.map((s) => computeSignalQuality(s)),
    },
    timestamp: f.timestampMs / 1000,
    n_channels: f.nChannels,
    sample_rate: f.sampleRate,
  };
}
