/**
 * EEGWaveformCanvas — raw ECG-style oscilloscope for 4 Muse channels.
 *
 * Uses a canvas element (not SVG) for 30fps updates without React re-renders.
 * Maintains a circular buffer of `256 × windowSec` samples per channel.
 *
 * Props
 * -----
 * signals    – 4×N float[][] from the WebSocket frame (TP9/AF7/AF8/TP10)
 * windowSec  – visible time window in seconds (default 5)
 * height     – canvas height in px (default 280)
 */

import { useEffect, useRef } from "react";

/* ── Config ─────────────────────────────────────────────────── */
const FS = 256;                    // Muse sample rate (Hz)
const CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"];
const CHANNEL_COLORS = [
  "hsl(200, 70%, 55%)",   // TP9  — cyan-blue
  "hsl(152, 60%, 48%)",   // AF7  — green
  "hsl(38,  85%, 58%)",   // AF8  — amber
  "hsl(262, 45%, 65%)",   // TP10 — purple
];
const BG_COLOR    = "hsl(220, 22%, 6%)";
const GRID_COLOR  = "hsl(220, 18%, 14%)";
const LABEL_COLOR = "hsl(220, 12%, 42%)";

/* ── Types ──────────────────────────────────────────────────── */
interface Props {
  signals?: number[][] | null;
  windowSec?: number;
  height?: number;
}

/* ── Component ──────────────────────────────────────────────── */
export function EEGWaveformCanvas({
  signals,
  windowSec = 5,
  height = 280,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Circular buffer per channel — pre-sized to avoid GC pressure
  const bufLen   = FS * windowSec;
  const bufRef   = useRef<Float32Array[]>(
    CHANNEL_NAMES.map(() => new Float32Array(bufLen))
  );
  const writePos = useRef(0);           // next write index (shared across channels)
  const rafId    = useRef<number>(0);

  // ── Ingest new samples — listen to unthrottled CustomEvent ──────
  // The WebSocket hook throttles React state to 1.5s for UI readability,
  // but dispatches a raw "eeg-signals" CustomEvent on every frame (4Hz).
  // Listening here keeps the circular buffer full so the canvas shows
  // continuous waveforms instead of flat lines between UI updates.
  useEffect(() => {
    const handler = (e: Event) => {
      const sigs = (e as CustomEvent<number[][]>).detail;
      if (!Array.isArray(sigs)) return;
      const buf  = bufRef.current;
      const wpos = writePos.current;
      for (let ch = 0; ch < Math.min(sigs.length, 4); ch++) {
        const chunk = sigs[ch];
        if (!Array.isArray(chunk)) continue;
        for (let i = 0; i < chunk.length; i++) {
          buf[ch][(wpos + i) % bufLen] = chunk[i];
        }
      }
      writePos.current = (wpos + (sigs[0] ?? []).length) % bufLen;
    };
    window.addEventListener("eeg-signals", handler);
    return () => window.removeEventListener("eeg-signals", handler);
  }, [bufLen]);

  // Fallback: also ingest from the (throttled) React prop so the canvas
  // still works if someone passes signals directly without the hook.
  useEffect(() => {
    if (!signals || !Array.isArray(signals)) return;
    const buf   = bufRef.current;
    let   wpos  = writePos.current;
    for (let ch = 0; ch < Math.min(signals.length, 4); ch++) {
      const chunk = signals[ch];
      if (!Array.isArray(chunk)) continue;
      for (let i = 0; i < chunk.length; i++) {
        buf[ch][(wpos + i) % bufLen] = chunk[i];
      }
    }
    writePos.current = (wpos + (signals[0] ?? []).length) % bufLen;
  }, [signals, bufLen]);

  // ── Draw loop (requestAnimationFrame at ~30fps) ───────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    function draw() {
      if (!canvas || !ctx) return;
      const W = canvas.width;
      const H = canvas.height;
      const nCh = 4;
      const trackH = H / nCh;

      // Background
      ctx.fillStyle = BG_COLOR;
      ctx.fillRect(0, 0, W, H);

      // Horizontal grid lines per channel
      ctx.strokeStyle = GRID_COLOR;
      ctx.lineWidth = 0.5;
      for (let ch = 1; ch < nCh; ch++) {
        const y = ch * trackH;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
      }

      // Draw each channel
      const buf  = bufRef.current;
      const wpos = writePos.current;

      for (let ch = 0; ch < nCh; ch++) {
        const chBuf = buf[ch];
        const midY  = (ch + 0.5) * trackH;

        // Auto-gain: find max absolute value in buffer
        let maxAbs = 1e-6;
        for (let i = 0; i < bufLen; i++) {
          const v = Math.abs(chBuf[i]);
          if (v > maxAbs) maxAbs = v;
        }
        const gain = (trackH * 0.4) / maxAbs;

        ctx.beginPath();
        ctx.strokeStyle = CHANNEL_COLORS[ch];
        ctx.lineWidth   = 1.2;

        for (let px = 0; px < W; px++) {
          // Map pixel x → circular buffer index (oldest → newest = left → right)
          const sampleIdx = (wpos - bufLen + Math.floor((px / W) * bufLen) + bufLen) % bufLen;
          const y = midY - chBuf[sampleIdx] * gain;
          if (px === 0) ctx.moveTo(px, y);
          else          ctx.lineTo(px, y);
        }
        ctx.stroke();

        // Channel label
        ctx.fillStyle  = LABEL_COLOR;
        ctx.font       = "11px monospace";
        ctx.textAlign  = "left";
        ctx.fillText(CHANNEL_NAMES[ch], 8, ch * trackH + 14);

        // Color swatch dot
        ctx.beginPath();
        ctx.arc(80, ch * trackH + 9, 4, 0, Math.PI * 2);
        ctx.fillStyle = CHANNEL_COLORS[ch];
        ctx.fill();
      }

      rafId.current = requestAnimationFrame(draw);
    }

    rafId.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafId.current);
  }, [bufLen]);

  // ── Resize observer to keep canvas pixel-perfect ─────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => {
      canvas.width  = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    });
    observer.observe(canvas);
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    return () => observer.disconnect();
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height, display: "block", borderRadius: "0.5rem" }}
    />
  );
}
