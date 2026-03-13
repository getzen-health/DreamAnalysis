import { useEffect, useRef } from "react";

interface SpectrogramChartProps {
  coefficients: number[][];
  frequencies: number[];
  times: number[];
  events?: {
    sleep_spindles?: Array<{ start: number; end: number; amplitude: number }>;
    k_complexes?: Array<{ time: number; amplitude: number }>;
  };
}

// Improved viridis color map: dark purple → teal → yellow
function viridisColor(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  // Keyframes at t=0, 0.25, 0.5, 0.75, 1.0
  const stops: [number, number, number][] = [
    [68,   1,  84],   // 0.00 dark purple
    [59,  82, 139],   // 0.25 deep blue
    [33, 145, 140],   // 0.50 teal
    [94, 201,  98],   // 0.75 green
    [253, 231,  37],  // 1.00 yellow
  ];
  const n = stops.length - 1;
  const i = Math.min(Math.floor(t * n), n - 1);
  const f = t * n - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [
    Math.round(a[0] + f * (b[0] - a[0])),
    Math.round(a[1] + f * (b[1] - a[1])),
    Math.round(a[2] + f * (b[2] - a[2])),
  ];
}

export function SpectrogramChart({
  coefficients,
  frequencies,
  times,
  events,
}: SpectrogramChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !coefficients.length || !times.length) return;

    // Match canvas resolution to container + device pixel ratio (sharp on retina)
    const dpr = window.devicePixelRatio || 1;
    const displayW = container.clientWidth || 600;
    const displayH = 200;
    canvas.width  = displayW * dpr;
    canvas.height = displayH * dpr;
    canvas.style.width  = `${displayW}px`;
    canvas.style.height = `${displayH}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const nFreqs = coefficients.length;
    const nTimes = coefficients[0]?.length || 0;
    if (nTimes === 0) return;

    const margin = { top: 8, right: 50, bottom: 28, left: 42 };
    const plotW = displayW - margin.left - margin.right;
    const plotH = displayH - margin.top - margin.bottom;

    ctx.clearRect(0, 0, displayW, displayH);

    // Find power range with percentile clipping for better contrast
    const allVals: number[] = [];
    for (let f = 0; f < nFreqs; f++)
      for (let t = 0; t < nTimes; t++)
        allVals.push(coefficients[f][t]);
    allVals.sort((a, b) => a - b);
    const lo = allVals[Math.floor(allVals.length * 0.02)] ?? 0;
    const hi = allVals[Math.floor(allVals.length * 0.98)] ?? 1;
    const range = hi - lo || 1;

    // Use ImageData for fast pixel-level rendering
    const imgData = ctx.createImageData(plotW, plotH);
    const data = imgData.data;

    for (let py = 0; py < plotH; py++) {
      // Map pixel row → frequency index (y flipped: low freq at bottom)
      const fIdx = Math.floor(((plotH - 1 - py) / plotH) * nFreqs);
      const fi = Math.min(fIdx, nFreqs - 1);
      for (let px = 0; px < plotW; px++) {
        const tIdx = Math.floor((px / plotW) * nTimes);
        const ti = Math.min(tIdx, nTimes - 1);
        const norm = Math.max(0, Math.min(1, (coefficients[fi][ti] - lo) / range));
        const [r, g, b] = viridisColor(norm);
        const i4 = (py * plotW + px) * 4;
        data[i4]     = r;
        data[i4 + 1] = g;
        data[i4 + 2] = b;
        data[i4 + 3] = 255;
      }
    }
    ctx.putImageData(imgData, margin.left, margin.top);

    // Draw event markers
    if (events && times.length > 1) {
      const timeRange = times[times.length - 1] - times[0];
      if (timeRange > 0) {
        const timeToX = (t: number) =>
          margin.left + ((t - times[0]) / timeRange) * plotW;

        if (events.sleep_spindles?.length) {
          ctx.strokeStyle = "rgba(0,255,255,0.9)";
          ctx.lineWidth = 1.5;
          for (const sp of events.sleep_spindles) {
            const x1 = timeToX(sp.start);
            const x2 = timeToX(sp.end);
            const y = margin.top + plotH * 0.28;
            ctx.beginPath();
            ctx.moveTo(x1, y - 5); ctx.lineTo(x1, y + 5);
            ctx.moveTo(x1, y); ctx.lineTo(x2, y);
            ctx.moveTo(x2, y - 5); ctx.lineTo(x2, y + 5);
            ctx.stroke();
          }
        }

        if (events.k_complexes?.length) {
          ctx.fillStyle = "rgba(255,0,255,0.9)";
          for (const kc of events.k_complexes) {
            const x = timeToX(kc.time);
            const y = margin.top + plotH * 0.88;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
          }
        }
      }
    }

    // Y-axis — frequency labels
    ctx.fillStyle = "rgba(200,200,220,0.85)";
    ctx.font = `${10}px monospace`;
    ctx.textAlign = "right";
    const nLabels = 5;
    for (let i = 0; i <= nLabels; i++) {
      const fIdx = Math.round((i / nLabels) * (nFreqs - 1));
      const y = margin.top + (1 - i / nLabels) * plotH;
      ctx.fillText(`${Math.round(frequencies[fIdx] ?? 0)}Hz`, margin.left - 4, y + 3);
    }

    // X-axis — time labels
    ctx.textAlign = "center";
    const nTimeLabels = Math.min(6, nTimes);
    for (let i = 0; i <= nTimeLabels; i++) {
      const tIdx = Math.round((i / nTimeLabels) * (nTimes - 1));
      const x = margin.left + (i / nTimeLabels) * plotW;
      ctx.fillText(`${(times[tIdx] ?? 0).toFixed(1)}s`, x, displayH - 6);
    }

    // Color bar
    ctx.textAlign = "right";
    const barX = displayW - margin.right + 10;
    const barW = 10;
    for (let i = 0; i < plotH; i++) {
      const norm = 1 - i / plotH;
      const [r, g, b] = viridisColor(norm);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(barX, margin.top + i, barW, 1);
    }
    // Color bar labels
    ctx.fillStyle = "rgba(200,200,220,0.75)";
    ctx.font = "9px monospace";
    ctx.textAlign = "left";
    ctx.fillText("max", barX + barW + 2, margin.top + 8);
    ctx.fillText("min", barX + barW + 2, margin.top + plotH);

  }, [coefficients, frequencies, times, events]);

  return (
    <div ref={containerRef} className="w-full" style={{ height: 200 }}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label="EEG spectrogram — frequency power over time, dark purple (low) to yellow (high), with sleep spindle and K-complex event markers"
        className="rounded"
      />
    </div>
  );
}
