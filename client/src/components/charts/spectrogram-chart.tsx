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

// Viridis-style color scale
function viridisColor(t: number): string {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(68 + t * (253 - 68));
  const g = Math.round(1 + t * (231 - 1));
  const b = Math.round(84 + t * (37 - 84));
  return `rgb(${r},${g},${b})`;
}

export function SpectrogramChart({
  coefficients,
  frequencies,
  times,
  events,
}: SpectrogramChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !coefficients.length || !times.length) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const nFreqs = coefficients.length;
    const nTimes = coefficients[0]?.length || 0;
    if (nTimes === 0) return;

    const width = canvas.width;
    const height = canvas.height;
    const margin = { top: 10, right: 40, bottom: 30, left: 45 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    ctx.clearRect(0, 0, width, height);

    // Find power range for normalization
    let minPower = Infinity;
    let maxPower = -Infinity;
    for (let f = 0; f < nFreqs; f++) {
      for (let t = 0; t < nTimes; t++) {
        const v = coefficients[f][t];
        if (v < minPower) minPower = v;
        if (v > maxPower) maxPower = v;
      }
    }
    const range = maxPower - minPower || 1;

    // Draw spectrogram heatmap
    const cellW = plotW / nTimes;
    const cellH = plotH / nFreqs;

    for (let f = 0; f < nFreqs; f++) {
      for (let t = 0; t < nTimes; t++) {
        const norm = (coefficients[f][t] - minPower) / range;
        ctx.fillStyle = viridisColor(norm);
        // Flip y-axis so low frequencies at bottom
        const x = margin.left + t * cellW;
        const y = margin.top + (nFreqs - 1 - f) * cellH;
        ctx.fillRect(x, y, Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    // Draw event markers
    if (events) {
      const timeRange = times[times.length - 1] - times[0];
      const timeToX = (t: number) =>
        margin.left + ((t - times[0]) / timeRange) * plotW;

      // Sleep spindles (cyan markers)
      if (events.sleep_spindles) {
        ctx.strokeStyle = "rgba(0, 255, 255, 0.9)";
        ctx.lineWidth = 2;
        for (const sp of events.sleep_spindles) {
          const x1 = timeToX(sp.start);
          const x2 = timeToX(sp.end);
          const y = margin.top + plotH * 0.3; // ~13Hz region
          ctx.beginPath();
          ctx.moveTo(x1, y - 4);
          ctx.lineTo(x1, y + 4);
          ctx.moveTo(x1, y);
          ctx.lineTo(x2, y);
          ctx.moveTo(x2, y - 4);
          ctx.lineTo(x2, y + 4);
          ctx.stroke();
        }
      }

      // K-complexes (magenta markers)
      if (events.k_complexes) {
        ctx.fillStyle = "rgba(255, 0, 255, 0.9)";
        for (const kc of events.k_complexes) {
          const x = timeToX(kc.time);
          const y = margin.top + plotH * 0.9; // Low frequency region
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }

    // Axes labels
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.font = "10px monospace";

    // Y-axis (frequency)
    ctx.textAlign = "right";
    const freqStep = Math.max(1, Math.floor(nFreqs / 5));
    for (let f = 0; f < nFreqs; f += freqStep) {
      const y = margin.top + (nFreqs - 1 - f) * cellH + cellH / 2;
      ctx.fillText(`${frequencies[f]}Hz`, margin.left - 4, y + 3);
    }

    // X-axis (time)
    ctx.textAlign = "center";
    const timeStep = Math.max(1, Math.floor(nTimes / 6));
    for (let t = 0; t < nTimes; t += timeStep) {
      const x = margin.left + t * cellW + cellW / 2;
      ctx.fillText(`${times[t].toFixed(1)}s`, x, height - 5);
    }

    // Color bar
    const barW = 12;
    const barH = plotH;
    const barX = width - margin.right + 8;
    for (let i = 0; i < barH; i++) {
      const norm = 1 - i / barH;
      ctx.fillStyle = viridisColor(norm);
      ctx.fillRect(barX, margin.top + i, barW, 1);
    }
  }, [coefficients, frequencies, times, events]);

  return (
    <div className="spectrogram-container">
      <canvas
        ref={canvasRef}
        width={600}
        height={200}
        className="w-full h-48 rounded"
        data-testid="chart-spectrogram"
      />
    </div>
  );
}
