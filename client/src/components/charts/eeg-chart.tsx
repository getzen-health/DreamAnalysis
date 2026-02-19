import { useEffect, useRef } from "react";
import { Chart, ChartConfiguration, registerables } from "chart.js";

Chart.register(...registerables);

interface EEGChartProps {
  alphaWaves: number[];
  betaWaves: number[];
}

export function EEGChart({ alphaWaves, betaWaves }: EEGChartProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstance = useRef<Chart | null>(null);

  // Create chart once on mount
  useEffect(() => {
    if (!chartRef.current) return;
    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    const config: ChartConfiguration = {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Alpha Waves',
            data: [],
            borderColor: 'hsl(195, 100%, 50%)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
          },
          {
            label: 'Beta Waves',
            data: [],
            borderColor: 'hsl(270, 70%, 65%)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            labels: { color: 'rgba(255, 255, 255, 0.7)' }
          }
        },
        scales: {
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
            ticks: { color: 'rgba(255, 255, 255, 0.7)' }
          },
          x: {
            display: false
          }
        }
      }
    };

    chartInstance.current = new Chart(ctx, config);

    return () => {
      chartInstance.current?.destroy();
      chartInstance.current = null;
    };
  }, []);

  // Update data in-place without destroying the chart
  useEffect(() => {
    const chart = chartInstance.current;
    if (!chart) return;

    const labels = Array.from({ length: alphaWaves.length }, (_, i) => i);
    chart.data.labels = labels;
    chart.data.datasets[0].data = alphaWaves;
    chart.data.datasets[1].data = betaWaves;

    // Auto-scale Y axis based on actual signal range
    const allValues = [...alphaWaves, ...betaWaves];
    if (allValues.length > 0) {
      const minVal = Math.min(...allValues);
      const maxVal = Math.max(...allValues);
      const padding = Math.max(10, (maxVal - minVal) * 0.15);
      const yScale = chart.options.scales?.y;
      if (yScale) {
        yScale.min = minVal - padding;
        yScale.max = maxVal + padding;
      }
    }

    chart.update('none');
  }, [alphaWaves, betaWaves]);

  return (
    <div className="chart-container h-64">
      <canvas ref={chartRef} data-testid="chart-eeg" />
    </div>
  );
}
