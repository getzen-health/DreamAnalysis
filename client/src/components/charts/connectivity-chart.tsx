import { useEffect, useRef } from "react";

interface ConnectivityChartProps {
  matrix: number[][];
  graphMetrics?: {
    clustering_coefficient: number;
    avg_path_length: number;
    small_world_index: number;
    hub_nodes: number[];
    degree_centrality?: number[];
  };
  directedFlow?: {
    granger?: { significant_pairs: Array<{ from: number; to: number; strength: number }> };
  };
  channelLabels?: string[];
}

// 10-20 system approximate positions (normalized 0-1) for up to 8 channels
const DEFAULT_POSITIONS: Array<[number, number]> = [
  [0.35, 0.15], // Fp1
  [0.65, 0.15], // Fp2
  [0.15, 0.45], // C3
  [0.85, 0.45], // C4
  [0.25, 0.75], // P3
  [0.75, 0.75], // P4
  [0.35, 0.9],  // O1
  [0.65, 0.9],  // O2
];

const DEFAULT_LABELS = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"];

export function ConnectivityChart({
  matrix,
  graphMetrics,
  directedFlow,
  channelLabels,
}: ConnectivityChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const n = matrix.length;
  const labels = channelLabels || DEFAULT_LABELS.slice(0, n);
  const positions = DEFAULT_POSITIONS.slice(0, n);

  const width = 400;
  const height = 350;
  const padding = 30;

  // Scale positions to SVG coordinates
  const nodeX = (i: number) => padding + positions[i][0] * (width - 2 * padding);
  const nodeY = (i: number) => padding + positions[i][1] * (height - 2 * padding);

  // Find max connectivity for normalization
  let maxConn = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && matrix[i][j] > maxConn) maxConn = matrix[i][j];
    }
  }

  // Degree centrality for node sizing
  const degrees = graphMetrics?.degree_centrality || matrix.map((row) => {
    const sum = row.reduce((a, b) => a + b, 0);
    return sum / (n - 1);
  });

  const hubNodes = new Set(graphMetrics?.hub_nodes || []);

  // Threshold for showing edges (top 60% of connections)
  const threshold = maxConn * 0.3;

  return (
    <div className="connectivity-chart">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        data-testid="chart-connectivity"
      >
        {/* Head outline */}
        <ellipse
          cx={width / 2}
          cy={height / 2}
          rx={width / 2 - 10}
          ry={height / 2 - 10}
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth={1}
        />

        {/* Edges */}
        {matrix.map((row, i) =>
          row.map((val, j) => {
            if (i >= j || val < threshold) return null;
            const opacity = val / maxConn;
            const strokeWidth = 1 + opacity * 3;

            // Check if this is a directed Granger pair
            const isDirected = directedFlow?.granger?.significant_pairs?.some(
              (p) => (p.from === i && p.to === j) || (p.from === j && p.to === i)
            );

            return (
              <line
                key={`${i}-${j}`}
                x1={nodeX(i)}
                y1={nodeY(i)}
                x2={nodeX(j)}
                y2={nodeY(j)}
                stroke={isDirected ? "hsl(195, 100%, 50%)" : "hsl(270, 70%, 65%)"}
                strokeWidth={strokeWidth}
                opacity={opacity * 0.8}
              />
            );
          })
        )}

        {/* Directed flow arrows */}
        {directedFlow?.granger?.significant_pairs?.map((pair, idx) => {
          if (pair.from >= n || pair.to >= n) return null;
          const x1 = nodeX(pair.from);
          const y1 = nodeY(pair.from);
          const x2 = nodeX(pair.to);
          const y2 = nodeY(pair.to);
          const mx = (x1 + x2) / 2;
          const my = (y1 + y2) / 2;
          const angle = Math.atan2(y2 - y1, x2 - x1);
          const arrowSize = 6;

          return (
            <polygon
              key={`arrow-${idx}`}
              points={`${mx},${my} ${mx - arrowSize * Math.cos(angle - 0.4)},${my - arrowSize * Math.sin(angle - 0.4)} ${mx - arrowSize * Math.cos(angle + 0.4)},${my - arrowSize * Math.sin(angle + 0.4)}`}
              fill="hsl(195, 100%, 50%)"
              opacity={pair.strength * 0.8}
            />
          );
        })}

        {/* Nodes */}
        {positions.map((_, i) => {
          const r = 8 + (degrees[i] || 0) * 10;
          const isHub = hubNodes.has(i);

          return (
            <g key={`node-${i}`}>
              <circle
                cx={nodeX(i)}
                cy={nodeY(i)}
                r={r}
                fill={isHub ? "hsl(195, 100%, 50%)" : "hsl(270, 70%, 65%)"}
                opacity={0.8}
                stroke={isHub ? "hsl(195, 100%, 70%)" : "none"}
                strokeWidth={isHub ? 2 : 0}
              />
              <text
                x={nodeX(i)}
                y={nodeY(i) + 3}
                textAnchor="middle"
                fill="white"
                fontSize={9}
                fontFamily="monospace"
              >
                {labels[i]}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
