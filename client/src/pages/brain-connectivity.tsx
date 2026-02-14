import { useState, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Network, Loader2, GitBranch } from "lucide-react";
import { ConnectivityChart } from "@/components/charts/connectivity-chart";
import { analyzeConnectivity, simulateEEG } from "@/lib/ml-api";

interface ConnectivityResult {
  connectivity_matrix: number[][];
  graph_metrics: {
    clustering_coefficient: number;
    avg_path_length: number;
    small_world_index: number;
    hub_nodes: number[];
    modularity: number;
    degree_centrality?: number[];
  };
  directed_flow: {
    granger: {
      matrix: number[][];
      significant_pairs: Array<{ from: number; to: number; strength: number }>;
    };
    dtf_matrix: number[][];
    dominant_direction: string;
  };
}

export default function BrainConnectivity() {
  const [result, setResult] = useState<ConnectivityResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [nChannels, setNChannels] = useState("4");
  const [method, setMethod] = useState("coherence");

  const runAnalysis = useCallback(async () => {
    setLoading(true);
    try {
      // Generate multi-channel simulated data
      const sim = await simulateEEG("rest", 10, 256, parseInt(nChannels));
      const data = await analyzeConnectivity(sim.signals, 256);
      setResult(data);
    } catch (e) {
      console.error("Connectivity analysis failed:", e);
    } finally {
      setLoading(false);
    }
  }, [nChannels]);

  return (
    <main className="p-4 md:p-6 space-y-6">
      <div className="flex items-center gap-3">
        <Network className="h-6 w-6 text-primary" />
        <h2 className="text-xl font-futuristic font-bold">Brain Network Connectivity</h2>
      </div>

      {/* Controls */}
      <Card className="glass-card p-4 rounded-xl">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="text-sm text-foreground/70">Channels:</span>
            <Select value={nChannels} onValueChange={setNChannels}>
              <SelectTrigger className="w-20 h-8 text-xs bg-card/50 border-primary/20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="2">2</SelectItem>
                <SelectItem value="4">4</SelectItem>
                <SelectItem value="6">6</SelectItem>
                <SelectItem value="8">8</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-foreground/70">Method:</span>
            <Select value={method} onValueChange={setMethod}>
              <SelectTrigger className="w-32 h-8 text-xs bg-card/50 border-primary/20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="coherence">Coherence</SelectItem>
                <SelectItem value="granger">Granger Causality</SelectItem>
                <SelectItem value="dtf">DTF</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button
            onClick={runAnalysis}
            disabled={loading}
            className="bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <GitBranch className="h-4 w-4 mr-2" />
            )}
            Analyze
          </Button>
        </div>
      </Card>

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Connectivity Graph */}
          <Card className="glass-card p-6 rounded-xl hover-glow">
            <h3 className="text-lg font-futuristic font-semibold mb-4">
              Connectivity Graph
            </h3>
            <ConnectivityChart
              matrix={result.connectivity_matrix}
              graphMetrics={result.graph_metrics}
              directedFlow={result.directed_flow}
            />
            <div className="flex items-center gap-4 mt-4 text-xs text-foreground/50">
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-[hsl(195,100%,50%)] inline-block" /> Directed
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-[hsl(270,70%,65%)] inline-block" /> Undirected
              </span>
              <span>Flow: {result.directed_flow.dominant_direction.replace(/_/g, " ")}</span>
            </div>
          </Card>

          {/* Graph Metrics */}
          <div className="space-y-4">
            <Card className="glass-card p-6 rounded-xl hover-glow">
              <h3 className="text-lg font-futuristic font-semibold mb-4">Graph Metrics</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-foreground/50">Clustering Coefficient</p>
                  <p className="text-xl font-mono font-bold text-primary">
                    {result.graph_metrics.clustering_coefficient.toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-foreground/50">Avg Path Length</p>
                  <p className="text-xl font-mono font-bold text-secondary">
                    {result.graph_metrics.avg_path_length.toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-foreground/50">Small-World Index</p>
                  <p className="text-xl font-mono font-bold text-warning">
                    {result.graph_metrics.small_world_index.toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-foreground/50">Modularity</p>
                  <p className="text-xl font-mono font-bold text-success">
                    {result.graph_metrics.modularity.toFixed(3)}
                  </p>
                </div>
              </div>
            </Card>

            <Card className="glass-card p-6 rounded-xl hover-glow">
              <h3 className="text-sm font-futuristic font-semibold mb-3">Hub Nodes</h3>
              <div className="flex gap-2 flex-wrap">
                {result.graph_metrics.hub_nodes.length > 0 ? (
                  result.graph_metrics.hub_nodes.map((node) => (
                    <span
                      key={node}
                      className="px-2 py-1 rounded bg-primary/10 border border-primary/30 text-primary text-xs font-mono"
                    >
                      Ch {node}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-foreground/50">No hub nodes detected</span>
                )}
              </div>
            </Card>

            {/* Connectivity Matrix Heatmap */}
            <Card className="glass-card p-6 rounded-xl hover-glow">
              <h3 className="text-sm font-futuristic font-semibold mb-3">Connectivity Matrix</h3>
              <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${result.connectivity_matrix.length}, 1fr)` }}>
                {result.connectivity_matrix.map((row, i) =>
                  row.map((val, j) => {
                    const intensity = Math.round(val * 255);
                    const bg = i === j
                      ? "rgb(50,50,50)"
                      : `rgb(${Math.round(68 + val * 185)},${Math.round(1 + val * 230)},${Math.round(84 - val * 47)})`;
                    return (
                      <div
                        key={`${i}-${j}`}
                        className="aspect-square rounded-sm"
                        style={{ backgroundColor: bg }}
                        title={`Ch${i}-Ch${j}: ${val.toFixed(3)}`}
                      />
                    );
                  })
                )}
              </div>
            </Card>
          </div>
        </div>
      )}
    </main>
  );
}
