interface BrainBandsProps {
  bandPowers: Record<string, number>;
}

const BAND_CONFIG = [
  { name: "delta", label: "Delta", range: "0.5-4 Hz", color: "hsl(270, 70%, 65%)", description: "Deep Sleep" },
  { name: "theta", label: "Theta", range: "4-8 Hz", color: "hsl(195, 100%, 50%)", description: "Relaxation" },
  { name: "alpha", label: "Alpha", range: "8-12 Hz", color: "hsl(120, 100%, 55%)", description: "Calm Focus" },
  { name: "beta", label: "Beta", range: "12-30 Hz", color: "hsl(45, 100%, 50%)", description: "Active Thinking" },
  { name: "gamma", label: "Gamma", range: "30-100 Hz", color: "hsl(0, 80%, 50%)", description: "High Cognition" },
];

export function BrainBands({ bandPowers }: BrainBandsProps) {
  return (
    <div className="space-y-4">
      {BAND_CONFIG.map(band => {
        const power = (bandPowers[band.name] || 0) * 100;
        return (
          <div key={band.name}>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full" style={{ background: band.color }} />
                <span className="text-sm font-medium">{band.label}</span>
                <span className="text-[10px] text-foreground/40">{band.range}</span>
              </div>
              <span className="text-xs font-mono" style={{ color: band.color }}>
                {power.toFixed(1)}%
              </span>
            </div>
            <div className="w-full h-2.5 bg-card/50 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${Math.min(power, 100)}%`, background: band.color }}
              />
            </div>
            <p className="text-[10px] text-foreground/40 mt-0.5">{band.description}</p>
          </div>
        );
      })}
    </div>
  );
}
