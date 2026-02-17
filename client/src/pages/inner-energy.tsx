import { useState, useEffect, useCallback } from "react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Sparkles, Eye, Activity } from "lucide-react";

interface ChakraState {
  name: string;
  sanskrit: string;
  activation: number;
  color: string;
}

interface EnergyState {
  chakras: ChakraState[];
  meditationDepth: number;
  meditationStage: string;
  consciousnessLevel: number;
  consciousnessName: string;
  thirdEyeActivation: number;
  dominantEnergy: string;
  guidance: string;
}

const CHAKRA_COLORS: Record<string, string> = {
  root: "#ef4444",
  sacral: "#f97316",
  solar_plexus: "#eab308",
  heart: "#22c55e",
  throat: "#3b82f6",
  third_eye: "#6366f1",
  crown: "#a855f7",
};

const CHAKRA_INFO = [
  { key: "root", name: "Root", sanskrit: "Muladhara" },
  { key: "sacral", name: "Sacral", sanskrit: "Svadhisthana" },
  { key: "solar_plexus", name: "Solar Plexus", sanskrit: "Manipura" },
  { key: "heart", name: "Heart", sanskrit: "Anahata" },
  { key: "throat", name: "Throat", sanskrit: "Vishuddha" },
  { key: "third_eye", name: "Third Eye", sanskrit: "Ajna" },
  { key: "crown", name: "Crown", sanskrit: "Sahasrara" },
];

function simulateEnergy(): EnergyState {
  const chakras = CHAKRA_INFO.map((c) => ({
    name: c.name,
    sanskrit: c.sanskrit,
    activation: Math.round(20 + Math.random() * 60),
    color: CHAKRA_COLORS[c.key],
  }));

  const maxChakra = chakras.reduce((max, c) =>
    c.activation > max.activation ? c : max
  );

  const meditationDepth = +(1 + Math.random() * 6).toFixed(1);
  const stages = [
    "Settling",
    "Relaxation",
    "Light Absorption",
    "Deep Absorption",
    "Jhana",
  ];
  const stageIdx = Math.min(
    stages.length - 1,
    Math.floor(meditationDepth / 2)
  );

  const consciousness = Math.round(100 + Math.random() * 400);
  const levels = ["Survival", "Desire", "Willpower", "Love", "Expression", "Insight", "Unity"];
  const levelIdx = Math.min(
    levels.length - 1,
    Math.floor(consciousness / 150)
  );

  const guidances = [
    "Focus on grounding. Feel the earth beneath you.",
    "Let creative energy flow. Don't resist the stream.",
    "Your heart center is open. Practice compassion.",
    "Speak your truth today. Your throat is activated.",
    "Inner vision is strong. Trust your intuition.",
    "Higher awareness is accessible. Stay present.",
  ];

  return {
    chakras,
    meditationDepth,
    meditationStage: stages[stageIdx],
    consciousnessLevel: consciousness,
    consciousnessName: levels[levelIdx],
    thirdEyeActivation: Math.round(10 + Math.random() * 50),
    dominantEnergy: maxChakra.name,
    guidance: guidances[Math.floor(Math.random() * guidances.length)],
  };
}

export default function InnerEnergy() {
  const [energy, setEnergy] = useState<EnergyState>(simulateEnergy);

  const update = useCallback(() => {
    setEnergy((prev) => {
      const next = simulateEnergy();
      // Smooth transitions
      return {
        ...next,
        chakras: next.chakras.map((c, i) => ({
          ...c,
          activation: Math.round(
            prev.chakras[i].activation * 0.7 + c.activation * 0.3
          ),
        })),
        meditationDepth: +(
          prev.meditationDepth * 0.7 +
          next.meditationDepth * 0.3
        ).toFixed(1),
        consciousnessLevel: Math.round(
          prev.consciousnessLevel * 0.7 + next.consciousnessLevel * 0.3
        ),
        thirdEyeActivation: Math.round(
          prev.thirdEyeActivation * 0.7 + next.thirdEyeActivation * 0.3
        ),
      };
    });
  }, []);

  useEffect(() => {
    const interval = setInterval(update, 4000);
    return () => clearInterval(interval);
  }, [update]);

  return (
    <main className="p-6 space-y-6 max-w-4xl">
      {/* Guidance */}
      <div className="bg-secondary/5 border border-secondary/15 rounded-xl p-4">
        <div className="flex items-start gap-3">
          <Sparkles className="h-5 w-5 text-secondary mt-0.5 shrink-0" />
          <p className="text-sm text-foreground/80">{energy.guidance}</p>
        </div>
      </div>

      {/* Chakra Activations */}
      <Card className="glass-card p-6">
        <h3 className="text-base font-medium mb-5">Energy Centers</h3>
        <div className="space-y-3">
          {[...energy.chakras].reverse().map((chakra) => (
            <div key={chakra.name} className="flex items-center gap-3">
              <div
                className="w-3 h-3 rounded-full shrink-0"
                style={{ backgroundColor: chakra.color }}
              />
              <div className="w-24 shrink-0">
                <div className="text-sm font-medium">{chakra.name}</div>
                <div className="text-xs text-muted-foreground">
                  {chakra.sanskrit}
                </div>
              </div>
              <div className="flex-1">
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-1000"
                    style={{
                      width: `${chakra.activation}%`,
                      backgroundColor: chakra.color,
                      opacity: 0.7,
                    }}
                  />
                </div>
              </div>
              <span className="text-xs font-mono text-muted-foreground w-8 text-right">
                {chakra.activation}
              </span>
            </div>
          ))}
        </div>
        <div className="mt-4 pt-3 border-t border-border text-sm text-muted-foreground">
          Dominant: <span className="text-foreground">{energy.dominantEnergy}</span>
        </div>
      </Card>

      {/* Meditation & Consciousness */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Card className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-medium">Meditation Depth</h3>
          </div>
          <div className="flex items-baseline gap-2 mb-2">
            <span className="text-2xl font-semibold">
              {energy.meditationDepth}
            </span>
            <span className="text-sm text-muted-foreground">/ 10</span>
          </div>
          <Badge variant="secondary" className="text-xs">
            {energy.meditationStage}
          </Badge>
        </Card>

        <Card className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Eye className="h-4 w-4 text-secondary" />
            <h3 className="text-sm font-medium">Consciousness</h3>
          </div>
          <div className="flex items-baseline gap-2 mb-2">
            <span className="text-2xl font-semibold">
              {energy.consciousnessLevel}
            </span>
            <span className="text-sm text-muted-foreground">/ 1000</span>
          </div>
          <Badge variant="secondary" className="text-xs">
            {energy.consciousnessName}
          </Badge>
        </Card>
      </div>

      {/* Third Eye */}
      <Card className="glass-card p-5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-medium mb-1">Third Eye Activation</h3>
            <p className="text-xs text-muted-foreground">
              Gamma + high-beta activity in prefrontal cortex
            </p>
          </div>
          <span className="text-lg font-mono">
            {energy.thirdEyeActivation}%
          </span>
        </div>
        <Progress value={energy.thirdEyeActivation} className="h-2 mt-3" />
      </Card>
    </main>
  );
}
