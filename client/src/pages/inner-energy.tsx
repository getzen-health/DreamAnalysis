import { useMemo, useState, useEffect, useRef } from "react";
import { getParticipantId } from "@/lib/participant";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScoreCircle } from "@/components/score-circle";
import { Sparkles, Activity, Radio } from "lucide-react";
import { useDevice } from "@/hooks/use-device";
import { useQuery } from "@tanstack/react-query";

const CURRENT_USER = getParticipantId();

/** Derive rough chakra activations from voice valence + arousal when no EEG. */
function voiceChakraActivations(valence: number, arousal: number, stress: number): number[] {
  // valence [-1,1] → positive = heart/crown active; arousal [0,1] → energy level
  const v = (valence + 1) / 2;     // 0→1
  const a = arousal;                // 0→1
  const s = Math.min(1, stress); // stress_from_watch is already 0-1
  return [
    Math.round((1 - a) * 60 + 15),                 // Root — grounded when low arousal
    Math.round(v * 50 + 15),                        // Sacral — creativity via positive valence
    Math.round((v + a) / 2 * 55 + 10),             // Solar Plexus
    Math.round(v * 60 + 10),                        // Heart — positive valence
    Math.round(a * 50 + 15),                        // Throat — arousal → expression
    Math.round((s > 0.5 ? s * 50 : a * 40) + 10),  // Third Eye
    Math.round(v * 40 + 5),                         // Crown — positive state
  ];
}

interface ChakraState {
  name: string;
  sanskrit: string;
  meaning: string;
  wave: string;
  activation: number;
  color: string;
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
  { key: "root",        name: "Root",         sanskrit: "Muladhara",     meaning: "Grounded & secure",       wave: "Delta waves · deep sleep rhythm" },
  { key: "sacral",      name: "Sacral",        sanskrit: "Svadhisthana",  meaning: "Creative & emotional",    wave: "Theta waves · dreamy / imaginative" },
  { key: "solar_plexus",name: "Solar Plexus",  sanskrit: "Manipura",      meaning: "Confident & focused",     wave: "Alpha + Theta · calm alertness" },
  { key: "heart",       name: "Heart",         sanskrit: "Anahata",       meaning: "Calm & compassionate",    wave: "Alpha waves · relaxed awareness" },
  { key: "throat",      name: "Throat",        sanskrit: "Vishuddha",     meaning: "Alert & expressive",      wave: "Beta waves · active thinking" },
  { key: "third_eye",   name: "Third Eye",     sanskrit: "Ajna",          meaning: "Insight & intuition",     wave: "High-beta · heightened focus" },
  { key: "crown",       name: "Crown",         sanskrit: "Sahasrara",     meaning: "Deep awareness",          wave: "Gamma waves · peak consciousness" },
];

function getGuidance(dominant: string, meditationDepth: string): string {
  const guides: Record<string, string> = {
    Root: "Focus on grounding. Feel the earth beneath you. Strong delta activity indicates deep stability.",
    Sacral: "Let creative energy flow. Your theta waves suggest heightened imagination and openness.",
    "Solar Plexus": "Your willpower center is active. Channel this steady alpha-theta blend into focused intention.",
    Heart: "Your heart center is open. Strong alpha waves indicate emotional balance and compassion.",
    Throat: "Speak your truth today. Beta activity shows your communication center is energized.",
    "Third Eye": "Inner vision is strong. High-frequency activity suggests heightened intuition and insight.",
    Crown: "Higher awareness is accessible. Gamma waves indicate transcendent awareness. Stay present.",
  };
  const base = guides[dominant] || "Your energy centers are balanced. Continue observing with gentle awareness.";
  if (meditationDepth === "Deep Absorption" || meditationDepth === "Jhana") {
    return base + " You've reached a deep meditative state — maintain this awareness.";
  }
  return base;
}

export default function InnerEnergy() {
  const { latestFrame, state: deviceState } = useDevice();
  const isStreaming = deviceState === "streaming";
  const analysis = latestFrame?.analysis;

  const { data: latestVoice } = useQuery<Record<string, unknown> | null>({
    queryKey: ["voice-inner-energy", CURRENT_USER],
    queryFn: async () => {
      const res = await fetch(`/api/ml/voice-watch/latest/${CURRENT_USER}`);
      if (!res.ok) return null;
      const data = await res.json();
      if (!data || Array.isArray(data) || typeof data !== "object") return null;
      return data as Record<string, unknown>;
    },
    staleTime: 60_000,
    retry: false,
    enabled: !isStreaming,
  });

  const hasVoice = !isStreaming && !!latestVoice;

  const bandPowers = analysis?.band_powers ?? {};
  const meditation = analysis?.meditation;
  const attention = analysis?.attention;
  const flowState = analysis?.flow_state;

  // Derive chakra activations from band powers
  const chakras: ChakraState[] = useMemo(() => {
    const d = (bandPowers.delta ?? 0) * 100;
    const t = (bandPowers.theta ?? 0) * 100;
    const a = (bandPowers.alpha ?? 0) * 100;
    const b = (bandPowers.beta ?? 0) * 100;
    const g = (bandPowers.gamma ?? 0) * 100;

    const activations = [
      Math.min(95, Math.round(d * 1.2)),        // Root — delta
      Math.min(95, Math.round(t * 1.1)),         // Sacral — theta
      Math.min(95, Math.round((a + t) * 0.6)),   // Solar Plexus — alpha/theta
      Math.min(95, Math.round(a * 1.2)),          // Heart — alpha
      Math.min(95, Math.round(b * 1.0)),          // Throat — beta
      Math.min(95, Math.round((b + g) * 0.7)),    // Third Eye — high beta + gamma
      Math.min(95, Math.round(g * 1.5)),           // Crown — gamma
    ];

    let voiceFallback: number[] | null = null;
    if (hasVoice && latestVoice) {
      const vv = (latestVoice.valence as number) ?? 0;
      const va = (latestVoice.arousal as number) ?? 0.5;
      const vs = (latestVoice.stress_from_watch as number) ?? 0.5;
      voiceFallback = voiceChakraActivations(vv, va, vs);
    }

    return CHAKRA_INFO.map((c, i) => ({
      name: c.name,
      sanskrit: c.sanskrit,
      meaning: c.meaning,
      wave: c.wave,
      activation: isStreaming ? activations[i] : (voiceFallback ? voiceFallback[i] : 0),
      color: CHAKRA_COLORS[c.key],
    }));
  }, [bandPowers.delta, bandPowers.theta, bandPowers.alpha, bandPowers.beta, bandPowers.gamma, isStreaming, hasVoice, latestVoice]);

  // Meditation depth from meditation model
  const meditationScore = meditation?.meditation_score ?? 0;
  const voiceArousal = (latestVoice?.arousal as number) ?? 0.5;
  const voiceValence = (latestVoice?.valence as number) ?? 0;
  const voiceFocusIndex = (latestVoice?.focus_index as number) ?? 0.4;
  const meditationPercent = isStreaming
    ? Math.round(meditationScore * 100)
    : hasVoice
      ? Math.round(Math.max(10, (1 - voiceArousal) * 60 + ((voiceValence + 1) / 2) * 20))
      : 0;
  const meditationStage = meditation?.depth ?? "—";

  // Consciousness level derived from attention + flow + meditation
  const consciousnessRaw = isStreaming
    ? (attention?.attention_score ?? 0) * 30 + (flowState?.flow_score ?? 0) * 40 + meditationScore * 30
    : hasVoice
      ? ((voiceValence + 1) / 2) * 40 + voiceArousal * 30
      : 0;
  const consciousnessPercent = Math.round(Math.min(100, consciousnessRaw));
  const consciousnessLevels = ["Survival", "Desire", "Willpower", "Love", "Expression", "Insight", "Unity"];
  const consciousnessName = consciousnessLevels[Math.min(6, Math.floor(consciousnessPercent / 15))] || "—";

  // Third eye activation from gamma + high beta (or voice focus when no EEG)
  const thirdEyeActivation = isStreaming
    ? Math.round(Math.min(100, ((bandPowers.gamma ?? 0) + (bandPowers.beta ?? 0) * 0.3) * 150))
    : hasVoice
      ? Math.round(Math.max(0, voiceFocusIndex * 80))
      : 0;

  // Dominant energy center
  const dominantChakra = chakras.reduce((max, c) =>
    c.activation > max.activation ? c : max, chakras[0]);

  // Throttle guidance text to 10s so user can read it
  const [guidance, setGuidance] = useState("Connect your Muse 2 to begin reading your energy centers from live EEG data.");
  const guidanceTimerRef = useRef(0);
  const GUIDANCE_THROTTLE = 10_000;

  useEffect(() => {
    if (!isStreaming) {
      if (hasVoice && latestVoice) {
        const emotion = (latestVoice.emotion as string) ?? "neutral";
        const valence = (latestVoice.valence as number) ?? 0;
        const voiceGuide = valence > 0.2
          ? `Voice check-in shows a ${emotion} state with positive energy. ${getGuidance(dominantChakra.name, "—")} Connect EEG for deeper readings.`
          : valence < -0.1
          ? `Voice shows a ${emotion} state. Your heart center may benefit from a breathing pause. ${getGuidance("Heart", "—")}`
          : `Voice baseline captured (${emotion}). ${getGuidance(dominantChakra.name, "—")} Connect Muse 2 for live EEG energy readings.`;
        setGuidance(voiceGuide);
      } else {
        setGuidance("Connect your Muse 2 to begin reading your energy centers from live EEG data.");
      }
      return;
    }
    const now = Date.now();
    if (now - guidanceTimerRef.current < GUIDANCE_THROTTLE && guidance !== "Connect your Muse 2 to begin reading your energy centers from live EEG data.") return;
    guidanceTimerRef.current = now;
    setGuidance(getGuidance(dominantChakra.name, meditationStage));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestFrame?.timestamp, hasVoice]);

  return (
    <main className="p-6 space-y-6 max-w-5xl">
      {/* Page header */}
      <div>
        <h1 className="text-xl font-semibold mb-1">Inner Energy</h1>
        <p className="text-sm text-muted-foreground leading-relaxed max-w-2xl">
          Your brainwaves are mapped to traditional energy centres (chakras). Different brainwave frequencies correspond to different mental states —
          slow waves (delta/theta) reflect deep rest and creativity; fast waves (beta/gamma) reflect active thinking and alertness.
          This is a mindfulness-inspired view of your live EEG data, not a medical reading.
        </p>
      </div>

      {/* Connection Banner */}
      {!isStreaming && (
        <div className="p-4 rounded-xl border border-warning/30 bg-warning/5 text-sm text-warning flex items-center gap-3">
          <Radio className="h-4 w-4 shrink-0" />
          {hasVoice
            ? "Showing voice-derived energy estimate. Connect Muse 2 for live EEG chakra readings."
            : "Connect your Muse 2 from the sidebar to see live energy data."}
        </div>
      )}

      {/* Guidance */}
      <div className="ai-insight-card">
        <div className="flex items-start gap-3">
          <Sparkles className="h-5 w-5 text-primary mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-foreground mb-1">Energy Guidance</p>
            <p className="text-sm text-muted-foreground leading-relaxed">{guidance}</p>
          </div>
        </div>
      </div>

      {/* Score Gauges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="score-card p-5 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={meditationPercent}
            label="Meditation"
            gradientId="grad-meditation"
            colorFrom="hsl(152, 60%, 48%)"
            colorTo="hsl(200, 70%, 55%)"
            size="md"
          />
          <Badge variant="secondary" className="text-xs mt-2">
            {isStreaming ? meditationStage : hasVoice ? "Voice-derived" : "—"}
          </Badge>
          <p className="text-[10px] text-muted-foreground mt-1 text-center">How calm & inward your mind is right now</p>
        </div>

        <div className="score-card p-5 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={consciousnessPercent}
            label="Awareness"
            gradientId="grad-consciousness"
            colorFrom="hsl(262, 45%, 65%)"
            colorTo="hsl(320, 55%, 60%)"
            size="md"
          />
          <Badge variant="secondary" className="text-xs mt-2">
            {(isStreaming || hasVoice) ? consciousnessName : "—"}
          </Badge>
          <p className="text-[10px] text-muted-foreground mt-1 text-center">How present & aware you feel (attention + flow + calm)</p>
        </div>

        <div className="score-card p-5 flex flex-col items-center hover-glow">
          <ScoreCircle
            value={thirdEyeActivation}
            label="Clarity"
            gradientId="grad-thirdeye"
            colorFrom="hsl(240, 55%, 60%)"
            colorTo="hsl(280, 50%, 60%)"
            size="md"
          />
          <p className="text-[10px] text-muted-foreground mt-2 text-center">
            Intuition & insight — from fast brainwaves in the forehead region
          </p>
        </div>
      </div>

      {/* Chakra Activations */}
      <Card className="glass-card p-6 hover-glow">
        <h3 className="text-sm font-medium mb-5 flex items-center gap-2">
          <Activity className="h-4 w-4 text-accent" />
          Energy Centers
          {isStreaming && (
            <span className="ml-auto text-[10px] font-mono text-primary animate-pulse">LIVE</span>
          )}
        </h3>
        <div className="space-y-4">
          {[...chakras].reverse().map((chakra) => (
            <div key={chakra.name} className="flex items-start gap-3">
              <div
                className="w-3 h-3 rounded-full shrink-0 mt-1.5"
                style={{
                  backgroundColor: chakra.color,
                  boxShadow: `0 0 6px ${chakra.color}66`,
                }}
              />
              <div className="w-28 shrink-0">
                <div className="text-sm font-medium">{chakra.name}</div>
                <div className="text-[10px] text-muted-foreground/60">{chakra.sanskrit}</div>
                <div className="text-[10px] text-muted-foreground mt-0.5">{chakra.meaning}</div>
              </div>
              <div className="flex-1 pt-1">
                <div className="h-2.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-1000"
                    style={{
                      width: `${chakra.activation}%`,
                      background: `linear-gradient(90deg, ${chakra.color}88, ${chakra.color})`,
                    }}
                  />
                </div>
                <div className="text-[9px] text-muted-foreground/50 mt-0.5">{chakra.wave}</div>
              </div>
              <span className="text-xs font-mono text-muted-foreground w-8 text-right pt-1">
                {chakra.activation}
              </span>
            </div>
          ))}
        </div>
        {(isStreaming || hasVoice) && (
          <div className="mt-4 pt-3 border-t border-border/30 text-sm text-muted-foreground">
            Most active: <span className="text-foreground font-medium">{dominantChakra.name}</span>
            <span className="text-muted-foreground/60 text-xs ml-2">— {dominantChakra.meaning}</span>
            {!isStreaming && <span className="text-muted-foreground/40 text-xs ml-2">(voice estimate)</span>}
          </div>
        )}
      </Card>
    </main>
  );
}
