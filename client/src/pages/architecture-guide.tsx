import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Brain,
  Activity,
  Heart,
  Moon,
  Sparkles,
  Zap,
  Headphones,
  MessageSquare,
  BarChart2,
  Network,
  HeartPulse,
  Lightbulb,
  Clock,
  Settings,
  Radio,
  Wifi,
  ArrowDown,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Eye,
  Battery,
  Gauge,
  Shield,
  Smile,
  Waves,
} from "lucide-react";

/* ---------- types ---------- */
interface SectionProps {
  id: string;
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

/* ---------- collapsible section ---------- */
function Section({ id, title, icon, children, defaultOpen = false }: SectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Card className="glass-card rounded-xl overflow-hidden hover-glow">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 p-5 text-left transition-colors hover:bg-muted/20"
      >
        <span className="text-primary">{icon}</span>
        <h3 className="text-base font-semibold flex-1">{title}</h3>
        {open ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>
      {open && <div className="px-5 pb-5 space-y-4">{children}</div>}
    </Card>
  );
}

/* ---------- small info card ---------- */
function InfoCard({
  icon,
  title,
  value,
  description,
  color = "text-primary",
}: {
  icon: React.ReactNode;
  title: string;
  value: string;
  description: string;
  color?: string;
}) {
  return (
    <div
      className="p-4 rounded-xl"
      style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}
    >
      <div className="flex items-center gap-2 mb-2">
        <span className={color}>{icon}</span>
        <span className="text-xs text-muted-foreground font-medium">{title}</span>
      </div>
      <p className={`text-sm font-semibold ${color}`}>{value}</p>
      <p className="text-xs text-muted-foreground mt-1 leading-relaxed">{description}</p>
    </div>
  );
}

/* ---------- score formula row ---------- */
function ScoreFormula({
  name,
  formula,
  meaning,
  color,
}: {
  name: string;
  formula: string;
  meaning: string;
  color: string;
}) {
  return (
    <div className="flex gap-4 items-start py-3 border-b border-border/20 last:border-0">
      <div className="w-28 shrink-0">
        <span className={`text-sm font-semibold ${color}`}>{name}</span>
      </div>
      <div className="flex-1">
        <code className="text-xs font-mono text-foreground/80 bg-muted/30 px-2 py-0.5 rounded">
          {formula}
        </code>
        <p className="text-xs text-muted-foreground mt-1">{meaning}</p>
      </div>
    </div>
  );
}

/* ========== Main Component ========== */
export default function ArchitectureGuide() {
  return (
    <main className="p-6 space-y-4 max-w-4xl">
      {/* Hero */}
      <div className="ai-insight-card">
        <div className="flex items-start gap-3">
          <div
            className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
            style={{ background: "linear-gradient(135deg, hsl(152,60%,48%,0.3), hsl(38,85%,58%,0.3))" }}
          >
            <Brain className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground">Svapnastra Architecture Guide</h2>
            <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
              Complete breakdown of how every page, component, score, and ML model works.
              Expand each section below to learn what each part does and how the data flows.
            </p>
          </div>
        </div>
      </div>

      {/* ──────────── 1. DATA FLOW ──────────── */}
      <Section id="data-flow" title="How Data Flows" icon={<Wifi className="h-5 w-5" />} defaultOpen>
        <div className="space-y-3">
          {[
            {
              step: "1",
              label: "Muse 2 Headband",
              detail: "4 EEG channels (TP9, AF7, AF8, TP10) at 256 Hz via Bluetooth",
              color: "bg-blue-500",
            },
            {
              step: "2",
              label: "BrainFlow v5.20.1",
              detail: "Reads raw EEG signals from Muse 2 (board ID 38, native Bluetooth)",
              color: "bg-cyan-500",
            },
            {
              step: "3",
              label: "FastAPI Backend (port 8000)",
              detail: "12 ML models process each chunk: sleep, emotion, dream, flow, creativity, memory, drowsiness, cognitive load, attention, stress, lucid dream, meditation",
              color: "bg-purple-500",
            },
            {
              step: "4",
              label: "WebSocket /ws/eeg-stream @ 4Hz",
              detail: "Sends JSON frames with all 12 model outputs, band powers, signal quality, emotion shifts, and coherence data",
              color: "bg-green-500",
            },
            {
              step: "5",
              label: "DeviceProvider (React Context)",
              detail: "Stores latestFrame globally — every page reads from useDevice() hook",
              color: "bg-amber-500",
            },
            {
              step: "6",
              label: "Pages render live data",
              detail: "Each page reads latestFrame.analysis.* — no fake data, all zeros when not streaming",
              color: "bg-rose-500",
            },
          ].map((item, i) => (
            <div key={i}>
              <div className="flex items-start gap-3">
                <div
                  className={`w-7 h-7 rounded-lg ${item.color} flex items-center justify-center shrink-0 text-white text-xs font-bold`}
                >
                  {item.step}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-foreground">{item.label}</p>
                  <p className="text-xs text-muted-foreground">{item.detail}</p>
                </div>
              </div>
              {i < 5 && (
                <div className="ml-3 py-1">
                  <ArrowDown className="h-3 w-3 text-muted-foreground/40" />
                </div>
              )}
            </div>
          ))}
        </div>
      </Section>

      {/* ──────────── 2. THE 12 ML MODELS ──────────── */}
      <Section id="ml-models" title="12 ML Models" icon={<Sparkles className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground mb-3">
          Every WebSocket frame runs your EEG through all 12 models. Each returns a score (0-1) and a state label.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <InfoCard
            icon={<Moon className="h-4 w-4" />}
            title="Sleep Staging"
            value="stage, confidence"
            description="Classifies: Wake, N1 (Light), N2 (Sleep), N3 (Deep), REM. Uses delta/theta/alpha ratios and spindle detection."
            color="text-blue-400"
          />
          <InfoCard
            icon={<Smile className="h-4 w-4" />}
            title="Emotion"
            value="emotion, valence, arousal"
            description="6 emotions: happy, sad, angry, fearful, relaxed, focused. Plus stress_index, focus_index, relaxation_index (all 0-1)."
            color="text-pink-400"
          />
          <InfoCard
            icon={<Moon className="h-4 w-4" />}
            title="Dream Detection"
            value="is_dreaming, probability"
            description="Detects dream states via REM-like patterns. Outputs: rem_likelihood, dream_intensity, lucidity_estimate (all 0-1)."
            color="text-purple-400"
          />
          <InfoCard
            icon={<Zap className="h-4 w-4" />}
            title="Flow State"
            value="in_flow, flow_score"
            description="Detects optimal performance state: high focus + moderate arousal + low stress. Flow score 0-1."
            color="text-green-400"
          />
          <InfoCard
            icon={<Lightbulb className="h-4 w-4" />}
            title="Creativity"
            value="state, creativity_score"
            description="Measures divergent thinking via theta-alpha ratio. High theta + moderate alpha = creative state."
            color="text-amber-400"
          />
          <InfoCard
            icon={<Brain className="h-4 w-4" />}
            title="Memory Encoding"
            value="state, encoding_score"
            description="Detects active memory consolidation. Theta bursts + gamma coupling = information being encoded."
            color="text-cyan-400"
          />
          <InfoCard
            icon={<Eye className="h-4 w-4" />}
            title="Attention"
            value="state, attention_score"
            description="Sustained attention via prefrontal beta activity. High beta + low theta = focused attention."
            color="text-emerald-400"
          />
          <InfoCard
            icon={<Battery className="h-4 w-4" />}
            title="Drowsiness"
            value="state, drowsiness_index"
            description="Detects fatigue: rising theta + dropping alpha + increasing slow eye movements."
            color="text-orange-400"
          />
          <InfoCard
            icon={<Gauge className="h-4 w-4" />}
            title="Cognitive Load"
            value="level, load_index"
            description="Mental workload from frontal theta power. High theta + high beta = cognitive overload."
            color="text-red-400"
          />
          <InfoCard
            icon={<Shield className="h-4 w-4" />}
            title="Stress"
            value="level, stress_index"
            description="Physiological stress: high beta/alpha ratio + frontal asymmetry + reduced alpha power."
            color="text-rose-400"
          />
          <InfoCard
            icon={<Heart className="h-4 w-4" />}
            title="Meditation"
            value="depth, meditation_score"
            description="Meditation depth: Surface, Light, Deep Absorption, Jhana. Strong alpha coherence + low beta."
            color="text-violet-400"
          />
          <InfoCard
            icon={<Moon className="h-4 w-4" />}
            title="Lucid Dream"
            value="state, lucidity_score"
            description="Only active during REM. Detects awareness within dreams via gamma bursts in frontal regions."
            color="text-fuchsia-400"
          />
        </div>
      </Section>

      {/* ──────────── 3. BAND POWERS ──────────── */}
      <Section id="band-powers" title="EEG Band Powers" icon={<Waves className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground mb-3">
          Raw EEG is decomposed into 5 frequency bands. All values are relative power (0-1 scale, shown as 0-100%).
        </p>
        <div className="space-y-3">
          {[
            { band: "Delta", range: "0.5-4 Hz", color: "hsl(270, 70%, 65%)", meaning: "Deep sleep, unconscious processing, healing. Dominant in N3 (deep sleep)." },
            { band: "Theta", range: "4-8 Hz", color: "hsl(195, 100%, 50%)", meaning: "Relaxation, creativity, meditation, drowsiness. High in REM sleep and deep meditation." },
            { band: "Alpha", range: "8-12 Hz", color: "hsl(120, 100%, 55%)", meaning: "Calm focus, relaxation with awareness. Eyes-closed rest. Bridge between conscious and subconscious." },
            { band: "Beta", range: "12-30 Hz", color: "hsl(45, 100%, 50%)", meaning: "Active thinking, problem-solving, anxiety. High during concentration, stress, and mental effort." },
            { band: "Gamma", range: "30-100 Hz", color: "hsl(0, 80%, 50%)", meaning: "Higher cognition, memory binding, peak awareness. Associated with 'aha' moments and transcendent states." },
          ].map((b) => (
            <div key={b.band} className="flex items-start gap-3 py-2">
              <div className="w-3 h-3 rounded-full shrink-0 mt-1" style={{ background: b.color }} />
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold">{b.band}</span>
                  <span className="text-[10px] text-muted-foreground font-mono">{b.range}</span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{b.meaning}</p>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* ──────────── 4. PAGES EXPLAINED ──────────── */}
      <Section id="dashboard" title="Dashboard (/)" icon={<Brain className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground mb-3">
          The main overview page with 3 score circles, mood timeline, brain wave sparklines, energy centers, and emotional shift alerts.
        </p>
        <h4 className="text-sm font-semibold text-foreground mb-2">Score Formulas</h4>
        <div className="rounded-lg overflow-hidden" style={{ background: "hsl(220, 22%, 7%)" }}>
          <div className="p-4 space-y-0">
            <ScoreFormula
              name="Wellness"
              formula="relaxation × 0.4 + (100 - stress) × 0.35 + focus × 0.25"
              meaning="Overall emotional wellbeing. High relaxation + low stress + decent focus = high wellness."
              color="text-green-400"
            />
            <ScoreFormula
              name="Sleep"
              formula="sleep_staging.confidence × 100"
              meaning="How confidently the ML model classifies your current sleep stage. High = clear EEG pattern."
              color="text-blue-400"
            />
            <ScoreFormula
              name="Brain"
              formula="focus × 0.4 + relaxation × 0.3 + (100 - arousal × 60) × 0.3"
              meaning="Cognitive balance. High focus + calm relaxation + moderate (not extreme) arousal."
              color="text-purple-400"
            />
          </div>
        </div>

        <h4 className="text-sm font-semibold text-foreground mt-4 mb-2">Emotional Shift Alerts</h4>
        <p className="text-xs text-muted-foreground mb-2">
          The EmotionShiftDetector watches EEG for pre-conscious changes — your brain's electrical signature shifts 2-8 seconds BEFORE you consciously notice a mood change. Like how animals sense human emotional shifts.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {[
            { type: "Approaching Anxiety", sign: "Beta rising + alpha dropping + stress increasing", color: "text-warning" },
            { type: "Approaching Calm", sign: "Alpha rising + beta dropping + calm ratio up", color: "text-success" },
            { type: "Approaching Sadness", sign: "Valence dropping + arousal dropping + theta rising", color: "text-blue-400" },
            { type: "Approaching Focus", sign: "Entropy dropping + beta structured + theta down", color: "text-emerald-400" },
            { type: "Approaching Joy", sign: "Valence rising + gamma bursts", color: "text-amber-400" },
            { type: "Turbulence", sign: "High variability in valence + arousal", color: "text-rose-400" },
          ].map((s) => (
            <div key={s.type} className="p-3 rounded-lg" style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}>
              <p className={`text-xs font-semibold ${s.color}`}>{s.type}</p>
              <p className="text-[10px] text-muted-foreground mt-0.5">{s.sign}</p>
            </div>
          ))}
        </div>

        <h4 className="text-sm font-semibold text-foreground mt-4 mb-2">Chakra Energy Bars</h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {[
            { chakra: "Root", band: "Delta", formula: "delta × 1.2", color: "text-red-400" },
            { chakra: "Sacral", band: "Theta", formula: "theta × 1.1", color: "text-orange-400" },
            { chakra: "Solar Plexus", band: "Alpha + Theta", formula: "(alpha + theta) × 0.6", color: "text-yellow-400" },
            { chakra: "Heart", band: "Alpha", formula: "alpha × 1.2", color: "text-green-400" },
            { chakra: "Throat", band: "Beta", formula: "beta × 1.0", color: "text-blue-400" },
            { chakra: "Third Eye", band: "Beta + Gamma", formula: "(beta + gamma) × 0.7", color: "text-indigo-400" },
            { chakra: "Crown", band: "Gamma", formula: "gamma × 1.5", color: "text-purple-400" },
          ].map((c) => (
            <div key={c.chakra} className="flex items-center gap-2 text-xs py-1">
              <span className={`font-medium w-24 ${c.color}`}>{c.chakra}</span>
              <span className="text-muted-foreground">{c.band}</span>
              <ArrowRight className="h-3 w-3 text-muted-foreground/40" />
              <code className="font-mono text-foreground/70 text-[10px]">{c.formula}</code>
            </div>
          ))}
        </div>
      </Section>

      <Section id="emotion-lab" title="Emotion Lab (/emotions)" icon={<Heart className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground mb-3">
          Deep dive into your emotional state with 5 panels.
        </p>
        <div className="space-y-3">
          <div className="p-3 rounded-lg" style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}>
            <p className="text-sm font-medium text-foreground">Emotion Wheel</p>
            <p className="text-xs text-muted-foreground mt-1">
              Radial SVG with 6 emotions (happy, sad, angry, fearful, relaxed, focused). Each node's distance from center = its probability from the emotion model. Dominant emotion pulses and is larger.
            </p>
          </div>
          <div className="p-3 rounded-lg" style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}>
            <p className="text-sm font-medium text-foreground">Mental State Metrics</p>
            <p className="text-xs text-muted-foreground mt-1">
              <strong>Stress</strong> (0-100): High beta relative to alpha. <strong>Focus</strong> (0-100): Structured beta + low theta.
              <strong> Relaxation</strong> (0-100): High alpha + low beta. <strong>Valence</strong> (-1 to +1): Negative = sad/angry, Positive = happy.
              <strong> Arousal</strong> (0-1): Low = calm, High = excited/anxious.
            </p>
          </div>
          <div className="p-3 rounded-lg" style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}>
            <p className="text-sm font-medium text-foreground">Valence-Arousal Scatter</p>
            <p className="text-xs text-muted-foreground mt-1">
              2D emotional space. X-axis = valence (sad↔happy), Y-axis = arousal (calm↔excited). Your emotional trajectory is plotted — dots grow larger over time. Cluster in upper-right = excited + positive. Lower-left = calm + negative.
            </p>
          </div>
        </div>
      </Section>

      <Section id="inner-energy" title="Inner Energy (/inner-energy)" icon={<Sparkles className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground mb-3">
          Spiritual/meditative interpretation of EEG data through the chakra system.
        </p>
        <div className="rounded-lg overflow-hidden" style={{ background: "hsl(220, 22%, 7%)" }}>
          <div className="p-4 space-y-0">
            <ScoreFormula
              name="Meditation"
              formula="meditation_model.meditation_score × 100"
              meaning="ML meditation model output. Depth labels: Surface, Light, Deep Absorption, Jhana."
              color="text-green-400"
            />
            <ScoreFormula
              name="Consciousness"
              formula="attention × 30 + flow × 40 + meditation × 30"
              meaning="Composite awareness level. Maps to: Survival → Desire → Willpower → Love → Expression → Insight → Unity."
              color="text-purple-400"
            />
            <ScoreFormula
              name="Third Eye"
              formula="(gamma + beta × 0.3) × 150"
              meaning="Prefrontal high-frequency activity. Associated with intuition and insight."
              color="text-indigo-400"
            />
          </div>
        </div>
      </Section>

      <Section id="brain-monitor" title="Brain Monitor (/brain-monitor)" icon={<Activity className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground mb-3">
          The technical EEG view showing all 12 ML model outputs simultaneously.
        </p>
        <div className="space-y-3">
          <div className="p-3 rounded-lg" style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}>
            <p className="text-sm font-medium text-foreground">EEG Brain Waves</p>
            <p className="text-xs text-muted-foreground mt-1">
              Real-time waveform chart showing alpha + beta waves. Includes signal quality indicator (SQI %), session recording controls, and source label (DEVICE vs SIMULATION).
            </p>
          </div>
          <div className="p-3 rounded-lg" style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}>
            <p className="text-sm font-medium text-foreground">12-Model Panel</p>
            <p className="text-xs text-muted-foreground mt-1">
              Grid of cards — one per ML model. Each shows: current state label, confidence bar (0-100%), and model-specific details. Only visible when streaming.
            </p>
          </div>
          <div className="p-3 rounded-lg" style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}>
            <p className="text-sm font-medium text-foreground">Wavelet Spectrogram</p>
            <p className="text-xs text-muted-foreground mt-1">
              Time-frequency analysis updated every 2s. Shows sleep spindles, K-complexes, and DWT energy distribution across frequency bands.
            </p>
          </div>
          <div className="p-3 rounded-lg" style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}>
            <p className="text-sm font-medium text-foreground">Electrode Grid</p>
            <p className="text-xs text-muted-foreground mt-1">
              8x8 grid showing per-channel signal quality. Green = good (80%+), Yellow = weak (60-80%), Red = error (&lt;60%).
            </p>
          </div>
        </div>
      </Section>

      <Section id="dreams" title="Dream Detection (/dreams)" icon={<Moon className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground mb-3">
          Live dream state monitoring with 4 score gauges and episode tracking.
        </p>
        <div className="rounded-lg overflow-hidden" style={{ background: "hsl(220, 22%, 7%)" }}>
          <div className="p-4 space-y-0">
            <ScoreFormula
              name="Dream Prob"
              formula="dream_detection.probability × 100"
              meaning="How likely you're in a dream state. Based on theta dominance + eye movement patterns."
              color="text-purple-400"
            />
            <ScoreFormula
              name="REM"
              formula="dream_detection.rem_likelihood × 100"
              meaning="Probability of REM sleep stage. Rapid eye movements + muscle atonia + mixed-frequency EEG."
              color="text-green-400"
            />
            <ScoreFormula
              name="Intensity"
              formula="dream_detection.dream_intensity × 100"
              meaning="How vivid/active the dream is. Higher gamma + theta bursts = more intense dreaming."
              color="text-amber-400"
            />
            <ScoreFormula
              name="Lucidity"
              formula="dream_detection.lucidity_estimate × 100"
              meaning="Awareness within the dream. Frontal gamma activity during REM = lucid dreaming potential."
              color="text-blue-400"
            />
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          <strong>Dream Episodes</strong> are auto-detected when <code>is_dreaming</code> transitions from false → true → false. Each episode records: start time, duration, intensity, lucidity, REM probability, and sleep stage.
        </p>
      </Section>

      <Section id="dream-patterns" title="Dream Patterns (/dream-patterns)" icon={<BarChart2 className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground">
          Session-level dream analytics. Shows a <strong>hypnogram</strong> (sleep architecture chart: N3=deep at bottom, Wake at top),
          <strong> dream activity</strong> (intensity bar chart), <strong>REM likelihood</strong> trend (area chart), and <strong>REM cycle progression</strong> (intensity + lucidity + duration across completed cycles).
          Natural pattern: later REM cycles are longer and more intense.
        </p>
      </Section>

      <Section id="health" title="Health Analytics (/health-analytics)" icon={<HeartPulse className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground mb-3">
          Holistic health dashboard pulling from all 12 models.
        </p>
        <div className="rounded-lg overflow-hidden" style={{ background: "hsl(220, 22%, 7%)" }}>
          <div className="p-4 space-y-0">
            <ScoreFormula
              name="Brain Health"
              formula="focus × 0.25 + relax × 0.25 + (100 - stress) × 0.25 + flow × 0.25"
              meaning="Balanced cognitive function across all dimensions."
              color="text-green-400"
            />
            <ScoreFormula
              name="Cognitive"
              formula="focus × 0.3 + creativity × 0.25 + memory × 0.25 + (100 - drowsiness) × 0.2"
              meaning="Mental performance capacity. How sharp your brain is right now."
              color="text-purple-400"
            />
            <ScoreFormula
              name="Wellbeing"
              formula="relax × 0.35 + (100 - stress) × 0.35 + flow × 0.3"
              meaning="Emotional balance and inner peace. Low stress + high relaxation + flow state."
              color="text-amber-400"
            />
          </div>
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          Also shows 8 individual metrics (Stress, Focus, Flow, Relaxation, Creativity, Memory, Cognitive Load, Drowsiness) with dynamic AI insights that trigger based on thresholds.
        </p>
      </Section>

      <Section id="insights" title="Insights (/insights)" icon={<Lightbulb className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground mb-3">
          AI-generated brain insights based on your current state. Up to 4 insight cards shown simultaneously.
        </p>
        <div className="space-y-2">
          {[
            { condition: "Focus > 60%", insight: "High Focus State — ideal for analytical tasks", color: "text-primary" },
            { condition: "Focus < 30%", insight: "Low Focus — consider a break", color: "text-warning" },
            { condition: "Creativity > 50%", insight: "Creative State — theta-alpha ratio high, good for brainstorming", color: "text-success" },
            { condition: "Stress > 50%", insight: "Elevated Stress — try breathing exercise", color: "text-warning" },
            { condition: "Relaxation > 60%", insight: "Calm & Balanced — parasympathetic state", color: "text-success" },
            { condition: "Flow > 60%", insight: "Flow State Achieved — minimize interruptions", color: "text-success" },
            { condition: "Dream Prob > 40%", insight: "Dream-Like Patterns — hypnagogic state", color: "text-secondary" },
            { condition: "Meditation > 50%", insight: "Deep Meditative State — strong alpha coherence", color: "text-primary" },
          ].map((item) => (
            <div key={item.condition} className="flex items-center gap-3 text-xs py-1">
              <code className="font-mono text-muted-foreground w-36 shrink-0">{item.condition}</code>
              <ArrowRight className="h-3 w-3 text-muted-foreground/40 shrink-0" />
              <span className={item.color}>{item.insight}</span>
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          Also shows a <strong>6-axis radar chart</strong> (Focus, Creativity, Relaxation, Memory, Flow, Meditation) and <strong>band power trends</strong> over time.
        </p>
      </Section>

      <Section id="neurofeedback" title="Neurofeedback (/neurofeedback)" icon={<Headphones className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground mb-3">
          Real-time brain training in 4 phases.
        </p>
        <div className="space-y-2">
          {[
            {
              phase: "1. Protocol Selection",
              detail: "Choose: Alpha Enhancement (increase calm focus), SMR Training (sensorimotor rhythm), Theta/Beta Ratio (ADHD protocol), Alpha Asymmetry (mood balance). Toggle audio feedback.",
            },
            {
              phase: "2. Calibration (30s)",
              detail: "Measures your baseline brain activity. Relax and breathe normally. Progress bar shows calibration completion.",
            },
            {
              phase: "3. Training",
              detail: "Live score gauge (0-100). Your band powers are sent to the backend every 1s and compared to your calibrated baseline. Score > threshold = reward (audio tone at 523Hz C5 + green flash). Track rewards count and streak.",
            },
            {
              phase: "4. Summary",
              detail: "Session stats: total rewards, reward rate %, average score, best streak.",
            },
          ].map((p) => (
            <div key={p.phase} className="p-3 rounded-lg" style={{ background: "hsl(220, 22%, 8%)", border: "1px solid hsl(220, 18%, 13%)" }}>
              <p className="text-sm font-medium text-foreground">{p.phase}</p>
              <p className="text-xs text-muted-foreground mt-1">{p.detail}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section id="connectivity" title="Brain Connectivity (/brain-connectivity)" icon={<Network className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground">
          Network analysis of brain region connections. Select channels (4/8/16) and method (coherence).
          Shows <strong>connectivity matrix</strong> heatmap, <strong>graph metrics</strong> (clustering coefficient, average path length, small-world index, modularity),
          and <strong>directed flow</strong> (Granger causality, DTF matrix, dominant information flow direction).
        </p>
      </Section>

      <Section id="ai-companion" title="AI Companion (/ai-companion)" icon={<MessageSquare className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground">
          Chat interface with an EEG-aware AI assistant. Can reference your current brain state during conversation — e.g., "I notice your stress levels are elevated, would you like to try a breathing exercise?"
        </p>
      </Section>

      <Section id="sessions" title="Sessions (/sessions)" icon={<Clock className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground">
          Browse and manage recorded EEG sessions. View session details, export as CSV for external analysis, or delete old sessions. Sessions are recorded via the Brain Monitor page's recording controls.
        </p>
      </Section>

      <Section id="settings" title="Settings (/settings)" icon={<Settings className="h-5 w-5" />}>
        <p className="text-xs text-muted-foreground">
          App configuration: hardware settings (electrode count, sampling rate), stress alert threshold, visual effects toggles,
          local vs server inference, data encryption, calibration wizard for model personalization, ML benchmark viewer, theme toggle, and data export/delete.
        </p>
      </Section>

      {/* ──────────── FOOTER ──────────── */}
      <div className="text-center py-6">
        <p className="text-xs text-muted-foreground">
          All data is 100% live from your Muse 2 headband. No fake data, no simulations.
        </p>
        <p className="text-[10px] text-muted-foreground/50 mt-1">
          Svapnastra &middot; 12 ML Models &middot; 13 Pages &middot; 4Hz Real-Time Streaming
        </p>
      </div>
    </main>
  );
}
