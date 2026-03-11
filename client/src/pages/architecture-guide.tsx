import { useState, type ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import {
  Brain,
  ChevronDown,
  ChevronUp,
  Compass,
  FlaskConical,
  GitBranch,
  HeartPulse,
  Lightbulb,
  Rocket,
  Sparkles,
  Target,
  Waves,
} from "lucide-react";

type SectionProps = {
  title: string;
  icon: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
};

function Section({ title, icon, defaultOpen = false, children }: SectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <Card className="overflow-hidden border-border/50 bg-background/80 backdrop-blur">
      <button
        className="flex w-full items-center gap-3 px-5 py-4 text-left transition-colors hover:bg-muted/20"
        onClick={() => setOpen((value) => !value)}
      >
        <span className="text-primary">{icon}</span>
        <div className="flex-1">
          <h2 className="text-base font-semibold">{title}</h2>
        </div>
        {open ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>
      {open && <div className="border-t border-border/40 px-5 py-5">{children}</div>}
    </Card>
  );
}

function StatCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="rounded-2xl border border-border/50 bg-muted/20 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-foreground">{value}</p>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{detail}</p>
    </div>
  );
}

const platformLayers = [
  {
    title: "Product layer",
    body:
      "A React application for live EEG dashboards, dream journaling, food logging, neurofeedback, AI companion flows, and a structured research journey.",
  },
  {
    title: "Inference layer",
    body:
      "A Python ML backend processes Muse 2 EEG streams, extracts features, runs model chains, and returns live state estimates over REST and WebSocket.",
  },
  {
    title: "Research layer",
    body:
      "Protocol documents, study routes, data export, benchmarks, and issue-driven experimentation turn the app into a research operations surface, not just a consumer dashboard.",
  },
];

const modelEvolution = [
  {
    phase: "Phase 1",
    title: "Heuristic-first EEG interpretation",
    body:
      "The foundation was interpretable EEG biomarker logic: FAA, band power ratios, sleep signatures, stress asymmetry, and rule-based dream or meditation indicators.",
  },
  {
    phase: "Phase 2",
    title: "Cross-dataset supervised training",
    body:
      "The project then consolidated public EEG datasets into a unified emotion training path, moving from isolated experiments toward cross-subject generalization.",
  },
  {
    phase: "Phase 3",
    title: "Personalization and stability",
    body:
      "Baseline calibration, running normalization, EMA smoothing, epoch gating, and online adaptation were added to make live outputs less noisy and more usable per person.",
  },
  {
    phase: "Phase 4",
    title: "Multimodal and edge-ready expansion",
    body:
      "The codebase expanded into voice, health, ONNX browser inference, BLE streaming, and mobile execution so the system could operate outside a lab-like desktop setup.",
  },
  {
    phase: "Phase 5",
    title: "Research-platform acceleration",
    body:
      "Recent issue-driven work pushed into foundation models, domain adaptation, neuroadaptive learning, seizure or tinnitus workflows, digital phenotyping, and broader BCI experiments.",
  },
];

const issueHighlights = [
  "Core reliability: auth for ML/WebSocket, WebSocket state fixes, security headers, env validation, CI, docker-compose, and frontend tests.",
  "Signal quality and live UX: artifact warnings, 4-second epoch gating, EMA smoothing, baseline calibration, connection warmup handling, and simulation fallback.",
  "Research-grade model growth: REVE, FEMBA, CSCL, domain adaptation, few-shot personalization, TSception fallback, dual-epoch processing, and Dream database integration.",
  "Multimodal scope: EEG plus voice, Apple Health, PPG, health summaries, mental health screening, and AI-generated explanations.",
  "Intervention and application scope: neurofeedback, emotion regulation loops, learning optimization, neurogaming, music therapy, tinnitus, seizure risk, and digital phenotyping.",
];

const useCases = [
  "Personal self-tracking: seeing how stress, focus, sleep, food, and dreams interact across a day.",
  "Research participation: collecting longitudinal day-night data for the continuity-hypothesis and food-emotion-dream program.",
  "Adaptive interventions: deciding when to breathe, rest, focus, change music, or enter neurofeedback based on brain state.",
  "Education and productivity: attention, fatigue, engagement, and learning-stage experiments for tutoring or study optimization.",
  "Clinical-adjacent exploration: early prototypes around seizure detection, tinnitus support, pain biomarkers, brain health, and phenotyping.",
];

const projectionSteps = [
  "Tell one clear story first: wearable EEG that connects daytime emotion, eating behavior, and nighttime dreams in natural settings.",
  "Separate what is production-ready from what is research-stage so credibility does not get diluted by the breadth of experiments.",
  "Publish benchmark snapshots and study protocol summaries in plain language, then use the app as the live demonstration surface.",
  "Lead with differentiated assets: Muse 2 support, longitudinal design, dream continuity framing, and multimodal personalization.",
  "Package distinct audiences: participant study page, researcher methods page, and partner/demo page instead of one blended pitch.",
];

const nextSteps = [
  "Validate the food-emotion module with human-subject data instead of leaving it as theory-backed but unvalidated heuristics.",
  "Narrow the flagship paper and demo to a small number of claims that can be defended with current evidence.",
  "Promote the dormant architecture page into a maintained project dossier and keep it synced with benchmarks, protocol revisions, and issue closures.",
  "Turn the open GitHub queue into themed milestones such as cognition, clinical screening, multimodal fusion, and assistive BCI.",
  "Add source-level references or deep links from this page to the benchmark dashboard, protocol docs, and study routes for easier external sharing.",
];

const liveModelStack = [
  {
    name: "Emotion classifier",
    detail:
      "Primary live path: unified LightGBM trained with global PCA (85 -> 80 features) across 11 datasets. Repository status lists 71.52% cross-subject CV on 187,751 samples. Runtime fallback chain also includes TSception CNN and feature heuristics when full model conditions are not met.",
  },
  {
    name: "Sleep staging",
    detail:
      "Classical supervised model path for Wake, N1, N2, N3, and REM. Status file reports 92.98% benchmark accuracy for the active path.",
  },
  {
    name: "Dream detector",
    detail:
      "Gradient-boosting style detector layered on top of sleep-state reasoning to estimate dream-relevant REM-like periods. Status file reports 97.20% benchmark accuracy.",
  },
  {
    name: "Cognitive state models",
    detail:
      "Drowsiness, cognitive load, attention, stress, lucid dream, and meditation are trained and/or heuristic-assisted models. Status reports include drowsiness 81.72% CV, cognitive load 65.72% CV, attention 63.87% CV, stress 59.64% CV, lucid dream 61.85% CV, and meditation 61.13% CV.",
  },
  {
    name: "Signal hygiene models",
    detail:
      "Artifact classifier uses LightGBM with 96.47% CV on synthetic artifact classes, while the denoising autoencoder reports +2.29 dB SNR improvement. These are used to protect downstream inference quality.",
  },
  {
    name: "Adaptive control models",
    detail:
      "A PPO reinforcement-learning threshold agent adjusts neurofeedback difficulty in real time. The status file reports a 67% live reward rate for the target flow zone.",
  },
];

const technicalPipeline = [
  "Acquisition: Muse 2 dry-electrode EEG, typically 4 channels, streamed through BrainFlow or native BLE decoding paths.",
  "Preprocessing: artifact handling, denoising, re-referencing, signal-quality scoring, filtering, and feature extraction from band powers, asymmetry metrics, Hjorth parameters, entropy-style features, and temporal summaries.",
  "Epoch logic: a 4-second sliding epoch buffer with overlap is used for live inference. The frontend also uses epoch gating so unstable early frames are not presented as confident results.",
  "Normalization: running z-score normalization and baseline calibration adapt the live stream to within-session drift and per-user resting-state differences.",
  "Inference orchestration: the Python backend runs parallel inference for major models and exposes outputs over REST and WebSocket to the React client.",
  "Client rendering: the React app consumes device context, model outputs, health sync, offline storage, and intervention triggers to update dashboards and study pages.",
];

const datasets = [
  "DEAP, DREAMER, GAMEEMO, DENS, FACED, SEED-IV, EEG-ER, STEW, Muse-Subconscious, EmoKey, and EAV are referenced in the current emotion training path.",
  "Mental Attention and STEW are explicitly used in the newer cognitive training scripts for drowsiness and cognitive-load style models.",
  "DREAM Database integration is called out for dream detection and sleep staging expansion.",
  "Several newer models are still trained partly on synthetic or proxy data, which matters when interpreting benchmark strength versus real-world validity.",
];

const researchOnlyModels = [
  "Foundation-model direction: REVE, FEMBA, CSCL, and other cross-subject transfer approaches appear in closed issues as major architecture experiments rather than mature end-user features.",
  "Personalization direction: FACE few-shot adapters, prototypical few-shot learning, online learning, and domain adaptation all indicate a shift toward subject-specific calibration rather than one-model-fits-all inference.",
  "Multimodal direction: TMNet EEG+voice fusion, Apple Health HRV integration, PPG stress augmentation, SenseVoice, DistilHuBERT, and voice mental-health biomarkers extend the platform beyond EEG-only reasoning.",
  "Clinical-adjacent direction: seizure detection, pre-ictal prediction, tinnitus protocols, pain biomarkers, cognitive reserve, developmental brain maturation, and digital phenotyping are present in the issue stream and API surface, but they should be communicated as research-stage work, not validated clinical products.",
];

const modelComparisonRows = [
  {
    model: "Emotion classifier",
    algorithm: "LightGBM + PCA, TSception fallback, heuristics fallback",
    data: "11 EEG datasets, cross-subject",
    metric: "71.52% CV",
    status: "Live core path",
  },
  {
    model: "Sleep staging",
    algorithm: "RF / LightGBM-style supervised classifier",
    data: "ISRUC and sleep datasets",
    metric: "92.98%",
    status: "Live core path",
  },
  {
    model: "Dream detector",
    algorithm: "Gradient boosting + sleep-state logic",
    data: "Dream/sleep benchmarks",
    metric: "97.20%",
    status: "Live core path",
  },
  {
    model: "Drowsiness",
    algorithm: "LightGBM",
    data: "Mental Attention + synthetic",
    metric: "81.72% CV",
    status: "Live cognitive path",
  },
  {
    model: "Cognitive load",
    algorithm: "LightGBM",
    data: "STEW + synthetic",
    metric: "65.72% CV",
    status: "Live cognitive path",
  },
  {
    model: "Attention",
    algorithm: "LightGBM",
    data: "Synthetic + DEAP proxy",
    metric: "63.87% CV",
    status: "Live cognitive path",
  },
  {
    model: "Stress",
    algorithm: "LightGBM",
    data: "Synthetic + DEAP proxy",
    metric: "59.64% CV",
    status: "Live cognitive path",
  },
  {
    model: "Lucid dream",
    algorithm: "LightGBM",
    data: "Synthetic, noise-augmented",
    metric: "61.85% CV",
    status: "Live/research hybrid",
  },
  {
    model: "Meditation",
    algorithm: "LightGBM",
    data: "Synthetic, noise-augmented",
    metric: "61.13% CV",
    status: "Live/research hybrid",
  },
  {
    model: "Artifact classifier",
    algorithm: "LightGBM",
    data: "Synthetic artifact classes",
    metric: "96.47% CV",
    status: "Quality-control path",
  },
  {
    model: "Denoising autoencoder",
    algorithm: "PyTorch autoencoder",
    data: "Synthetic paired noise-clean data",
    metric: "+2.29 dB SNR",
    status: "Quality-control path",
  },
  {
    model: "RL threshold agent",
    algorithm: "PPO actor-critic",
    data: "Synthetic neurofeedback environment",
    metric: "67% reward rate",
    status: "Adaptive intervention path",
  },
];

const milestoneTimeline = [
  {
    phase: "Foundation and hardening",
    issues: "#1, #4, #5, #6, #7, #10, #11",
    detail:
      "The early repo work focused on auth, WebSocket correctness, security headers, env validation, local stack setup, CI, and mobile/PWA readiness. This is the stage where the platform became operational rather than purely aspirational.",
  },
  {
    phase: "Live UX stabilization",
    issues: "#14, #15, #16, #17, #18",
    detail:
      "Baseline calibration, EMA smoothing, epoch_ready gating, artifact warnings, and missing tests improved live interpretability and reduced presentation of noisy states.",
  },
  {
    phase: "Multimodal and foundation-model expansion",
    issues: "#19, #21, #22, #23, #24, #26, #28, #29, #30, #31, #33, #34, #36, #40",
    detail:
      "This phase added REVE, FEMBA, EEG+voice fusion, Apple Health fusion, fast voice emotion models, health summaries, self-supervised adaptation, artifact-rejection upgrades, dual-epoch logic, GRU sleep staging, and cross-subject contrastive learning.",
  },
  {
    phase: "Research daemon acceleration",
    issues: "#41 through #58, plus #68 through #71",
    detail:
      "The repository moved into rapid issue-driven expansion across mental-health voice screening, neurofeedback, federated learning, SHAP explainability, PPG fusion, DreamNet analysis, binaural feedback, microstates, entropy, graph models, and emotion trajectories.",
  },
  {
    phase: "Clinical-adjacent and adaptive BCI wave",
    issues: "#91 through #118",
    detail:
      "This wave introduced microsleep, pain biomarkers, EEG authentication, study optimization, digital phenotyping, empathy, N400 detection, neural efficiency, tinnitus, mindfulness quality, learning-stage classification, adaptive tutoring, neurogaming, brain maturation, seizure detection, domain adaptation, few-shot personalization, and CSCL.",
  },
  {
    phase: "Current open frontier",
    issues: "#159 through #168",
    detail:
      "The open queue now points toward language processing, gaze prediction, altered-consciousness tracking, neurodegeneration markers, neurostimulation guidance, motor-intention decoding, imagined speech, and Parkinsonian screening.",
  },
];

export default function ArchitectureGuide() {
  return (
    <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
      <div className="rounded-3xl border border-emerald-500/20 bg-[radial-gradient(circle_at_top_left,_rgba(16,185,129,0.18),_transparent_40%),linear-gradient(135deg,rgba(10,14,22,0.98),rgba(14,18,30,0.94))] p-6 shadow-2xl">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-emerald-300">
            Project Dossier
          </Badge>
          <Badge variant="outline" className="border-sky-500/30 bg-sky-500/10 text-sky-300">
            Repo + GitHub reviewed
          </Badge>
          <Badge variant="outline" className="border-amber-500/30 bg-amber-500/10 text-amber-300">
            Snapshot: March 9, 2026
          </Badge>
        </div>

        <div className="mt-5 grid gap-6 lg:grid-cols-[1.45fr_0.95fr]">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
              AntarAI is a wearable EEG research and product platform.
            </h1>
            <p className="mt-4 max-w-3xl text-sm leading-7 text-muted-foreground sm:text-base">
              At its core, this project reads brain activity from a Muse 2 headset and turns it into
              live interpretations of emotion, focus, stress, sleep, dream-relevant states, and
              intervention opportunities. Around that core, the repository has grown into a larger
              platform for longitudinal dream research, multimodal wellness experiments, and emerging
              adaptive BCI workflows.
            </p>
            <p className="mt-4 max-w-3xl text-sm leading-7 text-muted-foreground sm:text-base">
              Verified facts on this page come from the repository, study documents, status files,
              and the GitHub issue tracker. Interpretive statements are based on those sources and are
              presented as synthesis rather than direct claims from a single file.
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1">
            <StatCard
              label="What it is"
              value="EEG app + research stack"
              detail="It is both an end-user application and an experimentation environment for new brain-state models."
            />
            <StatCard
              label="Flagship idea"
              value="Emotion -> eating -> dreams"
              detail="The strongest original thread in the codebase is the day-night research program linking waking emotion, food behavior, and dream content."
            />
          </div>
        </div>
      </div>

      <div className="mt-6 grid gap-4 md:grid-cols-3">
        {platformLayers.map((item) => (
          <Card key={item.title} className="border-border/50 bg-background/70">
            <CardContent className="pt-5">
              <h3 className="text-sm font-semibold text-foreground">{item.title}</h3>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">{item.body}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="mt-6 space-y-4">
        <Section
          title="What Exactly This Project Is About"
          icon={<Brain className="h-5 w-5" />}
          defaultOpen
        >
          <div className="space-y-4 text-sm leading-7 text-muted-foreground">
            <p>
              AntarAI is not only an EEG dashboard. It is a
              full pipeline that starts with wearable brain sensing, processes those signals into
              interpretable features, runs multiple model families, and then surfaces the results in
              pages for daily use, longitudinal tracking, guided studies, and adaptive interventions.
            </p>
            <p>
              The repository’s own documents show a progression from a core real-time EEG application
              into a broader research system. The README describes the technical architecture. The
              status file shows the product surface area and the widening model catalog. The IRB draft
              clarifies the central scientific question: whether naturalistic daytime EEG emotion
              markers predict dream emotion that night, and whether eating behavior changes that link.
            </p>
            <p>
              Inference: the most defensible identity for the project today is a consumer-device EEG
              platform built to study continuity between waking emotion, health behavior, and dream
              life while also incubating new BCI applications around that data stream.
            </p>
          </div>
        </Section>

        <Section title="What Research Is Going On" icon={<FlaskConical className="h-5 w-5" />} defaultOpen>
          <div className="space-y-4 text-sm leading-7 text-muted-foreground">
            <p>
              The primary research program is the 30-day longitudinal study described in
              `irb/COMBINED_STUDY_PROTOCOL_DRAFT.md`. Its central aims are to test the continuity
              hypothesis of dreaming with objective daytime EEG markers, examine eating behavior as a
              moderator of the emotion-to-dream pathway, estimate nightmare risk prospectively, and
              measure how stable consumer-EEG emotion biomarkers are within the same person across time.
            </p>
            <p>
              Surrounding that flagship study is a much broader issue-driven research engine. Recent
              closed issues added or expanded work in digital phenotyping, emotion regulation,
              neurofeedback, sleep architecture, voice fusion, health fusion, engagement detection,
              neuroadaptive tutoring, deception detection, tinnitus protocols, seizure detection, and
              adaptive learning.
            </p>
            <p>
              That means the project now spans three research bands: core dream and emotion science,
              multimodal brain-health sensing, and experimental BCI applications. The first band is the
              clearest narrative. The second and third bands show ambition, but they also increase the
              need to communicate maturity levels carefully.
            </p>
          </div>
        </Section>

        <Section title="How The Models Have Changed" icon={<Waves className="h-5 w-5" />}>
          <div className="space-y-4">
            {modelEvolution.map((item) => (
              <div key={item.phase} className="rounded-2xl border border-border/40 bg-muted/15 p-4">
                <div className="flex items-center gap-3">
                  <Badge variant="outline" className="border-primary/30 bg-primary/10 text-primary">
                    {item.phase}
                  </Badge>
                  <h3 className="text-sm font-semibold text-foreground">{item.title}</h3>
                </div>
                <p className="mt-3 text-sm leading-7 text-muted-foreground">{item.body}</p>
              </div>
            ))}
            <p className="text-sm leading-7 text-muted-foreground">
              Concretely, the training layer moved from legacy multi-algorithm scripts toward a unified
              LightGBM training path, newer cognitive model batch training, ONNX export, and web-side
              inference. The live system also gained personalization machinery such as baseline
              calibration, per-user adaptation, running normalization, and TSception fallback.
            </p>
          </div>
        </Section>

        <Section title="Technical Inference Pipeline" icon={<Brain className="h-5 w-5" />}>
          <div className="space-y-3">
            {technicalPipeline.map((item) => (
              <div key={item} className="rounded-xl border border-border/40 bg-muted/15 px-4 py-3 text-sm leading-7 text-muted-foreground">
                {item}
              </div>
            ))}
          </div>
        </Section>

        <Section title="Exact Models In The Active Stack" icon={<Sparkles className="h-5 w-5" />}>
          <div className="space-y-4">
            {liveModelStack.map((item) => (
              <div key={item.name} className="rounded-2xl border border-border/40 bg-muted/15 p-4">
                <h3 className="text-sm font-semibold text-foreground">{item.name}</h3>
                <p className="mt-2 text-sm leading-7 text-muted-foreground">{item.detail}</p>
              </div>
            ))}
            <p className="text-sm leading-7 text-muted-foreground">
              Important distinction: some of these models are directly in the live user path today,
              while others are exposed through specific API routes, benchmark dashboards, or research
              endpoints rather than every default UI page. The repository mixes deployed inference and
              active experimentation in the same codebase.
            </p>
          </div>
        </Section>

        <Section title="Model Comparison Table" icon={<Target className="h-5 w-5" />}>
          <div className="overflow-x-auto rounded-2xl border border-border/40">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-muted/30 text-muted-foreground">
                <tr>
                  <th className="px-4 py-3 font-medium">Model</th>
                  <th className="px-4 py-3 font-medium">Algorithm / family</th>
                  <th className="px-4 py-3 font-medium">Primary data basis</th>
                  <th className="px-4 py-3 font-medium">Reported metric</th>
                  <th className="px-4 py-3 font-medium">Role</th>
                </tr>
              </thead>
              <tbody>
                {modelComparisonRows.map((row, index) => (
                  <tr
                    key={row.model}
                    className={index % 2 === 0 ? "bg-background/60" : "bg-muted/10"}
                  >
                    <td className="px-4 py-3 align-top text-foreground">{row.model}</td>
                    <td className="px-4 py-3 align-top text-muted-foreground">{row.algorithm}</td>
                    <td className="px-4 py-3 align-top text-muted-foreground">{row.data}</td>
                    <td className="px-4 py-3 align-top text-muted-foreground">{row.metric}</td>
                    <td className="px-4 py-3 align-top text-muted-foreground">{row.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-sm leading-7 text-muted-foreground">
            This table is intentionally selective. It focuses on the models most important for
            understanding the currently documented live stack and the immediate product-research
            interface, rather than every experimental route in the repository.
          </p>
        </Section>

        <Section title="Training Data And Benchmark Context" icon={<FlaskConical className="h-5 w-5" />}>
          <div className="space-y-3">
            {datasets.map((item) => (
              <div key={item} className="rounded-xl border border-border/40 bg-muted/15 px-4 py-3 text-sm leading-7 text-muted-foreground">
                {item}
              </div>
            ))}
            <p className="text-sm leading-7 text-muted-foreground">
              The technical implication is that the repository contains a heterogeneous model stack:
              some models are trained on large cross-subject datasets, some on narrow real datasets,
              and some on synthetic or theory-derived labels. That is one of the most important things
              to understand before presenting the project externally.
            </p>
          </div>
        </Section>

        <Section title="What GitHub Issues Say About New Advancements" icon={<GitBranch className="h-5 w-5" />}>
          <div className="space-y-4 text-sm leading-7 text-muted-foreground">
            <p>
              GitHub snapshot reviewed on March 9, 2026: 119 total issues, 109 closed and 10 open.
              Most of the tracker is research-led rather than bug-led, which is unusual for a normal
              product repository and confirms that the repo is being used as a live research backlog.
            </p>
            <p>
              Of those issues, 103 are labeled as research, 18 involve health-data integration, 9
              involve voice emotion, 8 are multimodal, and 6 explicitly target foundation-model style
              integrations. Only 4 issues are labeled as bugs. That distribution shows the center of
              gravity has shifted from app stabilization to research acceleration.
            </p>
            <div className="space-y-2">
              {issueHighlights.map((item) => (
                <div key={item} className="rounded-xl border border-border/40 bg-muted/15 px-4 py-3">
                  {item}
                </div>
              ))}
            </div>
            <p>
              The current open queue points toward the next frontier: language processing from EEG,
              visual attention and gaze prediction, altered-consciousness analysis, dementia-screening
              style biomarkers, neurostimulation guidance, assistive motor-intention decoding,
              imagined-speech control, and Parkinsonian screening. Inference: the repository is moving
              from a wellness-and-dream platform toward a broader consumer-EEG research lab.
            </p>
          </div>
        </Section>

        <Section title="Research Models Versus Product Models" icon={<GitBranch className="h-5 w-5" />}>
          <div className="space-y-3">
            {researchOnlyModels.map((item) => (
              <div key={item} className="rounded-xl border border-border/40 bg-muted/15 px-4 py-3 text-sm leading-7 text-muted-foreground">
                {item}
              </div>
            ))}
            <p className="text-sm leading-7 text-muted-foreground">
              Inference: the repo should be described as a layered model ecosystem. One layer supports
              current live product experiences. Another layer exists as research APIs, experimental
              routes, benchmark artifacts, and issue-closed implementations that still need stricter
              validation or tighter productization.
            </p>
          </div>
        </Section>

        <Section title="Issue Timeline To Research Milestones" icon={<Compass className="h-5 w-5" />}>
          <div className="space-y-4">
            {milestoneTimeline.map((item) => (
              <div key={item.phase} className="rounded-2xl border border-border/40 bg-muted/15 p-4">
                <div className="flex flex-wrap items-center gap-3">
                  <h3 className="text-sm font-semibold text-foreground">{item.phase}</h3>
                  <Badge variant="outline" className="border-sky-500/30 bg-sky-500/10 text-sky-300">
                    {item.issues}
                  </Badge>
                </div>
                <p className="mt-3 text-sm leading-7 text-muted-foreground">{item.detail}</p>
              </div>
            ))}
            <p className="text-sm leading-7 text-muted-foreground">
              Interpreting the timeline at a high level: the repo first became reliable, then became
              usable, then became adaptive and multimodal, and is now being pushed toward a much wider
              BCI and brain-health research surface.
            </p>
          </div>
        </Section>

        <Section title="Use Cases We Have Right Now" icon={<Target className="h-5 w-5" />}>
          <div className="grid gap-3 md:grid-cols-2">
            {useCases.map((item) => (
              <div key={item} className="rounded-2xl border border-border/40 bg-muted/15 p-4 text-sm leading-7 text-muted-foreground">
                {item}
              </div>
            ))}
          </div>
        </Section>

        <Section title="How To Project This To The World" icon={<Rocket className="h-5 w-5" />}>
          <div className="space-y-3 text-sm leading-7 text-muted-foreground">
            {projectionSteps.map((item) => (
              <div key={item} className="rounded-xl border border-border/40 bg-muted/15 px-4 py-3">
                {item}
              </div>
            ))}
            <p>
              The strongest external framing is not “we have many models.” The stronger message is:
              this platform makes consumer EEG useful for understanding the full emotional day, from
              live state to behavior to sleep and dreams, while also serving as a launchpad for future
              adaptive brain-health applications.
            </p>
          </div>
        </Section>

        <Section title="What The Next Steps Should Be" icon={<Compass className="h-5 w-5" />}>
          <div className="space-y-3 text-sm leading-7 text-muted-foreground">
            {nextSteps.map((item) => (
              <div key={item} className="rounded-xl border border-border/40 bg-muted/15 px-4 py-3">
                {item}
              </div>
            ))}
          </div>
        </Section>

        <Section title="Plain-English Summary" icon={<Lightbulb className="h-5 w-5" />} defaultOpen>
          <div className="space-y-4 text-sm leading-7 text-muted-foreground">
            <p>
              This project started as a brain-computer interface app for reading live Muse 2 EEG and
              showing meaningful mental-state estimates. It has since become a much larger system that
              mixes product features, human-subject study infrastructure, mobile sensing, model
              benchmarking, and rapid research prototyping.
            </p>
            <p>
              The most original and communicable part is the emotional day-night story: what you feel
              in the day, how you eat, how you sleep, and what you dream may be measurable as one
              connected chain. The codebase already has enough pieces to tell that story well. The main
              strategic job now is to keep that story central while organizing the expanding research
              backlog into clear, credible milestones.
            </p>
            <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-4">
              <div className="flex items-center gap-2 text-emerald-300">
                <Sparkles className="h-4 w-4" />
                <span className="text-sm font-semibold">Best one-line positioning</span>
              </div>
              <p className="mt-2 text-sm leading-7 text-muted-foreground">
                AntarAI is a consumer-EEG platform that turns brain activity into a
                measurable story about emotion, behavior, sleep, and dreams.
              </p>
            </div>
          </div>
        </Section>

        <Card className="border-sky-500/20 bg-sky-500/5">
          <CardContent className="pt-5">
            <div className="flex items-center gap-2 text-sky-300">
              <HeartPulse className="h-4 w-4" />
              <h2 className="text-sm font-semibold">Page Basis</h2>
            </div>
            <p className="mt-3 text-sm leading-7 text-muted-foreground">
              This page was synthesized from `README.md`, `STATUS.md`,
              `irb/COMBINED_STUDY_PROTOCOL_DRAFT.md`, local git history, and the GitHub issue tracker
              for `LakshmiSravyaVedantham/DreamAnalysis`.
            </p>
            <p className="mt-3 text-sm leading-7 text-muted-foreground">
              Related execution plan: `docs/plans/2026-03-09-food-emotion-human-validation-plan.md`
            </p>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
