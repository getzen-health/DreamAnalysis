/**
 * LearnMoreOverlay — Bevel-style contextual bottom sheet explaining health metrics.
 * Each metric has: title, description, how it's calculated, tips to improve.
 */

import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import {
  HeartPulse, Moon, Activity, Brain, Zap, Utensils, TrendingUp, Battery,
  type LucideIcon,
} from "lucide-react";

interface MetricInfo {
  title: string;
  icon: LucideIcon;
  description: string;
  howCalculated: string;
  tips: string[];
  sources: string[];
}

const METRICS: Record<string, MetricInfo> = {
  recovery: {
    title: "Recovery Score",
    icon: HeartPulse,
    description: "How ready your body is to perform. A high recovery means your nervous system has recovered from recent strain and you can handle a challenging day.",
    howCalculated: "Combines HRV (heart rate variability), resting heart rate, sleep quality and duration, and recent strain levels.",
    tips: [
      "Prioritize 7-9 hours of sleep",
      "Reduce alcohol — it suppresses HRV for 24-48h",
      "Balance high-strain days with active recovery",
      "Stay hydrated — dehydration drops HRV",
    ],
    sources: ["Plews et al. (2013) — HRV-guided training", "Buchheit (2014) — Monitoring training status with HR measures"],
  },
  sleep: {
    title: "Sleep Score",
    icon: Moon,
    description: "Quality of your sleep last night. Factors in how long you slept, time in deep and REM stages, and how many times you woke up.",
    howCalculated: "Weighted average: 40% duration (vs 7-9h target), 30% deep sleep ratio, 20% REM ratio, 10% sleep efficiency (time asleep vs time in bed).",
    tips: [
      "Keep a consistent bedtime — even on weekends",
      "No screens 30 min before bed",
      "Keep bedroom cool (65-68°F / 18-20°C)",
      "Avoid caffeine after 2 PM",
      "Morning sunlight exposure sets your circadian clock",
    ],
    sources: ["Walker (2017) — Why We Sleep", "Huberman Lab — Sleep optimization protocols"],
  },
  strain: {
    title: "Strain Score",
    icon: Activity,
    description: "How much physical stress your body accumulated today. High strain means your cardiovascular and nervous systems worked hard.",
    howCalculated: "Based on time in elevated heart rate zones, exercise duration, workout intensity, and overall activity level. Adjusted for your baseline fitness.",
    tips: [
      "Match strain to recovery — high strain on high recovery days",
      "Include rest days in your training plan",
      "Active recovery (walking, yoga) on low recovery days",
      "Monitor your acute:chronic load ratio to prevent injury",
    ],
    sources: ["Banister (1991) — Training impulse model", "Gabbett (2016) — Training-injury prevention paradox"],
  },
  stress: {
    title: "Stress Score",
    icon: Brain,
    description: "Your current physiological stress level. Elevated stress shows up as increased heart rate, reduced HRV, and specific brainwave patterns.",
    howCalculated: "Combines heart rate relative to resting, HRV suppression, high-beta brainwave activity (20-30 Hz), and self-reported mood data.",
    tips: [
      "Box breathing (4-4-4-4) activates parasympathetic response",
      "10-min walk outdoors reduces cortisol",
      "Social connection lowers stress hormones",
      "Regular exercise builds stress resilience",
    ],
    sources: ["McEwen (2017) — Neurobiological mechanisms of stress resilience", "Thayer & Lane (2009) — HRV and stress"],
  },
  nutrition: {
    title: "Nutrition Score",
    icon: Utensils,
    description: "How well your food intake supports your brain and body today. Considers calorie targets, macro balance, and nutrient quality.",
    howCalculated: "Scored against AHEI (Alternate Healthy Eating Index): positive contributors (vegetables, fruits, omega-3s, whole grains) vs negative (processed food, excess sugar, sodium).",
    tips: [
      "Eat protein at every meal for sustained energy",
      "Omega-3 rich foods (salmon, walnuts) support brain function",
      "Complex carbs before focus work, not simple sugars",
      "Stay hydrated — even 1-2% dehydration impairs cognition",
    ],
    sources: ["AHEI-2010 scoring system", "Gomez-Pinilla (2008) — Brain foods"],
  },
  energy: {
    title: "Energy Bank",
    icon: Battery,
    description: "Your body's overall energy reserves — like a battery. It fills with sleep and recovery, and drains with strain and stress.",
    howCalculated: "Composite of all scores: recovery charges it, strain and stress drain it. Tracks cumulative balance over recent days, not just today.",
    tips: [
      "Think of it as a bank account — don't overdraw",
      "Big training days need recovery deposits",
      "Sleep is the #1 way to recharge",
      "Chronic low energy bank = overtraining risk",
    ],
    sources: ["Meeusen et al. (2013) — Prevention, diagnosis and treatment of overtraining syndrome"],
  },
  focus: {
    title: "Focus Index",
    icon: Zap,
    description: "How engaged and concentrated your brain is right now. High focus means sustained attention with minimal mind-wandering.",
    howCalculated: "Beta/alpha brainwave ratio (high beta = active thinking), theta suppression (low theta = less drowsiness), and voluntary attention markers.",
    tips: [
      "Work in 25-50 min focused blocks with breaks",
      "Eliminate notifications during deep work",
      "Morning is usually peak focus time (9-11 AM)",
      "Regular exercise improves baseline focus",
    ],
    sources: ["Klimesch (1999) — Alpha and theta oscillations", "Newport (2016) — Deep Work"],
  },
  mood: {
    title: "Mood / Valence",
    icon: TrendingUp,
    description: "Your emotional valence — how positive or negative you're feeling. Measured from brain asymmetry, voice tone, and self-reports.",
    howCalculated: "Frontal alpha asymmetry (FAA): more left-brain activity = approach/positive, more right = withdrawal/negative. Blended with voice emotion analysis and manual mood logs.",
    tips: [
      "Physical activity reliably improves mood",
      "Social interaction boosts positive valence",
      "Gratitude journaling shifts attention to positive",
      "Nature exposure reduces negative mood",
    ],
    sources: ["Davidson (1992) — Frontal alpha asymmetry and emotion", "Russell (1980) — Circumplex model of affect"],
  },
  "brain-energy": {
    title: "Brain Energy",
    icon: Battery,
    description: "Your neural arousal level — how energised and activated your brain is right now. Distinct from physical energy: you can feel bodily tired but mentally alert, or vice versa.",
    howCalculated: "Derived from EEG arousal index: beta/(alpha+beta) ratio across all 4 channels. High beta relative to alpha indicates an activated, energised brain state. Averaged across the current session.",
    tips: [
      "Caffeine boosts beta power — take it 30 min before focus work",
      "Bright light exposure in the morning elevates neural arousal",
      "Cold water face splash causes an immediate arousal spike",
      "High brain energy + low stress = flow state — use it for hard problems",
    ],
    sources: ["Barry et al. (2007) — EEG differences in eyes-open vs closed", "Oken et al. (2006) — Vigilance, sleep deprivation and EEG"],
  },
};

interface Props {
  metric: string;
  open: boolean;
  onClose: () => void;
}

export function LearnMoreOverlay({ metric, open, onClose }: Props) {
  const info = METRICS[metric];
  if (!info) return null;

  const Icon = info.icon;

  return (
    <Sheet open={open} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <SheetContent side="bottom" className="max-h-[80vh] overflow-y-auto rounded-t-2xl">
        <SheetHeader className="pb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Icon className="w-5 h-5 text-primary" />
            </div>
            <SheetTitle className="text-lg">{info.title}</SheetTitle>
          </div>
        </SheetHeader>

        <div className="space-y-5 pb-6">
          <p className="text-sm text-muted-foreground leading-relaxed">
            {info.description}
          </p>

          <div>
            <h4 className="text-xs font-semibold text-foreground/70 uppercase tracking-wider mb-2">
              How it's calculated
            </h4>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {info.howCalculated}
            </p>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-foreground/70 uppercase tracking-wider mb-2">
              Tips to improve
            </h4>
            <ul className="space-y-2">
              {info.tips.map((tip, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                  <span className="text-primary mt-0.5">•</span>
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-foreground/70 uppercase tracking-wider mb-2">
              Research basis
            </h4>
            <ul className="space-y-1">
              {info.sources.map((src, i) => (
                <li key={i} className="text-xs text-muted-foreground/60 italic">
                  {src}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
