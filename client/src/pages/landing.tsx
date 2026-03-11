import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { NeuralBackground } from "@/components/neural-background";
import { Brain, Moon, Heart, Bot, Eye, BarChart3, Mic, Wifi, ChevronRight, Watch } from "lucide-react";

const features = [
  {
    icon: Mic,
    title: "Voice Check-Ins",
    description: "Start with a short voice check-in to estimate emotion, energy, stress, and daily readiness in seconds.",
    color: "primary",
  },
  {
    icon: Watch,
    title: "Watch + Health Signals",
    description: "Blend heart rate, HRV, sleep, SpO2, and daily activity into readiness and recovery insights when health data is available.",
    color: "secondary",
  },
  {
    icon: Heart,
    title: "Daily Readiness",
    description: "Track stress, focus, recovery, and emotional balance from voice and health inputs without needing a headset.",
    color: "success",
  },
  {
    icon: Brain,
    title: "Optional EEG Upgrade",
    description: "Add Muse 2 later for live neural sensing, calibration, neurofeedback, and advanced brain-state tools.",
    color: "accent",
  },
  {
    icon: Moon,
    title: "Dream Journal",
    description: "Capture dreams by voice or text, then track themes, sleep quality, and next-day mood over time.",
    color: "warning",
  },
  {
    icon: Eye,
    title: "Dream Visualization",
    description: "Generate artwork from dream descriptions and build a visual archive of your inner world.",
    color: "primary",
  },
  {
    icon: Bot,
    title: "AI Wellness Companion",
    description: "Use guided breathing, reflection, and mood support tools throughout the day from the same voice-first flow.",
    color: "secondary",
  },
  {
    icon: BarChart3,
    title: "Trend Tracking",
    description: "See daily and weekly changes in mood, sleep, focus, and recovery as your data builds up.",
    color: "success",
  },
  {
    icon: Wifi,
    title: "Installable App",
    description: "Use it like a daily health app, keep journaling offline, and sync when connected.",
    color: "success",
  },
];

const colorClasses: Record<string, string> = {
  primary: "bg-primary/10 border-primary/30 text-primary",
  secondary: "bg-secondary/10 border-secondary/30 text-secondary",
  success: "bg-success/10 border-success/30 text-success",
  accent: "bg-accent/10 border-accent/30 text-accent",
  warning: "bg-warning/10 border-warning/30 text-warning",
};

export default function Landing() {
  return (
    <div className="min-h-screen bg-background text-foreground overflow-hidden">
      <NeuralBackground />

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center px-4">
        <div className="max-w-4xl mx-auto text-center z-10">
          <Badge variant="outline" className="mb-6 border-secondary/50 text-secondary px-4 py-1.5">
            Voice + Watch First. EEG Optional.
          </Badge>

          <h1 className="text-5xl md:text-7xl font-futuristic font-bold mb-6">
            <span className="text-gradient">AntarAI</span>
          </h1>

          <p className="text-lg md:text-xl text-foreground/60 max-w-2xl mx-auto mb-8 leading-relaxed">
            A daily readiness app built around voice check-ins, health signals, sleep, and dream tracking.
            Add Muse 2 later if you want live EEG and deeper neural features.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/onboarding-new">
              <Button size="lg" className="bg-gradient-to-r from-primary to-secondary text-primary-foreground hover-glow px-8 text-lg">
                Get Started Free
                <ChevronRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
            <Link href="/architecture-guide">
              <Button size="lg" variant="outline" className="border-primary/30 text-primary hover:bg-primary/10 px-8 text-lg">
                <Brain className="mr-2 h-5 w-5" />
                Live Demo
              </Button>
            </Link>
          </div>

          <div className="flex items-center justify-center gap-8 mt-12 text-foreground/40">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary font-mono">5</div>
              <div className="text-xs">Sleep Signals</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-secondary font-mono">6</div>
              <div className="text-xs">Emotions</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-success font-mono">4</div>
              <div className="text-xs">Health Inputs</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-warning font-mono">24/7</div>
              <div className="text-xs">Daily Tracking</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="relative py-24 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-futuristic font-bold mb-4">
              <span className="text-gradient">One Daily Flow</span>, Multiple Inputs
            </h2>
            <p className="text-foreground/60 max-w-xl mx-auto">
              Start with voice and health now. Layer in dreams, recovery, and optional EEG only when you want more depth.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, i) => {
              const Icon = feature.icon;
              return (
                <Card key={i} className="glass-card p-6 rounded-xl hover-glow group cursor-pointer transition-all duration-300 hover:scale-[1.02]">
                  <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 border ${colorClasses[feature.color]}`}>
                    <Icon className="h-6 w-6" />
                  </div>
                  <h3 className="font-futuristic font-semibold mb-2 text-sm">{feature.title}</h3>
                  <p className="text-xs text-foreground/60 leading-relaxed">{feature.description}</p>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative py-24 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <Card className="glass-card p-12 rounded-2xl hover-glow">
            <Brain className="h-16 w-16 mx-auto text-primary mb-6" />
            <h2 className="text-3xl font-futuristic font-bold mb-4">
              Start Your <span className="text-gradient">Daily Readiness Flow</span>
            </h2>
            <p className="text-foreground/60 mb-8 max-w-lg mx-auto">
              Begin with a voice check-in, connect health data when available, and add EEG later if you want live neural sensing.
            </p>
            <Link href="/onboarding-new">
              <Button size="lg" className="bg-gradient-to-r from-primary to-secondary text-primary-foreground hover-glow px-12 text-lg">
                Create Free Account
                <ChevronRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 px-4">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            <span className="font-futuristic text-sm font-bold text-gradient">AntarAI</span>
          </div>
          <div className="flex items-center gap-6 text-xs text-foreground/40">
            <span>Built with AI + Neuroscience</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
