import { useLocation } from "wouter";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Brain, Clock, Shield, ChevronRight, Headphones, Zap } from "lucide-react";

export default function StudyLanding() {
  const [, navigate] = useLocation();

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-2xl mx-auto px-4 py-12 space-y-10">

        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex justify-center">
            <Brain className="h-12 w-12 text-primary" />
          </div>
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
            Neural Dream Workshop Pilot Study
          </h1>
          <p className="text-muted-foreground text-base md:text-lg leading-relaxed max-w-xl mx-auto">
            We are running a small pilot study to understand how the brain responds to
            everyday stress and food. Your EEG data will help train better models and
            contribute to a published academic research paper.
          </p>
          <Badge variant="outline" className="border-green-500/50 text-green-400 px-4 py-1.5 text-sm">
            <Shield className="h-3.5 w-3.5 mr-1.5" />
            Anonymous &amp; Voluntary
          </Badge>
        </div>

        {/* What you will do */}
        <Card>
          <CardContent className="pt-6 space-y-4">
            <h2 className="font-semibold text-lg">What you will do</h2>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-0.5">
                  <Headphones className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <p className="font-medium text-sm">Wear the Muse 2 headband</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    A consumer EEG device that measures brain wave patterns non-invasively.
                    No gels, no discomfort.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-orange-500/10 flex items-center justify-center shrink-0 mt-0.5">
                  <Zap className="h-4 w-4 text-orange-400" />
                </div>
                <div>
                  <p className="font-medium text-sm">Complete a stress session — 25 min</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    A resting baseline, a short work period with the headband on, a guided
                    breathing exercise, and a post-session survey.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-secondary/10 flex items-center justify-center shrink-0 mt-0.5">
                  <Brain className="h-4 w-4 text-secondary" />
                </div>
                <div>
                  <p className="font-medium text-sm">Complete a food session — 40 min</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Pre-meal EEG baseline, eat your normal meal, a post-meal EEG window,
                    and a brief food survey.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Time requirement */}
        <Card className="border-muted">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <Clock className="h-5 w-5 text-muted-foreground shrink-0" />
              <div>
                <p className="font-medium text-sm">Time required: 25–40 min per session</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Sessions can be done on different days. You do not need to complete both
                  in one sitting.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Privacy note */}
        <Card className="border-green-500/20 bg-green-500/5">
          <CardContent className="pt-6 space-y-2">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-green-400" />
              <p className="font-medium text-sm text-green-400">Anonymous &amp; Voluntary</p>
            </div>
            <ul className="space-y-1.5 text-sm text-muted-foreground">
              <li>You are identified only by a participant code — no name, no email.</li>
              <li>Data is used exclusively for academic research.</li>
              <li>You can withdraw at any time without consequence.</li>
            </ul>
          </CardContent>
        </Card>

        {/* CTA */}
        <div className="flex justify-center">
          <Button
            size="lg"
            className="px-10 text-base"
            onClick={() => navigate("/study/consent")}
          >
            Join the Study
            <ChevronRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
}
