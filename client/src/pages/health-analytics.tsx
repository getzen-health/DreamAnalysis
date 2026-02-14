import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Heart,
  Bed,
  CheckCircle,
  TriangleAlert,
  Lightbulb,
  Moon,
} from "lucide-react";
import { useMetrics } from "@/hooks/use-metrics";
import { generateHealthInsights } from "@/lib/data-simulation";

export default function HealthAnalytics() {
  const { currentMetrics } = useMetrics();

  const iconMap: Record<string, typeof CheckCircle> = {
    "check-circle": CheckCircle,
    "exclamation-triangle": TriangleAlert,
    lightbulb: Lightbulb,
    moon: Moon,
  };

  const colorClasses: Record<string, string> = {
    success: "bg-success/10 border-success/30 text-success",
    warning: "bg-warning/10 border-warning/30 text-warning",
    info: "bg-primary/10 border-primary/30 text-primary",
    secondary: "bg-secondary/10 border-secondary/30 text-secondary",
  };

  return (
    <main className="p-4 md:p-6 space-y-6">
      {/* Health Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-4">
            Daily Steps
          </h3>
          <div className="text-3xl font-bold text-primary font-mono mb-2">
            {currentMetrics.dailySteps?.toLocaleString()}
          </div>
          <div className="text-sm text-foreground/70 mb-4">
            Target: 10,000 steps
          </div>
          <Progress
            value={((currentMetrics.dailySteps || 0) / 10000) * 100}
            className="h-2"
          />
        </Card>

        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-4">
            Sleep Duration
          </h3>
          <div className="text-3xl font-bold text-secondary font-mono mb-2">
            {Math.floor(currentMetrics.sleepDuration || 0)}h{" "}
            {Math.round(((currentMetrics.sleepDuration || 0) % 1) * 60)}m
          </div>
          <div className="text-sm text-foreground/70 mb-4">
            Recommended: 8h
          </div>
          <div className="flex items-center space-x-2">
            <Bed className="text-secondary h-4 w-4" />
            <span className="text-sm text-secondary">Good quality sleep</span>
          </div>
        </Card>

        <Card className="glass-card p-6 rounded-xl hover-glow">
          <h3 className="text-lg font-futuristic font-semibold mb-4">
            Avg Heart Rate
          </h3>
          <div className="text-3xl font-bold text-success font-mono mb-2">
            {currentMetrics.heartRate}
          </div>
          <div className="text-sm text-foreground/70 mb-4">
            Resting: 65-75 BPM
          </div>
          <div className="flex items-center space-x-2">
            <Heart className="text-success h-4 w-4" />
            <span className="text-sm text-success">Optimal range</span>
          </div>
        </Card>
      </div>

      {/* Personalized Insights */}
      <Card className="glass-card p-6 rounded-xl hover-glow">
        <h3 className="text-lg font-futuristic font-semibold mb-6">
          AI-Powered Health Insights
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {generateHealthInsights().map((insight, index) => {
            const Icon = iconMap[insight.icon];
            return (
              <div
                key={index}
                className={`flex items-start space-x-3 p-4 rounded-lg border ${colorClasses[insight.type] || ""}`}
              >
                {Icon && <Icon className="mt-1 h-5 w-5" />}
                <div>
                  <h4 className="font-semibold mb-1">{insight.title}</h4>
                  <p className="text-sm text-foreground/80">
                    {insight.description}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </Card>
    </main>
  );
}
