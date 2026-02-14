import { MetricCard } from "@/components/metric-card";
import { MoodChart } from "@/components/charts/mood-chart";
import { SleepChart } from "@/components/charts/sleep-chart";
import { AIAnalysis } from "@/components/ai-analysis";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Heart,
  AlertTriangle,
  Bed,
  Brain,
  BarChart3,
  Moon,
  Star,
} from "lucide-react";
import { useMetrics } from "@/hooks/use-metrics";

export default function Dashboard() {
  const { currentMetrics, moodData, sleepData, userId } = useMetrics();

  return (
    <main className="p-4 md:p-6 space-y-6">
      {/* Real-time Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Heart Rate"
          value={currentMetrics.heartRate}
          unit="BPM"
          change="+2.3%"
          icon={Heart}
          color="success"
          animated={true}
        >
          <div className="flex space-x-1">
            {[0, 0.1, 0.2, 0.3].map((delay, index) => (
              <div
                key={index}
                className="w-1 h-8 bg-success/30 rounded animate-brain-wave"
                style={{
                  animationDelay: `${delay}s`,
                  backgroundColor:
                    index === 3 ? "var(--success)" : undefined,
                }}
              />
            ))}
          </div>
        </MetricCard>

        <MetricCard
          title="Stress Level"
          value={currentMetrics.stressLevel}
          unit="STRESS"
          change="-5.2%"
          icon={AlertTriangle}
          color="warning"
        >
          <Progress value={currentMetrics.stressLevel} className="h-2" />
        </MetricCard>

        <MetricCard
          title="Sleep Quality"
          value={currentMetrics.sleepQuality}
          unit="QUALITY"
          change="+12%"
          icon={Bed}
          color="secondary"
        >
          <div className="flex space-x-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <Star
                key={star}
                className={`h-3 w-3 ${star <= 4 ? "text-secondary fill-secondary" : "text-secondary/30"}`}
              />
            ))}
          </div>
        </MetricCard>

        <MetricCard
          title="Neural Activity"
          value={currentMetrics.neuralActivity}
          unit="NEURAL"
          change="+8.7%"
          icon={Brain}
          color="primary"
        >
          <div className="relative">
            {[0, 25, 50, 75].map((left, index) => (
              <div
                key={index}
                className="neural-particle"
                style={{
                  left: `${left}%`,
                  animationDelay: `${index * 0.5}s`,
                }}
              />
            ))}
          </div>
        </MetricCard>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-futuristic font-semibold">
              7-Day Mood Timeline
            </h3>
            <BarChart3 className="text-primary" />
          </div>
          <MoodChart data={moodData} />
        </Card>

        <Card className="glass-card p-6 rounded-xl hover-glow">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-futuristic font-semibold">
              Sleep & Dream Activity
            </h3>
            <Moon className="text-secondary" />
          </div>
          <SleepChart data={sleepData} />
        </Card>
      </div>

      {/* AI Analysis Section */}
      <AIAnalysis userId={userId} />
    </main>
  );
}
