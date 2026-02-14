import { useState, useEffect, ReactNode } from "react";
import { useLocation } from "wouter";
import { Sidebar } from "@/components/sidebar";
import { NeuralBackground } from "@/components/neural-background";
import { Button } from "@/components/ui/button";
import { Download } from "lucide-react";
import { useMetrics } from "@/hooks/use-metrics";
import { useTheme } from "@/hooks/use-theme";

const routeTitles: Record<string, string> = {
  "/": "Neural Dashboard",
  "/brain-monitor": "Real-Time Monitor",
  "/health-analytics": "Health Analytics",
  "/dream-journal": "Dream Journal",
  "/dream-patterns": "Dream Patterns",
  "/emotion-lab": "Emotion Lab",
  "/ai-companion": "AI Companion",
  "/insights": "Insights",
  "/settings": "Settings",
};

interface AppLayoutProps {
  children: ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [location] = useLocation();
  const { userId } = useMetrics();
  const { theme } = useTheme();

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const pageTitle = routeTitles[location] || "Neural Dashboard";

  const handleDataExport = async () => {
    try {
      const response = await fetch(`/api/export/${userId}`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "neural_data.csv";
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Export failed:", error);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <NeuralBackground />

      <Sidebar />

      {/* Main Content */}
      <div className="md:ml-64 min-h-screen">
        {/* Header */}
        <header className="glass-card border-b border-primary/20 p-4 md:p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2
                className="text-2xl font-futuristic font-bold text-gradient"
                data-testid="page-title"
              >
                {pageTitle}
              </h2>
              <p className="text-foreground/70 text-sm" data-testid="current-time">
                {currentTime.toLocaleDateString("en-US", {
                  weekday: "long",
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                })}{" "}
                -{" "}
                {currentTime.toLocaleTimeString("en-US", { hour12: false })}
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 glass-card px-3 py-2 rounded-lg">
                <div className="status-indicator w-2 h-2"></div>
                <span className="text-sm font-mono text-success">
                  98.7% Signal
                </span>
              </div>
              <Button
                variant="outline"
                size="icon"
                className="glass-card border-primary/20 hover-glow"
                onClick={handleDataExport}
                data-testid="button-export"
              >
                <Download className="h-4 w-4 text-primary" />
              </Button>
            </div>
          </div>
        </header>

        {/* Page Content */}
        {children}
      </div>
    </div>
  );
}
