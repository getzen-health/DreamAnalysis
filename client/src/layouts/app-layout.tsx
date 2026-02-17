import { useState, useEffect, ReactNode } from "react";
import { useLocation } from "wouter";
import { Sidebar } from "@/components/sidebar";
import { NeuralBackground } from "@/components/neural-background";

const routeTitles: Record<string, string> = {
  "/": "Dashboard",
  "/emotions": "Emotion Lab",
  "/inner-energy": "Inner Energy",
  "/brain-monitor": "Brain Monitor",
  "/brain-connectivity": "Brain Connectivity",
  "/dreams": "Dream Journal",
  "/dream-patterns": "Dream Patterns",
  "/health-analytics": "Health Analytics",
  "/neurofeedback": "Neurofeedback",
  "/ai-companion": "AI Companion",
  "/insights": "Insights",
  "/sessions": "Sessions",
  "/settings": "Settings",
};

function getGreeting(): string {
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  return "Good evening";
}

interface AppLayoutProps {
  children: ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [location] = useLocation();

  useEffect(() => {
    const interval = setInterval(() => setCurrentTime(new Date()), 60000);
    return () => clearInterval(interval);
  }, []);

  const pageTitle = routeTitles[location] || "Dashboard";
  const dateStr = currentTime.toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
  });

  return (
    <div className="min-h-screen bg-background text-foreground">
      <NeuralBackground />
      <Sidebar />

      <div className="md:ml-56 min-h-screen">
        <header
          className="sticky top-0 z-30 border-b px-6 py-3"
          style={{
            background: "hsl(222, 25%, 6%, 0.85)",
            backdropFilter: "blur(12px)",
            WebkitBackdropFilter: "blur(12px)",
            borderColor: "hsl(220, 18%, 15%, 0.5)",
          }}
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-lg font-semibold text-foreground">
                {getGreeting()}
              </h1>
              <p className="text-xs text-muted-foreground">
                {pageTitle} &middot; {dateStr}
              </p>
            </div>
          </div>
        </header>

        {children}
      </div>
    </div>
  );
}
