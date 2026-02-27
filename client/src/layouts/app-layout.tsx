import { useState, useEffect, ReactNode } from "react";
import { useLocation } from "wouter";
import { Sidebar } from "@/components/sidebar";
import { NeuralBackground } from "@/components/neural-background";
import { InterventionBanner } from "@/components/intervention-banner";
import OfflineSyncBanner from "@/components/offline-sync-banner";
import { useHealthSync } from "@/hooks/use-health-sync";
import { registerNativePush } from "@/lib/native-push";

const routeTitles: Record<string, string> = {
  "/": "Dashboard",
  "/emotions": "Emotion Lab",
  "/inner-energy": "Inner Energy",
  "/brain-monitor": "Brain Monitor",
  "/brain-connectivity": "Brain Connectivity",
  "/dreams": "Dream Detection",
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
  // Start HealthKit / Health Connect auto-sync on first mount (no-op on web)
  useHealthSync();

  // Register for native push notifications on iOS/Android (no-op on web)
  useEffect(() => {
    registerNativePush().catch(() => {});
    // Handle navigation triggered by tapping a native push notification
    const onNavigate = (e: Event) => {
      const route = (e as CustomEvent<{ route: string }>).detail.route;
      if (route) window.location.href = route;
    };
    window.addEventListener("native-push-navigate", onNavigate);
    return () => window.removeEventListener("native-push-navigate", onNavigate);
  }, []);

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

      {/* md:ml-56 = sidebar width; on mobile sidebar overlays so no margin needed */}
      <div className="md:ml-56 min-h-screen overflow-x-hidden">
        <header
          className="sticky top-0 z-30 border-b pl-14 pr-4 py-3 md:px-6"
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

        {/* pb-safe: extra bottom padding on devices with home indicator */}
        <div className="pb-[env(safe-area-inset-bottom,0px)]">
          {children}
        </div>
      </div>

      {/* Real-time intervention notifications — floats above all content */}
      <InterventionBanner />
      {/* Offline indicator + auto-sync when connection restores */}
      <OfflineSyncBanner />
    </div>
  );
}
