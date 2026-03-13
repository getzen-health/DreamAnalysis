import { useState, useEffect, ReactNode } from "react";
import { useLocation } from "wouter";
import { motion, AnimatePresence } from "framer-motion";
import { Sidebar } from "@/components/sidebar";
import { NeuralBackground } from "@/components/neural-background";
import { InterventionBanner } from "@/components/intervention-banner";
import OfflineSyncBanner from "@/components/offline-sync-banner";
import { BottomTabs } from "@/components/bottom-tabs";
import { useHealthSync } from "@/hooks/use-health-sync";
import { usePullRefresh } from "@/hooks/use-pull-refresh";
import { registerNativePush } from "@/lib/native-push";
import { useAuth } from "@/hooks/use-auth";
import { useTheme } from "@/hooks/use-theme";
import { pingBackend } from "@/lib/ml-api";
import { Loader2, Sun, Moon } from "lucide-react";

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
  "/architecture-guide": "Project Guide",
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

  // Keep-alive: ping ML backend every 14 min to prevent Render free-tier sleep
  const { user } = useAuth();
  useEffect(() => {
    if (!user) return;
    const FOURTEEN_MIN = 14 * 60 * 1000;
    const id = setInterval(() => {
      if (document.visibilityState === "visible") {
        pingBackend(5_000).catch(() => {});
      }
    }, FOURTEEN_MIN);
    return () => clearInterval(id);
  }, [user]);

  const { ref: pullRef, pullDistance, refreshing } = usePullRefresh<HTMLDivElement>();
  const { theme, setTheme } = useTheme();

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
      <main ref={pullRef} className="md:ml-56 min-h-screen overflow-x-hidden" role="main">
        <header
          className="sticky top-0 z-30 border-b md:px-6"
          style={{
            background: "hsl(222, 25%, 5%, 0.92)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            borderColor: "hsl(220, 18%, 13%, 0.5)",
            paddingTop: "env(safe-area-inset-top, 0px)",
          }}
        >
          <div className="flex items-center justify-between pl-14 pr-4 py-2.5 md:pl-4 md:py-3">
            {/* Left: title */}
            <div>
              <p className="text-[15px] font-semibold text-foreground leading-tight">
                {pageTitle}
              </p>
              <p className="text-[11px] text-muted-foreground/60 leading-tight">
                {dateStr}
              </p>
            </div>
            {/* Right: theme toggle */}
            <button
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="w-9 h-9 flex items-center justify-center rounded-xl text-muted-foreground hover:text-foreground hover:bg-muted/40 active:bg-muted/60 transition-colors"
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </header>

        {/* Pull-to-refresh indicator */}
        {(pullDistance > 0 || refreshing) && (
          <div
            className="flex items-center justify-center overflow-hidden transition-[height] duration-200"
            style={{ height: refreshing ? 48 : pullDistance }}
          >
            <Loader2
              className={`h-5 w-5 text-primary ${refreshing ? "animate-spin" : ""}`}
              style={{
                opacity: refreshing ? 1 : Math.min(1, pullDistance / 40),
                transform: `rotate(${pullDistance * 3}deg)`,
              }}
            />
          </div>
        )}

        {/* pb-safe: extra bottom padding on devices with home indicator + bottom tab bar on mobile */}
        <div className="pb-[calc(env(safe-area-inset-bottom,0px)+3.5rem)] md:pb-[env(safe-area-inset-bottom,0px)]">
          <AnimatePresence mode="wait">
            <motion.div
              key={location}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>

      {/* Bottom tab bar — mobile only */}
      <BottomTabs />
      {/* Real-time intervention notifications — floats above all content */}
      <InterventionBanner />
      {/* Offline indicator + auto-sync when connection restores */}
      <OfflineSyncBanner />
    </div>
  );
}
