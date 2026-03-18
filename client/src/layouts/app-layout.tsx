import { useState, useEffect, ReactNode } from "react";
import { useLocation } from "wouter";
import { motion, AnimatePresence } from "framer-motion";
import { InterventionBanner } from "@/components/intervention-banner";
import OfflineSyncBanner from "@/components/offline-sync-banner";
import { BottomTabs } from "@/components/bottom-tabs";
import { useHealthSync } from "@/hooks/use-health-sync";
import { usePullRefresh } from "@/hooks/use-pull-refresh";
import { registerNativePush } from "@/lib/native-push";
import { useAuth } from "@/hooks/use-auth";
import { useTheme } from "@/hooks/use-theme";
import { useKeyboardScroll } from "@/hooks/use-keyboard-scroll";
import { pingBackend } from "@/lib/ml-api";
import { Loader2, Sun, Moon, ChevronLeft } from "lucide-react";

const routeTitles: Record<string, string> = {
  "/": "Today",
  "/discover": "Discover",
  "/nutrition": "Nutrition",
  "/you": "You",
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
  "/workout": "Workout",
  "/body-metrics": "Body Metrics",
  "/food-emotion": "Food & Mood",
  "/supplements": "Supplements",
  "/habits": "Habits",
  "/wellness": "Wellness",
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
  // On mobile, scroll focused inputs into view when the virtual keyboard opens
  useKeyboardScroll();

  // Android hardware back button: navigate back instead of exiting the app.
  // In Capacitor WebView, the hardware back button fires "backbutton" on document.
  // On root/dashboard, let the default behavior (exit) happen.
  useEffect(() => {
    const handleBackButton = (e: Event) => {
      if (location === "/") {
        // On dashboard — let default behavior exit the app
        return;
      }
      e.preventDefault();
      window.history.back();
    };
    document.addEventListener("backbutton", handleBackButton);
    return () => document.removeEventListener("backbutton", handleBackButton);
  }, [location]);

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
      <main ref={pullRef} className="min-h-screen overflow-x-hidden" role="main">
        {/* Mobile top bar with back button — shown on subpages only */}
        {location !== "/" && (
          <header
            className="sticky top-0 z-30 border-b flex items-center gap-2 px-3 py-2.5"
            style={{
              background: theme === "dark" ? "hsl(260, 20%, 7%, 0.92)" : "hsl(40, 30%, 97%, 0.92)",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              borderColor: theme === "dark" ? "hsl(255, 10%, 18%, 0.5)" : "hsl(40, 15%, 87%, 0.5)",
              paddingTop: "env(safe-area-inset-top, 0px)",
            }}
          >
            <button
              onClick={() => window.history.back()}
              className="w-9 h-9 flex items-center justify-center rounded-xl text-muted-foreground hover:text-foreground active:bg-muted/40 transition-colors -ml-1"
              aria-label="Go back"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            <p className="text-[14px] font-semibold text-foreground leading-tight truncate">
              {pageTitle}
            </p>
            <div className="ml-auto">
              <button
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                className="w-9 h-9 flex items-center justify-center rounded-xl text-muted-foreground hover:text-foreground active:bg-muted/40 transition-colors"
                aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              >
                {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </button>
            </div>
          </header>
        )}

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
        {/* Tab bar is 56px tall — 3.75rem */}
        <div className="pb-[calc(env(safe-area-inset-bottom,0px)+3.75rem)] md:pb-[env(safe-area-inset-bottom,0px)]">
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
