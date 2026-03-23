import { lazy, Suspense, useEffect, useState, type ReactNode } from "react";
import * as Sentry from "@sentry/react";
import { cleanExpiredLocalStorage } from "@/lib/storage-cleanup";
import { Switch, Route, useLocation } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "./hooks/use-theme";
import { AuthProvider, useAuth } from "./hooks/use-auth";
import { DeviceProvider } from "./hooks/use-device";
import { MLConnectionProvider } from "@/hooks/use-ml-connection";
import { VoiceCacheProvider } from "@/hooks/use-voice-cache";
import { MLWarmupScreen } from "@/components/ml-warmup-screen";
import { NeuralBackground } from "@/components/neural-background";
import AppLayout from "./layouts/app-layout";

// ── Core pages — only the 4 main tabs are static (always needed) ──────────
import Landing from "@/pages/landing";
import AuthPage from "@/pages/auth";
import Today from "@/pages/today";
import Discover from "@/pages/discover";
import Nutrition from "@/pages/nutrition";
import You from "@/pages/you";
import NotFound from "@/pages/not-found";
import { sbGetSetting } from "./lib/supabase-store";

// ── Everything else lazy-loaded — faster startup ──────────────────────────
const ForgotPasswordPage = lazy(() => import("@/pages/forgot-password"));
const ResetPasswordPage  = lazy(() => import("@/pages/reset-password"));
const Dashboard          = lazy(() => import("@/pages/dashboard"));
const BrainMonitor       = lazy(() => import("@/pages/brain-monitor"));
const DreamJournal       = lazy(() => import("@/pages/dream-journal"));
const AICompanionPage    = lazy(() => import("@/pages/ai-companion"));
const Biofeedback        = lazy(() => import("@/pages/biofeedback"));
const SessionHistory     = lazy(() => import("@/pages/session-history"));
const SettingsPage       = lazy(() => import("@/pages/settings"));
const CalibrationPage    = lazy(() => import("@/pages/calibration"));
const DailyBrainReport   = lazy(() => import("@/pages/daily-brain-report"));
const Onboarding         = lazy(() => import("@/pages/onboarding"));

// ── Intent selection page — lazy loaded ────────────────────────────────────
const IntentSelect = lazy(() => import("@/pages/intent-select"));

// ── Study pages — lazy loaded ───────────────────────────────────────────────
const StudyLanding       = lazy(() => import("@/pages/study/StudyLanding"));
const StudyConsent       = lazy(() => import("@/pages/study/StudyConsent"));
const StudyProfile       = lazy(() => import("@/pages/study/StudyProfile"));
const StudySession       = lazy(() => import("@/pages/study/StudySession"));
const StudySessionStress = lazy(() => import("@/pages/study/StudySessionStress"));
const StudySessionFood   = lazy(() => import("@/pages/study/StudySessionFood"));
const StudyComplete      = lazy(() => import("@/pages/study/StudyComplete"));
const StudyAdmin         = lazy(() => import("@/pages/study/StudyAdmin"));

// ── New consolidated pages — lazy loaded ─────────────────────────────────────
const StressTrends           = lazy(() => import("@/pages/stress-trends"));
const FocusTrends            = lazy(() => import("@/pages/focus-trends"));
const SleepPage              = lazy(() => import("@/pages/sleep"));
const HealthPage             = lazy(() => import("@/pages/health"));
const HeartRatePage          = lazy(() => import("@/pages/heart-rate"));
const StepsPage              = lazy(() => import("@/pages/steps"));
const BrainTabs              = lazy(() => import("@/pages/brain-tabs"));

// ── Heavy / rarely-visited pages — lazy loaded ─────────────────────────────
const BrainConnectivity      = lazy(() => import("@/pages/brain-connectivity"));
const HealthAnalytics        = lazy(() => import("@/pages/health-analytics"));
const Neurofeedback          = lazy(() => import("@/pages/neurofeedback"));
const Insights               = lazy(() => import("@/pages/insights"));
const InnerEnergy            = lazy(() => import("@/pages/inner-energy"));
const FormalBenchmarksDashboard = lazy(() => import("@/pages/formal-benchmarks-dashboard"));
const FoodEmotion            = lazy(() => import("@/pages/food-emotion"));
const DeviceSetup            = lazy(() => import("@/pages/device-setup"));
const ResearchEnroll         = lazy(() => import("@/pages/research-enroll"));
const ResearchMorning        = lazy(() => import("@/pages/research-morning"));
const ResearchHub            = lazy(() => import("@/pages/research-hub"));
const ResearchDaytime        = lazy(() => import("@/pages/research-daytime"));
const ResearchEvening        = lazy(() => import("@/pages/research-evening"));
const FoodLog                = lazy(() => import("@/pages/food-log"));
const SleepSession           = lazy(() => import("@/pages/sleep-session"));
const WeeklyBrainSummary     = lazy(() => import("@/pages/weekly-brain-summary"));
const PersonalRecords        = lazy(() => import("@/pages/personal-records"));
const PrivacyPolicy          = lazy(() => import("@/pages/privacy-policy"));
const ArchitectureGuide      = lazy(() => import("@/pages/architecture-guide"));
const SupplementsPage        = lazy(() => import("@/pages/supplements"));
const EmotionalIntelligence  = lazy(() => import("@/pages/emotional-intelligence"));
const EmotionalFitness       = lazy(() => import("@/pages/emotional-fitness"));
const SleepMusic             = lazy(() => import("@/components/sleep-stories"));
const CbtiModule             = lazy(() => import("@/pages/cbti-module"));
const BodyMetrics            = lazy(() => import("@/pages/body-metrics"));
const WorkoutPage            = lazy(() => import("@/pages/workout"));
const HabitsPage             = lazy(() => import("@/pages/habits"));
const WellnessPage           = lazy(() => import("@/pages/wellness"));
const ScoresDashboard        = lazy(() => import("@/pages/scores-dashboard"));
const ExportPage             = lazy(() => import("@/pages/export"));
const HelpPage               = lazy(() => import("@/pages/help"));
const ConnectedAssets        = lazy(() => import("@/pages/connected-assets"));
const NotificationsPage      = lazy(() => import("@/pages/notifications"));
const CouplesMeditation      = lazy(() => import("@/pages/couples-meditation"));
const AchievementsPage       = lazy(() => import("@/pages/achievements"));
const ConsentSettings        = lazy(() => import("@/pages/consent-settings"));
const QuickSession           = lazy(() => import("@/pages/quick-session"));
const TPBMSession            = lazy(() => import("@/pages/tpbm-session"));
const DeepWork               = lazy(() => import("@/pages/deep-work"));
const PainTracker            = lazy(() => import("@/pages/pain-tracker"));
const CommunityPage          = lazy(() => import("@/pages/community"));

// ── Error Boundary — Sentry-wrapped; reports crashes + shows branded fallback ──
function SentryErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <Sentry.ErrorBoundary
      fallback={({ resetError }) => (
        <div className="min-h-[60vh] flex flex-col items-center justify-center gap-6 bg-background text-foreground p-8">
          {/* AntarAI logo for brand consistency */}
          <div className="relative">
            <img src="/logo-antarai.svg" alt="AntarAI" className="h-12 w-12 opacity-70" />
          </div>

          <div className="text-center space-y-2 max-w-md">
            <h2 className="text-lg font-semibold text-foreground">Something went wrong</h2>
            <p className="text-sm text-muted-foreground">The app encountered an error. Your data is safe.</p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-3">
            <button
              className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition-colors"
              onClick={() => { resetError(); window.location.reload(); }}
            >
              Reload Page
            </button>
            <button
              className="px-4 py-2 rounded-lg border border-border text-xs font-medium text-foreground hover:bg-muted transition-colors"
              onClick={() => { resetError(); window.location.href = "/"; }}
            >
              Go Home
            </button>
            <button
              className="px-4 py-2 rounded-lg border border-destructive/30 text-xs font-medium text-destructive hover:bg-destructive/10 transition-colors"
              onClick={() => { localStorage.clear(); window.location.href = "/"; }}
            >
              Clear Data & Reload
            </button>
          </div>
        </div>
      )}
    >
      {children}
    </Sentry.ErrorBoundary>
  );
}

// Branded splash screen shown while a lazy chunk loads
function PageLoader() {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center gap-4 bg-background">
      {/* AntarAI logo */}
      <div className="relative">
        <img src="/logo-antarai.svg" alt="AntarAI" className="h-12 w-12" />
        <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl animate-pulse" />
      </div>
      {/* App name */}
      <p className="text-lg font-semibold text-primary tracking-tight">AntarAI</p>
      {/* Subtle spinner */}
      <div className="h-5 w-5 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
    </div>
  );
}

// Redirects new users to onboarding FIRST, then requires auth.
// Flow: fresh install → /onboarding → /auth → dashboard
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();
  const [location, setLocation] = useLocation();

  // Onboarding check runs BEFORE auth — new users see onboarding first, not a login wall
  if (location !== "/onboarding" && !sbGetSetting("ndw_onboarding_complete")) {
    setLocation("/onboarding");
    return null;
  }

  if (isLoading) return <PageLoader />;
  if (!user) {
    setLocation("/auth");
    return null;
  }
  return <>{children}</>;
}

// Simple redirect component for legacy onboarding routes
function RedirectTo({ to }: { to: string }) {
  const [, setLocation] = useLocation();
  useEffect(() => { setLocation(to); }, [to, setLocation]);
  return null;
}

function AppRoutes() {
  const [location] = useLocation();
  return (
    <SentryErrorBoundary key={location}>
    <Suspense fallback={<PageLoader />}>
    <Switch>
      <Route path="/welcome" component={Landing} />
      <Route path="/auth" component={AuthPage} />
      <Route path="/forgot-password" component={ForgotPasswordPage} />
      <Route path="/reset-password" component={ResetPasswordPage} />
      <Route path="/onboarding-new">
        {/* Legacy route — redirect to unified onboarding */}
        <RedirectTo to="/onboarding" />
      </Route>
      <Route path="/architecture-guide">
        <AppLayout><ArchitectureGuide /></AppLayout>
      </Route>
      <Route path="/">
        <ProtectedRoute><AppLayout><Today /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/discover">
        <ProtectedRoute><AppLayout><Discover /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/nutrition">
        <ProtectedRoute><AppLayout><Nutrition /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/you">
        <ProtectedRoute><AppLayout><You /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/stress">
        <ProtectedRoute><AppLayout><StressTrends /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/focus">
        <ProtectedRoute><AppLayout><FocusTrends /></AppLayout></ProtectedRoute>
      </Route>
      {/* Consolidated pages */}
      <Route path="/sleep">
        <ProtectedRoute><AppLayout><SleepPage /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/health">
        <ProtectedRoute><AppLayout><HealthPage /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/heart-rate">
        <ProtectedRoute><AppLayout><HeartRatePage /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/steps">
        <ProtectedRoute><AppLayout><StepsPage /></AppLayout></ProtectedRoute>
      </Route>
      {/* Redirects for removed pages → brain */}
      <Route path="/emotions"><RedirectTo to="/brain-monitor" /></Route>
      <Route path="/mood"><RedirectTo to="/brain-monitor" /></Route>
      <Route path="/journal"><RedirectTo to="/brain-monitor" /></Route>
      {/* Bottom tab route aliases */}
      <Route path="/trends">
        <ProtectedRoute><AppLayout><HealthAnalytics /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/brain-monitor">
        <ProtectedRoute><AppLayout><BrainTabs /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/brain-connectivity">
        <ProtectedRoute><AppLayout><BrainConnectivity /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/dreams">
        <ProtectedRoute><AppLayout><DreamJournal /></AppLayout></ProtectedRoute>
      </Route>

      <Route path="/health-analytics">
        <ProtectedRoute><AppLayout><HealthAnalytics /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/neurofeedback">
        <ProtectedRoute><AppLayout><Neurofeedback /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/ai-companion">
        <ProtectedRoute><AppLayout><AICompanionPage /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/insights">
        <ProtectedRoute><AppLayout><Insights /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/inner-energy">
        <ProtectedRoute><AppLayout><InnerEnergy /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/biofeedback">
        <ProtectedRoute><AppLayout><Biofeedback /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/sessions">
        <ProtectedRoute><AppLayout><SessionHistory /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/settings">
        <ProtectedRoute><AppLayout><SettingsPage /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/calibration">
        <ProtectedRoute><AppLayout><CalibrationPage /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/benchmarks">
        <ProtectedRoute><AppLayout><FormalBenchmarksDashboard /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/records">
        <ProtectedRoute><AppLayout><PersonalRecords /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/food">
        <ProtectedRoute><AppLayout><Nutrition /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/food-emotion">
        <ProtectedRoute><AppLayout><FoodEmotion /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/device-setup">
        <ProtectedRoute><AppLayout><DeviceSetup /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/research/enroll">
        <ProtectedRoute><AppLayout><ResearchEnroll /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/research/morning">
        <ProtectedRoute><AppLayout><ResearchMorning /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/research/daytime">
        <ProtectedRoute><AppLayout><ResearchDaytime /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/research/evening">
        <ProtectedRoute><AppLayout><ResearchEvening /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/research">
        <ProtectedRoute><AppLayout><ResearchHub /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/food-log">
        <ProtectedRoute><AppLayout><Nutrition /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/sleep-session">
        <ProtectedRoute><AppLayout><SleepSession /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/brain-report">
        <ProtectedRoute><AppLayout><DailyBrainReport /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/weekly-summary">
        <ProtectedRoute><AppLayout><WeeklyBrainSummary /></AppLayout></ProtectedRoute>
      </Route>
      {/* Study pages — public, no auth required */}
      <Route path="/study/session">
        <Suspense fallback={<PageLoader />}><StudySession /></Suspense>
      </Route>
      <Route path="/study/session/stress"><StudySessionStress /></Route>
      <Route path="/study/session/food"><StudySessionFood /></Route>
      <Route path="/study/complete"><StudyComplete /></Route>
      <Route path="/study/admin"><StudyAdmin /></Route>
      <Route path="/study/consent"><StudyConsent /></Route>
      <Route path="/study/profile"><StudyProfile /></Route>
      <Route path="/study"><StudyLanding /></Route>
      {/* Fullscreen onboarding — no sidebar */}
      <Route path="/welcome-intro">
        {/* Legacy route — redirect to unified onboarding */}
        <RedirectTo to="/onboarding" />
      </Route>
      <Route path="/onboarding" component={Onboarding} />
      <Route path="/quick-session">
        <ProtectedRoute><QuickSession /></ProtectedRoute>
      </Route>
      <Route path="/deep-work">
        <ProtectedRoute><AppLayout><DeepWork /></AppLayout></ProtectedRoute>
      </Route>
      {/* Intent selection — study vs explore, shown after first login */}
      <Route path="/intent">
        <Suspense fallback={<PageLoader />}>
          <IntentSelect />
        </Suspense>
      </Route>
      <Route path="/supplements">
        <Suspense fallback={<PageLoader />}>
          <AppLayout><SupplementsPage /></AppLayout>
        </Suspense>
      </Route>
      <Route path="/emotional-intelligence">
        <Suspense fallback={<PageLoader />}>
          <AppLayout><EmotionalIntelligence /></AppLayout>
        </Suspense>
      </Route>
      <Route path="/emotional-fitness">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><EmotionalFitness /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/sleep-music">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><SleepMusic /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/cbti">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><CbtiModule /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/body-metrics">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><BodyMetrics /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/workout">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><WorkoutPage /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/habits">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><HabitsPage /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/wellness">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><WellnessPage /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/scores">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><ScoresDashboard /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/export">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><ExportPage /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/help">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><HelpPage /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/connected-assets">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><ConnectedAssets /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/notifications">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><NotificationsPage /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/couples-meditation">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><CouplesMeditation /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      <Route path="/achievements">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><AchievementsPage /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      {/* tPBM session tracking */}
      <Route path="/tpbm">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><TPBMSession /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      {/* Pain/migraine tracking from frontal EEG */}
      <Route path="/pain-tracker">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><PainTracker /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      {/* Community features — anonymous mood sharing, challenges, streaks */}
      <Route path="/community">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><CommunityPage /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      {/* Biometric consent settings — per-modality data collection toggles */}
      <Route path="/consent-settings">
        <ProtectedRoute>
          <Suspense fallback={<PageLoader />}>
            <AppLayout><ConsentSettings /></AppLayout>
          </Suspense>
        </ProtectedRoute>
      </Route>
      {/* Public route — no auth required (needed for App Store / HealthKit) */}
      <Route path="/privacy">
        <Suspense fallback={<PageLoader />}>
          <AppLayout><PrivacyPolicy /></AppLayout>
        </Suspense>
      </Route>
      <Route component={NotFound} />
    </Switch>
    </Suspense>
    </SentryErrorBoundary>
  );
}

const PUBLIC_ROUTES = new Set(["/auth", "/forgot-password", "/reset-password", "/welcome", "/onboarding", "/welcome-intro", "/onboarding-new"]);

// Separated so it can access useAuth (must be inside AuthProvider)
function AppShell() {
  const [warmupDismissed, setWarmupDismissed] = useState(false);
  const [location] = useLocation();
  const { user, isLoading } = useAuth();
  const isPublicRoute = PUBLIC_ROUTES.has(location);

  useEffect(() => {
    cleanExpiredLocalStorage();
  }, []);

  // Only show warmup screen for authenticated users on non-public routes
  const showWarmup = !warmupDismissed && !isPublicRoute && !isLoading && !!user;

  return (
    <TooltipProvider>
      {showWarmup && (
        <MLWarmupScreen onSimulationMode={() => setWarmupDismissed(true)} />
      )}
      <AppRoutes />
      <Toaster />
    </TooltipProvider>
  );
}

function App() {
  const pathname = typeof window !== "undefined" ? window.location.pathname : "/";

  if (pathname === "/architecture-guide") {
    return (
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <TooltipProvider>
            <div className="min-h-screen bg-background text-foreground">
              <NeuralBackground />
              <Suspense fallback={<PageLoader />}>
                <ArchitectureGuide />
              </Suspense>
              <Toaster />
            </div>
          </TooltipProvider>
        </ThemeProvider>
      </QueryClientProvider>
    );
  }

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <MLConnectionProvider>
            <VoiceCacheProvider>
              <DeviceProvider>
                <AppShell />
              </DeviceProvider>
            </VoiceCacheProvider>
          </MLConnectionProvider>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
