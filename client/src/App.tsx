import { lazy, Suspense, Component, useState, type ReactNode, type ErrorInfo } from "react";
import { Switch, Route, useLocation } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "./hooks/use-theme";
import { AuthProvider, useAuth } from "./hooks/use-auth";
import { DeviceProvider } from "./hooks/use-device";
import { MLConnectionProvider } from "@/hooks/use-ml-connection";
import { MLWarmupScreen } from "@/components/ml-warmup-screen";
import { NeuralBackground } from "@/components/neural-background";
import AppLayout from "./layouts/app-layout";

// ── Core journey pages — static imports (always needed on first load) ──────
import Landing from "@/pages/landing";
import AuthPage from "@/pages/auth";
import ForgotPasswordPage from "@/pages/forgot-password";
import ResetPasswordPage from "@/pages/reset-password";
import Dashboard from "@/pages/dashboard";
import EmotionLab from "@/pages/emotion-lab";
import BrainMonitor from "@/pages/brain-monitor";
import DreamJournal from "@/pages/dream-journal";
import AICompanionPage from "@/pages/ai-companion";
import Biofeedback from "@/pages/biofeedback";
import SessionHistory from "@/pages/session-history";
import SettingsPage from "@/pages/settings";
import CalibrationPage from "@/pages/calibration";
import DailyBrainReport from "@/pages/daily-brain-report";
import Onboarding from "@/pages/onboarding";
import OnboardingNew from "@/pages/onboarding-new";
import NotFound from "@/pages/not-found";

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

// ── Error Boundary — prevents a single page crash from taking down the whole app ──
class ErrorBoundary extends Component<
  { children: ReactNode; fallback?: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode; fallback?: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }
  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div className="min-h-[60vh] flex flex-col items-center justify-center gap-4 text-muted-foreground text-sm p-8">
          <p>Something went wrong on this page.</p>
          <button
            className="px-4 py-2 rounded-md bg-primary text-primary-foreground text-xs"
            onClick={() => { this.setState({ hasError: false, error: null }); window.location.reload(); }}
          >
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// Minimal fallback shown while a lazy chunk loads
function PageLoader() {
  return (
    <div className="min-h-[60vh] flex items-center justify-center text-muted-foreground text-sm">
      Loading…
    </div>
  );
}

// Redirects unauthenticated users to /auth
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();
  const [, setLocation] = useLocation();
  if (isLoading) return <PageLoader />;
  if (!user) {
    setLocation("/auth");
    return null;
  }
  return <>{children}</>;
}

function AppRoutes() {
  const [location] = useLocation();
  return (
    <ErrorBoundary key={location}>
    <Suspense fallback={<PageLoader />}>
    <Switch>
      <Route path="/welcome" component={Landing} />
      <Route path="/auth" component={AuthPage} />
      <Route path="/forgot-password" component={ForgotPasswordPage} />
      <Route path="/reset-password" component={ResetPasswordPage} />
      <Route path="/onboarding-new" component={OnboardingNew} />
      <Route path="/architecture-guide">
        <AppLayout><ArchitectureGuide /></AppLayout>
      </Route>
      <Route path="/">
        <ProtectedRoute><AppLayout><Dashboard /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/emotions">
        <ProtectedRoute><AppLayout><EmotionLab /></AppLayout></ProtectedRoute>
      </Route>
      <Route path="/brain-monitor">
        <ProtectedRoute><AppLayout><BrainMonitor /></AppLayout></ProtectedRoute>
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
        <ProtectedRoute><AppLayout><FoodLog /></AppLayout></ProtectedRoute>
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
        <ProtectedRoute><AppLayout><FoodLog /></AppLayout></ProtectedRoute>
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
      <Route path="/onboarding" component={Onboarding} />
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
      {/* Public route — no auth required (needed for App Store / HealthKit) */}
      <Route path="/privacy">
        <Suspense fallback={<PageLoader />}>
          <AppLayout><PrivacyPolicy /></AppLayout>
        </Suspense>
      </Route>
      <Route component={NotFound} />
    </Switch>
    </Suspense>
    </ErrorBoundary>
  );
}

const PUBLIC_ROUTES = new Set(["/auth", "/forgot-password", "/reset-password", "/welcome", "/onboarding-new"]);

function App() {
  const [warmupDismissed, setWarmupDismissed] = useState(false);
  const [location] = useLocation();
  const pathname = typeof window !== "undefined" ? window.location.pathname : "/";
  const isPublicRoute = PUBLIC_ROUTES.has(location);

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
            <DeviceProvider>
              <TooltipProvider>
                {!warmupDismissed && !isPublicRoute && (
                  <MLWarmupScreen onSimulationMode={() => setWarmupDismissed(true)} />
                )}
                <AppRoutes />
                <Toaster />
              </TooltipProvider>
            </DeviceProvider>
          </MLConnectionProvider>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
