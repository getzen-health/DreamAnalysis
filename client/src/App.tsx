import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "./hooks/use-theme";
import { AuthProvider } from "./hooks/use-auth";
import { DeviceProvider } from "./hooks/use-device";
import AppLayout from "./layouts/app-layout";
import Landing from "@/pages/landing";
import AuthPage from "@/pages/auth";
import Dashboard from "@/pages/dashboard";
import EmotionLab from "@/pages/emotion-lab";
import BrainMonitor from "@/pages/brain-monitor";
import BrainConnectivity from "@/pages/brain-connectivity";
import DreamJournal from "@/pages/dream-journal";
import HealthAnalytics from "@/pages/health-analytics";
import Neurofeedback from "@/pages/neurofeedback";
import AICompanionPage from "@/pages/ai-companion";
import Insights from "@/pages/insights";
import InnerEnergy from "@/pages/inner-energy";
import Biofeedback from "@/pages/biofeedback";
import SessionHistory from "@/pages/session-history";
import SettingsPage from "@/pages/settings";
import CalibrationPage from "@/pages/calibration";
import FormalBenchmarksDashboard from "@/pages/formal-benchmarks-dashboard";
import FoodEmotion from "@/pages/food-emotion";
import DeviceSetup from "@/pages/device-setup";
import ResearchEnroll from "@/pages/research-enroll";
import ResearchMorning from "@/pages/research-morning";
import NotFound from "@/pages/not-found";

function AppRoutes() {
  return (
    <Switch>
      <Route path="/welcome" component={Landing} />
      <Route path="/auth" component={AuthPage} />
      <Route path="/">
        <AppLayout><Dashboard /></AppLayout>
      </Route>
      <Route path="/emotions">
        <AppLayout><EmotionLab /></AppLayout>
      </Route>
      <Route path="/brain-monitor">
        <AppLayout><BrainMonitor /></AppLayout>
      </Route>
      <Route path="/brain-connectivity">
        <AppLayout><BrainConnectivity /></AppLayout>
      </Route>
      <Route path="/dreams">
        <AppLayout><DreamJournal /></AppLayout>
      </Route>

      <Route path="/health-analytics">
        <AppLayout><HealthAnalytics /></AppLayout>
      </Route>
      <Route path="/neurofeedback">
        <AppLayout><Neurofeedback /></AppLayout>
      </Route>
      <Route path="/ai-companion">
        <AppLayout><AICompanionPage /></AppLayout>
      </Route>
      <Route path="/insights">
        <AppLayout><Insights /></AppLayout>
      </Route>
      <Route path="/inner-energy">
        <AppLayout><InnerEnergy /></AppLayout>
      </Route>
      <Route path="/biofeedback">
        <AppLayout><Biofeedback /></AppLayout>
      </Route>
      <Route path="/sessions">
        <AppLayout><SessionHistory /></AppLayout>
      </Route>
      <Route path="/settings">
        <AppLayout><SettingsPage /></AppLayout>
      </Route>
      <Route path="/calibration">
        <AppLayout><CalibrationPage /></AppLayout>
      </Route>
      <Route path="/benchmarks">
        <AppLayout><FormalBenchmarksDashboard /></AppLayout>
      </Route>
      <Route path="/food">
        <AppLayout><FoodEmotion /></AppLayout>
      </Route>
      <Route path="/device-setup">
        <AppLayout><DeviceSetup /></AppLayout>
      </Route>
      <Route path="/research/enroll">
        <AppLayout><ResearchEnroll /></AppLayout>
      </Route>
      <Route path="/research/morning">
        <AppLayout><ResearchMorning /></AppLayout>
      </Route>
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <DeviceProvider>
            <TooltipProvider>
              <Toaster />
              <AppRoutes />
            </TooltipProvider>
          </DeviceProvider>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
