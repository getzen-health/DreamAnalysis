import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "./hooks/use-theme";
import { AuthProvider } from "./hooks/use-auth";
import AppLayout from "./layouts/app-layout";
import Landing from "@/pages/landing";
import AuthPage from "@/pages/auth";
import Dashboard from "@/pages/dashboard";
import EmotionLab from "@/pages/emotion-lab";
import InnerEnergy from "@/pages/inner-energy";
import SessionHistory from "@/pages/session-history";
import SettingsPage from "@/pages/settings";
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
      <Route path="/inner-energy">
        <AppLayout><InnerEnergy /></AppLayout>
      </Route>
      <Route path="/sessions">
        <AppLayout><SessionHistory /></AppLayout>
      </Route>
      <Route path="/settings">
        <AppLayout><SettingsPage /></AppLayout>
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
          <TooltipProvider>
            <Toaster />
            <AppRoutes />
          </TooltipProvider>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
