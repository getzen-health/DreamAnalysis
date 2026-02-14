import { useState } from "react";
import { Link, useLocation } from "wouter";
import {
  Brain,
  ChartLine,
  Activity,
  BarChart3,
  Moon,
  Bot,
  Settings,
  Menu,
  Zap,
  TrendingUp,
  Sparkles,
  Headphones,
  Clock,
  Network,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks/use-mobile";
import { useDevice } from "@/hooks/use-device";
import { DeviceConnection } from "@/components/device-connection";

const navigationItems = [
  { path: "/", label: "Dashboard", icon: ChartLine },
  { path: "/brain-monitor", label: "Real-Time Monitor", icon: Activity },
  { path: "/health-analytics", label: "Health Analytics", icon: BarChart3 },
  { path: "/dream-journal", label: "Dream Journal", icon: Moon },
  { path: "/dream-patterns", label: "Dream Patterns", icon: TrendingUp },
  { path: "/emotion-lab", label: "Emotion Lab", icon: Zap },
  { path: "/neurofeedback", label: "Neurofeedback", icon: Headphones },
  { path: "/brain-connectivity", label: "Connectivity", icon: Network },
  { path: "/sessions", label: "Sessions", icon: Clock },
  { path: "/ai-companion", label: "AI Companion", icon: Bot },
  { path: "/insights", label: "Insights", icon: Sparkles },
  { path: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const [deviceModalOpen, setDeviceModalOpen] = useState(false);
  const isMobile = useIsMobile();
  const [location] = useLocation();
  const device = useDevice();

  const statusLabel =
    device.state === "streaming"
      ? "STREAMING"
      : device.state === "connected"
        ? "CONNECTED"
        : device.state === "connecting"
          ? "CONNECTING..."
          : "DISCONNECTED";

  const statusColor =
    device.state === "streaming"
      ? "text-primary"
      : device.state === "connected"
        ? "text-success"
        : device.state === "connecting"
          ? "text-warning"
          : "text-foreground/50";

  return (
    <>
      {/* Mobile Menu Toggle */}
      {isMobile && (
        <Button
          variant="outline"
          size="icon"
          className="fixed top-4 left-4 z-50 glass-card border-primary/20 hover-glow"
          onClick={() => setIsOpen(!isOpen)}
          data-testid="button-mobile-menu"
        >
          <Menu className="h-5 w-5 text-primary" />
        </Button>
      )}

      {/* Sidebar */}
      <div
        className={`fixed left-0 top-0 w-64 h-screen glass-card border-r border-primary/20 z-40 transition-transform duration-300 ${
          isMobile
            ? isOpen
              ? "translate-x-0"
              : "-translate-x-full"
            : "translate-x-0"
        }`}
        data-testid="sidebar-navigation"
      >
        <div className="p-6">
          {/* Logo */}
          <div className="flex items-center mb-8">
            <div className="w-10 h-10 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center mr-3">
              <Brain className="text-white text-lg" />
            </div>
            <div>
              <h1 className="font-futuristic text-lg font-bold text-gradient">
                Neural Dream
              </h1>
              <p className="text-xs text-secondary">Weaver v2.1</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="space-y-2">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const isActive =
                location === item.path ||
                (item.path !== "/" && location.startsWith(item.path));

              return (
                <Link
                  key={item.path}
                  href={item.path}
                  onClick={() => {
                    if (isMobile) {
                      setIsOpen(false);
                    }
                  }}
                  className={`w-full flex items-center px-4 py-3 rounded-lg transition-all ${
                    isActive
                      ? "bg-primary/10 text-primary border border-primary/30 hover-glow"
                      : "text-foreground/70 hover:bg-card hover:text-foreground"
                  }`}
                  data-testid={`nav-${item.path === "/" ? "dashboard" : item.path.slice(1)}`}
                >
                  <Icon className="mr-3 h-5 w-5" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* BCI Status */}
          <div className="mt-8 pt-6 border-t border-border">
            <button
              onClick={() => setDeviceModalOpen(true)}
              className="w-full text-left hover:bg-card/50 rounded-lg p-2 -m-2 transition-colors cursor-pointer"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-foreground/70">BCI Status</span>
                <div
                  className={`w-2 h-2 rounded-full ${
                    device.state === "streaming"
                      ? "bg-primary animate-pulse"
                      : device.state === "connected"
                        ? "bg-success"
                        : device.state === "connecting"
                          ? "bg-warning animate-pulse"
                          : "bg-foreground/30"
                  }`}
                  data-testid="status-indicator"
                ></div>
              </div>
              <p className={`text-xs font-mono ${statusColor}`}>
                {statusLabel}
              </p>
              {device.deviceStatus && device.state !== "disconnected" && (
                <p className="text-xs text-foreground/40 mt-1">
                  {device.deviceStatus.n_channels}ch | {device.deviceStatus.sample_rate}Hz
                </p>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Overlay */}
      {isMobile && isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30"
          onClick={() => setIsOpen(false)}
          data-testid="sidebar-overlay"
        />
      )}

      {/* Device Connection Modal */}
      <DeviceConnection
        open={deviceModalOpen}
        onOpenChange={setDeviceModalOpen}
        device={device}
      />
    </>
  );
}
