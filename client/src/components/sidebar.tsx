import { useState } from "react";
import { Link, useLocation } from "wouter";
import {
  Brain,
  Heart,
  Sparkles,
  Activity,
  Moon,
  BarChart2,
  Network,
  HeartPulse,
  Headphones,
  MessageSquare,
  Clock,
  Settings,
  Menu,
  Lightbulb,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks/use-mobile";
import { useDevice } from "@/hooks/use-device";
import { DeviceConnection } from "@/components/device-connection";

interface NavItem {
  path: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const sections: NavSection[] = [
  {
    title: "Core",
    items: [
      { path: "/", label: "Dashboard", icon: Brain },
      { path: "/emotions", label: "Emotions", icon: Heart },
      { path: "/inner-energy", label: "Inner Energy", icon: Sparkles },
    ],
  },
  {
    title: "Analysis",
    items: [
      { path: "/brain-monitor", label: "Brain Monitor", icon: Activity },
      { path: "/dreams", label: "Dream Detection", icon: Moon },
      { path: "/dream-patterns", label: "Dream Patterns", icon: BarChart2 },
      { path: "/brain-connectivity", label: "Connectivity", icon: Network },
      { path: "/health-analytics", label: "Health", icon: HeartPulse },
    ],
  },
  {
    title: "Tools",
    items: [
      { path: "/neurofeedback", label: "Neurofeedback", icon: Headphones },
      { path: "/ai-companion", label: "AI Companion", icon: MessageSquare },
      { path: "/insights", label: "Insights", icon: Lightbulb },
      { path: "/sessions", label: "Sessions", icon: Clock },
    ],
  },
];

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const [deviceModalOpen, setDeviceModalOpen] = useState(false);
  const isMobile = useIsMobile();
  const [location] = useLocation();
  const device = useDevice();

  const isConnected =
    device.state === "streaming" || device.state === "connected";

  return (
    <>
      {isMobile && (
        <Button
          variant="ghost"
          size="icon"
          className="fixed top-4 left-4 z-50"
          onClick={() => setIsOpen(!isOpen)}
        >
          <Menu className="h-5 w-5" />
        </Button>
      )}

      <div
        className={`fixed left-0 top-0 w-56 h-screen z-40 transition-transform duration-200 border-r ${
          isMobile
            ? isOpen
              ? "translate-x-0"
              : "-translate-x-full"
            : "translate-x-0"
        }`}
        style={{
          background: "hsl(222, 25%, 5%)",
          borderColor: "hsl(220, 20%, 10%)",
        }}
      >
        <div className="flex flex-col h-full overflow-y-auto">
          {/* Logo */}
          <div className="flex items-center px-5 pt-5 pb-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center mr-3"
              style={{ background: "linear-gradient(135deg, hsl(152,60%,48%), hsl(38,85%,58%))" }}>
              <Brain className="h-4 w-4 text-white" />
            </div>
            <span className="text-base font-semibold text-gradient">
              Svapnastra
            </span>
          </div>

          {/* Navigation Sections */}
          <nav className="flex-1 px-2 pb-4">
            {sections.map((section) => (
              <div key={section.title}>
                <div className="nav-section-label">{section.title}</div>
                {section.items.map((item) => {
                  const Icon = item.icon;
                  const isActive =
                    location === item.path ||
                    (item.path !== "/" && location.startsWith(item.path));

                  return (
                    <Link
                      key={item.path}
                      href={item.path}
                      onClick={() => isMobile && setIsOpen(false)}
                      className={`flex items-center px-3 py-2 rounded-lg text-[13px] transition-all mb-0.5 ${
                        isActive
                          ? "text-primary font-medium"
                          : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                      }`}
                      style={
                        isActive
                          ? {
                              background:
                                "linear-gradient(135deg, hsl(152,60%,48%,0.12), hsl(152,60%,48%,0.04))",
                              borderLeft: "2px solid hsl(152,60%,48%)",
                            }
                          : undefined
                      }
                    >
                      <Icon className="mr-3 h-4 w-4 shrink-0" />
                      <span>{item.label}</span>
                    </Link>
                  );
                })}
              </div>
            ))}

            {/* Settings (separate) */}
            <div className="mt-2 pt-2 border-t border-border/30">
              <Link
                href="/settings"
                onClick={() => isMobile && setIsOpen(false)}
                className={`flex items-center px-3 py-2 rounded-lg text-[13px] transition-all ${
                  location === "/settings"
                    ? "text-primary font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                }`}
                style={
                  location === "/settings"
                    ? {
                        background:
                          "linear-gradient(135deg, hsl(152,60%,48%,0.12), hsl(152,60%,48%,0.04))",
                        borderLeft: "2px solid hsl(152,60%,48%)",
                      }
                    : undefined
                }
              >
                <Settings className="mr-3 h-4 w-4 shrink-0" />
                <span>Settings</span>
              </Link>
            </div>
          </nav>

          {/* BCI Status */}
          <button
            onClick={() => setDeviceModalOpen(true)}
            className="mx-3 mb-4 p-3 rounded-xl text-left transition-colors cursor-pointer hover:bg-muted/30"
            style={{
              background: "hsl(220, 22%, 8%)",
              border: "1px solid hsl(220, 18%, 13%)",
            }}
          >
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full shrink-0 ${
                  isConnected ? "bg-success" : "bg-muted-foreground/40"
                }`}
                style={isConnected ? { boxShadow: "0 0 6px hsl(152,60%,48%)" } : undefined}
              />
              <span className="text-xs text-muted-foreground">
                {device.state === "streaming"
                  ? "Streaming"
                  : isConnected
                    ? "Connected"
                    : "Connect Device"}
              </span>
            </div>
            {device.deviceStatus && isConnected && (
              <p className="text-[10px] text-muted-foreground/60 mt-1 ml-4">
                {device.deviceStatus.n_channels}ch | {device.deviceStatus.sample_rate}Hz
              </p>
            )}
          </button>
        </div>
      </div>

      {isMobile && isOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-30 backdrop-blur-sm"
          onClick={() => setIsOpen(false)}
        />
      )}

      <DeviceConnection
        open={deviceModalOpen}
        onOpenChange={setDeviceModalOpen}
        device={device}
      />
    </>
  );
}
