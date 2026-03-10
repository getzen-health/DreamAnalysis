import { useState } from "react";
import { Link, useLocation } from "wouter";
import {
  Brain,
  Sun,
  Moon,
  Settings,
  Menu,
  Wind,
  Utensils,
  BedDouble,
  LogOut,
  LayoutDashboard,
  Activity,
  Network,
  Sparkles,
  Radio,
  SlidersHorizontal,
  Lightbulb,
  BarChart2,
  History,
  MessageCircle,
  Leaf,
  CalendarDays,
  Bluetooth,
  Trophy,
  Pill,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks/use-mobile";
import { useDevice } from "@/hooks/use-device";
import { DeviceConnection } from "@/components/device-connection";
import { useTheme } from "@/hooks/use-theme";
import { useAuth } from "@/hooks/use-auth";
import { useMLConnection } from "@/hooks/use-ml-connection";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

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
    title: "Voice & Watch",
    items: [
      { path: "/",               label: "Dashboard",      icon: LayoutDashboard },
      { path: "/brain-report",   label: "Daily Report",   icon: Sun },
      { path: "/weekly-summary", label: "Weekly Summary", icon: CalendarDays },
    ],
  },
  {
    title: "Mind & Recovery",
    items: [
      { path: "/emotional-intelligence", label: "EI Dashboard",   icon: Brain },
      { path: "/emotions",        label: "Emotions",         icon: Brain },
      { path: "/insights",        label: "Insights",         icon: Lightbulb },
      { path: "/health-analytics",label: "Health Analytics", icon: BarChart2 },
      { path: "/sessions",        label: "History",          icon: History },
    ],
  },
  {
    title: "Health & Life",
    items: [
      { path: "/food",          label: "Food & Mood",   icon: Utensils },
      { path: "/supplements",   label: "Supplements",   icon: Pill },
      { path: "/dreams",        label: "Dreams",        icon: Moon },
      { path: "/sleep-session", label: "Sleep",         icon: BedDouble },
      { path: "/biofeedback",   label: "Breathe",       icon: Wind },
    ],
  },
  {
    title: "Support",
    items: [
      { path: "/ai-companion", label: "AI Companion", icon: MessageCircle },
      { path: "/records",      label: "My Records",   icon: Trophy },
    ],
  },
  {
    title: "Add EEG Later",
    items: [
      { path: "/brain-monitor",      label: "Brain Monitor",   icon: Activity },
      { path: "/brain-connectivity", label: "Connectivity",    icon: Network },
      { path: "/inner-energy",       label: "Inner Energy",    icon: Sparkles },
      { path: "/neurofeedback",      label: "Neurofeedback",   icon: Radio },
      { path: "/calibration",        label: "EEG Calibration", icon: SlidersHorizontal },
      { path: "/device-setup",       label: "EEG Setup",       icon: Bluetooth },
    ],
  },
];

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const [deviceModalOpen, setDeviceModalOpen] = useState(false);
  const isMobile = useIsMobile();
  const [location] = useLocation();
  const device = useDevice();
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const { user, logout } = useAuth();

  const isConnected =
    device.state === "streaming" || device.state === "connected";

  const { status: mlStatus, latencyMs, reconnect: mlReconnect } = useMLConnection();

  const mlDotColor =
    mlStatus === "ready"   ? "bg-green-500" :
    mlStatus === "error"   ? "bg-red-500" :
                             "bg-amber-500 animate-pulse";

  const mlLabel =
    mlStatus === "ready"   ? `ML: Connected${latencyMs ? ` (${latencyMs}ms)` : ""}` :
    mlStatus === "error"   ? "ML: Unreachable — click to retry" :
                             "ML: Warming up...";

  return (
    <>
      {isMobile && (
        <Button
          variant="ghost"
          size="icon"
          // top-[calc(16px+env(safe-area-inset-top))] pushes below iOS notch/Dynamic Island
          className="fixed z-50 w-11 h-11"
          style={{
            top: "calc(12px + env(safe-area-inset-top, 0px))",
            left: "calc(12px + env(safe-area-inset-left, 0px))",
          }}
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
          background: isDark ? "hsl(222, 25%, 5%)" : "hsl(0, 0%, 100%)",
          borderColor: isDark ? "hsl(220, 20%, 10%)" : "hsl(220, 14%, 90%)",
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
            {sections.map((section, si) => (
              <div key={section.title || "main"} className={si > 0 ? "mt-1" : ""}>
                {section.title && (
                  <div className="px-3 pt-3 pb-1 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/40">
                    {section.title}
                  </div>
                )}
                {section.items.map((item) => {
                  const Icon = item.icon;
                  const isActive =
                    location === item.path ||
                    (item.path !== "/" && location.startsWith(item.path + "/"));

                  return (
                    <Link
                      key={item.path}
                      href={item.path}
                      onClick={() => isMobile && setIsOpen(false)}
                      className={`flex items-center px-3 py-2.5 rounded-lg text-[13px] transition-all mb-0.5 min-h-[44px] ${
                        isActive
                          ? "text-primary font-medium"
                          : isDark
                            ? "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                            : "text-muted-foreground hover:text-foreground/70 hover:bg-muted/40"
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
                className={`flex items-center px-3 py-2.5 rounded-lg text-[13px] transition-all min-h-[44px] ${
                  location === "/settings"
                    ? "text-primary font-medium"
                    : isDark
                      ? "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                      : "text-muted-foreground hover:text-foreground/70 hover:bg-muted/40"
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
            className="mx-3 mb-3 p-3 rounded-xl text-left transition-colors cursor-pointer hover:bg-muted/30"
            style={{
              background: isDark ? "hsl(220, 22%, 8%)" : "hsl(220, 14%, 96%)",
              border: isDark ? "1px solid hsl(220, 18%, 13%)" : "1px solid hsl(220, 14%, 88%)",
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
                  ? (device.selectedDevice === "synthetic" ? "Demo Mode" : "Streaming")
                  : isConnected
                    ? "Connected"
                    : "EEG Optional"}
              </span>
            </div>
            {device.deviceStatus && isConnected && (
              <p className="text-[10px] text-muted-foreground/60 mt-1 ml-4">
                {device.deviceStatus.n_channels}ch | {device.deviceStatus.sample_rate}Hz
              </p>
            )}
          </button>

          {/* ML Connection Status */}
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                className="flex items-center gap-2 px-3 py-1.5 rounded-md text-xs text-muted-foreground hover:text-foreground transition-colors w-full"
                onClick={mlStatus === "error" ? mlReconnect : undefined}
              >
                <span className={`w-2 h-2 rounded-full shrink-0 ${mlDotColor}`} />
                <span>
                  {mlStatus === "ready" ? "ML Ready" :
                   mlStatus === "error" ? "ML Offline" :
                                          "ML Starting"}
                </span>
              </button>
            </TooltipTrigger>
            <TooltipContent side="right">
              <p className="text-xs">{mlLabel}</p>
            </TooltipContent>
          </Tooltip>

          {/* User avatar + logout — padded above home indicator bar */}
          {user && (
            <div
              className="mx-3 px-3 py-2 rounded-xl flex items-center gap-2"
              style={{
                marginBottom: "calc(1rem + env(safe-area-inset-bottom, 0px))",
                background: isDark ? "hsl(220, 22%, 7%)" : "hsl(220, 14%, 95%)",
                border: isDark ? "1px solid hsl(220, 18%, 12%)" : "1px solid hsl(220, 14%, 87%)",
              }}
            >
              <div
                className="w-7 h-7 rounded-full flex items-center justify-center shrink-0 text-[11px] font-semibold text-white"
                style={{ background: "linear-gradient(135deg, hsl(152,60%,40%), hsl(38,85%,50%))" }}
              >
                {user.username.charAt(0).toUpperCase()}
              </div>
              <span className="text-xs text-foreground/70 truncate flex-1">
                {user.username}
              </span>
              <button
                onClick={() => logout()}
                className="shrink-0 text-muted-foreground/50 hover:text-muted-foreground transition-colors"
                title="Sign out"
              >
                <LogOut className="h-3.5 w-3.5" />
              </button>
            </div>
          )}
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
