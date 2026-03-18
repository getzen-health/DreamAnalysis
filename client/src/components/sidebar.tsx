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
  CalendarDays,
  Bluetooth,
  Trophy,
  Pill,
  Scale,
  Dumbbell,
  ChevronDown,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks/use-mobile";
import { useDevice } from "@/hooks/use-device";
import { DeviceConnection } from "@/components/device-connection";
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

// Always visible — core features for every user
const coreSections: NavSection[] = [
  {
    title: "",
    items: [
      { path: "/",               label: "Dashboard",      icon: LayoutDashboard },
      { path: "/brain-report",   label: "Daily Report",   icon: Sun },
      { path: "/emotions",       label: "Emotions",       icon: Brain },
      { path: "/food",           label: "Food & Mood",    icon: Utensils },
      { path: "/dreams",         label: "Dreams",         icon: Moon },
      { path: "/biofeedback",    label: "Breathe",        icon: Wind },
      { path: "/ai-companion",   label: "AI Companion",   icon: MessageCircle },
    ],
  },
];

// Collapsed by default — analytics & extras
const moreSections: NavSection[] = [
  {
    title: "More",
    items: [
      { path: "/weekly-summary",        label: "Weekly Summary",   icon: CalendarDays },
      { path: "/inner-energy",          label: "Inner Energy",     icon: Sparkles },
      { path: "/emotional-intelligence", label: "EI Dashboard",    icon: Brain },
      { path: "/insights",              label: "Insights",         icon: Lightbulb },
      { path: "/health-analytics",      label: "Health Analytics", icon: BarChart2 },
      { path: "/body-metrics",          label: "Body Metrics",     icon: Scale },
      { path: "/workout",              label: "Strength",         icon: Dumbbell },
      { path: "/sessions",              label: "History",          icon: History },
      { path: "/supplements",           label: "Supplements",      icon: Pill },
      { path: "/sleep-session",         label: "Sleep",            icon: BedDouble },
      { path: "/records",               label: "My Records",       icon: Trophy },
    ],
  },
];

// Only visible when EEG device is connected
const eegSections: NavSection[] = [
  {
    title: "EEG",
    items: [
      { path: "/brain-monitor",      label: "Brain Monitor",   icon: Activity },
      { path: "/brain-connectivity", label: "Connectivity",    icon: Network },
      { path: "/neurofeedback",      label: "Neurofeedback",   icon: Radio },
      { path: "/calibration",        label: "EEG Calibration", icon: SlidersHorizontal },
      { path: "/device-setup",       label: "EEG Setup",       icon: Bluetooth },
    ],
  },
];

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);
  const [deviceModalOpen, setDeviceModalOpen] = useState(false);
  const isMobile = useIsMobile();
  const [location] = useLocation();
  const device = useDevice();
  const { user, logout } = useAuth();

  const isConnected =
    device.state === "streaming" || device.state === "connected";

  // Auto-expand "More" when user is on one of those pages
  const moreVisible = moreOpen || moreSections[0].items.some(
    (item) => location === item.path || (item.path !== "/" && location.startsWith(item.path + "/"))
  );
  // Auto-expand EEG when user is on one of those pages
  const eegVisible = isConnected || eegSections[0].items.some(
    (item) => location === item.path || (item.path !== "/" && location.startsWith(item.path + "/"))
  );

  // Build visible sections
  const visibleSections = [
    ...coreSections,
    ...(moreVisible ? moreSections : []),
    ...(eegVisible ? eegSections : []),
  ];

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
      {/* Hamburger button — desktop only. Mobile uses bottom tabs. */}
      {!isMobile && (
        <Button
          variant="ghost"
          size="icon"
          className="fixed z-50 min-w-[44px] min-h-[44px] w-11 h-11 rounded-xl bg-background/80 backdrop-blur-md border border-border/40 shadow-lg"
          style={{
            top: "calc(12px + env(safe-area-inset-top, 0px))",
            left: "calc(12px + env(safe-area-inset-left, 0px))",
          }}
          onClick={() => setIsOpen(!isOpen)}
          aria-label={isOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={isOpen}
        >
          <Menu className="h-6 w-6" aria-hidden="true" />
        </Button>
      )}

      <div
        className={`fixed left-0 top-0 w-64 h-screen z-40 transition-transform duration-200 ease-out border-r ${
          isMobile
            ? "-translate-x-full"
            : "translate-x-0 w-56"
        }`}
        style={{
          background: "hsl(var(--background) / 0.97)",
          backdropFilter: isMobile ? "blur(20px)" : undefined,
          WebkitBackdropFilter: isMobile ? "blur(20px)" : undefined,
          borderColor: "hsl(var(--border))",
        }}
      >
        <div className="flex flex-col h-full overflow-y-auto">
          {/* Logo */}
          <div
            className="flex items-center px-5 pb-3"
            style={{ paddingTop: isMobile ? "calc(20px + env(safe-area-inset-top, 0px))" : "20px" }}
          >
            <div className="w-8 h-8 rounded-lg flex items-center justify-center mr-3"
              style={{ background: "linear-gradient(135deg, hsl(152,60%,48%), hsl(38,85%,58%))" }}>
              <Brain className="h-4 w-4 text-white" />
            </div>
            <span className="text-base font-semibold text-gradient">
              AntarAI
            </span>
          </div>

          {/* Navigation Sections */}
          <nav aria-label="Sidebar navigation" className="flex-1 px-2 pb-4">
            {visibleSections.map((section, si) => (
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
                          : "text-muted-foreground hover:text-foreground hover:bg-muted/40"
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

            {/* "More" toggle — only show when More section is collapsed */}
            {!moreVisible && (
              <button
                onClick={() => setMoreOpen(true)}
                className="flex items-center px-3 py-2.5 rounded-lg text-[13px] w-full mt-1 min-h-[44px] text-muted-foreground/60 hover:text-muted-foreground hover:bg-muted/30 transition-colors"
              >
                <ChevronDown className="mr-3 h-4 w-4 shrink-0" />
                <span>More</span>
              </button>
            )}

            {/* Settings (separate) */}
            <div className="mt-2 pt-2 border-t border-border/30">
              <Link
                href="/settings"
                onClick={() => isMobile && setIsOpen(false)}
                className={`flex items-center px-3 py-2.5 rounded-lg text-[13px] transition-all min-h-[44px] ${
                  location === "/settings"
                    ? "text-primary font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/40"
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
              background: "hsl(var(--muted))",
              border: "1px solid hsl(var(--border))",
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
                background: "hsl(var(--muted))",
                border: "1px solid hsl(var(--border))",
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

      {/* Sidebar overlay — desktop only when sidebar is open over content */}
      {!isMobile && isOpen && (
        <div
          className="fixed inset-0 z-30 animate-in fade-in duration-200"
          style={{
            background: "rgba(0, 0, 0, 0.5)",
            backdropFilter: "blur(4px)",
            WebkitBackdropFilter: "blur(4px)",
          }}
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
