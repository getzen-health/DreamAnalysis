import { useState } from "react";
import { Link, useLocation } from "wouter";
import {
  Brain,
  Sun,
  Moon,
  BarChart3,
  MessageSquare,
  Settings,
  Menu,
  Sparkles,
  Wind,
  Utensils,
  FlaskConical,
  Radio,
  BedDouble,
  LogOut,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks/use-mobile";
import { useDevice } from "@/hooks/use-device";
import { DeviceConnection } from "@/components/device-connection";
import { useTheme } from "@/hooks/use-theme";
import { useAuth } from "@/hooks/use-auth";

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
    title: "Today",
    items: [
      { path: "/brain-report",    label: "Brain Report",   icon: Sun },
      { path: "/weekly-summary", label: "Week in Review", icon: BarChart3 },
      { path: "/",               label: "Brain State",    icon: Brain },
    ],
  },
  {
    title: "States",
    items: [
      { path: "/sleep-session", label: "Sleep",         icon: BedDouble },
      { path: "/dreams",        label: "Dreams",        icon: Moon },
      { path: "/inner-energy",  label: "Spiritual",     icon: Sparkles },
      { path: "/food",          label: "Food",          icon: Utensils },
      { path: "/biofeedback",   label: "Breathe",       icon: Wind },
    ],
  },
  {
    title: "Research",
    items: [
      { path: "/research",      label: "Study Hub",     icon: FlaskConical },
    ],
  },
  {
    title: "Tools",
    items: [
      { path: "/device-setup",  label: "Connect Device", icon: Radio },
      { path: "/ai-companion",  label: "AI Companion",   icon: MessageSquare },
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
