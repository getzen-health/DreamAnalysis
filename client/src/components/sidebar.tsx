import { useState } from "react";
import { Link, useLocation } from "wouter";
import {
  Brain,
  Heart,
  Sparkles,
  Clock,
  Settings,
  Menu,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useIsMobile } from "@/hooks/use-mobile";
import { useDevice } from "@/hooks/use-device";
import { DeviceConnection } from "@/components/device-connection";

const navigationItems = [
  { path: "/", label: "Dashboard", icon: Brain },
  { path: "/emotions", label: "Emotions", icon: Heart },
  { path: "/inner-energy", label: "Inner Energy", icon: Sparkles },
  { path: "/sessions", label: "Sessions", icon: Clock },
  { path: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(false);
  const [deviceModalOpen, setDeviceModalOpen] = useState(false);
  const isMobile = useIsMobile();
  const [location] = useLocation();
  const device = useDevice();

  const isConnected = device.state === "streaming" || device.state === "connected";

  return (
    <>
      {/* Mobile Menu Toggle */}
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

      {/* Sidebar */}
      <div
        className={`fixed left-0 top-0 w-56 h-screen bg-sidebar border-r border-border z-40 transition-transform duration-200 ${
          isMobile
            ? isOpen
              ? "translate-x-0"
              : "-translate-x-full"
            : "translate-x-0"
        }`}
      >
        <div className="p-5 flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center mb-8">
            <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center mr-3">
              <Brain className="h-4 w-4 text-primary" />
            </div>
            <span className="text-base font-semibold text-foreground">
              NeuralDream
            </span>
          </div>

          {/* Navigation */}
          <nav className="space-y-1 flex-1">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const isActive =
                location === item.path ||
                (item.path !== "/" && location.startsWith(item.path));

              return (
                <Link
                  key={item.path}
                  href={item.path}
                  onClick={() => isMobile && setIsOpen(false)}
                  className={`flex items-center px-3 py-2.5 rounded-lg text-sm transition-colors ${
                    isActive
                      ? "bg-primary/10 text-primary font-medium"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                >
                  <Icon className="mr-3 h-4 w-4" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* BCI Status */}
          <button
            onClick={() => setDeviceModalOpen(true)}
            className="mt-auto pt-4 border-t border-border text-left hover:bg-muted/50 rounded-lg p-2 -mx-1 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  isConnected ? "bg-success" : "bg-muted-foreground/40"
                }`}
              />
              <span className="text-xs text-muted-foreground">
                {isConnected ? "BCI Connected" : "No Device"}
              </span>
            </div>
          </button>
        </div>
      </div>

      {/* Mobile Overlay */}
      {isMobile && isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30"
          onClick={() => setIsOpen(false)}
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
