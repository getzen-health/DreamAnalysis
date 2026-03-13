import { Link, useLocation } from "wouter";
import {
  LayoutDashboard,
  Brain,
  Heart,
  Smile,
  CircleUser,
} from "lucide-react";
import { hapticLight } from "@/lib/haptics";

const tabs = [
  { path: "/",              icon: LayoutDashboard, label: "Home" },
  { path: "/brain-monitor", icon: Brain,           label: "Brain" },
  { path: "/emotions",      icon: Smile,           label: "Emotions" },
  { path: "/health-analytics", icon: Heart,        label: "Health" },
  { path: "/settings",      icon: CircleUser,      label: "Profile" },
];

export function BottomTabs() {
  const [location] = useLocation();

  return (
    <nav
      aria-label="Main navigation"
      className="fixed bottom-0 left-0 right-0 z-40 md:hidden border-t"
      style={{
        background: "hsl(222, 25%, 5%, 0.95)",
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        borderColor: "hsl(220, 18%, 13%, 0.6)",
        paddingBottom: "env(safe-area-inset-bottom, 0px)",
      }}
    >
      <div className="flex items-stretch justify-around" style={{ height: "52px" }}>
        {tabs.map((tab) => {
          const Icon = tab.icon;
          // Active if exact match, or if path starts with tab path (and tab path isn't "/")
          const isActive =
            location === tab.path ||
            (tab.path !== "/" && location.startsWith(tab.path));

          return (
            <Link
              key={tab.path}
              href={tab.path}
              onClick={() => hapticLight()}
              aria-current={isActive ? "page" : undefined}
              aria-label={tab.label}
              className="relative flex flex-col items-center justify-center gap-0.5 flex-1 py-1.5 transition-colors"
            >
              {/* Active indicator pill above icon */}
              {isActive && (
                <span
                  className="absolute top-0 left-1/2 -translate-x-1/2 rounded-full bg-primary"
                  style={{ width: 24, height: 2, borderRadius: 1 }}
                />
              )}

              <Icon
                className={`transition-colors ${
                  isActive
                    ? "text-primary"
                    : "text-muted-foreground/55"
                }`}
                style={{ width: 22, height: 22 }}
                aria-hidden="true"
                strokeWidth={isActive ? 2.25 : 1.75}
              />
              <span
                className={`text-[10px] leading-none tracking-tight transition-colors ${
                  isActive
                    ? "text-primary font-semibold"
                    : "text-muted-foreground/55 font-normal"
                }`}
              >
                {tab.label}
              </span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
