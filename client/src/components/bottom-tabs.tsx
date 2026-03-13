import { Link, useLocation } from "wouter";
import {
  Sun,
  BookOpen,
  TrendingUp,
  CircleUser,
} from "lucide-react";
import { hapticLight } from "@/lib/haptics";

const tabs = [
  { path: "/",        icon: Sun,        label: "Today"   },
  { path: "/journal", icon: BookOpen,   label: "Journal" },
  { path: "/trends",  icon: TrendingUp, label: "Trends"  },
  { path: "/you",     icon: CircleUser, label: "You"     },
];

export function BottomTabs() {
  const [location] = useLocation();

  return (
    <nav
      role="navigation"
      aria-label="Main navigation"
      className="fixed bottom-0 left-0 right-0 z-40 md:hidden border-t bg-background/95 border-border/60"
      style={{
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        paddingBottom: "env(safe-area-inset-bottom, 0px)",
      }}
    >
      <div className="flex items-stretch justify-around" style={{ height: "56px" }}>
        {tabs.map((tab) => {
          const Icon = tab.icon;
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
              className="relative flex flex-col items-center justify-center gap-0.5 flex-1 py-1.5 transition-all active:scale-95"
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
                style={{ width: 24, height: 24 }}
                aria-hidden="true"
                strokeWidth={isActive ? 2.5 : 1.75}
                fill={isActive ? "currentColor" : "none"}
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
