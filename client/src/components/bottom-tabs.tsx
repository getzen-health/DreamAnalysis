import { Link, useLocation } from "wouter";
import { Home, Activity, CircleUser } from "lucide-react";
import { hapticLight } from "@/lib/haptics";

const tabs = [
  { path: "/", icon: Home, label: "Home" },
  { path: "/sessions", icon: Activity, label: "Activity" },
  { path: "/settings", icon: CircleUser, label: "Profile" },
];

export function BottomTabs() {
  const [location] = useLocation();

  return (
    <nav
      aria-label="Main navigation"
      className="fixed bottom-0 left-0 right-0 z-40 md:hidden border-t"
      style={{
        background: "hsl(222, 25%, 6%, 0.92)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        borderColor: "hsl(220, 18%, 15%, 0.5)",
        paddingBottom: "env(safe-area-inset-bottom, 0px)",
      }}
    >
      <div className="flex items-center justify-around" style={{ height: "56px" }}>
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
              className={`flex flex-col items-center justify-center gap-1 min-w-[48px] min-h-[44px] px-3 py-1.5 rounded-lg transition-colors ${
                isActive
                  ? "text-primary"
                  : "text-muted-foreground/60 active:text-muted-foreground"
              }`}
            >
              <Icon
                className={`h-[22px] w-[22px] ${isActive ? "text-primary" : ""}`}
                aria-hidden="true"
                strokeWidth={isActive ? 2.5 : 1.75}
              />
              <span className={`text-[11px] leading-tight ${isActive ? "font-semibold" : "font-normal"}`}>
                {tab.label}
              </span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
