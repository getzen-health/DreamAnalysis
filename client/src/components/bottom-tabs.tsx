import { Link, useLocation } from "wouter";
import { LayoutDashboard, Brain, Moon, MessageCircle, Settings } from "lucide-react";
import { hapticLight } from "@/lib/haptics";

const tabs = [
  { path: "/", icon: LayoutDashboard, label: "Home" },
  { path: "/emotions", icon: Brain, label: "Emotions" },
  { path: "/dreams", icon: Moon, label: "Dreams" },
  { path: "/ai-companion", icon: MessageCircle, label: "AI Chat" },
  { path: "/settings", icon: Settings, label: "Settings" },
];

export function BottomTabs() {
  const [location] = useLocation();

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-40 md:hidden border-t"
      style={{
        background: "hsl(222, 25%, 6%, 0.92)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderColor: "hsl(220, 18%, 15%, 0.5)",
        paddingBottom: "env(safe-area-inset-bottom, 0px)",
      }}
    >
      <div className="flex items-center justify-around h-14">
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
              className={`flex flex-col items-center justify-center gap-0.5 w-16 py-1 transition-colors ${
                isActive
                  ? "text-primary"
                  : "text-muted-foreground/60"
              }`}
            >
              <Icon className={`h-5 w-5 ${isActive ? "text-primary" : ""}`} />
              <span className="text-[10px] leading-tight">{tab.label}</span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
