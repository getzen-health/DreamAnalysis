import { ReactNode } from "react";
import { useLocation } from "wouter";
import { Sidebar } from "@/components/sidebar";

const routeTitles: Record<string, string> = {
  "/": "Dashboard",
  "/emotions": "Emotions",
  "/inner-energy": "Inner Energy",
  "/sessions": "Sessions",
  "/settings": "Settings",
};

interface AppLayoutProps {
  children: ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const [location] = useLocation();
  const pageTitle = routeTitles[location] || "Dashboard";

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Sidebar />

      {/* Main Content */}
      <div className="md:ml-56 min-h-screen">
        {/* Header */}
        <header className="border-b border-border px-6 py-4">
          <h2 className="text-xl font-semibold text-foreground">
            {pageTitle}
          </h2>
        </header>

        {/* Page Content */}
        {children}
      </div>
    </div>
  );
}
