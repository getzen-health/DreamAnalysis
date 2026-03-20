import { createContext, useContext, useEffect, useState } from "react";

type Theme = "dark" | "light";
type ThemeSetting = "dark" | "light" | "auto";

type ThemeContextType = {
  theme: Theme;
  themeSetting: ThemeSetting;
  setTheme: (theme: ThemeSetting) => void;
};

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

function resolveAutoTheme(): Theme {
  // Use system preference first
  if (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: light)").matches) {
    return "light";
  }
  // Fallback: time-based (light 7am-7pm)
  const hour = new Date().getHours();
  return hour >= 7 && hour < 19 ? "light" : "dark";
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [themeSetting, setThemeSetting] = useState<ThemeSetting>("light");
  const [theme, setResolvedTheme] = useState<Theme>("light");

  // Load saved setting
  useEffect(() => {
    const saved = localStorage.getItem("neural-theme") as ThemeSetting | null;
    if (saved) {
      setThemeSetting(saved);
    }
  }, []);

  // Resolve actual theme from setting
  useEffect(() => {
    const resolved = themeSetting === "auto" ? resolveAutoTheme() : themeSetting;
    setResolvedTheme(resolved);
  }, [themeSetting]);

  // Listen for system theme changes when in auto mode
  useEffect(() => {
    if (themeSetting !== "auto") return;
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return;
    const handler = () => setResolvedTheme(resolveAutoTheme());
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [themeSetting]);

  function setTheme(setting: ThemeSetting) {
    setThemeSetting(setting);
    localStorage.setItem("neural-theme", setting);
  }

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    // Update mobile status bar color to match theme
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#0f1117" : "#ffffff");
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, themeSetting, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
