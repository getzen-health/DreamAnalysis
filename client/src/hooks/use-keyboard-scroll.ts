import { useEffect } from "react";

/** On mobile, scrolls focused inputs into view when the virtual keyboard opens. */
export function useKeyboardScroll() {
  useEffect(() => {
    if (!window.visualViewport) return;

    const handleResize = () => {
      const active = document.activeElement;
      if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) {
        setTimeout(() => {
          (active as HTMLElement).scrollIntoView({ behavior: "smooth", block: "center" });
        }, 100);
      }
    };

    window.visualViewport.addEventListener("resize", handleResize);
    return () => window.visualViewport?.removeEventListener("resize", handleResize);
  }, []);
}
