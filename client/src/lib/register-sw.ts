/**
 * register-sw.ts — registers the AntarAI service worker for offline-first caching.
 *
 * Call once at app mount. Safe to call multiple times (browsers deduplicate
 * registrations for the same script URL).
 *
 * The SW lives at /sw.js (client/public/sw.js) and is served at the root
 * path by Vite in dev and Vercel in production.
 */

export function registerServiceWorker(): void {
  if (!("serviceWorker" in navigator)) return;

  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/sw.js")
      .then((reg) => {
        // Check for SW updates every 30 minutes while the app is open
        setInterval(() => reg.update(), 30 * 60 * 1000);
      })
      .catch((err) => {
        // Non-fatal: app works without SW, just without offline caching
        console.warn("[SW] Registration failed:", err);
      });
  });
}
