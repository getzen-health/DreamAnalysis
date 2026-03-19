/**
 * AntarAI Service Worker — PWA offline mode with caching + background sync.
 *
 * Caching strategies:
 *   cache-first        — app shell (HTML, CSS, JS bundles, fonts, icons, images)
 *   network-first      — API routes with IndexedDB fallback for health/session/settings
 *   stale-while-revalidate — user profile and non-critical API responses
 *
 * Background sync tag: "sync-offline-data"
 */

const SHELL_CACHE   = "antarai-shell-v1";
const API_CACHE     = "antarai-api-v1";
const PROFILE_CACHE = "antarai-profile-v1";

// ─── App shell assets to precache ─────────────────────────────────────────────
const PRECACHE_ASSETS = [
  "/",
  "/index.html",
  "/manifest.json",
  "/icon-192.png",
  "/icon-512.png",
  "/icon-180.png",
];

// ─── Route matchers ──────────────────────────────────────────────────────────

/** API routes that should be cached with network-first + IndexedDB fallback. */
const NETWORK_FIRST_API = [
  "/api/health",
  "/api/sessions",
  "/api/settings",
  "/api/voice-emotion",
  "/api/food-log",
];

/** API routes for stale-while-revalidate (user profile, non-critical). */
const STALE_WHILE_REVALIDATE_API = [
  "/api/user",
  "/api/profile",
  "/api/brain/history",
  "/api/brain/weekly-summary",
  "/api/brain/patterns",
  "/api/brain/today-totals",
  "/api/streaks/status",
  "/brain-report/streak",
];

function isAppShellRequest(url) {
  const { pathname } = new URL(url);
  // Static assets: JS/CSS bundles, fonts, icons, images
  if (
    pathname.match(/\.(js|css|woff2?|ttf|otf|eot|svg|png|jpg|jpeg|webp|ico|gif)$/)
  ) return true;
  // Navigation requests (SPA routes — all serve index.html)
  return !pathname.startsWith("/api/");
}

function matchesNetworkFirst(url) {
  const { pathname } = new URL(url);
  return NETWORK_FIRST_API.some((p) => pathname.startsWith(p));
}

function matchesStaleWhileRevalidate(url) {
  const { pathname } = new URL(url);
  return STALE_WHILE_REVALIDATE_API.some((p) => pathname.startsWith(p));
}

// ─── Install ──────────────────────────────────────────────────────────────────

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(PRECACHE_ASSETS))
      .then(() => self.skipWaiting())
  );
});

// ─── Activate ─────────────────────────────────────────────────────────────────

self.addEventListener("activate", (event) => {
  const CURRENT_CACHES = [SHELL_CACHE, API_CACHE, PROFILE_CACHE];
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => !CURRENT_CACHES.includes(k))
            .map((k) => caches.delete(k))
        )
      )
      .then(() => self.clients.claim())
  );
});

// ─── Fetch ────────────────────────────────────────────────────────────────────

self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Only intercept GET requests
  if (request.method !== "GET") return;

  const url = request.url;

  if (matchesNetworkFirst(url)) {
    event.respondWith(networkFirstWithCacheFallback(request));
  } else if (matchesStaleWhileRevalidate(url)) {
    event.respondWith(staleWhileRevalidate(request));
  } else if (isAppShellRequest(url)) {
    event.respondWith(cacheFirst(request));
  }
  // Anything else (WebSocket upgrades, non-GET API mutations) — let pass through
});

// ─── Caching strategies ───────────────────────────────────────────────────────

/**
 * Cache-first: serve from SHELL_CACHE; fetch + cache on miss.
 * For SPA navigation requests, always fall back to /index.html.
 */
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(SHELL_CACHE);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    // Navigation fallback — serve app shell so the SPA can render
    const shell = await caches.match("/index.html");
    if (shell) return shell;
    return new Response("Offline", { status: 503 });
  }
}

/**
 * Network-first with API cache fallback.
 * On network success: update API_CACHE.
 * On network failure: serve cached version.
 */
async function networkFirstWithCacheFallback(request) {
  const cache = await caches.open(API_CACHE);
  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await cache.match(request);
    if (cached) return cached;
    return new Response(
      JSON.stringify({ error: "offline", cached: false }),
      { status: 503, headers: { "Content-Type": "application/json" } }
    );
  }
}

/**
 * Stale-while-revalidate: respond immediately from PROFILE_CACHE (if any),
 * then fetch and update cache in the background.
 */
async function staleWhileRevalidate(request) {
  const cache = await caches.open(PROFILE_CACHE);
  const cached = await cache.match(request);

  // Kick off network fetch regardless — update cache in background
  const fetchPromise = fetch(request)
    .then((response) => {
      if (response.ok) cache.put(request, response.clone());
      return response;
    })
    .catch(() => null);

  return cached ?? (await fetchPromise) ?? new Response(
    JSON.stringify({ error: "offline" }),
    { status: 503, headers: { "Content-Type": "application/json" } }
  );
}

// ─── Background Sync ──────────────────────────────────────────────────────────

self.addEventListener("sync", (event) => {
  if (event.tag === "sync-offline-data") {
    event.waitUntil(broadcastSyncRequest());
  }
});

/**
 * Notify all open clients to run syncAll() — the actual IndexedDB sync logic
 * lives in offline-store.ts (client-side). The SW just triggers it.
 */
async function broadcastSyncRequest() {
  const clients = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
  for (const client of clients) {
    client.postMessage({ type: "SW_BACKGROUND_SYNC" });
  }
}

// ─── Push Notifications ───────────────────────────────────────────────────────

self.addEventListener("push", (event) => {
  const data = event.data ? event.data.json() : {};
  const title = data.title || "AntarAI";
  const options = {
    body: data.body || "Your morning brain report is ready.",
    icon: "/icon-192.png",
    badge: "/icon-192.png",
    tag: data.tag || "antarai-notification",
    renotify: true,
    data: { url: data.url || "/" },
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = event.notification.data?.url || "/";
  event.waitUntil(
    self.clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then((clientList) => {
        for (const client of clientList) {
          if (client.url.includes(url) && "focus" in client) return client.focus();
        }
        return self.clients.openWindow(url);
      })
  );
});
