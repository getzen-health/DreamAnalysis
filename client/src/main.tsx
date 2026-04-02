import { createRoot } from "react-dom/client";
import { initSentry } from "./lib/sentry";
import App from "./App";
import "./index.css";
import { flushPendingCorrections } from "@/lib/feedback-sync";
import { initSQLiteStore } from "@/lib/sqlite-store";
import { maybeBackup } from "@/lib/icloud-sync";

// Sentry crash analytics — only activates when VITE_SENTRY_DSN is set
initSentry();

// Datadog Browser RUM + Logs — only activates when both VITE_DD_APPLICATION_ID
// and VITE_DD_CLIENT_TOKEN are set at build time
const ddAppId = import.meta.env.VITE_DD_APPLICATION_ID as string | undefined;
const ddClientToken = import.meta.env.VITE_DD_CLIENT_TOKEN as string | undefined;

if (ddAppId && ddClientToken) {
  const ddSite = (import.meta.env.VITE_DD_SITE as string | undefined) ?? "datadoghq.com";
  const ddEnv = import.meta.env.MODE;

  import("@datadog/browser-rum").then(({ datadogRum }) => {
    datadogRum.init({
      applicationId: ddAppId,
      clientToken: ddClientToken,
      site: ddSite,
      service: "neural-dream-workshop-ui",
      env: ddEnv,
      version: "1.0.0",
      sessionSampleRate: 100,
      sessionReplaySampleRate: 20,
      trackUserInteractions: true,
      trackResources: true,
      trackLongTasks: true,
      defaultPrivacyLevel: "mask-user-input",
    });
  });

  import("@datadog/browser-logs").then(({ datadogLogs }) => {
    datadogLogs.init({
      clientToken: ddClientToken,
      site: ddSite,
      service: "neural-dream-workshop-ui",
      env: ddEnv,
      forwardErrorsToLogs: true,
      sessionSampleRate: 100,
    });
  });
}

createRoot(document.getElementById("root")!).render(<App />);

// Flush any queued corrections from previous offline sessions
flushPendingCorrections().catch(() => {});

// On-device encrypted SQLite store (native only — no-op on web)
initSQLiteStore().then(available => {
  if (available) maybeBackup().catch(() => {}); // trigger iCloud backup if iOS + 12h since last
}).catch(() => {});
