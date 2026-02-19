import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

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
