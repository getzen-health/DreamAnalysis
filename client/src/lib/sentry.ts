import * as Sentry from "@sentry/capacitor";
import * as SentryReact from "@sentry/react";

export function initSentry() {
  // Only init in production or if DSN is set
  const dsn = import.meta.env.VITE_SENTRY_DSN;
  if (!dsn) return;

  Sentry.init(
    {
      dsn,
      release: `antarai@${import.meta.env.VITE_APP_VERSION || "1.0.0"}`,
      environment: import.meta.env.MODE,
      integrations: [SentryReact.browserTracingIntegration()],
      tracesSampleRate: 0.2,
      // Never send PHI in breadcrumbs
      beforeBreadcrumb(breadcrumb) {
        if (breadcrumb.category === "console") return null;
        return breadcrumb;
      },
      // Scrub sensitive data
      beforeSend(event) {
        if (event.request?.data) delete event.request.data;
        return event;
      },
    },
    SentryReact.init,
  );
}
