// Datadog APM tracer — must be the first import in server/index.ts
// Only activates when DD_AGENT_HOST or DD_API_KEY is set in the environment.
// Safe no-op when neither variable is present.

import tracer from "dd-trace";

const enabled = !!(process.env.DD_AGENT_HOST || process.env.DD_API_KEY);

if (enabled) {
  tracer.init({
    hostname: process.env.DD_AGENT_HOST ?? "localhost",
    service: process.env.DD_SERVICE ?? "neural-dream-workshop-api",
    env: process.env.DD_ENV ?? process.env.NODE_ENV ?? "development",
    version: process.env.DD_VERSION ?? "1.0.0",
    logInjection: true,
    runtimeMetrics: true,
  });
}

export default tracer;
