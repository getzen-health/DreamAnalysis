/**
 * GET /api/ping-ml
 *
 * Lightweight keep-alive endpoint. Pings the Render ML backend so it
 * doesn't spin down from inactivity.
 *
 * Usage:
 *   Vercel Pro plan: add this to vercel.json "crons" (already done below).
 *   Vercel Hobby plan: point a free external cron at this URL, e.g.
 *     https://cron-job.org  or  https://uptimerobot.com
 *     Target URL: https://dream-analysis.vercel.app/api/ping-ml
 *     Interval: every 14 minutes
 */

import type { VercelRequest, VercelResponse } from "@vercel/node";

const ML_URL =
  process.env.VITE_ML_API_URL ||
  process.env.ML_API_URL ||
  "https://neural-dream-ml.onrender.com";

export default async function handler(
  _req: VercelRequest,
  res: VercelResponse,
) {
  const start = Date.now();

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15_000);

    const r = await fetch(`${ML_URL}/health`, { signal: controller.signal });
    clearTimeout(timeout);

    const latency = Date.now() - start;
    const body = r.ok ? await r.json().catch(() => ({})) : {};

    return res.status(200).json({
      ok: r.ok,
      status: r.status,
      latencyMs: latency,
      ml: body,
      ts: new Date().toISOString(),
    });
  } catch (err) {
    return res.status(200).json({
      ok: false,
      error: err instanceof Error ? err.message : "unreachable",
      latencyMs: Date.now() - start,
      ts: new Date().toISOString(),
    });
  }
}
