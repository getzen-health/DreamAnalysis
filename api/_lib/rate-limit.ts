import { sql } from 'drizzle-orm';
import type { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * Check and increment a rate limit counter backed by the rate_limit_entries table.
 *
 * Uses a single upsert query:
 *  - If no row exists, inserts with count=1 and window_start=now.
 *  - If the row exists and the window has expired, resets count=1 and window_start=now.
 *  - If the row exists and the window is active, increments count.
 *
 * Returns { allowed: true } if under the limit, or
 *         { allowed: false, retryAfterSeconds } if the limit is exceeded.
 */
export async function checkRateLimit(
  db: any,
  key: string,
  maxAttempts: number,
  windowMinutes: number,
): Promise<{ allowed: boolean; retryAfterSeconds?: number }> {
  // Single atomic upsert: insert or update, then return current count + window_start.
  // Use make_interval() to safely pass the window as a parameterized integer.
  const rows = await db.execute(sql`
    INSERT INTO rate_limit_entries (key, count, window_start)
    VALUES (${key}, 1, NOW())
    ON CONFLICT (key) DO UPDATE SET
      count = CASE
        WHEN rate_limit_entries.window_start + make_interval(mins => ${windowMinutes}) <= NOW()
        THEN 1
        ELSE rate_limit_entries.count + 1
      END,
      window_start = CASE
        WHEN rate_limit_entries.window_start + make_interval(mins => ${windowMinutes}) <= NOW()
        THEN NOW()
        ELSE rate_limit_entries.window_start
      END
    RETURNING count, window_start
  `);

  const row = rows.rows?.[0] ?? rows[0];
  if (!row) {
    // If somehow no row returned, allow the request (fail open).
    return { allowed: true };
  }

  const count = Number(row.count);
  const windowStart = new Date(row.window_start);

  if (count > maxAttempts) {
    const windowEndMs = windowStart.getTime() + windowMinutes * 60 * 1000;
    const retryAfterSeconds = Math.max(1, Math.ceil((windowEndMs - Date.now()) / 1000));
    return { allowed: false, retryAfterSeconds };
  }

  return { allowed: true };
}

/**
 * Extract the client IP from a Vercel request.
 * Vercel sets x-forwarded-for and x-real-ip headers.
 */
export function getClientIp(req: VercelRequest): string {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string') {
    return forwarded.split(',')[0].trim();
  }
  const realIp = req.headers['x-real-ip'];
  if (typeof realIp === 'string') {
    return realIp.trim();
  }
  // Fallback — unlikely in Vercel but safe default
  return (req.socket?.remoteAddress ?? 'unknown').trim();
}

/**
 * Send a 429 Too Many Requests response with Retry-After header.
 */
export function tooManyRequests(
  res: VercelResponse,
  retryAfterSeconds: number,
  message: string = 'Too many requests. Please try again later.',
): VercelResponse {
  res.setHeader('Retry-After', String(retryAfterSeconds));
  return res.status(429).json({ error: message });
}
