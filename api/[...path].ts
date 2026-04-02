/**
 * Unified API catch-all for Vercel Hobby plan (max 12 serverless functions).
 * Routes all /api/* requests to the appropriate handler inline.
 *
 * Routes handled:
 *   POST   /api/auth/register
 *   POST   /api/auth/login
 *   GET    /api/auth/me
 *   POST   /api/auth/logout
 *   POST   /api/auth/forgot-password
 *   POST   /api/auth/reset-password
 *   POST   /api/dreams/create
 *   GET    /api/dreams/list
 *   GET    /api/dreams/analytics
 *   POST   /api/dreams/generate-image
 *   POST   /api/dream-analysis
 *   POST   /api/dream-analysis/multi-pass
 *   GET    /api/dream-analysis/:userId
 *   POST   /api/ai-chat
 *   GET    /api/ai-chat/:userId
 *   POST   /api/emotions/record
 *   GET    /api/emotions/history
 *   POST   /api/health-metrics
 *   GET    /api/health-metrics/:userId
 *   POST   /api/health-samples              (ingest HealthKit / Health Connect time-series)
 *   GET    /api/health-samples/:userId      (?metric=heart_rate&days=7)
 *   GET    /api/settings/:userId
 *   POST   /api/settings/:userId
 *   PUT    /api/settings/:userId
 *   GET    /api/export/:userId
 *   GET    /api/insights/weekly
 *   POST   /api/notifications/subscribe
 *   POST   /api/analyze-mood
 *   POST   /api/food/analyze
 *   GET    /api/food/logs/:userId
 *   POST   /api/study/enroll
 *   GET    /api/study/status/:userId
 *   POST   /api/study/morning
 *   POST   /api/study/daytime
 *   POST   /api/study/evening
 *   GET    /api/study/history/:userId
 *   POST   /api/study/withdraw
 *   POST   /api/study/consent              (pilot study)
 *   POST   /api/study/session/start        (pilot study)
 *   POST   /api/study/session/complete     (pilot study)
 *   GET    /api/study/admin/participants   (pilot study, admin required)
 *   GET    /api/study/admin/sessions       (pilot study, admin required)
 *   GET    /api/study/admin/stats          (pilot study, admin required)
 *   GET    /api/study/admin/export-csv     (pilot study, admin required)
 *   POST   /api/readings                  (store voice/food/health/eeg reading)
 *   GET    /api/readings/:userId          (export readings for training)
 *   ALL    /api/ml/*                      (ML backend proxy → FastAPI)
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { scrypt, randomBytes, timingSafeEqual } from 'crypto';
import { promisify } from 'util';
import { eq, desc, asc, and, gte, lt, sql, isNull } from 'drizzle-orm';

import { z } from 'zod';
import { success, error, badRequest, methodNotAllowed, unauthorized } from './_lib/response.js';
import { generateToken, setAuthCookie, clearAuthCookie, requireAuth, requireOwner, requireAdmin } from './_lib/auth.js';
import { checkRateLimit, getClientIp, tooManyRequests } from './_lib/rate-limit.js';

// Lazy-load heavy modules at handler runtime to avoid Vercel cold-start crash.
// drizzle-orm/neon-http and openai can crash the Node.js process if loaded at
// module-init time in Vercel's serverless environment.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let schema: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _dbGetter: (() => any) | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _openaiGetter: (() => any) | null = null;

async function loadModules() {
  if (schema) return;
  const [schemaModule, dbModule, openaiModule] = await Promise.all([
    import('../shared/schema.js'),
    import('./_lib/db.js'),
    import('./_lib/openai.js'),
  ]);
  schema = schemaModule;
  _dbGetter = dbModule.getDb;
  _openaiGetter = openaiModule.getOpenAIClient;
}

// Synchronous wrappers — safe to call after loadModules() has resolved
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getDb(): any {
  if (!_dbGetter) throw new Error('Modules not initialized');
  return _dbGetter();
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getOpenAIClient(): any {
  if (!_openaiGetter) throw new Error('Modules not initialized');
  return _openaiGetter();
}

const scryptAsync = promisify(scrypt);

// ── Body parser ──────────────────────────────────────────────────────────────
// Vercel's runtime has a known bug where req.body throws "Invalid JSON" and the
// request stream is already consumed. As a workaround, clients send the body
// base64-encoded in the x-body-b64 header, which bypasses Vercel's broken parser.
async function parseRequestBody(req: VercelRequest): Promise<unknown> {
  // Strategy 1 (primary): read body from x-body-b64 header
  const b64 = req.headers['x-body-b64'] as string | undefined;
  if (b64) {
    try {
      const raw = Buffer.from(b64, 'base64').toString('utf8');
      if (raw) return JSON.parse(raw);
    } catch { /* fall through */ }
  }

  // Strategy 2: direct body access (works on some Vercel runtimes)
  try {
    const b = (req as any).body;
    if (b !== undefined && b !== null) {
      if (typeof b === 'string' && b.length > 0) {
        try { return JSON.parse(b); } catch { /* fall through */ }
      }
      if (typeof b === 'object' && Object.keys(b as object).length > 0) return b;
    }
  } catch { /* getter threw */ }

  // Strategy 3: read stream (fallback)
  try {
    const chunks: Buffer[] = [];
    await new Promise<void>((resolve) => {
      const timeout = setTimeout(resolve, 2000);
      req.on('data', (chunk: Buffer) => chunks.push(chunk));
      req.on('end', () => { clearTimeout(timeout); resolve(); });
      req.on('error', () => { clearTimeout(timeout); resolve(); });
    });
    if (chunks.length > 0) {
      const raw = Buffer.concat(chunks).toString('utf8');
      return JSON.parse(raw);
    }
  } catch { /* fall through */ }

  return {};
}

// ── Helpers ─────────────────────────────────────────────────────────────────

async function hashPassword(password: string): Promise<string> {
  const salt = randomBytes(16).toString('hex');
  const derived = (await scryptAsync(password, salt, 64)) as Buffer;
  return `${salt}:${derived.toString('hex')}`;
}

async function verifyPassword(stored: string, supplied: string): Promise<boolean> {
  const [salt, hash] = stored.split(':');
  if (!salt || !hash) return false;
  const derived = (await scryptAsync(supplied, salt, 64)) as Buffer;
  return timingSafeEqual(derived, Buffer.from(hash, 'hex'));
}

// ── Route handlers ───────────────────────────────────────────────────────────

async function authRegister(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  try {
    // Rate limit: 20 registrations per IP per hour
    const db = getDb();
    const ip = getClientIp(req);
    const rl = await checkRateLimit(db, `register:${ip}`, 20, 60);
    if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);

    // Use parsed body (may be pre-set by early parser, or parse again as fallback)
    const body = (req.body && typeof req.body === 'object' && Object.keys(req.body).length > 0)
      ? req.body : await parseRequestBody(req);
    const { username, password, email } = body as { username?: string; password?: string; email?: string };
    if (!username || typeof username !== 'string' || username.trim().length < 3)
      return badRequest(res, 'Username must be at least 3 characters');
    if (username.trim().length > 50)
      return badRequest(res, 'Username must be ≤50 characters');
    if (!password || typeof password !== 'string' || password.length < 6)
      return badRequest(res, 'Password must be at least 6 characters');
    if (password.length > 128)
      return badRequest(res, 'Password must be ≤128 characters');
    const [existing] = await db.select().from(schema.users).where(eq(schema.users.username, username.trim().toLowerCase()));
    if (existing) return badRequest(res, 'Username already exists');
    const normalizedEmail = email?.trim().toLowerCase() || null;
    if (normalizedEmail) {
      const [emailExists] = await db.select().from(schema.users).where(eq(schema.users.email, normalizedEmail));
      if (emailExists) return badRequest(res, 'An account with this email already exists');
    }
    const [user] = await db.insert(schema.users).values({
      username: username.trim().toLowerCase(), password: await hashPassword(password), email: normalizedEmail,
    }).returning();
    if (!user) throw new Error('Insert returned no rows — check DB schema/migrations');
    const token = generateToken({ userId: user.id, username: user.username, role: (user.role as 'user' | 'admin') ?? 'user' });
    setAuthCookie(res, token);
    const { password: _, ...safe } = user;
    return success(res, { user: safe, token }, 201);
  } catch (err: any) {
    console.error('[authRegister]', err?.message ?? err, err?.stack);
    return error(res, `Registration failed: ${err?.message || 'Unknown error'}`, 500);
  }
}

async function authLogin(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  try {
    // Rate limit: 10 login attempts per IP per 15 minutes
    const db = getDb();
    const ip = getClientIp(req);
    const rl = await checkRateLimit(db, `login:${ip}`, 10, 15);
    if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);

    const body = (req.body && typeof req.body === 'object' && Object.keys(req.body).length > 0)
      ? req.body : await parseRequestBody(req);
    const { username, password } = body as { username?: string; password?: string };
    if (!username || !password) return badRequest(res, 'Username and password required');
    if (typeof password !== 'string' || password.length > 128) return badRequest(res, 'Invalid credentials');
    const [user] = await db.select().from(schema.users).where(eq(schema.users.username, username.trim().slice(0, 50)));
    if (!user || !(await verifyPassword(user.password, password)))
      return unauthorized(res, 'Invalid username or password');
    const token = generateToken({ userId: user.id, username: user.username, role: (user.role as 'user' | 'admin') ?? 'user' });
    setAuthCookie(res, token);
    const { password: _, ...safe } = user;
    return success(res, { user: safe, token });
  } catch (err: any) {
    console.error('[authLogin]', err?.message ?? err, err?.stack);
    return error(res, `Login failed: ${err?.message || 'Unknown error'}`, 500);
  }
}

async function authMe(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const payload = requireAuth(req, res);
  if (!payload) return;
  const db = getDb();
  const [user] = await db.select().from(schema.users).where(eq(schema.users.id, payload.userId));
  if (!user) return error(res, 'User not found', 404);
  const { password: _, ...safe } = user;
  return success(res, safe);
}

async function authLogout(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  clearAuthCookie(res);
  return success(res, { message: 'Logged out successfully' });
}

async function authForgotPassword(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const GENERIC = 'If that email exists, a reset link was sent';
  try {
    const body = await parseRequestBody(req) as Record<string, unknown>;
    const email = typeof body.email === 'string' ? body.email.trim().toLowerCase() : '';
    if (!email) return res.status(200).json({ message: GENERIC });

    // Rate limit: 5 forgot-password attempts per email per hour
    const db = getDb();
    const rl = await checkRateLimit(db, `forgot-password:${email}`, 5, 60);
    if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
    const [user] = await db.select().from(schema.users)
      .where(eq(schema.users.email, email)).limit(1);

    if (user) {
      const token = randomBytes(32).toString('hex');
      const expiresAt = new Date(Date.now() + 60 * 60 * 1000); // 1 hour
      await db.insert(schema.passwordResetTokens).values({ userId: user.id, token, expiresAt });

      const appUrl = process.env.APP_URL ?? 'https://dream-analysis.vercel.app';
      const resetUrl = `${appUrl}/reset-password?token=${token}`;

      if (process.env.GMAIL_APP_PASSWORD) {
        const nodemailer = (await import('nodemailer')).default;
        const transporter = nodemailer.createTransport({
          service: 'gmail',
          auth: {
            user: process.env.GMAIL_USER ?? 'lakshmisravya.vedantham@gmail.com',
            pass: process.env.GMAIL_APP_PASSWORD,
          },
        });
        try {
          await transporter.sendMail({
            from: `"AntarAI" <${process.env.GMAIL_USER ?? 'lakshmisravya.vedantham@gmail.com'}>`,
            to: user.email ?? email,
            subject: 'Reset your password',
            text: `Click the link to reset your password (valid 1 hour):\n\n${resetUrl}\n\nIf you did not request this, ignore this email.`,
            html: `<p>Click the link below to reset your password (valid 1 hour):</p><p><a href="${resetUrl}">${resetUrl}</a></p><p>If you did not request this, ignore this email.</p>`,
          });
        } catch (mailErr: any) {
          console.error('[authForgotPassword] email send failed:', mailErr?.message ?? mailErr);
        }
      }
    }

    return res.status(200).json({ message: GENERIC });
  } catch (err: any) {
    console.error('[authForgotPassword]', err?.message ?? err);
    return res.status(200).json({ message: GENERIC });
  }
}

async function authResetPassword(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  try {
    const body = await parseRequestBody(req) as Record<string, unknown>;
    const token = typeof body.token === 'string' ? body.token : '';
    const newPassword = typeof body.newPassword === 'string' ? body.newPassword : '';
    if (!token || !newPassword)
      return badRequest(res, 'Token and new password required');
    if (newPassword.length < 6) return badRequest(res, 'Password must be at least 6 characters');
    if (newPassword.length > 128) return badRequest(res, 'Password must be ≤128 characters');

    const db = getDb();
    const now = new Date();
    const [row] = await db.select().from(schema.passwordResetTokens)
      .where(
        and(
          eq(schema.passwordResetTokens.token, token),
          gte(schema.passwordResetTokens.expiresAt, now),
          isNull(schema.passwordResetTokens.usedAt),
        )
      ).limit(1);

    if (!row) return res.status(400).json({ message: 'Invalid or expired reset token' });

    const hashed = await hashPassword(newPassword);
    await db.update(schema.users).set({ password: hashed }).where(eq(schema.users.id, row.userId));
    await db.update(schema.passwordResetTokens)
      .set({ usedAt: now })
      .where(eq(schema.passwordResetTokens.id, row.id));

    return success(res, { message: 'Password updated successfully' });
  } catch (err: any) {
    console.error('[authResetPassword]', err?.message ?? err);
    return error(res, 'Password reset failed. Please try again.', 500);
  }
}

async function dreamsCreate(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { tags, sleepQuality, sleepDuration } = req.body;
  const userId = authPayload.userId;
  const dreamText = typeof req.body.dreamText === 'string' ? req.body.dreamText.trim() : '';
  if (!dreamText) return badRequest(res, 'dreamText is required');
  if (dreamText.length > 10000) return badRequest(res, 'dreamText exceeds max length (10000 chars)');
  // Validate optional numeric fields
  const safeSleepQuality = sleepQuality != null ? Number(sleepQuality) : null;
  if (safeSleepQuality !== null && (!isFinite(safeSleepQuality) || safeSleepQuality < 0 || safeSleepQuality > 10)) {
    return badRequest(res, 'sleepQuality must be 0–10');
  }
  const safeSleepDuration = sleepDuration != null ? Number(sleepDuration) : null;
  if (safeSleepDuration !== null && (!isFinite(safeSleepDuration) || safeSleepDuration < 0 || safeSleepDuration > 24)) {
    return badRequest(res, 'sleepDuration must be 0–24 hours');
  }
  // Validate optional tags array — cap at 20 tags, each ≤50 chars
  const safeTags: string[] = Array.isArray(tags)
    ? tags.slice(0, 20).filter((t: unknown) => typeof t === 'string').map((t: string) => t.trim().slice(0, 50)).filter(Boolean)
    : [];
  const db = getDb();
  const rl = await checkRateLimit(db, `dreams-create:${userId}`, 20, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!, 'Too many dream submissions. Please wait before trying again.');
  const openai = getOpenAIClient();
  const recentDreams = await db.select({ dreamText: schema.dreamAnalysis.dreamText, symbols: schema.dreamAnalysis.symbols })
    .from(schema.dreamAnalysis).where(eq(schema.dreamAnalysis.userId, userId))
    .orderBy(desc(schema.dreamAnalysis.timestamp)).limit(5);
  const historyCtx = recentDreams.length > 0
    ? `\n\nRecent dream themes: ${recentDreams.map((d: (typeof recentDreams)[number]) => (d.symbols as string[] | null)?.join(', ') || 'unknown').join('; ')}`
    : '';
  let analysis: Record<string, unknown> = {};
  try {
    const resp = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: `You are an expert dream analyst combining Jungian, Freudian, and neuroscience perspectives. Respond with JSON: {"symbols":[],"emotions":[{"emotion":"","intensity":0}],"analysis":"","lucidityScore":1,"themes":[],"wakingLifeConnections":"","recurringPatterns":""}${historyCtx}` },
        { role: 'user', content: `Analyze this dream: ${dreamText}` },
      ],
      response_format: { type: 'json_object' },
    });
    analysis = JSON.parse(resp.choices[0].message.content || '{}');
  } catch (err) {
    console.error('[dreamsCreate AI]', err instanceof Error ? err.message : err);
  }
  const [entry] = await db.insert(schema.dreamAnalysis).values({
    userId, dreamText, symbols: analysis.symbols || [], emotions: analysis.emotions || [],
    aiAnalysis: analysis.analysis || '', lucidityScore: analysis.lucidityScore || null,
    sleepQuality: safeSleepQuality, sleepDuration: safeSleepDuration, tags: safeTags,
  }).returning();
  if (analysis.symbols && (analysis.symbols as string[]).length > 0) {
    const symbolRows = (analysis.symbols as string[]).map((sym) => ({ userId, symbol: sym, meaning: null, frequency: 1 }));
    await db.insert(schema.dreamSymbols).values(symbolRows).onConflictDoNothing();
  }
  return success(res, { ...entry, themes: analysis.themes, wakingLifeConnections: analysis.wakingLifeConnections }, 201);
}

async function dreamsList(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const userId = req.query.userId as string;
  if (!userId) return error(res, 'userId required', 400);
  if (!requireOwner(req, res, userId)) return;
  const page = Math.max(parseInt(req.query.page as string) || 1, 1);
  const limit = Math.min(Math.max(parseInt(req.query.limit as string) || 20, 1), 100);
  const db = getDb();
  const dreams = await db.select().from(schema.dreamAnalysis)
    .where(eq(schema.dreamAnalysis.userId, userId))
    .orderBy(desc(schema.dreamAnalysis.timestamp)).limit(limit).offset((page - 1) * limit);
  return success(res, { dreams, page, limit });
}

async function dreamsAnalytics(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const userId = req.query.userId as string;
  if (!userId) return error(res, 'userId required', 400);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const [dreams, symbols] = await Promise.all([
    db.select().from(schema.dreamAnalysis).where(eq(schema.dreamAnalysis.userId, userId)).orderBy(desc(schema.dreamAnalysis.timestamp)).limit(1000),
    db.select().from(schema.dreamSymbols).where(eq(schema.dreamSymbols.userId, userId)).limit(500),
  ]);
  const total = dreams.length;
  const tagCounts: Record<string, number> = {};
  const emotionCounts: Record<string, number> = {};
  dreams.forEach((d: (typeof dreams)[number]) => {
    (d.tags as string[] | null)?.forEach(t => { tagCounts[t] = (tagCounts[t] || 0) + 1; });
    (d.emotions as Array<{ emotion: string }> | null)?.forEach(e => { emotionCounts[e.emotion] = (emotionCounts[e.emotion] || 0) + 1; });
  });
  return success(res, {
    totalDreams: total,
    avgSleepQuality: Math.round(dreams.reduce((s: number, d: (typeof dreams)[number]) => s + (d.sleepQuality || 0), 0) / Math.max(total, 1) * 10) / 10,
    avgLucidity: Math.round(dreams.reduce((s: number, d: (typeof dreams)[number]) => s + (d.lucidityScore || 0), 0) / Math.max(total, 1) * 10) / 10,
    tagDistribution: tagCounts, emotionDistribution: emotionCounts,
    topSymbols: symbols.sort((a: (typeof symbols)[number], b: (typeof symbols)[number]) => (b.frequency || 0) - (a.frequency || 0)).slice(0, 10),
  });
}

async function dreamsGenerateImage(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { dreamId } = req.body;
  if (!dreamId) return badRequest(res, 'dreamId required');
  const db = getDb();
  // Rate limit: 10 image generations per user per hour
  const rl = await checkRateLimit(db, `dream-image:${authPayload.userId}`, 10, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!, 'Too many image generation requests. Please wait before trying again.');
  const [dream] = await db.select().from(schema.dreamAnalysis).where(eq(schema.dreamAnalysis.id, dreamId));
  if (!dream) return error(res, 'Dream not found', 404);
  if (dream.userId !== authPayload.userId) return error(res, 'Forbidden', 403);
  // Pollinations AI — free, no API key, returns a stable URL for the prompt
  const prompt = `Surreal dreamlike digital art: ${dream.dreamText.substring(0, 400)}. Style: ethereal, mystical, glowing colors, cosmic atmosphere. No text.`;
  const imageUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}?width=1024&height=1024&nologo=true&seed=${Date.now()}`;
  await db.update(schema.dreamAnalysis).set({ imageUrl }).where(eq(schema.dreamAnalysis.id, dreamId));
  return success(res, { imageUrl });
}

async function dreamAnalysisPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const userId = authPayload.userId;
  const dreamText = typeof req.body?.dreamText === 'string' ? req.body.dreamText.trim() : '';
  if (!dreamText) return badRequest(res, 'Missing dreamText');
  if (dreamText.length > 10000) return badRequest(res, 'dreamText exceeds max length (10000 chars)');
  const db = getDb();
  const rl = await checkRateLimit(db, `dream-analysis:${userId}`, 20, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!, 'Too many dream analysis requests. Please wait before trying again.');
  const openai = getOpenAIClient();
  let analysis: Record<string, unknown> = {};
  try {
    const resp = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: 'You are a dream analysis expert. Respond with JSON: {"symbols":[],"emotions":[{"emotion":"","intensity":0}],"analysis":""}' },
        { role: 'user', content: `Analyze this dream: ${dreamText}` },
      ],
      response_format: { type: 'json_object' },
    });
    analysis = JSON.parse(resp.choices[0].message.content || '{}');
  } catch (err) {
    console.error('[dreamAnalysisPost AI]', err instanceof Error ? err.message : err);
  }
  const [entry] = await db.insert(schema.dreamAnalysis).values({
    userId, dreamText, symbols: analysis.symbols || [], emotions: analysis.emotions || [], aiAnalysis: analysis.analysis || '',
  }).returning();
  return success(res, entry, 201);
}

async function dreamAnalysisGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const rows = await db.select().from(schema.dreamAnalysis)
    .where(eq(schema.dreamAnalysis.userId, userId)).orderBy(desc(schema.dreamAnalysis.timestamp)).limit(20);
  return success(res, rows);
}

async function dreamAnalysisMultiPassPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { dreamText, recentThemes } = req.body;
  if (!dreamText || typeof dreamText !== 'string') return badRequest(res, 'Missing dreamText');
  if (dreamText.length > 10000) return badRequest(res, 'dreamText exceeds max length (10000 chars)');
  // Rate limit: 15 multi-pass analyses per IP per hour (3 Groq calls each)
  const rl = await checkRateLimit(getDb(), `dream-multipass:${getClientIp(req)}`, 15, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!, 'Too many dream analysis requests. Please wait before trying again.');

  const openai = getOpenAIClient();
  const safeThemes = Array.isArray(recentThemes)
    ? recentThemes.slice(0, 10).map((t: unknown) => typeof t === 'string' ? t.trim().slice(0, 100) : '').filter(Boolean)
    : [];
  const recentCtx = safeThemes.length > 0
    ? `\nRecent dream themes for continuity: ${safeThemes.join(', ')}`
    : '';

  try {
    // Pass 1: Extract themes and symbols
    const r1 = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: 'You are a dream analysis expert. Extract themes, symbols, and a brief summary. Return only valid JSON.' },
        { role: 'user', content: `Extract key themes, symbolic elements, and write a 2-3 sentence summary.\nReturn JSON: {"themes":[],"symbols":[],"summary":""}${recentCtx}\n\nDream: ${dreamText}` },
      ],
      response_format: { type: 'json_object' },
    });
    let pass1 = { themes: [] as string[], symbols: [] as string[], summary: '' };
    try { pass1 = JSON.parse(r1.choices[0].message.content || '{}'); } catch { /* defaults */ }

    // Pass 2: Interpret symbols and emotional tone
    const r2 = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: 'You are a dream interpretation specialist combining Jungian and neuroscience perspectives. Return only valid JSON.' },
        { role: 'user', content: `Given themes: ${JSON.stringify(pass1.themes)}, symbols: ${JSON.stringify(pass1.symbols)}\nInterpret each symbol, determine emotional tone, suggest waking life connections, identify lucidity indicators.\nReturn JSON: {"symbols":[{"symbol":"","meaning":""}],"emotionalTone":"","connections":[],"lucidityIndicators":[]}\n\nDream: ${dreamText}` },
      ],
      response_format: { type: 'json_object' },
    });
    let pass2 = { symbols: [] as { symbol: string; meaning: string }[], emotionalTone: 'neutral', connections: [] as string[], lucidityIndicators: [] as string[] };
    try { pass2 = JSON.parse(r2.choices[0].message.content || '{}'); } catch { /* defaults */ }

    // Pass 3: Actionable insight
    const r3 = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: 'You are a clinical dream analyst. Provide specific, actionable insights. Return only valid JSON.' },
        { role: 'user', content: `Summary: ${pass1.summary}\nThemes: ${(pass1.themes || []).join(', ')}\nTone: ${pass2.emotionalTone}\nSymbols: ${JSON.stringify(pass2.symbols)}\n\nWrite one specific, actionable insight.\nReturn JSON: {"actionableInsight":""}` },
      ],
      response_format: { type: 'json_object' },
    });
    let pass3 = { actionableInsight: '' };
    try { pass3 = JSON.parse(r3.choices[0].message.content || '{}'); } catch { /* defaults */ }

    return success(res, {
      summary: pass1.summary || '',
      themes: Array.isArray(pass1.themes) ? pass1.themes : [],
      symbols: Array.isArray(pass2.symbols) ? pass2.symbols : [],
      emotionalTone: pass2.emotionalTone || 'neutral',
      connections: Array.isArray(pass2.connections) ? pass2.connections : [],
      lucidityIndicators: Array.isArray(pass2.lucidityIndicators) ? pass2.lucidityIndicators : [],
      actionableInsight: pass3.actionableInsight || '',
    });
  } catch (err) {
    // Single-pass fallback
    try {
      const fallback = await openai.chat.completions.create({
        model: 'llama-3.3-70b-versatile',
        messages: [
          { role: 'system', content: 'You are a dream analysis expert. Return JSON: {"summary":"","themes":[],"symbols":[{"symbol":"","meaning":""}],"emotionalTone":"","connections":[],"lucidityIndicators":[],"actionableInsight":""}' },
          { role: 'user', content: `Analyze this dream: ${dreamText}` },
        ],
        response_format: { type: 'json_object' },
      });
      const parsed = JSON.parse(fallback.choices[0].message.content || '{}');
      return success(res, {
        summary: parsed.summary || '',
        themes: Array.isArray(parsed.themes) ? parsed.themes : [],
        symbols: Array.isArray(parsed.symbols) ? parsed.symbols : [],
        emotionalTone: parsed.emotionalTone || 'neutral',
        connections: Array.isArray(parsed.connections) ? parsed.connections : [],
        lucidityIndicators: Array.isArray(parsed.lucidityIndicators) ? parsed.lucidityIndicators : [],
        actionableInsight: parsed.actionableInsight || '',
      });
    } catch (fallbackErr) {
      console.error('[dreamAnalyze fallback]', fallbackErr instanceof Error ? fallbackErr.message : fallbackErr);
      return error(res, 'Dream analysis failed', 500);
    }
  }
}

async function aiChatPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { message } = req.body;
  const userId = authPayload.userId;
  if (!message) return badRequest(res, 'Missing message');
  if (typeof message !== 'string' || message.length > 2000) return badRequest(res, 'Message must be a string of ≤2000 characters');
  const db = getDb();
  // Rate limit: 40 AI chat messages per user per hour
  const rlChat = await checkRateLimit(db, `ai-chat:${userId}`, 40, 60);
  if (!rlChat.allowed) return tooManyRequests(res, rlChat.retryAfterSeconds!, 'Too many messages. Please wait before sending more.');
  await db.insert(schema.aiChats).values({ userId, message, isUser: true });

  // Fetch last 10 messages for conversation context (5 turns)
  const [recentMetrics, recentHistory] = await Promise.all([
    db.select().from(schema.healthMetrics)
      .where(eq(schema.healthMetrics.userId, userId)).orderBy(desc(schema.healthMetrics.timestamp)).limit(5),
    db.select().from(schema.aiChats)
      .where(eq(schema.aiChats.userId, userId)).orderBy(desc(schema.aiChats.timestamp)).limit(11),
  ]);

  const ctx = recentMetrics.length > 0
    ? `Recent health data: HR ${recentMetrics[0].heartRate}, stress ${recentMetrics[0].stressLevel}, sleep quality ${recentMetrics[0].sleepQuality}.`
    : '';

  // Build conversation history (exclude the message just inserted, reverse to chronological)
  const historyMsgs: { role: 'user' | 'assistant'; content: string }[] = recentHistory
    .filter((c) => c.message !== message || !c.isUser) // exclude the just-saved user msg
    .slice(0, 10)
    .reverse()
    .map((c) => ({ role: c.isUser ? 'user' : 'assistant', content: c.message }));

  const openai = getOpenAIClient();
  let aiMsg: string;
  try {
    const resp = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: `You are an AI wellness companion for a Brain-Computer Interface system. ${ctx} Be supportive and concise.` },
        ...historyMsgs,
        { role: 'user', content: message },
      ],
    });
    aiMsg = resp.choices[0].message.content || "I'm here to help you with your wellness journey.";
  } catch (err) {
    console.error('[aiChat]', err instanceof Error ? err.message : err);
    aiMsg = "Sorry, I'm having trouble connecting right now. Please try again in a moment.";
  }
  const [chat] = await db.insert(schema.aiChats).values({ userId, message: aiMsg, isUser: false }).returning();
  return success(res, chat, 201);
}

async function aiChatGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const chats = await db.select().from(schema.aiChats)
    .where(eq(schema.aiChats.userId, userId)).orderBy(desc(schema.aiChats.timestamp)).limit(50);
  return success(res, chats);
}

async function emotionsRecord(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const userId = req.body?.userId as string | undefined;
  if (!userId) return badRequest(res, 'userId required');
  if (!requireOwner(req, res, userId)) return;
  const parsed = schema.insertEmotionReadingSchema.safeParse(req.body);
  if (!parsed.success) return badRequest(res, parsed.error.issues[0]?.message ?? 'Invalid emotion data');
  const db = getDb();
  // Rate limit: 120 emotion records per user per hour
  const rl = await checkRateLimit(db, `emotions-record:${userId}`, 120, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const [reading] = await db.insert(schema.emotionReadings).values(parsed.data).returning();
  return success(res, reading, 201);
}

async function emotionsCorrect(req: VercelRequest, res: VercelResponse, id: string) {
  if (req.method !== 'PATCH') return methodNotAllowed(res, ['PATCH']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { userCorrectedEmotion } = req.body;
  const validEmotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral', 'stress', 'focus', 'relaxed', 'excited'];
  if (!userCorrectedEmotion || !validEmotions.includes(userCorrectedEmotion))
    return badRequest(res, `userCorrectedEmotion must be one of: ${validEmotions.join(', ')}`);
  const db = getDb();
  const [existing] = await db.select({ id: schema.emotionReadings.id, userId: schema.emotionReadings.userId })
    .from(schema.emotionReadings).where(eq(schema.emotionReadings.id, id));
  if (!existing) return error(res, 'Reading not found', 404);
  if (existing.userId !== authPayload.userId) return unauthorized(res, 'Not your reading');
  const [updated] = await db.update(schema.emotionReadings)
    .set({ userCorrectedEmotion, userCorrectedAt: new Date() })
    .where(eq(schema.emotionReadings.id, id))
    .returning();
  return success(res, updated);
}

async function emotionsCorrectLatest(req: VercelRequest, res: VercelResponse, userId: string) {
  if (!requireOwner(req, res, userId)) return;
  if (req.method !== 'PATCH') return methodNotAllowed(res, ['PATCH']);
  const { userCorrectedEmotion } = req.body;
  const validEmotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral', 'stress', 'focus', 'relaxed', 'excited'];
  if (!userCorrectedEmotion || !validEmotions.includes(userCorrectedEmotion))
    return badRequest(res, `userCorrectedEmotion must be one of: ${validEmotions.join(', ')}`);
  const db = getDb();
  const [latest] = await db.select({ id: schema.emotionReadings.id })
    .from(schema.emotionReadings)
    .where(eq(schema.emotionReadings.userId, userId))
    .orderBy(desc(schema.emotionReadings.timestamp))
    .limit(1);
  if (!latest) return error(res, 'No readings found for this user', 404);
  const [updated] = await db.update(schema.emotionReadings)
    .set({ userCorrectedEmotion, userCorrectedAt: new Date() })
    .where(eq(schema.emotionReadings.id, latest.id))
    .returning();
  return success(res, updated);
}

async function emotionsHistory(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const userId = req.query.userId as string;
  if (!userId) return error(res, 'userId required', 400);
  if (!requireOwner(req, res, userId)) return;
  const limit = Math.min(Math.max(parseInt(req.query.limit as string) || 50, 1), 1000);
  const db = getDb();
  const rows = await db.select().from(schema.emotionReadings)
    .where(eq(schema.emotionReadings.userId, userId)).orderBy(desc(schema.emotionReadings.timestamp)).limit(limit);
  return success(res, rows);
}

async function healthMetricsPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const userId = req.body?.userId as string | undefined;
  if (!userId) return badRequest(res, 'userId required');
  if (!requireOwner(req, res, userId)) return;
  try {
    const data = schema.insertHealthMetricsSchema.parse(req.body);
    const db = getDb();
    // Rate limit: 200 health metric writes per user per hour
    const rl = await checkRateLimit(db, `health-metrics:${userId}`, 200, 60);
    if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
    const [row] = await db.insert(schema.healthMetrics).values(data).returning();
    return success(res, row, 201);
  } catch (e: any) {
    if (e.name === 'ZodError') return badRequest(res, 'Invalid health metrics data');
    throw e;
  }
}

async function healthMetricsGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const rows = await db.select().from(schema.healthMetrics)
    .where(eq(schema.healthMetrics.userId, userId)).orderBy(desc(schema.healthMetrics.timestamp)).limit(50);
  return success(res, rows);
}

async function healthSamplesPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  try {
    const db = getDb();
    const ip = getClientIp(req);
    // Rate limit: 50 batch uploads per IP per hour
    const rl = await checkRateLimit(db, `health-samples:${ip}`, 50, 60);
    if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
    const { user_id, samples } = req.body as {
      user_id: string;
      samples: Array<{
        source: string;
        metric: string;
        value: number;
        unit?: string;
        recorded_at: string;
        metadata?: Record<string, unknown>;
      }>;
    };
    if (!user_id || !Array.isArray(samples) || samples.length === 0) {
      return badRequest(res, 'user_id and samples[] required');
    }
    if (samples.length > 2000) {
      return badRequest(res, 'samples[] exceeds max batch size of 2000');
    }
    if (!requireOwner(req, res, user_id)) return;
    const rows = samples
      .filter((s) => typeof s.value === 'number' && !isNaN(s.value)
        && typeof s.source === 'string' && s.source.length > 0
        && typeof s.metric === 'string' && s.metric.length > 0)
      .map((s) => {
        const recordedAt = s.recorded_at ? new Date(s.recorded_at) : new Date();
        return {
          userId: user_id,
          source: s.source.slice(0, 100),
          metric: s.metric.slice(0, 100),
          value: s.value,
          unit: typeof s.unit === 'string' ? s.unit.slice(0, 50) : null,
          metadata: (s.metadata ?? null) as Record<string, unknown> | null,
          recordedAt: isNaN(recordedAt.getTime()) ? new Date() : recordedAt,
        };
      });
    if (rows.length > 0) {
      await db.insert(schema.healthSamples).values(rows).onConflictDoNothing();
    }
    return success(res, { inserted: rows.length }, 201);
  } catch (e) {
    console.error('health-samples POST failed:', e);
    return error(res, 'Failed to ingest health samples');
  }
}

async function healthSamplesGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  try {
    const db = getDb();
    const metric = (req.query.metric as string) || 'heart_rate';
    const days = Math.min(Math.max(parseInt((req.query.days as string) || '7', 10), 1), 365);
    const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

    const rows = await db
      .select({ value: schema.healthSamples.value, recordedAt: schema.healthSamples.recordedAt })
      .from(schema.healthSamples)
      .where(and(
        eq(schema.healthSamples.userId, userId),
        eq(schema.healthSamples.metric, metric),
        gte(schema.healthSamples.recordedAt, since),
      ))
      .orderBy(asc(schema.healthSamples.recordedAt))
      .limit(200);

    return res.json(rows.map((r: { value: number; recordedAt: Date }) => ({
      value: r.value,
      recorded_at: r.recordedAt.toISOString(),
    })));
  } catch (e) {
    console.error('health-samples GET failed:', e);
    return error(res, 'Failed to fetch health samples');
  }
}

async function settingsHandler(req: VercelRequest, res: VercelResponse, userId: string) {
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  if (req.method === 'GET') {
    const [s] = await db.select().from(schema.userSettings).where(eq(schema.userSettings.userId, userId)).limit(1);
    return success(res, s || null);
  }
  if (req.method === 'POST' || req.method === 'PUT') {
    try {
      const data = schema.insertUserSettingsSchema.parse(req.body);
      const [existing] = await db.select().from(schema.userSettings).where(eq(schema.userSettings.userId, userId)).limit(1);
      const [result] = existing
        ? await db.update(schema.userSettings).set(data).where(eq(schema.userSettings.userId, userId)).returning()
        : await db.insert(schema.userSettings).values({ ...data, userId }).returning();
      return success(res, result);
    } catch (e: any) {
      if (e.name === 'ZodError') return badRequest(res, 'Invalid settings data');
      throw e;
    }
  }
  return methodNotAllowed(res, ['GET', 'POST', 'PUT']);
}

async function exportHandler(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const type = (req.query.type as string) || 'health';
  const db = getDb();

  // ── Apple Health XML ────────────────────────────────────────────────────
  if (type === 'healthkit') {
    const metrics = await db.select().from(schema.healthMetrics)
      .where(eq(schema.healthMetrics.userId, userId)).orderBy(asc(schema.healthMetrics.timestamp)).limit(5000);
    const records = metrics.flatMap((m: (typeof metrics)[number]) => {
      const ts = new Date(m.timestamp).toISOString().replace('T', ' ').slice(0, 19) + ' +0000';
      const rows: string[] = [];
      if (m.heartRate)    rows.push(`  <Record type="HKQuantityTypeIdentifierHeartRate" sourceName="AntarAI" unit="count/min" creationDate="${ts}" startDate="${ts}" endDate="${ts}" value="${m.heartRate}"/>`);
      if (m.dailySteps)   rows.push(`  <Record type="HKQuantityTypeIdentifierStepCount" sourceName="AntarAI" unit="count" creationDate="${ts}" startDate="${ts}" endDate="${ts}" value="${m.dailySteps}"/>`);
      if (m.sleepDuration) rows.push(`  <Record type="HKCategoryTypeIdentifierSleepAnalysis" sourceName="AntarAI" creationDate="${ts}" startDate="${ts}" endDate="${ts}" value="HKCategoryValueSleepAnalysisAsleepCore"/>`);
      return rows;
    });
    const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE HealthData [\n<!ELEMENT HealthData (Record*)>\n<!ELEMENT Record EMPTY>\n<!ATTLIST Record type CDATA #REQUIRED sourceName CDATA #REQUIRED unit CDATA #IMPLIED creationDate CDATA #REQUIRED startDate CDATA #REQUIRED endDate CDATA #REQUIRED value CDATA #IMPLIED>\n]>\n<HealthData locale="en_US">\n${records.join('\n')}\n</HealthData>`;
    res.setHeader('Content-Type', 'application/xml');
    res.setHeader('Content-Disposition', `attachment; filename=neural_dream_healthkit_${new Date().toISOString().slice(0, 10)}.xml`);
    return res.send(xml);
  }

  // ── Dreams CSV ──────────────────────────────────────────────────────────
  if (type === 'dreams') {
    const dreams = await db.select().from(schema.dreamAnalysis)
      .where(eq(schema.dreamAnalysis.userId, userId)).orderBy(desc(schema.dreamAnalysis.timestamp)).limit(5000);
    if (dreams.length === 0) { res.setHeader('Content-Type', 'text/csv'); return res.send('timestamp,dreamText,symbols,aiAnalysis,lucidityScore\nNo dreams recorded yet'); }
    const escape = (s: unknown) => `"${String(s ?? '').replace(/"/g, '""')}"`;
    const rows = dreams.map((d: (typeof dreams)[number]) => [
      d.timestamp, escape(d.dreamText),
      escape((d.symbols as string[] | null)?.join('; ') ?? ''),
      escape(d.aiAnalysis ?? ''), d.lucidityScore ?? '',
    ].join(','));
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename=dream_analysis_${new Date().toISOString().slice(0, 10)}.csv`);
    return res.send(['timestamp,dreamText,symbols,aiAnalysis,lucidityScore', ...rows].join('\n'));
  }

  // ── Emotions CSV ────────────────────────────────────────────────────────
  if (type === 'emotions') {
    const readings = await db.select().from(schema.emotionReadings)
      .where(eq(schema.emotionReadings.userId, userId)).orderBy(desc(schema.emotionReadings.timestamp)).limit(5000);
    if (readings.length === 0) { res.setHeader('Content-Type', 'text/csv'); return res.send('No emotion data yet'); }
    const rows = readings.map((r: (typeof readings)[number]) => [
      r.timestamp, r.dominantEmotion, r.stress, r.happiness, r.focus, r.energy,
      r.valence ?? '', r.arousal ?? '', r.userCorrectedEmotion ?? '',
    ].join(','));
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename=emotion_readings_${new Date().toISOString().slice(0, 10)}.csv`);
    return res.send(['timestamp,dominantEmotion,stress,happiness,focus,energy,valence,arousal,userCorrectedEmotion', ...rows].join('\n'));
  }

  // ── All data CSV (multi-section) ────────────────────────────────────────
  if (type === 'all') {
    const [metrics, dreams, readings] = await Promise.all([
      db.select().from(schema.healthMetrics).where(eq(schema.healthMetrics.userId, userId)).orderBy(desc(schema.healthMetrics.timestamp)).limit(5000),
      db.select().from(schema.dreamAnalysis).where(eq(schema.dreamAnalysis.userId, userId)).orderBy(desc(schema.dreamAnalysis.timestamp)).limit(5000),
      db.select().from(schema.emotionReadings).where(eq(schema.emotionReadings.userId, userId)).orderBy(desc(schema.emotionReadings.timestamp)).limit(5000),
    ]);
    const escape = (s: unknown) => `"${String(s ?? '').replace(/"/g, '""')}"`;
    const sections = [
      '# HEALTH METRICS',
      ['timestamp,heartRate,stressLevel,sleepQuality,neuralActivity,dailySteps,sleepDuration',
        ...metrics.map((m: (typeof metrics)[number]) => [m.timestamp, m.heartRate, m.stressLevel, m.sleepQuality, m.neuralActivity, m.dailySteps, m.sleepDuration].join(','))].join('\n'),
      '\n# DREAM ANALYSIS',
      ['timestamp,dreamText,symbols,aiAnalysis',
        ...dreams.map((d: (typeof dreams)[number]) => [d.timestamp, escape(d.dreamText), escape((d.symbols as string[] | null)?.join('; ') ?? ''), escape(d.aiAnalysis ?? '')].join(','))].join('\n'),
      '\n# EMOTION READINGS',
      ['timestamp,dominantEmotion,stress,happiness,focus,energy,userCorrectedEmotion',
        ...readings.map((r: (typeof readings)[number]) => [r.timestamp, r.dominantEmotion, r.stress, r.happiness, r.focus, r.energy, r.userCorrectedEmotion ?? ''].join(','))].join('\n'),
    ];
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename=neural_dream_export_${new Date().toISOString().slice(0, 10)}.csv`);
    return res.send(sections.join('\n'));
  }

  // ── Health metrics CSV (default) ────────────────────────────────────────
  const metrics = await db.select().from(schema.healthMetrics)
    .where(eq(schema.healthMetrics.userId, userId)).orderBy(desc(schema.healthMetrics.timestamp)).limit(5000);
  if (metrics.length === 0) { res.setHeader('Content-Type', 'text/csv'); return res.send('No data available'); }
  const rows = metrics.map((m: (typeof metrics)[number]) => [m.timestamp, m.heartRate, m.stressLevel, m.sleepQuality, m.neuralActivity, m.dailySteps, m.sleepDuration].join(','));
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', `attachment; filename=health_metrics_${new Date().toISOString().slice(0, 10)}.csv`);
  return res.send(['timestamp,heartRate,stressLevel,sleepQuality,neuralActivity,dailySteps,sleepDuration', ...rows].join('\n'));
}

async function insightsWeekly(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const userId = req.query.userId as string;
  if (!userId) return error(res, 'userId required', 400);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  // Rate limit: 10 weekly insight requests per user per hour
  const rl = await checkRateLimit(db, `insights-weekly:${userId}`, 10, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!, 'Too many insight requests. Please wait before refreshing again.');
  const [dreams, emotions, metrics] = await Promise.all([
    db.select().from(schema.dreamAnalysis).where(eq(schema.dreamAnalysis.userId, userId)).orderBy(desc(schema.dreamAnalysis.timestamp)).limit(10),
    db.select().from(schema.emotionReadings).where(eq(schema.emotionReadings.userId, userId)).orderBy(desc(schema.emotionReadings.timestamp)).limit(50),
    db.select().from(schema.healthMetrics).where(eq(schema.healthMetrics.userId, userId)).orderBy(desc(schema.healthMetrics.timestamp)).limit(50),
  ]);
  const openai = getOpenAIClient();
  const ctx = {
    dreamCount: dreams.length,
    dreamSymbols: dreams.flatMap((d: (typeof dreams)[number]) => (d.symbols as string[]) || []),
    avgStress: emotions.length ? emotions.reduce((s: number, e: (typeof emotions)[number]) => s + e.stress, 0) / emotions.length : null,
    avgFocus: emotions.length ? emotions.reduce((s: number, e: (typeof emotions)[number]) => s + e.focus, 0) / emotions.length : null,
    avgSleepQuality: metrics.length ? metrics.reduce((s: number, m: (typeof metrics)[number]) => s + m.sleepQuality, 0) / metrics.length : null,
    dominantEmotions: emotions.slice(0, 10).map((e: (typeof emotions)[number]) => e.dominantEmotion),
  };
  try {
    const resp = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: 'You are an AI neuroscience wellness advisor. Generate 4 personalized weekly insights. Respond with JSON: {"insights":[{"title":"","description":"","type":"success|warning|info|secondary","icon":"brain|heart|moon|lightbulb"}],"weeklyScore":0,"recommendation":""}' },
        { role: 'user', content: `Generate weekly insights for: ${JSON.stringify(ctx)}` },
      ],
      response_format: { type: 'json_object' },
    });
    return success(res, JSON.parse(resp.choices[0].message.content || '{}'));
  } catch (err) {
    console.error('[insightsWeekly]', err instanceof Error ? err.message : err);
    return error(res, 'Failed to generate insights. Please try again.', 500);
  }
}

async function notificationsSubscribe(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { endpoint, keys } = req.body;
  const userId = authPayload.userId;
  if (!endpoint || !keys) return badRequest(res, 'endpoint and keys are required');
  if (typeof endpoint !== 'string' || endpoint.length > 2048) return badRequest(res, 'endpoint must be a string ≤2048 chars');
  try { new URL(endpoint); } catch { return badRequest(res, 'endpoint must be a valid URL'); }
  if (typeof keys !== 'object' || keys === null) return badRequest(res, 'keys must be an object');
  if (!keys.p256dh || typeof keys.p256dh !== 'string' || keys.p256dh.length > 256) return badRequest(res, 'keys.p256dh is required (max 256 chars)');
  if (!keys.auth || typeof keys.auth !== 'string' || keys.auth.length > 128) return badRequest(res, 'keys.auth is required (max 128 chars)');
  const db = getDb();
  // Rate limit: 10 subscription registrations per user per hour
  const rlSub = await checkRateLimit(db, `push-subscribe:${userId}`, 10, 60);
  if (!rlSub.allowed) return tooManyRequests(res, rlSub.retryAfterSeconds!);
  // Deduplicate: return existing record if this endpoint is already registered
  const [existing] = await db.select().from(schema.pushSubscriptions)
    .where(eq(schema.pushSubscriptions.endpoint, endpoint)).limit(1);
  if (existing) return success(res, existing);
  const [sub] = await db.insert(schema.pushSubscriptions).values({ userId, endpoint, keys }).returning();
  return success(res, sub, 201);
}

async function notificationsVapidPublicKey(_req: VercelRequest, res: VercelResponse) {
  const publicKey = process.env.VAPID_PUBLIC_KEY;
  if (!publicKey) return error(res, 'VAPID not configured', 503);
  return success(res, { publicKey });
}

async function notificationsTrigger(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const vapidPublic = process.env.VAPID_PUBLIC_KEY;
  const vapidPrivate = process.env.VAPID_PRIVATE_KEY;
  if (!vapidPublic || !vapidPrivate) return error(res, 'VAPID keys not configured', 503);

  const { userId: bodyUserId, title: rawTitle, body: rawBody, url: rawUrl } = req.body ?? {};
  // Always scope to the authenticated user — never allow broadcasting to all users
  const targetUserId = authPayload.userId;
  if (bodyUserId && bodyUserId !== targetUserId) return error(res, 'Forbidden', 403);
  // Sanitize notification content
  const title = typeof rawTitle === 'string' ? rawTitle.trim().slice(0, 100) : 'AntarAI';
  const notifBody = typeof rawBody === 'string' ? rawBody.trim().slice(0, 200) : 'Good morning! Your brain report is ready.';
  const url = typeof rawUrl === 'string' ? rawUrl.trim().slice(0, 200) : '/brain-report';

  const db = getDb();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let webPush: any;
  try {
    const m = await import('web-push');
    webPush = m.default ?? m;
  } catch (e) {
    return error(res, `web-push import failed: ${(e as Error).message}`, 500);
  }
  try {
    webPush.setVapidDetails('mailto:noreply@dream-analysis.vercel.app', vapidPublic.trim(), vapidPrivate.trim());
  } catch (e) {
    return error(res, `VAPID setup failed: ${(e as Error).message}`, 500);
  }

  // Fetch subscriptions scoped to the authenticated user only
  const subs = await db.select().from(schema.pushSubscriptions).where(eq(schema.pushSubscriptions.userId, targetUserId));

  if (subs.length === 0) return success(res, { sent: 0, message: 'No subscriptions found' });

  const payload = JSON.stringify({ title, body: notifBody, url });
  const results = await Promise.allSettled(
    subs.map((sub: { endpoint: string; keys: unknown }) =>
      webPush.sendNotification(
        { endpoint: sub.endpoint, keys: sub.keys as { p256dh: string; auth: string } },
        payload,
      )
    )
  );

  const sent = results.filter(r => r.status === 'fulfilled').length;
  const failed = results.length - sent;
  return success(res, { sent, failed, total: subs.length });
}

async function analyzeMood(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const rawText = req.body?.text;
  const text = typeof rawText === 'string' ? rawText.trim() : '';
  if (!text) return badRequest(res, 'Missing text to analyze');
  if (text.length > 2000) return badRequest(res, 'Text must be a string of ≤2000 characters');
  // Rate limit: 60 mood analyses per IP per hour
  const rlMood = await checkRateLimit(getDb(), `analyze-mood:${getClientIp(req)}`, 60, 60);
  if (!rlMood.allowed) return tooManyRequests(res, rlMood.retryAfterSeconds!);
  const openai = getOpenAIClient();
  try {
    const resp = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: 'Analyze the mood from text. Respond with JSON: {"mood":"","stressLevel":0,"emotions":[],"recommendations":[]}' },
        { role: 'user', content: text },
      ],
      response_format: { type: 'json_object' },
    });
    const parsed = JSON.parse(resp.choices[0].message.content || '{}');
    return success(res, parsed);
  } catch (err) {
    console.error('[analyzeMood]', err instanceof Error ? err.message : err);
    return error(res, 'Mood analysis failed', 500);
  }
}

// ── Food log ─────────────────────────────────────────────────────────────────

const FOOD_JSON_SCHEMA = `{
  "foodItems": [{"name":"...","portion":"...","calories":0,"carbs_g":0,"protein_g":0,"fat_g":0}],
  "totalCalories": 0,
  "dominantMacro": "carbs|protein|fat|balanced",
  "glycemicImpact": "low|medium|high",
  "moodImpact": "2-sentence prediction of mood/energy 2-4 hours later",
  "dreamRelevance": "2-sentence note on how this may affect tonight's sleep or dream vividness",
  "summary": "One plain-English sentence describing what was eaten",
  "vitamins": {"vitamin_d_mcg":0,"vitamin_b12_mcg":0,"vitamin_c_mg":0,"iron_mg":0,"magnesium_mg":0,"zinc_mg":0,"omega3_g":0}
}`;

async function foodAnalyze(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const body = req.body ?? {};
  const { imageBase64, textDescription, mealType, moodBefore, notes } = body;
  const userId = authPayload.userId;
  if (!textDescription && !imageBase64) return badRequest(res, 'Provide a photo or describe what you ate');
  if (textDescription && (typeof textDescription !== 'string' || textDescription.length > 1000)) return badRequest(res, 'Description must be ≤1000 characters');
  // ~10 MB base64 limit — prevents memory exhaustion and excessive vision API costs
  if (imageBase64 && (typeof imageBase64 !== 'string' || imageBase64.length > 10_485_760)) return badRequest(res, 'Image exceeds 10 MB limit');
  // Rate limit: 30 food analyses per user per hour
  const rlFood = await checkRateLimit(getDb(), `food-analyze:${userId}`, 30, 60);
  if (!rlFood.allowed) return tooManyRequests(res, rlFood.retryAfterSeconds!, 'Too many food analysis requests. Please wait before trying again.');

  const db = getDb();
  const jsonPrompt = `Estimate nutrition and return ONLY valid JSON with this exact shape:\n${FOOD_JSON_SCHEMA}`;

  try {
    let content: string;

    if (imageBase64) {
      // Vision path: GPT-4o-mini analyzes the food photo directly
      const openaiKey = process.env.OPENAI_API_KEY;
      if (!openaiKey) return error(res, 'Vision AI not configured (OPENAI_API_KEY missing)', 503);
      const { default: OpenAI } = await import('openai');
      const visionClient = new OpenAI({ apiKey: openaiKey });
      const textPart = textDescription
        ? `The user describes their ${mealType ?? 'meal'}: "${textDescription}". Also analyze the photo.`
        : `Analyze this ${mealType ?? 'meal'} photo.`;
      const resp = await visionClient.chat.completions.create({
        model: 'llama-3.3-70b-versatile',
        messages: [{ role: 'user', content: [
          { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${imageBase64}`, detail: 'low' } },
          { type: 'text', text: `${textPart}\n\n${jsonPrompt}` },
        ] }],
        max_tokens: 1024,
      });
      content = resp.choices[0].message.content ?? '{}';
    } else {
      // Text-only path: Cerebras Llama 3.1
      let openai;
      try { openai = getOpenAIClient(); } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'AI service not configured';
        return error(res, `AI service unavailable: ${msg}`, 503);
      }
      const resp = await openai.chat.completions.create({
        model: 'llama-3.3-70b-versatile',
        messages: [{ role: 'user', content: `The user describes their ${mealType ?? 'meal'}: "${textDescription}"\n\n${jsonPrompt}` }],
        response_format: { type: 'json_object' },
      });
      content = resp.choices[0].message.content ?? '{}';
    }

    // Strip markdown fences if present
    const stripped = content.replace(/```json?\n?/g, '').replace(/```/g, '').trim();
    let analysis;
    try {
      analysis = JSON.parse(stripped || '{}');
    } catch (parseErr) {
      console.error('Food analyze — AI returned invalid JSON:', stripped);
      return error(res, 'AI returned invalid response. Please try again.', 502);
    }

    const [log] = await db.insert(schema.foodLogs).values({
      userId,
      mealType: mealType ?? 'snack',
      foodItems: analysis.foodItems ?? [],
      totalCalories: typeof analysis.totalCalories === 'number' ? analysis.totalCalories : null,
      dominantMacro: typeof analysis.dominantMacro === 'string' ? analysis.dominantMacro : null,
      glycemicImpact: typeof analysis.glycemicImpact === 'string' ? analysis.glycemicImpact : null,
      aiMoodImpact: typeof analysis.moodImpact === 'string' ? analysis.moodImpact : null,
      aiDreamRelevance: typeof analysis.dreamRelevance === 'string' ? analysis.dreamRelevance : null,
      summary: typeof analysis.summary === 'string' ? analysis.summary : null,
      moodBefore: moodBefore ?? null,
      notes: notes ?? null,
    }).returning();

    // Push nutrition metrics to health pipeline (best-effort)
    const totalCalories = typeof analysis.totalCalories === 'number' ? analysis.totalCalories : 0;
    const foodItems = Array.isArray(analysis.foodItems) ? analysis.foodItems : [];
    const totalProtein = foodItems.reduce((s: number, f: any) => s + (f?.protein_g ?? 0), 0);
    const totalCarbs = foodItems.reduce((s: number, f: any) => s + (f?.carbs_g ?? 0), 0);
    const totalFat = foodItems.reduce((s: number, f: any) => s + (f?.fat_g ?? 0), 0);
    const supabaseUrl = process.env.SUPABASE_URL;
    if (supabaseUrl && totalCalories > 0) {
      try {
        const now = new Date().toISOString();
        const samples: { source: string; metric: string; value: number; unit: string; recorded_at: string }[] = [
          { source: 'manual', metric: 'total_calories', value: totalCalories, unit: 'kcal', recorded_at: now },
        ];
        if (totalProtein > 0) samples.push({ source: 'manual', metric: 'total_protein_g', value: totalProtein, unit: 'g', recorded_at: now });
        if (totalCarbs > 0) samples.push({ source: 'manual', metric: 'total_carbs_g', value: totalCarbs, unit: 'g', recorded_at: now });
        if (totalFat > 0) samples.push({ source: 'manual', metric: 'total_fat_g', value: totalFat, unit: 'g', recorded_at: now });
        await fetch(`${supabaseUrl}/functions/v1/ingest-health-data`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, samples }),
        });
      } catch (e) { console.error('Pipeline push failed:', e); }
    }

    // Also insert into health_samples table directly (dual-write for reliability)
    if (totalCalories > 0) {
      try {
        const now = new Date();
        await db.insert(schema.healthSamples).values({
          userId,
          source: 'manual',
          metric: 'total_calories',
          value: totalCalories,
          unit: 'kcal',
          recordedAt: now,
        }).onConflictDoNothing();
        if (totalProtein > 0) {
          await db.insert(schema.healthSamples).values({
            userId, source: 'manual', metric: 'total_protein_g', value: totalProtein, unit: 'g', recordedAt: now,
          }).onConflictDoNothing();
        }
        if (totalCarbs > 0) {
          await db.insert(schema.healthSamples).values({
            userId, source: 'manual', metric: 'total_carbs_g', value: totalCarbs, unit: 'g', recordedAt: now,
          }).onConflictDoNothing();
        }
        if (totalFat > 0) {
          await db.insert(schema.healthSamples).values({
            userId, source: 'manual', metric: 'total_fat_g', value: totalFat, unit: 'g', recordedAt: now,
          }).onConflictDoNothing();
        }
      } catch (e) { console.error('health_samples insert failed (non-fatal):', e); }
    }

    return success(res, { ...analysis, id: log.id, loggedAt: log.loggedAt }, 201);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : 'Unknown error';
    console.error('Food analyze error:', msg);
    if (msg.includes('API key') || msg.includes('auth') || msg.includes('401')) {
      return error(res, 'AI-powered meal suggestions require an active API key. Please check your configuration.', 503);
    }
    return error(res, `Food analysis failed: ${msg}`, 500);
  }
}

async function foodLogs(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const url = new URL(req.url ?? '', `http://${req.headers.host}`);
  const limit = Math.min(Math.max(parseInt(url.searchParams.get('limit') ?? '50', 10), 1), 200);
  const offset = Math.max(parseInt(url.searchParams.get('offset') ?? '0', 10), 0);
  const logs = await db.select().from(schema.foodLogs)
    .where(eq(schema.foodLogs.userId, userId))
    .orderBy(desc(schema.foodLogs.loggedAt))
    .limit(limit)
    .offset(offset);
  return success(res, logs);
}

// ── Brain history — reads emotion data from emotionReadings + userReadings ────

async function brainHistory(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const url = new URL(req.url ?? '', `http://${req.headers.host}`);
  const days = Math.min(Math.max(parseInt(url.searchParams.get('days') ?? '7', 10), 1), 365);
  const fromTs = new Date(Date.now() - days * 86400000);

  // Read from emotionReadings (EEG + voice check-in data)
  const readings = await db.select().from(schema.emotionReadings)
    .where(and(eq(schema.emotionReadings.userId, userId), gte(schema.emotionReadings.timestamp, fromTs)))
    .orderBy(desc(schema.emotionReadings.timestamp))
    .limit(2000);

  // Also read from userReadings (voice chat, manual, health)
  const userR = await db.select().from(schema.userReadings)
    .where(and(eq(schema.userReadings.userId, userId), gte(schema.userReadings.createdAt, fromTs)))
    .orderBy(desc(schema.userReadings.createdAt))
    .limit(500);

  // Merge into unified format
  const result = [
    ...readings.map((r: any) => ({
      stress: r.stress ?? 0,
      happiness: r.happiness ?? 0,
      focus: r.focus ?? 0,
      dominantEmotion: r.dominantEmotion ?? 'neutral',
      valence: r.valence ?? null,
      timestamp: r.timestamp?.toISOString?.() ?? r.timestamp,
    })),
    ...userR.map((r: any) => ({
      stress: r.stress ?? 0,
      happiness: r.valence != null ? Math.max(0, r.valence) : 0,
      focus: r.focus ?? 0,
      dominantEmotion: r.emotion ?? 'neutral',
      valence: r.valence ?? null,
      timestamp: r.createdAt?.toISOString?.() ?? r.createdAt,
    })),
  ];

  result.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  return success(res, result.slice(0, 2000));
}

// ── Brain today-totals — avg stress/focus/emotion since midnight ──────────────

async function brainTodayTotals(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const midnight = new Date();
  midnight.setHours(0, 0, 0, 0);

  const readings = await db.select().from(schema.emotionReadings)
    .where(and(eq(schema.emotionReadings.userId, userId), gte(schema.emotionReadings.timestamp, midnight)))
    .orderBy(desc(schema.emotionReadings.timestamp))
    .limit(2000);

  if (readings.length === 0) {
    return success(res, { userId, count: 0, avgStress: null, avgFocus: null, avgHappiness: null, avgEnergy: null, dominantEmotion: null });
  }

  const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const emotionCounts: Record<string, number> = {};
  readings.forEach((r: any) => {
    const emo = r.dominantEmotion ?? 'neutral';
    emotionCounts[emo] = (emotionCounts[emo] || 0) + 1;
  });
  const dominantEmotion = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;

  return success(res, {
    userId,
    count: readings.length,
    avgStress: avg(readings.map((r: any) => r.stress ?? 0)),
    avgFocus: avg(readings.map((r: any) => r.focus ?? 0)),
    avgHappiness: avg(readings.map((r: any) => r.happiness ?? 0)),
    avgEnergy: avg(readings.map((r: any) => r.energy ?? 0)),
    avgValence: avg(readings.filter((r: any) => r.valence != null).map((r: any) => r.valence)),
    avgArousal: avg(readings.filter((r: any) => r.arousal != null).map((r: any) => r.arousal)),
    dominantEmotion,
  });
}

// ── Brain at-this-time-yesterday — ±30 min window same time yesterday ────────

async function brainAtThisTimeYesterday(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const now = Date.now();
  const oneDayMs = 24 * 60 * 60 * 1000;
  const windowMs = 30 * 60 * 1000; // ±30 min
  const fromTs = new Date(now - oneDayMs - windowMs);
  const toTs = new Date(now - oneDayMs + windowMs);

  const readings = await db.select().from(schema.emotionReadings)
    .where(and(
      eq(schema.emotionReadings.userId, userId),
      gte(schema.emotionReadings.timestamp, fromTs),
      lt(schema.emotionReadings.timestamp, toTs),
    ))
    .orderBy(desc(schema.emotionReadings.timestamp))
    .limit(200);

  if (readings.length === 0) {
    return success(res, { userId, count: 0, avgStress: null, avgFocus: null, avgHappiness: null });
  }

  const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
  return success(res, {
    userId,
    count: readings.length,
    windowStart: fromTs.toISOString(),
    windowEnd: toTs.toISOString(),
    avgStress: avg(readings.map((r: any) => r.stress ?? 0)),
    avgFocus: avg(readings.map((r: any) => r.focus ?? 0)),
    avgHappiness: avg(readings.map((r: any) => r.happiness ?? 0)),
    avgEnergy: avg(readings.map((r: any) => r.energy ?? 0)),
    avgValence: avg(readings.filter((r: any) => r.valence != null).map((r: any) => r.valence)),
  });
}

const emotionReadingSchema = z.object({
  userId: z.string().min(1),
  sessionId: z.string().nullable().optional(),
  stress: z.number().min(0).max(1).default(0),
  happiness: z.number().min(0).max(1).default(0),
  focus: z.number().min(0).max(1).default(0),
  energy: z.number().min(0).max(1).default(0),
  dominantEmotion: z.string().max(50).default('neutral'),
  valence: z.number().min(-1).max(1).nullable().optional(),
  arousal: z.number().min(0).max(1).nullable().optional(),
});

async function emotionReadingsBatch(req: VercelRequest, res: VercelResponse) {
  const body = await parseRequestBody(req) as unknown;
  const parsed = z.object({ readings: z.array(emotionReadingSchema).min(1).max(50) }).safeParse(body);
  if (!parsed.success) return badRequest(res, parsed.error.issues[0]?.message ?? 'Invalid readings');
  const db = getDb();
  // Rate limit by IP: 200 batch calls per hour
  const ip = getClientIp(req);
  const rl = await checkRateLimit(db, `emotion-batch:${ip}`, 200, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const rows = parsed.data.readings.map((r) => ({
    userId: r.userId, sessionId: r.sessionId ?? null,
    stress: r.stress, happiness: r.happiness, focus: r.focus, energy: r.energy,
    dominantEmotion: r.dominantEmotion, valence: r.valence ?? null, arousal: r.arousal ?? null,
  }));
  try {
    await db.insert(schema.emotionReadings).values(rows).onConflictDoNothing();
  } catch (err) {
    console.error('[emotionReadingsBatch]', err instanceof Error ? err.message : err);
    return error(res, 'Failed to save readings', 500);
  }
  return success(res, { saved: rows.length }, 201);
}

const userReadingSchema = z.object({
  userId: z.string().min(1),
  source: z.enum(['voice', 'eeg', 'health', 'manual']).default('voice'),
  emotion: z.string().max(50).default('neutral'),
  valence: z.number().min(-1).max(1).nullable().optional(),
  arousal: z.number().min(0).max(1).nullable().optional(),
  stress: z.number().min(0).max(1).nullable().optional(),
  confidence: z.number().min(0).max(1).nullable().optional(),
  modelType: z.string().max(50).default('voice'),
});

async function userReadingsPost(req: VercelRequest, res: VercelResponse) {
  const body = await parseRequestBody(req) as unknown;
  const parsed = userReadingSchema.safeParse(body);
  if (!parsed.success) return badRequest(res, parsed.error.issues[0]?.message ?? 'Invalid reading data');
  const db = getDb();
  // Rate limit by IP: 200 readings per hour (anonymous participant support — no auth required)
  const ip = getClientIp(req);
  const rl = await checkRateLimit(db, `user-readings:${ip}`, 200, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const [row] = await db.insert(schema.userReadings).values({
    userId: parsed.data.userId, source: parsed.data.source, emotion: parsed.data.emotion,
    valence: parsed.data.valence ?? null, arousal: parsed.data.arousal ?? null,
    stress: parsed.data.stress ?? null, confidence: parsed.data.confidence ?? null,
    modelType: parsed.data.modelType,
  }).returning();
  return success(res, row, 201);
}

const foodLogSchema = z.object({
  userId: z.string().min(1),
  mealType: z.enum(['breakfast', 'lunch', 'dinner', 'snack', 'meal']).default('meal'),
  summary: z.string().max(500).nullable().optional(),
  totalCalories: z.number().min(0).max(10000).nullable().optional(),
  dominantMacro: z.enum(['protein', 'carbs', 'fat', 'balanced']).nullable().optional(),
  foodItems: z.array(z.unknown()).nullable().optional(),
});

async function foodLog(req: VercelRequest, res: VercelResponse) {
  // Authenticate — log goes to the authenticated user (ignore body userId)
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const parsed = foodLogSchema.safeParse(req.body);
  if (!parsed.success) return badRequest(res, parsed.error.issues[0]?.message ?? 'Invalid food log data');
  const db = getDb();
  // Rate limit: 50 food log entries per user per hour
  const rl = await checkRateLimit(db, `food-log:${authPayload.userId}`, 50, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const [row] = await db.insert(schema.foodLogs).values({
    userId: authPayload.userId, mealType: parsed.data.mealType,
    summary: parsed.data.summary ?? null, totalCalories: parsed.data.totalCalories ?? null,
    dominantMacro: parsed.data.dominantMacro ?? null, foodItems: parsed.data.foodItems ?? null,
  }).returning();
  return success(res, row, 201);
}

// ── Study helpers ─────────────────────────────────────────────────────────────

async function getActiveParticipant(userId: string) {
  const db = getDb();
  const [p] = await db.select().from(schema.studyParticipants)
    .where(and(
      eq(schema.studyParticipants.userId, userId),
      eq(schema.studyParticipants.status, 'active')
    )).limit(1);
  return p ?? null;
}

async function getOrCreateTodaySession(participant: typeof schema.studyParticipants.$inferSelect) {
  const db = getDb();
  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);
  const tomorrowStart = new Date(todayStart);
  tomorrowStart.setDate(tomorrowStart.getDate() + 1);

  const [existing] = await db.select().from(schema.studySessions)
    .where(and(
      eq(schema.studySessions.participantId, participant.id),
      gte(schema.studySessions.sessionDate, todayStart),
      lt(schema.studySessions.sessionDate, tomorrowStart)
    )).limit(1);
  if (existing) return existing;

  const [{ n }] = await db.select({ n: sql<number>`count(*)::int` })
    .from(schema.studySessions)
    .where(eq(schema.studySessions.participantId, participant.id));
  const dayNumber = (n ?? 0) + 1;

  const [created] = await db.insert(schema.studySessions).values({
    participantId: participant.id,
    studyCode: participant.studyCode,
    dayNumber,
    sessionDate: todayStart,
  }).returning();
  return created;
}

async function checkAndMarkValidDay(sessionId: string): Promise<boolean> {
  const db = getDb();
  const [session] = await db.select().from(schema.studySessions)
    .where(eq(schema.studySessions.id, sessionId));
  if (!session) return false;
  const done = [session.morningCompleted, session.daytimeCompleted, session.eveningCompleted]
    .filter(Boolean).length;
  const isValid = done >= 2;
  if (isValid && !session.validDay) {
    await db.update(schema.studySessions).set({ validDay: true }).where(eq(schema.studySessions.id, sessionId));
  }
  return isValid;
}

// ── Study route handlers ──────────────────────────────────────────────────────

async function studyEnroll(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { userId, studyId, consentVersion, overnightEegConsent,
          preferredMorningTime, preferredDaytimeTime, preferredEveningTime,
          consentFullName, consentInitials } = req.body;
  if (!userId || !studyId || !consentVersion)
    return badRequest(res, 'userId, studyId, and consentVersion are required');
  if (!requireOwner(req, res, userId)) return;
  const KNOWN_STUDY_IDS = ['svapnastra-beta-v1', 'svapnastra-beta-v2', 'emotional-day-night-v1', 'dream-analysis-v1'];
  if (!KNOWN_STUDY_IDS.includes(studyId))
    return badRequest(res, 'Unknown studyId');
  const already = await getActiveParticipant(userId);
  if (already) return res.status(409).json({ message: 'Already enrolled in an active study', studyCode: already.studyCode });
  const db = getDb();
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  const makeCode = () => Array.from({ length: 6 }, () => chars[Math.floor(Math.random() * chars.length)]).join('');
  let studyCode = makeCode();
  let collision = await db.select({ id: schema.studyParticipants.id }).from(schema.studyParticipants)
    .where(eq(schema.studyParticipants.studyCode, studyCode)).limit(1);
  while (collision.length > 0) {
    studyCode = makeCode();
    collision = await db.select({ id: schema.studyParticipants.id }).from(schema.studyParticipants)
      .where(eq(schema.studyParticipants.studyCode, studyCode)).limit(1);
  }
  const [participant] = await db.insert(schema.studyParticipants).values({
    userId, studyId, studyCode, consentVersion,
    consentSignedAt: new Date(),
    consentFullName: consentFullName ? String(consentFullName).trim().slice(0, 200) : null,
    consentInitials: consentInitials ? String(consentInitials).trim().slice(0, 10) : null,
    overnightEegConsent: overnightEegConsent ?? false,
    preferredMorningTime, preferredDaytimeTime, preferredEveningTime,
  }).returning();
  return success(res, { studyCode: participant.studyCode, enrolledAt: participant.enrolledAt }, 201);
}

async function studyStatus(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const participant = await getActiveParticipant(userId);
  if (!participant) return success(res, { enrolled: false });
  const db = getDb();
  const todayStart = new Date(); todayStart.setHours(0, 0, 0, 0);
  const tomorrowStart = new Date(todayStart); tomorrowStart.setDate(tomorrowStart.getDate() + 1);
  const [today] = await db.select().from(schema.studySessions)
    .where(and(
      eq(schema.studySessions.participantId, participant.id),
      gte(schema.studySessions.sessionDate, todayStart),
      lt(schema.studySessions.sessionDate, tomorrowStart)
    )).limit(1);
  return success(res, {
    enrolled: true,
    studyCode: participant.studyCode,
    completedDays: participant.completedDays,
    targetDays: participant.targetDays,
    todaySession: today ?? null,
    preferredTimes: {
      morning: participant.preferredMorningTime,
      daytime: participant.preferredDaytimeTime,
      evening: participant.preferredEveningTime,
    },
  });
}

async function studyMorning(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { userId, dreamText, noRecall, dreamValence, dreamArousal,
          nightmareFlag, sleepQuality, sleepHours,
          minutesFromWaking, currentMoodRating } = req.body;
  if (!userId) return badRequest(res, 'userId is required');
  if (!requireOwner(req, res, userId)) return;
  const participant = await getActiveParticipant(userId);
  if (!participant) return error(res, 'Not enrolled in an active study', 404);
  const session = await getOrCreateTodaySession(participant);
  if (session.morningCompleted) return res.status(409).json({ message: 'Morning entry already submitted today' });
  const db = getDb();
  const safeDreamText = typeof dreamText === 'string' ? dreamText.trim().slice(0, 5000) : null;
  await db.insert(schema.studyMorningEntries).values({
    sessionId: session.id,
    studyCode: participant.studyCode,
    dreamText: noRecall ? null : (safeDreamText ?? null),
    noRecall: noRecall ?? false,
    dreamValence, dreamArousal, nightmareFlag,
    sleepQuality, sleepHours, minutesFromWaking, currentMoodRating,
  });
  await db.update(schema.studySessions).set({ morningCompleted: true }).where(eq(schema.studySessions.id, session.id));
  await checkAndMarkValidDay(session.id);
  const needsSupport = typeof currentMoodRating === 'number' && currentMoodRating <= 2;
  return success(res, { success: true, dayNumber: session.dayNumber, needsSupport });
}

async function studyDaytime(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { userId, eegFeatures, faa, highBeta, fmt, sqiMean, eegDurationSec,
          samValence, samArousal, samStress, panasItems,
          sleepHoursReported, caffeineServings, significantEventYN } = req.body;
  if (!userId) return badRequest(res, 'userId is required');
  if (!requireOwner(req, res, userId)) return;
  const participant = await getActiveParticipant(userId);
  if (!participant) return error(res, 'Not enrolled in an active study', 404);
  const session = await getOrCreateTodaySession(participant);
  if (session.daytimeCompleted) return res.status(409).json({ message: 'Daytime entry already submitted today' });
  const db = getDb();
  await db.insert(schema.studyDaytimeEntries).values({
    sessionId: session.id,
    studyCode: participant.studyCode,
    eegFeatures, faa, highBeta, fmt, sqiMean, eegDurationSec,
    samValence, samArousal, samStress, panasItems,
    sleepHoursReported, caffeineServings, significantEventYN,
  });
  await db.update(schema.studySessions).set({ daytimeCompleted: true }).where(eq(schema.studySessions.id, session.id));
  await checkAndMarkValidDay(session.id);
  return success(res, { success: true, dayNumber: session.dayNumber });
}

async function studyEvening(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { userId, dayValence, dayArousal, peakEmotionIntensity,
          peakEmotionDirection, meals, emotionalEatingDay,
          cravingsToday, cravingTypes, exerciseLevel, alcoholDrinks,
          supplementsTaken, medicationsTaken, medicationsDetails,
          stressRightNow, readyForSleep } = req.body;
  if (!userId) return badRequest(res, 'userId is required');
  if (!requireOwner(req, res, userId)) return;
  const participant = await getActiveParticipant(userId);
  if (!participant) return error(res, 'Not enrolled in an active study', 404);
  const session = await getOrCreateTodaySession(participant);
  if (session.eveningCompleted) return res.status(409).json({ message: 'Evening entry already submitted today' });
  const db = getDb();
  await db.insert(schema.studyEveningEntries).values({
    sessionId: session.id,
    studyCode: participant.studyCode,
    dayValence, dayArousal, peakEmotionIntensity, peakEmotionDirection,
    meals, emotionalEatingDay, cravingsToday, cravingTypes,
    exerciseLevel, alcoholDrinks, supplementsTaken, medicationsTaken,
    medicationsDetails, stressRightNow, readyForSleep,
  });
  await db.update(schema.studySessions).set({ eveningCompleted: true }).where(eq(schema.studySessions.id, session.id));
  const isValid = await checkAndMarkValidDay(session.id);
  if (isValid) {
    await db.update(schema.studyParticipants)
      .set({ completedDays: sql`${schema.studyParticipants.completedDays} + 1` })
      .where(eq(schema.studyParticipants.id, participant.id));
  }
  const finalDays = (participant.completedDays ?? 0) + (isValid ? 1 : 0);
  return success(res, { success: true, validDay: isValid, completedDays: finalDays });
}

async function studyHistory(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const participant = await getActiveParticipant(userId);
  if (!participant) return success(res, []);
  const db = getDb();
  const sessions = await db.select().from(schema.studySessions)
    .where(eq(schema.studySessions.participantId, participant.id))
    .orderBy(asc(schema.studySessions.dayNumber));
  return success(res, sessions);
}

async function studyWithdraw(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { userId } = req.body;
  if (!userId) return badRequest(res, 'userId is required');
  if (!requireOwner(req, res, userId)) return;
  const participant = await getActiveParticipant(userId);
  if (!participant) return error(res, 'No active study enrollment found', 404);
  const db = getDb();
  await db.update(schema.studyParticipants)
    .set({ status: 'withdrawn', withdrawnAt: new Date() })
    .where(eq(schema.studyParticipants.id, participant.id));
  return success(res, { daysCompleted: participant.completedDays ?? 0, message: 'You have been withdrawn from the study. Thank you for your contribution.' });
}

// ── Pilot study routes ────────────────────────────────────────────────────────

async function pilotConsent(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { participant_code, age, diet_type, has_apple_watch, consent_text } = req.body;
  if (!participant_code || typeof participant_code !== 'string' || participant_code.length > 100) return badRequest(res, 'participant_code is required (max 100 chars)');
  const db = getDb();
  const rl = await checkRateLimit(db, `pilot-consent:${getClientIp(req)}`, 10, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const parsedAge = age != null ? Number(age) : null;
  if (parsedAge !== null && (!isFinite(parsedAge) || parsedAge < 0 || parsedAge > 120)) return badRequest(res, 'age must be 0–120');
  await db.insert(schema.pilotParticipants).values({
    participantCode:  participant_code.trim().slice(0, 100),
    age:              parsedAge,
    dietType:         typeof diet_type === 'string' ? diet_type.slice(0, 50) : null,
    hasAppleWatch:    has_apple_watch ? true : false,
    consentText:      typeof consent_text === 'string' ? consent_text.slice(0, 2000) : null,
    consentTimestamp: new Date(),
  }).onConflictDoNothing();
  return success(res, { success: true, participant_code });
}

async function pilotSessionStart(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { participant_code, block_type } = req.body;
  if (!participant_code || typeof participant_code !== 'string') return badRequest(res, 'participant_code is required');
  if (!block_type || typeof block_type !== 'string') return badRequest(res, 'block_type is required');
  const ALLOWED_BLOCK_TYPES = ['rest', 'task', 'music', 'meditation', 'breathing', 'control', 'intervention'];
  if (!ALLOWED_BLOCK_TYPES.includes(block_type)) return badRequest(res, `block_type must be one of: ${ALLOWED_BLOCK_TYPES.join(', ')}`);
  const db = getDb();
  const rl = await checkRateLimit(db, `pilot-session:${getClientIp(req)}`, 20, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const [row] = await db.insert(schema.pilotSessions).values({
    participantCode:       participant_code.trim().slice(0, 100),
    blockType:             block_type,
    interventionTriggered: false,
  }).returning({ id: schema.pilotSessions.id });
  return success(res, { session_id: row.id });
}

async function pilotSessionComplete(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { session_id, pre_eeg_json, post_eeg_json, eeg_features_json, survey_json, intervention_triggered } = req.body;
  if (session_id == null) return badRequest(res, 'session_id is required');
  // Enforce JSON field size caps to prevent oversized payloads in the DB
  const MAX_JSON_BYTES = 65536; // 64 KB per field
  for (const [name, val] of [['pre_eeg_json', pre_eeg_json], ['post_eeg_json', post_eeg_json], ['eeg_features_json', eeg_features_json], ['survey_json', survey_json]] as const) {
    if (val != null && JSON.stringify(val).length > MAX_JSON_BYTES) return badRequest(res, `${name} exceeds 64 KB limit`);
  }
  const db = getDb();
  // Rate limit: 20 completions per IP per hour
  const rl = await checkRateLimit(db, `pilot-complete:${getClientIp(req)}`, 20, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  await db.update(schema.pilotSessions)
    .set({
      preEegJson:            pre_eeg_json ?? null,
      postEegJson:           post_eeg_json ?? null,
      eegFeaturesJson:       eeg_features_json ?? null,
      surveyJson:            survey_json ?? null,
      interventionTriggered: intervention_triggered ? true : false,
    })
    .where(eq(schema.pilotSessions.id, Number(session_id)));
  return success(res, { success: true });
}

async function pilotAdminParticipants(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireAdmin(req, res)) return;
  const db = getDb();
  const rows = await db.select().from(schema.pilotParticipants).orderBy(desc(schema.pilotParticipants.createdAt));
  return success(res, rows);
}

async function pilotAdminSessions(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireAdmin(req, res)) return;
  const db = getDb();
  const rows = await db.select().from(schema.pilotSessions).orderBy(desc(schema.pilotSessions.createdAt));
  return success(res, rows);
}

async function pilotAdminStats(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireAdmin(req, res)) return;
  const db = getDb();
  const participants = await db.select().from(schema.pilotParticipants);
  const sessions = await db.select().from(schema.pilotSessions);

  const total_participants = participants.length;
  const total_sessions = sessions.length;
  const stress_sessions = sessions.filter((s: any) => s.blockType === 'stress').length;
  const food_sessions = sessions.filter((s: any) => s.blockType === 'food').length;
  const complete_sessions = sessions.filter((s: any) => !s.partial && s.surveyJson !== null).length;
  const partial_sessions = sessions.filter((s: any) => s.partial === true).length;

  const qualityScores = sessions.map((s: any) => s.dataQualityScore).filter((v: any): v is number => v != null);
  const avg_quality_score = qualityScores.length > 0 ? qualityScores.reduce((a: number, b: number) => a + b, 0) / qualityScores.length : 0;

  const durations = sessions.map((s: any) => s.durationSeconds).filter((v: any): v is number => v != null);
  const avg_duration_seconds = durations.length > 0 ? durations.reduce((a: number, b: number) => a + b, 0) / durations.length : 0;

  const stressReductions: number[] = [];
  for (const s of sessions) {
    if ((s as any).blockType !== 'stress') continue;
    const pre = (s as any).preEegJson as Record<string, unknown> | null;
    const post = (s as any).postEegJson as Record<string, unknown> | null;
    if (pre && typeof pre === 'object' && typeof pre.stress_level === 'number' &&
        post && typeof post === 'object' && typeof post.stress_level === 'number') {
      stressReductions.push(pre.stress_level - post.stress_level);
    }
  }
  const avg_stress_reduction = stressReductions.length > 0 ? stressReductions.reduce((a, b) => a + b, 0) / stressReductions.length : 0;

  return success(res, { total_participants, total_sessions, stress_sessions, food_sessions, complete_sessions, partial_sessions, avg_quality_score, avg_duration_seconds, avg_stress_reduction });
}

async function pilotAdminExportCsv(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireAdmin(req, res)) return;
  const db = getDb();
  const rows = await db
    .select({
      participantCode:       schema.pilotSessions.participantCode,
      blockType:             schema.pilotSessions.blockType,
      age:                   schema.pilotParticipants.age,
      dietType:              schema.pilotParticipants.dietType,
      hasAppleWatch:         schema.pilotParticipants.hasAppleWatch,
      interventionTriggered: schema.pilotSessions.interventionTriggered,
      preEegJson:            schema.pilotSessions.preEegJson,
      postEegJson:           schema.pilotSessions.postEegJson,
      surveyJson:            schema.pilotSessions.surveyJson,
    })
    .from(schema.pilotSessions)
    .leftJoin(schema.pilotParticipants, eq(schema.pilotSessions.participantCode, schema.pilotParticipants.participantCode))
    .orderBy(desc(schema.pilotSessions.createdAt));

  const EEG_BANDS = ['alpha', 'beta', 'theta', 'delta', 'gamma'] as const;
  const surveyKeys = new Set<string>();
  for (const row of rows) {
    const s = row.surveyJson as Record<string, unknown> | null;
    if (s && typeof s === 'object') {
      for (const [k, v] of Object.entries(s)) {
        if (typeof v === 'number') surveyKeys.add(k);
      }
    }
  }
  const sortedSurveyKeys = Array.from(surveyKeys).sort();
  const headers = [
    'participant_code', 'block_type', 'age', 'diet_type', 'has_apple_watch', 'intervention_triggered',
    ...EEG_BANDS.map(b => `pre_${b}`),
    ...EEG_BANDS.map(b => `post_${b}`),
    ...sortedSurveyKeys,
  ];
  const cell = (v: unknown): string => v == null ? '' : String(v).replace(/,/g, ';');
  const csvLines = [headers.join(',')];
  for (const row of rows) {
    const pre = row.preEegJson as Record<string, unknown> | null;
    const post = row.postEegJson as Record<string, unknown> | null;
    const survey = row.surveyJson as Record<string, unknown> | null;
    csvLines.push([
      cell(row.participantCode), cell(row.blockType), cell(row.age), cell(row.dietType),
      cell(row.hasAppleWatch), cell(row.interventionTriggered),
      ...EEG_BANDS.map(b => cell(pre?.[b])),
      ...EEG_BANDS.map(b => cell(post?.[b])),
      ...sortedSurveyKeys.map(k => cell(survey?.[k])),
    ].join(','));
  }
  const dateStr = new Date().toISOString().slice(0, 10);
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', `attachment; filename="study-data-${dateStr}.csv"`);
  return res.send(csvLines.join('\n'));
}

async function pilotSessionCheckpoint(req: VercelRequest, res: VercelResponse, sessionId: number) {
  if (req.method !== 'PATCH') return methodNotAllowed(res, ['PATCH']);
  const { pre_eeg_json, post_eeg_json, eeg_features_json, intervention_triggered, partial, phase_log } = req.body;
  // Enforce JSON field size caps
  const MAX_JSON_BYTES = 65536; // 64 KB per field
  for (const [name, val] of [['pre_eeg_json', pre_eeg_json], ['post_eeg_json', post_eeg_json], ['eeg_features_json', eeg_features_json], ['phase_log', phase_log]] as const) {
    if (val != null && JSON.stringify(val).length > MAX_JSON_BYTES) return badRequest(res, `${name} exceeds 64 KB limit`);
  }
  const db = getDb();
  // Rate limit: 60 checkpoints per IP per hour
  const rl = await checkRateLimit(db, `pilot-checkpoint:${getClientIp(req)}`, 60, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  await db.update(schema.pilotSessions)
    .set({
      ...(pre_eeg_json !== undefined      && { preEegJson: pre_eeg_json }),
      ...(post_eeg_json !== undefined     && { postEegJson: post_eeg_json }),
      ...(eeg_features_json !== undefined && { eegFeaturesJson: eeg_features_json }),
      ...(intervention_triggered !== undefined && { interventionTriggered: !!intervention_triggered }),
      ...(partial !== undefined           && { partial: !!partial }),
      ...(phase_log !== undefined         && { phaseLog: phase_log }),
      checkpointAt: new Date(),
    })
    .where(eq(schema.pilotSessions.id, sessionId));
  return success(res, { success: true });
}

async function userIntentGet(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const payload = requireAuth(req, res);
  if (!payload) return;
  const db = getDb();
  const [u] = await db.select({ intent: schema.users.intent }).from(schema.users).where(eq(schema.users.id, payload.userId));
  return success(res, { intent: u?.intent ?? null });
}

async function userIntentPatch(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'PATCH') return methodNotAllowed(res, ['PATCH']);
  const payload = requireAuth(req, res);
  if (!payload) return;
  const { intent } = req.body;
  if (!['study', 'explore'].includes(intent)) return badRequest(res, 'invalid intent');
  const db = getDb();
  const rl = await checkRateLimit(db, `user-intent:${payload.userId}`, 20, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  await db.update(schema.users).set({ intent }).where(eq(schema.users.id, payload.userId));
  return success(res, { success: true, intent });
}

// ── User readings (voice / food / health / EEG training data) ────────────────

async function readingsCreate(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  try {
    const body = req.body ?? {};
    const { userId, source, emotion, valence, arousal, stress, confidence, modelType, features } = body as {
      userId?: string;
      source?: string;
      emotion?: string;
      valence?: number;
      arousal?: number;
      stress?: number;
      confidence?: number;
      modelType?: string;
      features?: unknown;
    };

    if (!source || !['voice', 'food', 'health', 'eeg'].includes(source)) {
      return badRequest(res, 'source must be one of: voice, food, health, eeg');
    }
    if (userId !== undefined && (typeof userId !== 'string' || userId.length > 128)) {
      return badRequest(res, 'Invalid userId');
    }
    if (emotion !== undefined && (typeof emotion !== 'string' || emotion.length > 100)) {
      return badRequest(res, 'emotion must be a string ≤100 characters');
    }
    // Clamp numeric readings to valid ranges
    const clamp01 = (v: unknown) => typeof v === 'number' && isFinite(v) ? Math.max(0, Math.min(1, v)) : null;
    const clampN1 = (v: unknown) => typeof v === 'number' && isFinite(v) ? Math.max(-1, Math.min(1, v)) : null;
    // features must be a plain object/array, not a huge string
    if (features !== undefined && features !== null) {
      const featStr = JSON.stringify(features);
      if (featStr.length > 65536) return badRequest(res, 'features payload too large (max 64KB)');
    }

    const db = getDb();
    const rlKey = userId ? `readings-create:${userId}` : `readings-create:${getClientIp(req)}`;
    const rl = await checkRateLimit(db, rlKey, 120, 60);
    if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
    const [row] = await db.insert(schema.userReadings).values({
      userId: userId ?? null,
      source,
      emotion: emotion ?? null,
      valence: clampN1(valence),
      arousal: clamp01(arousal),
      stress: clamp01(stress),
      confidence: clamp01(confidence),
      modelType: modelType ?? null,
      features: features ?? null,
    }).returning();

    return success(res, row, 201);
  } catch (err: any) {
    console.error('[readingsCreate]', err?.message ?? err);
    return error(res, `Failed to store reading: ${err?.message || 'Unknown error'}`, 500);
  }
}

async function readingsList(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  try {
    const db = getDb();
    const sourceFilter = req.query.source as string | undefined;
    const limit = Math.min(Math.max(parseInt(req.query.limit as string) || 200, 1), 1000);

    const conditions = [eq(schema.userReadings.userId, userId)];
    if (sourceFilter && ['voice', 'food', 'health', 'eeg'].includes(sourceFilter)) {
      conditions.push(eq(schema.userReadings.source, sourceFilter));
    }

    const rows = await db
      .select()
      .from(schema.userReadings)
      .where(and(...conditions))
      .orderBy(desc(schema.userReadings.createdAt))
      .limit(limit);

    return success(res, { readings: rows, count: rows.length });
  } catch (err: any) {
    console.error('[readingsList]', err?.message ?? err);
    return error(res, `Failed to fetch readings: ${err?.message || 'Unknown error'}`, 500);
  }
}

// ── IRT (Image Rehearsal Therapy) sessions ────────────────────────────────────

async function irtSessionPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { originalDreamText, rewrittenEnding, rehearsalNote } = req.body;
  const userId = authPayload.userId;
  if (!originalDreamText || typeof originalDreamText !== 'string') return badRequest(res, 'originalDreamText required');
  if (!rewrittenEnding || typeof rewrittenEnding !== 'string') return badRequest(res, 'rewrittenEnding required');
  if (originalDreamText.length > 10000 || rewrittenEnding.length > 10000) return badRequest(res, 'Text too long (max 10000 chars)');
  const db = getDb();
  const rl = await checkRateLimit(db, `irt-session:${userId}`, 20, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const [row] = await db.insert(schema.irtSessions).values({
    userId,
    originalDreamText,
    rewrittenEnding,
    rehearsalNote: typeof rehearsalNote === 'string' ? rehearsalNote.substring(0, 1000) : null,
  }).returning();
  return success(res, row, 201);
}

// ── Research correlation ──────────────────────────────────────────────────────

async function researchCorrelation(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  // Find active participant record
  const [participant] = await db.select().from(schema.studyParticipants)
    .where(and(eq(schema.studyParticipants.userId, userId), eq(schema.studyParticipants.status, 'active')))
    .limit(1);
  if (!participant) return success(res, []);
  // Fetch sessions with morning + daytime entries
  const sessions = await db.select().from(schema.studySessions)
    .where(eq(schema.studySessions.participantId, participant.id))
    .orderBy(asc(schema.studySessions.dayNumber)).limit(30);
  if (sessions.length === 0) return success(res, []);
  const sessionIds = sessions.map((s) => s.id);
  const [morningEntries, daytimeEntries, foodLogs] = await Promise.all([
    db.select().from(schema.studyMorningEntries).where(inArray(schema.studyMorningEntries.sessionId, sessionIds)).limit(500),
    db.select().from(schema.studyDaytimeEntries).where(inArray(schema.studyDaytimeEntries.sessionId, sessionIds)).limit(500),
    db.select().from(schema.foodLogs).where(eq(schema.foodLogs.userId, userId)).orderBy(desc(schema.foodLogs.loggedAt)).limit(200),
  ]);
  const morningBySession = new Map(morningEntries.map((m) => [m.sessionId, m]));
  const daytimeBySession = new Map(daytimeEntries.map((d) => [d.sessionId, d]));
  const result = sessions.map((s) => {
    const morning = morningBySession.get(s.id);
    const daytime = daytimeBySession.get(s.id);
    const sessionDate = new Date(s.sessionDate).toISOString().slice(0, 10);
    const sessionFoods = foodLogs.filter((f) => f.loggedAt && new Date(f.loggedAt).toISOString().slice(0, 10) === sessionDate);
    return {
      dayNumber: s.dayNumber,
      sessionDate,
      validDay: s.validDay,
      morning: morning ? {
        dreamValence: morning.dreamValence ?? null,
        noRecall: morning.noRecall ?? false,
        nightmareFlag: morning.nightmareFlag ?? null,
        dreamSnippet: morning.dreamText ? morning.dreamText.substring(0, 200) : null,
        welfareScore: morning.currentMoodRating ?? null,
      } : null,
      daytime: daytime ? {
        samValence: daytime.samValence ?? null,
        samStress: daytime.samStress ?? null,
        faa: daytime.faa ?? null,
      } : null,
      foods: sessionFoods.map((f) => ({
        id: f.id,
        summary: f.summary,
        mealType: f.mealType,
        totalCalories: f.totalCalories,
        dominantMacro: f.dominantMacro,
        glycemicImpact: f.glycemicImpact as string | null ?? null,
        aiMoodImpact: null,
        aiDreamRelevance: null,
        loggedAt: f.loggedAt ? f.loggedAt.toISOString() : '',
      })),
    };
  });
  return success(res, result);
}

// ── Body metrics ─────────────────────────────────────────────────────────────

async function bodyMetricsLatest(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const [row] = await db.select().from(schema.bodyMetrics)
    .where(eq(schema.bodyMetrics.userId, userId))
    .orderBy(desc(schema.bodyMetrics.recordedAt)).limit(1);
  if (!row) return success(res, null);
  return success(res, row);
}

async function bodyMetricsList(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const days = Math.min(Math.max(parseInt(req.query.days as string) || 90, 1), 365);
  const since = new Date(Date.now() - days * 86_400_000);
  const db = getDb();
  const rows = await db.select().from(schema.bodyMetrics)
    .where(and(eq(schema.bodyMetrics.userId, userId), gte(schema.bodyMetrics.recordedAt, since)))
    .orderBy(desc(schema.bodyMetrics.recordedAt)).limit(200);
  return success(res, rows);
}

// ── Health samples (steps / heart-rate by metric) ─────────────────────────────

async function healthSamplesByMetric(req: VercelRequest, res: VercelResponse, userId: string, metric: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const days = Math.min(Math.max(parseInt(req.query.days as string) || 7, 1), 365);
  const since = new Date(Date.now() - days * 86_400_000);
  const db = getDb();
  const rows = await db.select({
    value: schema.healthSamples.value,
    recorded_at: schema.healthSamples.recordedAt,
  }).from(schema.healthSamples)
    .where(and(
      eq(schema.healthSamples.userId, userId),
      eq(schema.healthSamples.metric, metric),
      gte(schema.healthSamples.recordedAt, since),
    ))
    .orderBy(desc(schema.healthSamples.recordedAt))
    .limit(500);
  return success(res, rows);
}

// ── Device connections ───────────────────────────────────────────────────────

async function deviceConnectionsList(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const rows = await db.select({
    provider: schema.deviceConnections.provider,
    status: schema.deviceConnections.syncStatus,
    lastSyncAt: schema.deviceConnections.lastSyncAt,
    connectedAt: schema.deviceConnections.connectedAt,
  }).from(schema.deviceConnections)
    .where(eq(schema.deviceConnections.userId, userId));
  return success(res, rows);
}

async function devicesList(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const rows = await db.select({
    provider: schema.deviceConnections.provider,
    status: schema.deviceConnections.syncStatus,
    lastSyncAt: schema.deviceConnections.lastSyncAt,
    connectedAt: schema.deviceConnections.connectedAt,
  }).from(schema.deviceConnections)
    .where(eq(schema.deviceConnections.userId, userId));
  return success(res, { devices: rows.map((r) => ({ provider: r.provider, status: r.status, lastSyncAt: r.lastSyncAt, connectedAt: r.connectedAt })) });
}

// ── Workouts ─────────────────────────────────────────────────────────────────

async function workoutsGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const limit = Math.min(Math.max(parseInt(req.query.limit as string) || 20, 1), 100);
  const rows = await db.select().from(schema.workouts)
    .where(eq(schema.workouts.userId, userId))
    .orderBy(desc(schema.workouts.startedAt))
    .limit(limit);
  return success(res, rows);
}

async function workoutsPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { startedAt } = req.body;
  const workoutType = typeof req.body.workoutType === 'string' ? req.body.workoutType.trim() : '';
  if (!workoutType) return badRequest(res, 'workoutType required');
  if (workoutType.length > 50) return badRequest(res, 'workoutType must be ≤50 chars');
  const db = getDb();
  const rl = await checkRateLimit(db, `workouts-post:${authPayload.userId}`, 30, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const parsedStartedAt = startedAt ? new Date(startedAt) : new Date();
  const [row] = await db.insert(schema.workouts).values({
    userId: authPayload.userId,
    name: typeof req.body.name === 'string' ? req.body.name.trim().substring(0, 100) : null,
    workoutType,
    startedAt: isNaN(parsedStartedAt.getTime()) ? new Date() : parsedStartedAt,
    source: 'manual',
    notes: typeof req.body.notes === 'string' ? req.body.notes.trim().substring(0, 500) : null,
  }).returning();
  return success(res, row, 201);
}

async function workoutsPut(req: VercelRequest, res: VercelResponse, workoutId: string) {
  if (req.method !== 'PUT') return methodNotAllowed(res, ['PUT']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const db = getDb();
  const [existing] = await db.select({ userId: schema.workouts.userId }).from(schema.workouts)
    .where(eq(schema.workouts.id, workoutId)).limit(1);
  if (!existing) return error(res, 'Workout not found', 404);
  if (existing.userId !== authPayload.userId) return unauthorized(res, 'Not your workout');
  const { endedAt, durationMin, caloriesBurned, avgHr, maxHr, notes } = req.body;
  const parsedEndedAt = endedAt ? new Date(endedAt) : null;
  if (parsedEndedAt && isNaN(parsedEndedAt.getTime())) return badRequest(res, 'Invalid endedAt timestamp');
  const parsedDuration = durationMin != null ? Number(durationMin) : null;
  if (parsedDuration !== null && (!isFinite(parsedDuration) || parsedDuration < 0 || parsedDuration > 1440)) return badRequest(res, 'durationMin must be 0–1440');
  const parsedCal = caloriesBurned != null ? Number(caloriesBurned) : null;
  if (parsedCal !== null && (!isFinite(parsedCal) || parsedCal < 0 || parsedCal > 10000)) return badRequest(res, 'caloriesBurned must be 0–10000');
  const parsedAvgHr = avgHr != null ? Number(avgHr) : null;
  if (parsedAvgHr !== null && (!isFinite(parsedAvgHr) || parsedAvgHr < 20 || parsedAvgHr > 300)) return badRequest(res, 'avgHr must be 20–300 bpm');
  const parsedMaxHr = maxHr != null ? Number(maxHr) : null;
  if (parsedMaxHr !== null && (!isFinite(parsedMaxHr) || parsedMaxHr < 20 || parsedMaxHr > 300)) return badRequest(res, 'maxHr must be 20–300 bpm');
  const rl = await checkRateLimit(db, `workouts-put:${authPayload.userId}`, 60, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  await db.update(schema.workouts).set({
    endedAt: parsedEndedAt,
    durationMin: parsedDuration !== null ? String(parsedDuration) : null,
    caloriesBurned: parsedCal !== null ? String(parsedCal) : null,
    avgHr: parsedAvgHr !== null ? String(parsedAvgHr) : null,
    maxHr: parsedMaxHr !== null ? String(parsedMaxHr) : null,
    notes: typeof notes === 'string' ? notes.trim().substring(0, 500) : null,
  }).where(eq(schema.workouts.id, workoutId));
  return success(res, { updated: true });
}

async function workoutSetsPost(req: VercelRequest, res: VercelResponse, workoutId: string) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const db = getDb();
  const [existing] = await db.select({ userId: schema.workouts.userId }).from(schema.workouts)
    .where(eq(schema.workouts.id, workoutId)).limit(1);
  if (!existing) return error(res, 'Workout not found', 404);
  if (existing.userId !== authPayload.userId) return unauthorized(res, 'Not your workout');
  const rl = await checkRateLimit(db, `workout-sets:${authPayload.userId}`, 200, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const { exerciseId, setNumber, reps, weightKg, durationSec, rpe } = req.body;
  const parsedSetNumber = Math.max(1, Math.min(parseInt(setNumber) || 1, 999));
  const parsedReps = reps != null ? Math.max(0, Math.min(parseInt(reps) || 0, 9999)) : null;
  const parsedDurationSec = durationSec != null ? Math.max(0, Math.min(parseInt(durationSec) || 0, 86400)) : null;
  const parsedWeightKg = weightKg != null ? parseFloat(String(weightKg)) : null;
  if (parsedWeightKg !== null && (!isFinite(parsedWeightKg) || parsedWeightKg < 0 || parsedWeightKg > 10000)) {
    return badRequest(res, 'weightKg must be a non-negative number ≤10000');
  }
  const parsedRpe = rpe != null ? parseFloat(String(rpe)) : null;
  if (parsedRpe !== null && (!isFinite(parsedRpe) || parsedRpe < 1 || parsedRpe > 10)) {
    return badRequest(res, 'rpe must be 1–10');
  }
  const [row] = await db.insert(schema.workoutSets).values({
    workoutId,
    exerciseId: exerciseId ?? null,
    setNumber: parsedSetNumber,
    reps: parsedReps,
    weightKg: parsedWeightKg !== null ? String(parsedWeightKg) : null,
    durationSec: parsedDurationSec,
    rpe: parsedRpe !== null ? String(parsedRpe) : null,
    completed: true,
  }).returning();
  return success(res, row, 201);
}

async function workoutTemplatesGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const rows = await db.select().from(schema.workoutTemplates)
    .where(eq(schema.workoutTemplates.userId, userId))
    .orderBy(desc(schema.workoutTemplates.createdAt))
    .limit(50);
  return success(res, rows);
}

async function workoutTemplatesPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { name, description, exercises } = req.body;
  if (!name || typeof name !== 'string' || name.length > 100) return badRequest(res, 'name required (max 100 chars)');
  const db = getDb();
  const rl = await checkRateLimit(db, `workout-templates:${authPayload.userId}`, 30, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const [row] = await db.insert(schema.workoutTemplates).values({
    userId: authPayload.userId,
    name: name.trim(),
    description: typeof description === 'string' ? description.substring(0, 500) : null,
    exercises: Array.isArray(exercises) ? exercises : [],
  }).returning();
  return success(res, row, 201);
}

async function workoutTemplatesDelete(req: VercelRequest, res: VercelResponse, templateId: string) {
  if (req.method !== 'DELETE') return methodNotAllowed(res, ['DELETE']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const db = getDb();
  const [existing] = await db.select({ userId: schema.workoutTemplates.userId }).from(schema.workoutTemplates)
    .where(eq(schema.workoutTemplates.id, templateId)).limit(1);
  if (!existing) return error(res, 'Template not found', 404);
  if (existing.userId !== authPayload.userId) return unauthorized(res, 'Not your template');
  const rl = await checkRateLimit(db, `workout-templates-delete:${authPayload.userId}`, 30, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  await db.delete(schema.workoutTemplates).where(eq(schema.workoutTemplates.id, templateId));
  return success(res, { deleted: true });
}

// ── Habits ───────────────────────────────────────────────────────────────────

async function habitsGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const rows = await db.select().from(schema.habits)
    .where(and(eq(schema.habits.userId, userId), eq(schema.habits.isActive, true)))
    .orderBy(asc(schema.habits.createdAt)).limit(200);
  return success(res, rows);
}

async function habitsPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { name, category, icon, targetValue, unit } = req.body;
  if (!name || typeof name !== 'string' || name.length > 100) return badRequest(res, 'name required (max 100 chars)');
  if (category !== undefined && (typeof category !== 'string' || category.length > 50)) return badRequest(res, 'category must be a string ≤50 chars');
  if (icon !== undefined && (typeof icon !== 'string' || icon.length > 50)) return badRequest(res, 'icon must be a string ≤50 chars');
  if (unit !== undefined && (typeof unit !== 'string' || unit.length > 30)) return badRequest(res, 'unit must be a string ≤30 chars');
  const targetNum = targetValue != null ? Number(targetValue) : null;
  if (targetNum !== null && (!isFinite(targetNum) || targetNum < 0)) return badRequest(res, 'targetValue must be a non-negative number');
  const db = getDb();
  const rl = await checkRateLimit(db, `habits-post:${authPayload.userId}`, 50, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const [row] = await db.insert(schema.habits).values({
    userId: authPayload.userId,
    name: name.trim(),
    category: typeof category === 'string' ? category : null,
    icon: typeof icon === 'string' ? icon : null,
    targetValue: targetNum !== null ? String(targetNum) : null,
    unit: typeof unit === 'string' ? unit : null,
    isActive: true,
  }).returning();
  return success(res, row, 201);
}

async function habitsDelete(req: VercelRequest, res: VercelResponse, habitId: string) {
  if (req.method !== 'DELETE') return methodNotAllowed(res, ['DELETE']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const db = getDb();
  const [existing] = await db.select({ userId: schema.habits.userId }).from(schema.habits)
    .where(eq(schema.habits.id, habitId)).limit(1);
  if (!existing) return error(res, 'Habit not found', 404);
  if (existing.userId !== authPayload.userId) return unauthorized(res, 'Not your habit');
  const rl = await checkRateLimit(db, `habits-delete:${authPayload.userId}`, 50, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  // Soft delete
  await db.update(schema.habits).set({ isActive: false }).where(eq(schema.habits.id, habitId));
  return success(res, { deleted: true });
}

async function habitLogsGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const days = Math.min(Math.max(parseInt(req.query.days as string) || 7, 1), 90);
  const since = new Date(Date.now() - days * 86_400_000);
  const db = getDb();
  const rows = await db.select().from(schema.habitLogs)
    .where(and(eq(schema.habitLogs.userId, userId), gte(schema.habitLogs.loggedAt, since)))
    .orderBy(desc(schema.habitLogs.loggedAt))
    .limit(500);
  return success(res, rows);
}

async function habitLogsPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { habitId, value, note } = req.body;
  if (!habitId || typeof habitId !== 'string') return badRequest(res, 'habitId required');
  const val = Number(value);
  if (!isFinite(val) || val < 0) return badRequest(res, 'value must be a non-negative finite number');
  const db = getDb();
  const rl = await checkRateLimit(db, `habit-logs:${authPayload.userId}`, 200, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  // Verify the habit exists and belongs to the authenticated user
  const [habit] = await db.select({ userId: schema.habits.userId }).from(schema.habits)
    .where(eq(schema.habits.id, habitId)).limit(1);
  if (!habit) return error(res, 'Habit not found', 404);
  if (habit.userId !== authPayload.userId) return error(res, 'Forbidden', 403);
  const [row] = await db.insert(schema.habitLogs).values({
    userId: authPayload.userId,
    habitId,
    value: String(val),
    note: typeof note === 'string' ? note.substring(0, 200) : null,
  }).returning();
  return success(res, row, 201);
}

async function habitLogsStreaks(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const [userHabits, logs] = await Promise.all([
    db.select({ id: schema.habits.id }).from(schema.habits)
      .where(and(eq(schema.habits.userId, userId), eq(schema.habits.isActive, true))),
    db.select({ habitId: schema.habitLogs.habitId, loggedAt: schema.habitLogs.loggedAt })
      .from(schema.habitLogs).where(eq(schema.habitLogs.userId, userId))
      .orderBy(desc(schema.habitLogs.loggedAt)).limit(1000),
  ]);
  // Group log dates by habitId
  const logsByHabit = new Map<string, Set<string>>();
  for (const log of logs) {
    if (!log.habitId) continue;
    const date = new Date(log.loggedAt!).toISOString().slice(0, 10);
    if (!logsByHabit.has(log.habitId)) logsByHabit.set(log.habitId, new Set());
    logsByHabit.get(log.habitId)!.add(date);
  }
  // Compute current streak per habit
  const streaks: Record<string, number> = {};
  for (const { id } of userHabits) {
    const dates = logsByHabit.get(id);
    if (!dates) { streaks[id] = 0; continue; }
    let streak = 0;
    const d = new Date();
    while (dates.has(d.toISOString().slice(0, 10))) {
      streak++;
      d.setDate(d.getDate() - 1);
    }
    streaks[id] = streak;
  }
  return success(res, streaks);
}

// ── AI Coach ─────────────────────────────────────────────────────────────────

async function aiCoachPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { message, userId, history, context, memories, tone } = req.body;
  if (!message || typeof message !== 'string') return badRequest(res, 'message required');
  if (message.length > 2000) return badRequest(res, 'message must be ≤2000 characters');
  if (userId && (typeof userId !== 'string' || userId.length > 128)) return badRequest(res, 'Invalid userId');
  const ALLOWED_TONES = ['supportive', 'motivational', 'clinical', 'gentle', 'direct'];
  const safeTone = typeof tone === 'string' && ALLOWED_TONES.includes(tone) ? tone : 'supportive';
  const db = getDb();
  const rlKey = userId ? `ai-coach:${userId}` : `ai-coach:${getClientIp(req)}`;
  const rl = await checkRateLimit(db, rlKey, 30, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!, 'Too many coach requests. Please wait.');
  const openai = getOpenAIClient();
  const sysPrompt = [
    'You are a knowledgeable AI wellness coach specializing in sleep, dreams, and emotional health.',
    context ? `User context: ${String(context).substring(0, 500)}` : '',
    memories && Array.isArray(memories) && memories.length > 0
      ? `User memories: ${memories.slice(0, 5).map((m: unknown) => typeof m === 'string' ? m.trim().slice(0, 100) : '').filter(Boolean).join('; ')}` : '',
    `Coaching tone: ${safeTone}. Be concise and actionable.`,
  ].filter(Boolean).join(' ');

  const conversationMsgs: { role: 'user' | 'assistant'; content: string }[] = [];
  if (Array.isArray(history)) {
    for (const h of history.slice(-8)) {
      if (h && typeof h.message === 'string' && typeof h.isUser === 'boolean') {
        conversationMsgs.push({ role: h.isUser ? 'user' : 'assistant', content: h.message.substring(0, 500) });
      }
    }
  }
  try {
    const resp = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: sysPrompt },
        ...conversationMsgs,
        { role: 'user', content: message },
      ],
    });
    const reply = resp.choices[0].message.content || "I'm here to support your wellness journey.";
    return success(res, { message: reply }, 200);
  } catch (err) {
    console.error('[aiCoach]', err instanceof Error ? err.message : err);
    return error(res, 'Coach unavailable right now. Please try again.', 500);
  }
}

// ── Sleep alarm optimizer ─────────────────────────────────────────────────────

async function sleepAlarm(req: VercelRequest, res: VercelResponse, userId: string) {
  if (!requireOwner(req, res, userId)) return;
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const targetWake = req.query.targetWake as string;
  if (!targetWake) return badRequest(res, 'targetWake required (HH:MM)');
  const [hStr, mStr] = targetWake.split(':');
  const wakeH = parseInt(hStr, 10);
  const wakeM = parseInt(mStr, 10);
  if (isNaN(wakeH) || isNaN(wakeM) || wakeH < 0 || wakeH > 23 || wakeM < 0 || wakeM > 59) {
    return badRequest(res, 'targetWake must be HH:MM');
  }
  // Sleep cycle = 90 min; REM dominates cycles 4-6
  const CYCLE_MIN = 90;
  const now = new Date();
  const wake = new Date(now);
  wake.setHours(wakeH, wakeM, 0, 0);
  if (wake <= now) wake.setDate(wake.getDate() + 1);
  const totalMinutes = Math.round((wake.getTime() - now.getTime()) / 60000);
  const estimatedCycles = Math.floor(totalMinutes / CYCLE_MIN);
  // Optimal window: target the end of a REM cycle (slightly before wake)
  const optimalMid = new Date(wake.getTime() - (totalMinutes % CYCLE_MIN) * 60000);
  const optimalStart = new Date(optimalMid.getTime() - 15 * 60000);
  const optimalEnd = new Date(optimalMid.getTime() + 15 * 60000);
  const fmt = (d: Date) => d.toTimeString().slice(0, 5);
  const expectedStage = estimatedCycles >= 4 ? 'REM' : estimatedCycles >= 2 ? 'NREM2' : 'NREM3';
  const confidence = estimatedCycles >= 4 ? 0.85 : estimatedCycles >= 2 ? 0.7 : 0.5;
  const note = estimatedCycles >= 5
    ? 'Ideal sleep duration — expect vivid dreams and easy awakening.'
    : estimatedCycles >= 4
    ? 'Good sleep opportunity — aim to be in bed within 30 minutes.'
    : 'Limited cycles — prioritize falling asleep quickly.';
  return success(res, {
    optimalWindow: { start: fmt(optimalStart), end: fmt(optimalEnd), midpoint: fmt(optimalMid) },
    targetWake: targetWake,
    estimatedCycles,
    expectedStage,
    confidence,
    note,
  });
}

// ── Reality tests ────────────────────────────────────────────────────────────

async function realityTestPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { result, notes } = req.body;
  const userId = authPayload.userId;
  const validResults = ['dreaming', 'awake', 'unsure'];
  if (!result || !validResults.includes(result)) return badRequest(res, 'result must be dreaming, awake, or unsure');
  const db = getDb();
  const rl = await checkRateLimit(db, `reality-test:${userId}`, 200, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const [row] = await db.insert(schema.realityTests).values({
    userId, result,
    notes: typeof notes === 'string' ? notes.substring(0, 500) : null,
  }).returning();
  return success(res, row, 201);
}

async function realityTestGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const [total, todayRows] = await Promise.all([
    db.select({ count: sql<number>`count(*)::int` }).from(schema.realityTests).where(eq(schema.realityTests.userId, userId)),
    db.select({ timestamp: schema.realityTests.timestamp }).from(schema.realityTests)
      .where(and(eq(schema.realityTests.userId, userId), gte(schema.realityTests.timestamp, today)))
      .orderBy(desc(schema.realityTests.timestamp)),
  ]);
  // Compute day streak from recent history
  const recentDays = await db.select({ timestamp: schema.realityTests.timestamp })
    .from(schema.realityTests).where(eq(schema.realityTests.userId, userId))
    .orderBy(desc(schema.realityTests.timestamp)).limit(100);
  const daySet = new Set(recentDays.map((r) => new Date(r.timestamp).toISOString().slice(0, 10)));
  let streak = 0;
  const d = new Date();
  while (daySet.has(d.toISOString().slice(0, 10))) {
    streak++;
    d.setDate(d.getDate() - 1);
  }
  return success(res, { todayCount: todayRows.length, streak, totalCount: total[0]?.count ?? 0 });
}

// ── Cycle tracking ───────────────────────────────────────────────────────────

async function cyclePost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { date, flowLevel, symptoms, phase, contraception, basalTemp, notes } = req.body;
  if (!date || typeof date !== 'string') return badRequest(res, 'date required (YYYY-MM-DD)');
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || isNaN(new Date(date).getTime())) return badRequest(res, 'date must be a valid YYYY-MM-DD');
  const safeBasalTemp = basalTemp != null ? parseFloat(String(basalTemp)) : null;
  if (safeBasalTemp !== null && (!isFinite(safeBasalTemp) || safeBasalTemp < 30 || safeBasalTemp > 45)) return badRequest(res, 'basalTemp must be a realistic body temperature (30–45°C)');
  const db = getDb();
  const rl = await checkRateLimit(db, `cycle-post:${authPayload.userId}`, 50, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const [row] = await db.insert(schema.cycleTracking).values({
    userId: authPayload.userId, date,
    flowLevel: flowLevel ?? null,
    symptoms: Array.isArray(symptoms) ? symptoms : null,
    phase: phase ?? null,
    contraception: contraception ?? null,
    basalTemp: safeBasalTemp !== null ? String(safeBasalTemp) : null,
    notes: typeof notes === 'string' ? notes.substring(0, 500) : null,
  }).onConflictDoUpdate({
    target: [schema.cycleTracking.userId, schema.cycleTracking.date],
    set: { flowLevel: flowLevel ?? null, symptoms: Array.isArray(symptoms) ? symptoms : null, phase: phase ?? null, contraception: contraception ?? null, basalTemp: safeBasalTemp !== null ? String(safeBasalTemp) : null, notes: typeof notes === 'string' ? notes.substring(0, 500) : null },
  }).returning();
  return success(res, row, 201);
}

async function cycleGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const days = Math.min(Math.max(parseInt(req.query.days as string) || 90, 1), 365);
  const since = new Date(Date.now() - days * 86_400_000);
  const db = getDb();
  const rows = await db.select().from(schema.cycleTracking)
    .where(and(eq(schema.cycleTracking.userId, userId), gte(schema.cycleTracking.date, since.toISOString().slice(0, 10))))
    .orderBy(desc(schema.cycleTracking.date))
    .limit(500);
  return success(res, rows);
}

async function cyclePhase(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  // Find all period start entries (flow entries) ordered desc
  const rows = await db.select({ date: schema.cycleTracking.date })
    .from(schema.cycleTracking)
    .where(and(eq(schema.cycleTracking.userId, userId)))
    .orderBy(desc(schema.cycleTracking.date))
    .limit(20);
  if (rows.length === 0) return success(res, { currentPhase: 'unknown', dayOfCycle: 0, avgCycleLength: 28, lastPeriodStart: null, nextPeriodDate: null, periodStartCount: 0 });

  const lastPeriodStart = rows[0].date;
  const today = new Date().toISOString().slice(0, 10);
  const dayOfCycle = Math.max(1, Math.round((new Date(today).getTime() - new Date(lastPeriodStart).getTime()) / 86_400_000) + 1);
  const avgCycleLength = 28;
  const nextPeriodDate = new Date(new Date(lastPeriodStart).getTime() + avgCycleLength * 86_400_000).toISOString().slice(0, 10);
  const phase = dayOfCycle <= 5 ? 'menstrual' : dayOfCycle <= 13 ? 'follicular' : dayOfCycle <= 16 ? 'ovulation' : 'luteal';
  return success(res, { currentPhase: phase, dayOfCycle, avgCycleLength, lastPeriodStart, nextPeriodDate, periodStartCount: rows.length });
}

// ── Mood log ─────────────────────────────────────────────────────────────────

async function moodLogPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { moodScore, energyLevel, notes } = req.body;
  if (moodScore === undefined || moodScore === null) return badRequest(res, 'moodScore required');
  const score = Number(moodScore);
  if (isNaN(score) || score < 0 || score > 10) return badRequest(res, 'moodScore must be 0–10');
  const energyNum = energyLevel != null ? Number(energyLevel) : null;
  if (energyNum !== null && (!isFinite(energyNum) || energyNum < 0 || energyNum > 10)) {
    return badRequest(res, 'energyLevel must be 0–10');
  }
  const db = getDb();
  const rl = await checkRateLimit(db, `mood-log:${authPayload.userId}`, 100, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const [row] = await db.insert(schema.moodLogs).values({
    userId: authPayload.userId,
    moodScore: String(score),
    energyLevel: energyNum !== null ? String(energyNum) : null,
    notes: typeof notes === 'string' ? notes.trim().substring(0, 1000) : null,
  }).returning();
  return success(res, row, 201);
}

async function moodLogGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const days = Math.min(Math.max(parseInt(req.query.days as string) || 7, 1), 365);
  const since = new Date(Date.now() - days * 86_400_000);
  const db = getDb();
  const rows = await db.select().from(schema.moodLogs)
    .where(and(eq(schema.moodLogs.userId, userId), gte(schema.moodLogs.loggedAt, since)))
    .orderBy(desc(schema.moodLogs.loggedAt))
    .limit(500);
  return success(res, rows);
}

// ── Meal history ─────────────────────────────────────────────────────────────

async function mealHistoryList(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const limit = Math.min(Math.max(parseInt(req.query.limit as string) || 50, 1), 200);
  const rows = await db.select().from(schema.mealHistory)
    .where(eq(schema.mealHistory.userId, userId))
    .orderBy(desc(schema.mealHistory.createdAt))
    .limit(limit);
  return success(res, rows);
}

async function mealHistoryToggleFavorite(req: VercelRequest, res: VercelResponse, id: string) {
  if (req.method !== 'PATCH') return methodNotAllowed(res, ['PATCH']);
  const { isFavorite } = req.body;
  if (typeof isFavorite !== 'boolean') return badRequest(res, 'isFavorite must be a boolean');
  const db = getDb();
  const [existing] = await db.select({ id: schema.mealHistory.id, userId: schema.mealHistory.userId })
    .from(schema.mealHistory).where(eq(schema.mealHistory.id, id)).limit(1);
  if (!existing) return error(res, 'Meal not found', 404);
  if (!requireOwner(req, res, existing.userId ?? '')) return;
  const [updated] = await db.update(schema.mealHistory)
    .set({ isFavorite })
    .where(eq(schema.mealHistory.id, id))
    .returning();
  return success(res, updated);
}

// ── Dream frames + session complete ──────────────────────────────────────────

async function dreamFramesPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { sessionId, frames } = req.body || {};
  const userId = authPayload.userId;
  if (!sessionId || !Array.isArray(frames) || frames.length === 0) {
    return badRequest(res, 'sessionId and frames[] required');
  }
  if (frames.length > 500) return badRequest(res, 'frames[] exceeds max batch size of 500');
  const db = getDb();
  // Rate limit: 200 frame batches per user per hour
  const rl = await checkRateLimit(db, `dream-frames:${userId}`, 200, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);

  const rows = frames.map((f: any) => ({
    sessionId,
    userId,
    dreamIntensity: typeof f.dreamIntensity === 'number' ? Math.min(1, Math.max(0, f.dreamIntensity)) : 0.5,
    remLikelihood: typeof f.remLikelihood === 'number' ? Math.min(1, Math.max(0, f.remLikelihood)) : 0.5,
    valence: typeof f.valence === 'number' ? Math.min(1, Math.max(-1, f.valence)) : null,
    arousal: typeof f.arousal === 'number' ? Math.min(1, Math.max(0, f.arousal)) : null,
    lucidityScore: typeof f.lucidityScore === 'number' ? Math.min(1, Math.max(0, f.lucidityScore)) : null,
    lucidityState: typeof f.lucidityState === 'string' ? f.lucidityState.slice(0, 50) : null,
    thetaActivity: typeof f.thetaActivity === 'number' ? f.thetaActivity : null,
    betaActivation: typeof f.betaActivation === 'number' ? f.betaActivation : null,
    eyeMovementIndex: typeof f.eyeMovementIndex === 'number' ? f.eyeMovementIndex : null,
    dominantEmotion: typeof f.dominantEmotion === 'string' ? f.dominantEmotion.slice(0, 50) : null,
    timestamp: (() => { const d = f.timestamp ? new Date(f.timestamp) : new Date(); return isNaN(d.getTime()) ? new Date() : d; })(),
  }));
  await db.insert(schema.dreamFrames).values(rows).onConflictDoNothing();
  return success(res, { saved: rows.length }, 201);
}

async function dreamSessionComplete(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { sessionId } = req.body || {};
  const userId = authPayload.userId;
  if (!sessionId) return badRequest(res, 'sessionId required');
  const db = getDb();
  // Rate limit: 20 session completes per user per hour
  const rl = await checkRateLimit(db, `dream-session-complete:${userId}`, 20, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!, 'Too many session completions. Please wait.');

  const frames = await db.select().from(schema.dreamFrames)
    .where(and(eq(schema.dreamFrames.sessionId, sessionId), eq(schema.dreamFrames.userId, userId)))
    .orderBy(asc(schema.dreamFrames.timestamp))
    .limit(500);

  if (frames.length === 0) {
    return success(res, {
      message: 'No frames found for this session. Start a dream session to record EEG data.',
    });
  }

  // Aggregate frame stats
  const avgIntensity = frames.reduce((s, f) => s + (f.dreamIntensity ?? 0), 0) / frames.length;
  const peakLucidity = frames.reduce((m, f) => Math.max(m, f.lucidityScore ?? 0), 0);
  const durationMinutes = frames.length > 1
    ? (new Date(frames[frames.length - 1].timestamp).getTime() - new Date(frames[0].timestamp).getTime()) / 60000
    : 0;

  // Determine lucidity state
  let peakLucidityState = 'non_lucid';
  if (peakLucidity > 0.7) peakLucidityState = 'controlled';
  else if (peakLucidity > 0.5) peakLucidityState = 'lucid';
  else if (peakLucidity > 0.3) peakLucidityState = 'pre_lucid';

  const emotionCounts = new Map<string, number>();
  for (const f of frames) {
    if (f.dominantEmotion) emotionCounts.set(f.dominantEmotion, (emotionCounts.get(f.dominantEmotion) ?? 0) + 1);
  }
  const topEmotion = emotionCounts.size > 0
    ? [...emotionCounts.entries()].sort((a, b) => b[1] - a[1])[0][0]
    : 'neutral';

  // Generate AI narrative
  const openai = getOpenAIClient();
  let narrative = '';
  let primaryTheme = 'neutral';
  let keyInsight = '';
  let morningMoodPrediction = 'neutral';
  let eegSummary = '';

  try {
    const summary = `Session: ${frames.length} frames over ${Math.round(durationMinutes)} minutes. ` +
      `Avg dream intensity: ${avgIntensity.toFixed(2)}. Peak lucidity: ${peakLucidity.toFixed(2)} (${peakLucidityState}). ` +
      `Dominant emotion: ${topEmotion}.`;
    const resp = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: 'You are a dream analysis expert. Respond with JSON: {"narrative":"...","primaryTheme":"...","keyInsight":"...","morningMoodPrediction":"...","eegSummary":"..."}. Keep each field under 100 words.' },
        { role: 'user', content: `Analyze this dream session: ${summary}` },
      ],
      max_tokens: 400,
      temperature: 0.7,
      response_format: { type: 'json_object' },
    });
    const parsed = JSON.parse(resp.choices?.[0]?.message?.content ?? '{}');
    narrative = parsed.narrative ?? '';
    primaryTheme = parsed.primaryTheme ?? 'neutral';
    keyInsight = parsed.keyInsight ?? '';
    morningMoodPrediction = parsed.morningMoodPrediction ?? 'neutral';
    eegSummary = parsed.eegSummary ?? '';
  } catch {
    narrative = `Your ${Math.round(durationMinutes)}-minute dream session showed ${avgIntensity > 0.6 ? 'vivid' : 'moderate'} dream activity with ${topEmotion} as the dominant emotion.`;
    primaryTheme = topEmotion;
    eegSummary = `Peak lucidity: ${(peakLucidity * 100).toFixed(0)}% (${peakLucidityState}).`;
  }

  return success(res, {
    narrative,
    primaryTheme,
    keyInsight,
    morningMoodPrediction,
    eegSummary,
    episode: { durationMinutes: Math.round(durationMinutes), peakIntensity: Math.round(avgIntensity * 100) / 100, peakLucidityState },
  });
}

// ── Morning briefing (AI) ─────────────────────────────────────────────────────

async function morningBriefing(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const db = getDb();
  // Rate limit: 10 briefings per user per hour
  const rl = await checkRateLimit(db, `morning-briefing:${authPayload.userId}`, 5, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);

  const { sleepData, morningHrv, emotionSummary, patternSummaries, yesterdaySummary, dreamContext } = req.body || {};
  const safeYesterdaySummary = typeof yesterdaySummary === 'string' ? yesterdaySummary.trim().slice(0, 500) : null;
  const safeDreamInsight = typeof dreamContext?.keyInsight === 'string' ? dreamContext.keyInsight.trim().slice(0, 300) : null;

  const openai = getOpenAIClient();
  try {
    const context = [
      sleepData ? `Sleep: ${sleepData.totalHours ?? '?'}h total, efficiency ${sleepData.efficiency ?? '?'}%` : '',
      morningHrv ? `Morning HRV: ${morningHrv}ms` : '',
      emotionSummary ? `Yesterday emotions: ${emotionSummary.dominantLabel} dominant (stress ${(emotionSummary.avgStress * 100).toFixed(0)}%, focus ${(emotionSummary.avgFocus * 100).toFixed(0)}%)` : '',
      safeYesterdaySummary ? `Yesterday: ${safeYesterdaySummary}` : '',
      safeDreamInsight ? `Dream insight: ${safeDreamInsight}` : '',
    ].filter(Boolean).join('. ');

    const resp = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: 'You are a personal brain health coach. Given the user\'s overnight data, generate a morning briefing. Respond with JSON: {"stateSummary":"1-2 sentence state overview","actions":["action1","action2","action3"],"forecast":{"label":"positive|neutral|challenging","probability":0.7,"reason":"brief reason"}}' },
        { role: 'user', content: context || 'No health data available. Generate a general morning briefing.' },
      ],
      max_tokens: 300,
      temperature: 0.7,
      response_format: { type: 'json_object' },
    });
    const parsed = JSON.parse(resp.choices?.[0]?.message?.content ?? '{}');
    return success(res, {
      stateSummary: parsed.stateSummary ?? 'Good morning! Your brain is ready for the day.',
      actions: parsed.actions ?? ['Start with deep breathing', 'Review your goals', 'Hydrate'],
      forecast: parsed.forecast ?? { label: 'neutral', probability: 0.6, reason: 'Baseline conditions' },
    });
  } catch {
    return success(res, {
      stateSummary: 'Good morning! Start your day with intention.',
      actions: ['5 minutes of deep breathing', 'Set 3 priorities for today', 'Drink a glass of water'],
      forecast: { label: 'neutral', probability: 0.6, reason: 'Data unavailable' },
    });
  }
}

// ── Voice emotion sync ────────────────────────────────────────────────────────

async function voiceEmotionPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  const { valence, arousal, confidence, probabilities, timestamp } = req.body || {};
  const userId = authPayload.userId;
  const emotion = typeof req.body?.emotion === 'string' ? req.body.emotion.trim() : '';
  if (!emotion || emotion.length > 100) return badRequest(res, 'emotion must be a non-empty string ≤100 chars');
  const parsedTs = timestamp ? new Date(timestamp) : new Date();
  if (isNaN(parsedTs.getTime())) return badRequest(res, 'Invalid timestamp');
  const db = getDb();
  // Rate limit: 100 per user per hour
  const rl = await checkRateLimit(db, `voice-emotion:${userId}`, 100, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  const clamp01 = (v: unknown) => typeof v === 'number' && isFinite(v) ? Math.max(0, Math.min(1, v)) : null;
  const clampN1 = (v: unknown) => typeof v === 'number' && isFinite(v) ? Math.max(-1, Math.min(1, v)) : null;
  const [row] = await db.insert(schema.userReadings).values({
    userId,
    source: 'voice',
    emotion: emotion.slice(0, 50),
    valence: clampN1(valence),
    arousal: clamp01(arousal),
    stress: null,
    confidence: clamp01(confidence),
    modelType: 'voice',
    createdAt: parsedTs,
  }).returning();
  return success(res, row, 201);
}

// ── Community mood feed ───────────────────────────────────────────────────────

async function communityMoodFeed(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const db = getDb();
  const since = new Date(Date.now() - 24 * 60 * 60 * 1000); // last 24 hours
  // Community entries stored with sessionId = '_community' and userId = null
  const rows = await db.select({ dominantEmotion: schema.emotionReadings.dominantEmotion })
    .from(schema.emotionReadings)
    .where(and(
      isNull(schema.emotionReadings.userId),
      eq(schema.emotionReadings.sessionId, '_community'),
      gte(schema.emotionReadings.timestamp, since),
    ))
    .limit(2000);
  const counts: Record<string, number> = {};
  for (const r of rows) {
    if (r.dominantEmotion) counts[r.dominantEmotion] = (counts[r.dominantEmotion] ?? 0) + 1;
  }
  return success(res, { counts, total: rows.length });
}

async function communityShareMood(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { emotion } = req.body || {};
  const validEmotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral', 'stressed', 'relaxed', 'focused'];
  if (!emotion || !validEmotions.includes(emotion)) return badRequest(res, 'Invalid emotion');
  const db = getDb();
  // Rate limit: 5 shares per IP per hour to prevent flooding
  const ip = getClientIp(req);
  const rl = await checkRateLimit(db, `community-mood:${ip}`, 5, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  // Store anonymously: userId = null, sessionId = '_community'
  await db.insert(schema.emotionReadings).values({
    userId: null, sessionId: '_community',
    stress: 0.5, happiness: 0.5, focus: 0.5, energy: 0.5,
    dominantEmotion: emotion,
  });
  // Return updated counts
  return communityMoodFeed(req, res);
}

// ── GET /api/user — return authenticated user profile ────────────────────────

async function userGet(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const payload = requireAuth(req, res);
  if (!payload) return;
  const db = getDb();
  const [user] = await db.select().from(schema.users).where(eq(schema.users.id, payload.userId));
  if (!user) return error(res, 'User not found', 404);
  const { password: _, ...safe } = user;
  return success(res, safe);
}

// ── GET /api/glucose/current/:userId — CGM glucose stub ──────────────────────

async function glucoseCurrent(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  // No CGM integration yet — return null data so the UI gracefully shows "no data"
  const db = getDb();
  // Check if user has a CGM device connected
  const [cgmRow] = await db.select({ id: schema.deviceConnections.id })
    .from(schema.deviceConnections)
    .where(and(eq(schema.deviceConnections.userId, userId), eq(schema.deviceConnections.provider, 'cgm')))
    .limit(1);
  if (!cgmRow) return success(res, { current: null, trend: null });
  // Future: fetch live CGM data from connected provider
  return success(res, { current: null, trend: null });
}

// ── GET /api/food/mood-correlation/:userId — correlate food with mood ─────────

async function foodMoodCorrelation(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const days = Math.min(Math.max(parseInt(req.query.days as string) || 7, 1), 90);
  const db = getDb();
  const since = new Date(Date.now() - days * 86_400_000);

  const [foodData, moodData, emotionData] = await Promise.all([
    db.select({
      id: schema.foodLogs.id,
      name: schema.foodLogs.name,
      calories: schema.foodLogs.calories,
      protein: schema.foodLogs.protein,
      loggedAt: schema.foodLogs.loggedAt,
    }).from(schema.foodLogs)
      .where(and(eq(schema.foodLogs.userId, userId), gte(schema.foodLogs.loggedAt, since)))
      .orderBy(desc(schema.foodLogs.loggedAt))
      .limit(500),
    db.select({
      moodScore: schema.moodLogs.moodScore,
      loggedAt: schema.moodLogs.loggedAt,
    }).from(schema.moodLogs)
      .where(and(eq(schema.moodLogs.userId, userId), gte(schema.moodLogs.loggedAt, since)))
      .limit(500),
    db.select({
      dominantEmotion: schema.emotionReadings.dominantEmotion,
      timestamp: schema.emotionReadings.timestamp,
    }).from(schema.emotionReadings)
      .where(and(eq(schema.emotionReadings.userId, userId), gte(schema.emotionReadings.timestamp, since)))
      .limit(500),
  ]);

  const WINDOW_MS = 2 * 60 * 60 * 1000; // ±2 hours
  let matchedWithMood = 0;
  let matchedWithEmotion = 0;

  for (const food of foodData) {
    const foodTs = new Date(food.loggedAt!).getTime();
    if (moodData.some(m => Math.abs(new Date(m.loggedAt!).getTime() - foodTs) <= WINDOW_MS)) {
      matchedWithMood++;
    }
    if (emotionData.some(e => Math.abs(new Date(e.timestamp).getTime() - foodTs) <= WINDOW_MS)) {
      matchedWithEmotion++;
    }
  }

  return success(res, {
    entries: foodData,
    totalFoodLogs: foodData.length,
    matchedWithMood,
    matchedWithEmotion,
  });
}

// ── Dream weekly synthesis (AI) ───────────────────────────────────────────────

async function dreamWeeklySynthesis(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  // Rate limit: 10 syntheses per user per hour (calls LLM)
  const rl = await checkRateLimit(db, `dream-weekly-synthesis:${userId}`, 3, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!, 'Too many synthesis requests. Please wait before generating again.');

  const since = new Date(Date.now() - 7 * 86_400_000);
  const dreams = await db.select({
    dreamText: schema.dreamAnalysis.dreamText,
    themes: schema.dreamAnalysis.themes,
    emotionalArc: schema.dreamAnalysis.emotionalArc,
    keyInsight: schema.dreamAnalysis.keyInsight,
    threatSimulationIndex: schema.dreamAnalysis.threatSimulationIndex,
    symbols: schema.dreamAnalysis.symbols,
    timestamp: schema.dreamAnalysis.timestamp,
  }).from(schema.dreamAnalysis)
    .where(and(eq(schema.dreamAnalysis.userId, userId), gte(schema.dreamAnalysis.timestamp, since)))
    .orderBy(asc(schema.dreamAnalysis.timestamp))
    .limit(20);

  if (dreams.length === 0) {
    return success(res, {
      synthesis: 'No dreams recorded this week. Start journaling your dreams to receive a personalised weekly synthesis.',
      topThemes: [],
      dreamCount: 0,
      nightmareCount: 0,
      generatedAt: new Date().toISOString(),
    });
  }

  const nightmareCount = dreams.filter(d =>
    (d.threatSimulationIndex as number | null) !== null && (d.threatSimulationIndex as number) > 0.6
  ).length;

  // Build compact context for the LLM
  const dreamLines = dreams.map((d, i) => {
    const date = new Date(d.timestamp).toISOString().slice(0, 10);
    const parts = [`Dream ${i + 1} (${date})`];
    const themes = d.themes as string[] | null;
    if (Array.isArray(themes) && themes.length > 0) parts.push(`themes: ${themes.join(', ')}`);
    if (d.emotionalArc) parts.push(`arc: ${(d.emotionalArc as string).slice(0, 80)}`);
    if (d.keyInsight) parts.push(`insight: "${(d.keyInsight as string).slice(0, 100)}"`);
    if (d.dreamText) parts.push(`text: "${(d.dreamText as string).slice(0, 150)}"`);
    return parts.join(' | ');
  }).join('\n');

  // Aggregate top themes
  const themeCounts = new Map<string, number>();
  for (const d of dreams) {
    const th = d.themes as string[] | null;
    if (Array.isArray(th)) {
      for (const t of th) themeCounts.set(t, (themeCounts.get(t) ?? 0) + 1);
    }
  }
  const topThemes = [...themeCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5).map(([t]) => t);

  const openai = getOpenAIClient();
  let synthesis = '';
  try {
    const resp = await openai.chat.completions.create({
      model: 'llama-3.3-70b-versatile',
      messages: [
        {
          role: 'system',
          content: 'You are a thoughtful dream analyst. Write a 3-4 sentence personalised weekly dream synthesis. Focus on recurring emotional themes, growth patterns, and one key insight. Be warm, specific, and avoid generic statements.',
        },
        {
          role: 'user',
          content: `Here are my dreams from the past week:\n\n${dreamLines}\n\nWrite a personalised weekly synthesis narrative.`,
        },
      ],
      max_tokens: 300,
      temperature: 0.7,
    });
    synthesis = resp.choices?.[0]?.message?.content?.trim() ?? '';
  } catch {
    synthesis = `This week you recorded ${dreams.length} dream${dreams.length !== 1 ? 's' : ''}. ${topThemes.length > 0 ? `Recurring themes include: ${topThemes.slice(0, 3).join(', ')}.` : ''} ${nightmareCount > 0 ? `${nightmareCount} nightmare${nightmareCount !== 1 ? 's' : ''} were flagged — consider exploring IRT practices.` : 'No nightmares this week — a positive sign.'}`;
  }

  return success(res, {
    synthesis,
    topThemes,
    dreamCount: dreams.length,
    nightmareCount,
    generatedAt: new Date().toISOString(),
  });
}

// ── Dream patterns ────────────────────────────────────────────────────────────

async function dreamPatternsGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const days = Math.min(Math.max(parseInt(req.query.days as string) || 30, 1), 365);
  const db = getDb();
  const since = new Date(Date.now() - days * 86_400_000);
  const dreams = await db.select({
    id: schema.dreamAnalysis.id,
    dreamText: schema.dreamAnalysis.dreamText,
    symbols: schema.dreamAnalysis.symbols,
    themes: schema.dreamAnalysis.themes,
    lucidityScore: schema.dreamAnalysis.lucidityScore,
    sleepQuality: schema.dreamAnalysis.sleepQuality,
    tags: schema.dreamAnalysis.tags,
    threatSimulationIndex: schema.dreamAnalysis.threatSimulationIndex,
    keyInsight: schema.dreamAnalysis.keyInsight,
    aiAnalysis: schema.dreamAnalysis.aiAnalysis,
    timestamp: schema.dreamAnalysis.timestamp,
  }).from(schema.dreamAnalysis)
    .where(and(eq(schema.dreamAnalysis.userId, userId), gte(schema.dreamAnalysis.timestamp, since)))
    .orderBy(desc(schema.dreamAnalysis.timestamp))
    .limit(500);

  // Aggregate themes
  const themeCounts = new Map<string, number>();
  for (const d of dreams) {
    const th = d.themes as string[] | null;
    if (Array.isArray(th)) {
      for (const t of th) themeCounts.set(t, (themeCounts.get(t) ?? 0) + 1);
    }
  }
  const themes = [...themeCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10).map(([name, count]) => ({ name, count }));

  // Aggregate symbols
  const symCounts = new Map<string, number>();
  for (const d of dreams) {
    const syms = d.symbols as string[] | null;
    if (Array.isArray(syms)) {
      for (const s of syms) symCounts.set(s, (symCounts.get(s) ?? 0) + 1);
    }
  }
  const symbols = [...symCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10).map(([name, count]) => ({ name, count }));

  // Sentiment trend (approximated via threatSimulationIndex: lower = more positive)
  const sentimentTrend = dreams
    .filter(d => d.threatSimulationIndex !== null)
    .map(d => ({
      date: new Date(d.timestamp).toISOString().slice(0, 10),
      valence: 1 - (d.threatSimulationIndex as number),
    }))
    .reverse();

  // Nightmare detection: tagged or high threatSimulation
  const nightmareDates: string[] = [];
  for (const d of dreams) {
    const tags = d.tags as string[] | null;
    const isNightmare = (Array.isArray(tags) && tags.includes('nightmare')) ||
      ((d.threatSimulationIndex as number | null) !== null && (d.threatSimulationIndex as number) > 0.6);
    if (isNightmare) nightmareDates.push(new Date(d.timestamp).toISOString());
  }

  // Count by window
  const now = Date.now();
  const countInWindow = (windowMs: number) => dreams.filter(d => now - new Date(d.timestamp).getTime() < windowMs).length;

  const topInsights = dreams.slice(0, 5).map(d => ({
    insight: d.aiAnalysis ? d.aiAnalysis.slice(0, 120) : null,
    keyInsight: d.keyInsight ?? null,
    date: new Date(d.timestamp).toISOString(),
  }));

  return success(res, {
    period: days,
    entryCount: dreams.length,
    themes,
    symbols,
    sentimentTrend,
    topInsights,
    nightmareCount: nightmareDates.length,
    nightmareDates,
    counts: {
      last7: countInWindow(7 * 86_400_000),
      last30: countInWindow(30 * 86_400_000),
      last90: countInWindow(90 * 86_400_000),
    },
  });
}

// ── Dream symbols ─────────────────────────────────────────────────────────────

async function dreamSymbolsGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (!requireOwner(req, res, userId)) return;
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const db = getDb();
  const rows = await db.select().from(schema.dreamSymbols)
    .where(eq(schema.dreamSymbols.userId, userId))
    .orderBy(desc(schema.dreamSymbols.frequency))
    .limit(100);
  return success(res, rows);
}

// ── Dream quality trend ───────────────────────────────────────────────────────

async function dreamQualityTrend(req: VercelRequest, res: VercelResponse, userId: string) {
  if (!requireOwner(req, res, userId)) return;
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const days = Math.min(Math.max(parseInt(req.query.days as string) || 14, 1), 90);
  const db = getDb();
  const since = new Date(Date.now() - days * 86_400_000);
  const dreams = await db.select({
    lucidityScore: schema.dreamAnalysis.lucidityScore,
    sleepQuality: schema.dreamAnalysis.sleepQuality,
    threatSimulationIndex: schema.dreamAnalysis.threatSimulationIndex,
    dreamText: schema.dreamAnalysis.dreamText,
    timestamp: schema.dreamAnalysis.timestamp,
  }).from(schema.dreamAnalysis)
    .where(and(eq(schema.dreamAnalysis.userId, userId), gte(schema.dreamAnalysis.timestamp, since)))
    .orderBy(asc(schema.dreamAnalysis.timestamp))
    .limit(500);

  // Compute daily quality scores — blend lucidity(0-100) + sleepQuality(0-10->*10) + threatInversion
  const dayMap = new Map<string, { sum: number; count: number }>();
  for (const d of dreams) {
    const date = new Date(d.timestamp).toISOString().slice(0, 10);
    const parts: number[] = [];
    if (d.lucidityScore !== null && d.lucidityScore !== undefined) parts.push(Math.min(d.lucidityScore, 100));
    if (d.sleepQuality !== null && d.sleepQuality !== undefined) parts.push(Math.min(d.sleepQuality * 10, 100));
    if (d.threatSimulationIndex !== null && d.threatSimulationIndex !== undefined) parts.push((1 - (d.threatSimulationIndex as number)) * 100);
    if (d.dreamText) parts.push(60); // base score for having any entry
    if (parts.length === 0) continue;
    const score = parts.reduce((a, b) => a + b, 0) / parts.length;
    const entry = dayMap.get(date) ?? { sum: 0, count: 0 };
    entry.sum += score;
    entry.count++;
    dayMap.set(date, entry);
  }

  const trend = [...dayMap.entries()].map(([date, { sum, count }]) => ({
    date,
    score: Math.round(sum / count),
    count,
  }));

  const allScores = trend.map(t => t.score);
  const avgScore = allScores.length > 0 ? Math.round(allScores.reduce((a, b) => a + b, 0) / allScores.length) : null;
  const current = trend.length > 0 ? trend[trend.length - 1].score : null;

  return success(res, { current, avgScore, trend, totalDreams: dreams.length });
}

// ── Nightmare recurrence ──────────────────────────────────────────────────────

async function nightmareRecurrence(req: VercelRequest, res: VercelResponse, userId: string) {
  if (!requireOwner(req, res, userId)) return;
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const db = getDb();
  const since = new Date(Date.now() - 14 * 86_400_000);
  const dreams = await db.select({
    tags: schema.dreamAnalysis.tags,
    threatSimulationIndex: schema.dreamAnalysis.threatSimulationIndex,
    timestamp: schema.dreamAnalysis.timestamp,
  }).from(schema.dreamAnalysis)
    .where(and(eq(schema.dreamAnalysis.userId, userId), gte(schema.dreamAnalysis.timestamp, since)))
    .orderBy(desc(schema.dreamAnalysis.timestamp))
    .limit(200);

  const isNightmare = (d: typeof dreams[0]) => {
    const tags = d.tags as string[] | null;
    return (Array.isArray(tags) && tags.includes('nightmare')) ||
      ((d.threatSimulationIndex as number | null) !== null && (d.threatSimulationIndex as number) > 0.6);
  };

  const now = Date.now();
  const week1 = dreams.filter(d => now - new Date(d.timestamp).getTime() < 7 * 86_400_000 && isNightmare(d));
  const week2 = dreams.filter(d => {
    const age = now - new Date(d.timestamp).getTime();
    return age >= 7 * 86_400_000 && age < 14 * 86_400_000 && isNightmare(d);
  });

  const recentNightmares = week1.length;
  const olderNightmares = week2.length;
  const trend = recentNightmares > olderNightmares ? 'worsening' : recentNightmares < olderNightmares ? 'improving' : 'stable';
  const lastNightmare = week1[0] ?? week2[0];

  // IRT sessions
  const irtRows = await db.select({ createdAt: schema.irtSessions.createdAt })
    .from(schema.irtSessions)
    .where(eq(schema.irtSessions.userId, userId))
    .orderBy(desc(schema.irtSessions.createdAt))
    .limit(100);

  const irtSessionCount = irtRows.length;
  const lastIrtDate = irtRows[0]?.createdAt ? new Date(irtRows[0].createdAt).toISOString() : null;

  // Nightmares after most recent IRT session
  let postIrtNightmares = 0;
  if (lastIrtDate) {
    const lastIrtTs = new Date(lastIrtDate).getTime();
    postIrtNightmares = dreams.filter(d => new Date(d.timestamp).getTime() > lastIrtTs && isNightmare(d)).length;
  }

  return success(res, {
    recentNightmares,
    olderNightmares,
    trend,
    lastNightmareDate: lastNightmare ? new Date(lastNightmare.timestamp).toISOString() : null,
    irtSessionCount,
    lastIrtDate,
    postIrtNightmares,
  });
}

// ── Device connections (connect / sync / disconnect) ──────────────────────────

async function devicesConnect(req: VercelRequest, res: VercelResponse, provider: string) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const payload = requireAuth(req, res);
  if (!payload) return;
  const db = getDb();
  // Rate limit: 10 device connect attempts per user per hour
  const rl = await checkRateLimit(db, `devices-connect:${payload.userId}`, 10, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  // Upsert a pending connection — real OAuth flow would redirect to provider
  const [row] = await db.insert(schema.deviceConnections)
    .values({
      userId: payload.userId,
      provider,
      accessToken: 'pending',
      syncStatus: 'pending',
      connectedAt: new Date(),
    })
    .onConflictDoUpdate({
      target: [schema.deviceConnections.userId, schema.deviceConnections.provider],
      set: { syncStatus: 'pending', errorMessage: null },
    })
    .returning();
  return success(res, { message: `Connection initiated for ${provider}`, device: row, authUrl: null });
}

async function devicesSync(req: VercelRequest, res: VercelResponse, provider: string) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const payload = requireAuth(req, res);
  if (!payload) return;
  const db = getDb();
  // Rate limit: 20 sync requests per user per hour
  const rlSync = await checkRateLimit(db, `devices-sync:${payload.userId}`, 20, 60);
  if (!rlSync.allowed) return tooManyRequests(res, rlSync.retryAfterSeconds!);
  const [row] = await db.update(schema.deviceConnections)
    .set({ lastSyncAt: new Date(), syncStatus: 'active', errorMessage: null })
    .where(and(eq(schema.deviceConnections.userId, payload.userId), eq(schema.deviceConnections.provider, provider)))
    .returning();
  if (!row) return error(res, 'Device not connected', 404);
  return success(res, { message: `Sync triggered for ${provider}`, device: row });
}

async function devicesDisconnect(req: VercelRequest, res: VercelResponse, provider: string) {
  if (req.method !== 'DELETE') return methodNotAllowed(res, ['DELETE']);
  const payload = requireAuth(req, res);
  if (!payload) return;
  const db = getDb();
  await db.delete(schema.deviceConnections)
    .where(and(eq(schema.deviceConnections.userId, payload.userId), eq(schema.deviceConnections.provider, provider)));
  return success(res, { message: `Disconnected ${provider}` });
}

// ── Clear all user data ───────────────────────────────────────────────────────

async function clearUserData(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'DELETE') return methodNotAllowed(res, ['DELETE']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  // Rate limit: 2 full data wipes per hour to prevent accidental repeated deletion
  const rl = await checkRateLimit(db, `clear-user-data:${userId}`, 2, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);
  // Delete in dependency order (children before parents)
  await db.delete(schema.habitLogs).where(eq(schema.habitLogs.userId, userId));
  await db.delete(schema.habits).where(eq(schema.habits.userId, userId));
  // workoutSets references workoutId, so fetch user workout IDs first
  const userWorkouts = await db.select({ id: schema.workouts.id }).from(schema.workouts).where(eq(schema.workouts.userId, userId));
  if (userWorkouts.length > 0) {
    const { inArray } = await import('drizzle-orm');
    await db.delete(schema.workoutSets).where(inArray(schema.workoutSets.workoutId, userWorkouts.map((w: any) => w.id)));
  }
  await db.delete(schema.workouts).where(eq(schema.workouts.userId, userId));
  await db.delete(schema.exerciseHistory).where(eq(schema.exerciseHistory.userId, userId));
  await db.delete(schema.bodyMetrics).where(eq(schema.bodyMetrics.userId, userId));
  await db.delete(schema.cycleTracking).where(eq(schema.cycleTracking.userId, userId));
  await db.delete(schema.moodLogs).where(eq(schema.moodLogs.userId, userId));
  await db.delete(schema.mealHistory).where(eq(schema.mealHistory.userId, userId));
  await db.delete(schema.foodLogs).where(eq(schema.foodLogs.userId, userId));
  await db.delete(schema.dreamSymbols).where(eq(schema.dreamSymbols.userId, userId));
  await db.delete(schema.dreamAnalysis).where(eq(schema.dreamAnalysis.userId, userId));
  await db.delete(schema.emotionReadings).where(eq(schema.emotionReadings.userId, userId));
  await db.delete(schema.brainReadings).where(eq(schema.brainReadings.userId, userId));
  await db.delete(schema.healthSamples).where(eq(schema.healthSamples.userId, userId));
  await db.delete(schema.healthMetrics).where(eq(schema.healthMetrics.userId, userId));
  await db.delete(schema.aiChats).where(eq(schema.aiChats.userId, userId));
  await db.delete(schema.realityTests).where(eq(schema.realityTests.userId, userId));
  await db.delete(schema.irtSessions).where(eq(schema.irtSessions.userId, userId));
  await db.delete(schema.userReadings).where(eq(schema.userReadings.userId, userId));
  await db.delete(schema.userSettings).where(eq(schema.userSettings.userId, userId));
  return success(res, { message: 'All data cleared successfully' });
}

// ── Exercise history personal records ─────────────────────────────────────────

async function exerciseHistoryPrs(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  // Fetch PRs: best estimated1rm and bestWeightKg per exercise
  const rows = await db
    .select({
      exerciseId: schema.exerciseHistory.exerciseId,
      bestWeightKg: schema.exerciseHistory.bestWeightKg,
      bestReps: schema.exerciseHistory.bestReps,
      estimated1rm: schema.exerciseHistory.estimated1rm,
      totalVolume: schema.exerciseHistory.totalVolume,
      date: schema.exerciseHistory.date,
      exerciseName: schema.exercises.name,
      exerciseCategory: schema.exercises.category,
    })
    .from(schema.exerciseHistory)
    .leftJoin(schema.exercises, eq(schema.exerciseHistory.exerciseId, schema.exercises.id))
    .where(eq(schema.exerciseHistory.userId, userId))
    .orderBy(desc(schema.exerciseHistory.estimated1rm))
    .limit(200);
  return success(res, rows);
}

// ── Brain: yesterday insights ─────────────────────────────────────────────────

async function brainYesterdayInsights(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const now = new Date();
  const yesterdayStart = new Date(now);
  yesterdayStart.setDate(yesterdayStart.getDate() - 1);
  yesterdayStart.setHours(0, 0, 0, 0);
  const yesterdayEnd = new Date(yesterdayStart);
  yesterdayEnd.setHours(23, 59, 59, 999);

  const rows = await db
    .select({
      stress: schema.brainReadings.stress,
      focus: schema.brainReadings.focus,
      relaxation: schema.brainReadings.relaxation,
      dominantEmotion: schema.brainReadings.dominantEmotion,
      valence: schema.brainReadings.valence,
      timestamp: schema.brainReadings.timestamp,
    })
    .from(schema.brainReadings)
    .where(
      and(
        eq(schema.brainReadings.userId, userId),
        gte(schema.brainReadings.timestamp, yesterdayStart),
        lt(schema.brainReadings.timestamp, yesterdayEnd),
      )
    )
    .orderBy(desc(schema.brainReadings.timestamp))
    .limit(500);

  if (rows.length === 0) return success(res, null);

  const avg = (arr: (number | null)[]) => {
    const valid = arr.filter((v): v is number => v !== null && v !== undefined);
    return valid.length > 0 ? valid.reduce((a, b) => a + b, 0) / valid.length : null;
  };

  const emotionCounts = new Map<string, number>();
  for (const r of rows) {
    if (r.dominantEmotion) {
      emotionCounts.set(r.dominantEmotion, (emotionCounts.get(r.dominantEmotion) ?? 0) + 1);
    }
  }
  const dominantEmotion = emotionCounts.size > 0
    ? [...emotionCounts.entries()].sort((a, b) => b[1] - a[1])[0][0]
    : null;

  return success(res, {
    date: yesterdayStart.toISOString().slice(0, 10),
    readingCount: rows.length,
    avgStress: avg(rows.map(r => r.stress)),
    avgFocus: avg(rows.map(r => r.focus)),
    avgRelaxation: avg(rows.map(r => r.relaxation)),
    avgValence: avg(rows.map(r => r.valence)),
    dominantEmotion,
  });
}

// ── Brain: patterns ───────────────────────────────────────────────────────────

async function brainPatterns(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const days = Math.min(Math.max(parseInt(req.query.days as string) || 30, 1), 90);
  const since = new Date(Date.now() - days * 86_400_000);

  const rows = await db
    .select({
      stress: schema.brainReadings.stress,
      focus: schema.brainReadings.focus,
      relaxation: schema.brainReadings.relaxation,
      dominantEmotion: schema.brainReadings.dominantEmotion,
      timestamp: schema.brainReadings.timestamp,
    })
    .from(schema.brainReadings)
    .where(and(eq(schema.brainReadings.userId, userId), gte(schema.brainReadings.timestamp, since)))
    .orderBy(asc(schema.brainReadings.timestamp))
    .limit(2000);

  if (rows.length === 0) return success(res, { days, readingCount: 0, emotions: [], peakFocusHour: null, peakStressHour: null });

  // Emotion frequency
  const emotionCounts = new Map<string, number>();
  for (const r of rows) {
    if (r.dominantEmotion) {
      emotionCounts.set(r.dominantEmotion, (emotionCounts.get(r.dominantEmotion) ?? 0) + 1);
    }
  }
  const emotions = [...emotionCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([emotion, count]) => ({ emotion, count, pct: Math.round((count / rows.length) * 100) }));

  // Peak focus/stress by hour of day
  const hourFocus = new Array(24).fill(0).map(() => ({ sum: 0, count: 0 }));
  const hourStress = new Array(24).fill(0).map(() => ({ sum: 0, count: 0 }));
  for (const r of rows) {
    const h = new Date(r.timestamp).getHours();
    if (r.focus !== null && r.focus !== undefined) { hourFocus[h].sum += r.focus; hourFocus[h].count++; }
    if (r.stress !== null && r.stress !== undefined) { hourStress[h].sum += r.stress; hourStress[h].count++; }
  }
  const peakFocusHour = hourFocus.reduce((best, h, i) =>
    h.count > 0 && (best === null || h.sum / h.count > hourFocus[best].sum / hourFocus[best].count) ? i : best, null as number | null);
  const peakStressHour = hourStress.reduce((best, h, i) =>
    h.count > 0 && (best === null || h.sum / h.count > hourStress[best].sum / hourStress[best].count) ? i : best, null as number | null);

  return success(res, { days, readingCount: rows.length, emotions, peakFocusHour, peakStressHour });
}

// ── Demo data seeder ─────────────────────────────────────────────────────────

async function seedDemo(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res);
  const db = getDb();
  const payload = requireAuth(req, res);
  if (!payload) return;
  const userId = payload.userId;

  const now = Date.now();
  const day = 86_400_000;

  // 7 days of health metrics (morning + evening readings)
  const healthRows = [];
  for (let d = 6; d >= 0; d--) {
    const base = now - d * day;
    healthRows.push(
      { userId, heartRate: 62 + Math.round(Math.random() * 12), stressLevel: 3 + Math.round(Math.random() * 4), sleepQuality: 6 + Math.round(Math.random() * 3), neuralActivity: 55 + Math.round(Math.random() * 25), dailySteps: 5000 + Math.round(Math.random() * 5000), sleepDuration: 6 + Math.round(Math.random() * 2), timestamp: new Date(base - 8 * 3_600_000) },
      { userId, heartRate: 70 + Math.round(Math.random() * 15), stressLevel: 4 + Math.round(Math.random() * 5), sleepQuality: 5 + Math.round(Math.random() * 4), neuralActivity: 60 + Math.round(Math.random() * 30), dailySteps: 5000 + Math.round(Math.random() * 5000), sleepDuration: 7, timestamp: new Date(base) },
    );
  }
  await db.insert(schema.healthMetrics).values(healthRows).onConflictDoNothing();

  // 5 dream entries
  const dreamTexts = [
    { text: "Flying over a glowing city at dusk, feeling weightless and free. The skyline pulsed with violet light.", analysis: "Flying dreams correlate with elevated theta activity — your brain entered a hypnagogic state. The violet light imagery suggests right-hemisphere alpha dominance.", symbols: ["flying","city","light"] },
    { text: "Walking through a vast library where every book glowed from within. I could hear the thoughts inside.", analysis: "Library dreams indicate memory consolidation. Luminescent books represent knowledge your hippocampus is actively encoding during NREM sleep.", symbols: ["library","knowledge","light"] },
    { text: "Deep in a forest, a stream ran backwards. A wolf watched from the edge but felt friendly.", analysis: "Water symbolizes emotional flow; reversed water suggests processing conflicting emotions. The wolf represents primal intelligence — your amygdala processing instinct.", symbols: ["forest","water","wolf"] },
    { text: "Standing on a glass floor above clouds. Could see cities below. Felt completely calm.", analysis: "Height dreams with calm affect indicate strong prefrontal regulation. Glass floor represents clarity of conscious awareness during REM.", symbols: ["height","clouds","calm"] },
    { text: "Playing piano music I had never heard but somehow knew. Every note felt inevitable.", analysis: "Creative dreams during REM show the default mode network integrating disparate memories. Procedural creativity peaks in late-cycle REM.", symbols: ["music","creativity","memory"] },
  ];
  const dreamRows = dreamTexts.map((d, i) => ({
    userId, dreamText: d.text, symbols: d.symbols, aiAnalysis: d.analysis,
    lucidityScore: 30 + i * 12,
    sleepQuality: 6 + i,
    timestamp: new Date(now - (6 - i) * day - 7 * 3_600_000),
  }));
  await db.insert(schema.dreamAnalysis).values(dreamRows).onConflictDoNothing();

  // 5 emotion readings
  const emotions = ['happy','neutral','happy','focused','relaxed'] as const;
  const emotionRows = emotions.map((e, i) => ({
    userId, dominantEmotion: e,
    valence: 0.2 + i * 0.1, arousal: 0.4 + i * 0.05,
    stressIndex: 0.2 + i * 0.05, focusIndex: 0.5 + i * 0.08,
    relaxationIndex: 0.6 - i * 0.05,
    emotionProbabilities: { happy: 0.5, neutral: 0.2, sad: 0.1, angry: 0.05, fear: 0.05, surprise: 0.1 },
    bandPowers: { delta: 0.15, theta: 0.2, alpha: 0.3, beta: 0.25, gamma: 0.1 },
    timestamp: new Date(now - (4 - i) * day),
  }));
  await db.insert(schema.emotionReadings).values(emotionRows).onConflictDoNothing();

  return success(res, {
    seeded: {
      healthMetrics: healthRows.length,
      dreams: dreamRows.length,
      emotions: emotionRows.length,
    },
    message: 'Demo data loaded — refresh any page to see your data.',
  });
}

// ── Main router ──────────────────────────────────────────────────────────────

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS preflight
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, x-body-b64');
  if (req.method === 'OPTIONS') return res.status(200).end();

  // Lazy-load all heavy modules (schema, db, openai) before routing to avoid cold-start crash
  await loadModules();

  // Parse body early so all route handlers can safely access req.body.
  // Vercel's runtime has a bug where req.body getter throws "Invalid JSON" and
  // the stream is consumed. Use Object.defineProperty to force-override the getter.
  if (['POST', 'PUT', 'PATCH'].includes(req.method || '')) {
    const parsed = await parseRequestBody(req);
    try {
      Object.defineProperty(req, 'body', { value: parsed, writable: true, configurable: true });
    } catch {
      // If defineProperty fails (non-configurable), mutate in place
      (req as any).body = parsed;
    }
  }

  // Extract path segments from /api/seg0/seg1/...
  const url = req.url || '';
  const clean = url.split('?')[0].replace(/^\/api\//, '');
  const segs = clean.split('/').filter(Boolean);

  try {
    const [s0, s1] = segs;

    if (s0 === 'ping' || (s0 === 'health' && !s1)) return res.status(200).json({ ok: true, ts: Date.now() });

    if (s0 === 'auth') {
      if (s1 === 'register')        return await authRegister(req, res);
      if (s1 === 'login')           return await authLogin(req, res);
      if (s1 === 'me')              return await authMe(req, res);
      if (s1 === 'logout')          return await authLogout(req, res);
      if (s1 === 'forgot-password') return await authForgotPassword(req, res);
      if (s1 === 'reset-password')  return await authResetPassword(req, res);
    }

    if (s0 === 'dreams') {
      if (s1 === 'create')         return await dreamsCreate(req, res);
      if (s1 === 'list')           return await dreamsList(req, res);
      if (s1 === 'analytics')      return await dreamsAnalytics(req, res);
      if (s1 === 'generate-image') return await dreamsGenerateImage(req, res);
    }

    if (s0 === 'dream-analysis') {
      if (!s1) return await dreamAnalysisPost(req, res);
      if (s1 === 'multi-pass') return await dreamAnalysisMultiPassPost(req, res);
      return await dreamAnalysisGet(req, res, s1);
    }

    if (s0 === 'ai-chat') {
      if (!s1) return await aiChatPost(req, res);
      if (req.method === 'GET') return await aiChatGet(req, res, s1);
      return await aiChatPost(req, res);
    }

    if (s0 === 'emotions') {
      if (s1 === 'record')  return await emotionsRecord(req, res);
      if (s1 === 'history') return await emotionsHistory(req, res);
      if (s1 && segs[2] === 'correct') return await emotionsCorrect(req, res, s1);
      if (s1 === 'correct-latest' && segs[2]) return await emotionsCorrectLatest(req, res, segs[2]);
    }

    if (s0 === 'health-metrics') {
      if (!s1) return await healthMetricsPost(req, res);
      return await healthMetricsGet(req, res, s1);
    }

    if (s0 === 'health-samples') {
      if (!s1 && req.method === 'POST') return await healthSamplesPost(req, res);
      if (s1 && req.method === 'GET') return await healthSamplesGet(req, res, s1);
    }

    if (s0 === 'settings' && s1) {
      if (segs[2] === 'data' && req.method === 'DELETE') return await clearUserData(req, res, s1);
      return await settingsHandler(req, res, s1);
    }
    if (s0 === 'export'   && s1) return await exportHandler(req, res, s1);

    if (s0 === 'insights' && s1 === 'weekly') return await insightsWeekly(req, res);
    if (s0 === 'notifications') {
      if (s1 === 'subscribe')         return await notificationsSubscribe(req, res);
      if (s1 === 'vapid-public-key')  return await notificationsVapidPublicKey(req, res);
      if (s1 === 'brain-state-trigger' || s1 === 'trigger') return await notificationsTrigger(req, res);
    }
    if (s0 === 'analyze-mood') return await analyzeMood(req, res);

    if (s0 === 'food') {
      if (s1 === 'analyze') return await foodAnalyze(req, res);
      if (s1 === 'log' && req.method === 'POST') return await foodLog(req, res);
      if (s1 === 'logs' && segs[2]) return await foodLogs(req, res, segs[2]);
      if (s1 === 'mood-correlation' && segs[2] && req.method === 'GET') return await foodMoodCorrelation(req, res, segs[2]);
    }

    if (s0 === 'glucose' && s1 === 'current' && segs[2] && req.method === 'GET') {
      return await glucoseCurrent(req, res, segs[2]);
    }

    // Brain history — emotion/stress/focus readings over time
    if (s0 === 'brain' && s1 === 'history' && segs[2] && req.method === 'GET') {
      return await brainHistory(req, res, segs[2]);
    }

    // Brain today-totals — avg stress/focus/emotion since midnight
    if (s0 === 'brain' && s1 === 'today-totals' && segs[2] && req.method === 'GET') {
      return await brainTodayTotals(req, res, segs[2]);
    }

    // Brain at-this-time-yesterday — ±30 min window same time yesterday
    if (s0 === 'brain' && s1 === 'at-this-time-yesterday' && segs[2] && req.method === 'GET') {
      return await brainAtThisTimeYesterday(req, res, segs[2]);
    }

    // Brain yesterday-insights — daily aggregates for yesterday
    if (s0 === 'brain' && s1 === 'yesterday-insights' && segs[2] && req.method === 'GET') {
      return await brainYesterdayInsights(req, res, segs[2]);
    }

    // Brain patterns — emotion frequency + peak hours over N days
    if (s0 === 'brain' && s1 === 'patterns' && segs[2] && req.method === 'GET') {
      return await brainPatterns(req, res, segs[2]);
    }

    // Inner Score
    if (s0 === 'inner-score' && s1) {
      if (!requireOwner(req, res, s1)) return;
      if (segs[2] === 'history' && req.method === 'GET') return await innerScoreHistory(req, res, s1);
      if (req.method === 'POST') return await innerScorePost(req, res, s1);
      if (req.method === 'GET') return await innerScoreGet(req, res, s1);
    }

    // Streaks
    if (s0 === 'streaks') {
      if (s1 === 'checkin' && req.method === 'POST') return await streaksCheckin(req, res);
      if (s1 && req.method === 'GET') {
        if (!requireOwner(req, res, s1)) return;
        return await streaksGet(req, res, s1);
      }
    }

    // Emotion readings batch
    if (s0 === 'emotion-readings' && s1 === 'batch' && req.method === 'POST') {
      return await emotionReadingsBatch(req, res);
    }

    // User readings (voice/food/health/eeg)
    if (s0 === 'user-readings' && req.method === 'POST') {
      return await userReadingsPost(req, res);
    }

    if (s0 === 'study') {
      if (s1 === 'enroll')               return await studyEnroll(req, res);
      if (s1 === 'morning')              return await studyMorning(req, res);
      if (s1 === 'daytime')              return await studyDaytime(req, res);
      if (s1 === 'evening')              return await studyEvening(req, res);
      if (s1 === 'withdraw')             return await studyWithdraw(req, res);
      if (s1 === 'status' && segs[2])    return await studyStatus(req, res, segs[2]);
      if (s1 === 'history' && segs[2])   return await studyHistory(req, res, segs[2]);
      // Pilot study routes
      if (s1 === 'consent')                                    return await pilotConsent(req, res);
      if (s1 === 'session' && segs[2] === 'start')             return await pilotSessionStart(req, res);
      if (s1 === 'session' && segs[2] === 'complete')          return await pilotSessionComplete(req, res);
      if (s1 === 'session' && segs[2] && segs[3] === 'checkpoint') return await pilotSessionCheckpoint(req, res, Number(segs[2]));
      if (s1 === 'admin' && segs[2] === 'participants')        return await pilotAdminParticipants(req, res);
      if (s1 === 'admin' && segs[2] === 'sessions')            return await pilotAdminSessions(req, res);
      if (s1 === 'admin' && segs[2] === 'stats')              return await pilotAdminStats(req, res);
      if (s1 === 'admin' && segs[2] === 'export-csv')          return await pilotAdminExportCsv(req, res);
    }

    if (s0 === 'user') {
      if (!s1 && req.method === 'GET') return await userGet(req, res);
      if (s1 === 'intent' && req.method === 'GET')   return await userIntentGet(req, res);
      if (s1 === 'intent' && req.method === 'PATCH')  return await userIntentPatch(req, res);
    }

    if (s0 === 'readings') {
      if (!s1 && req.method === 'POST') return await readingsCreate(req, res);
      // /api/readings/:userId/correct-latest — alias used by today.tsx for voice emotion correction
      if (s1 && segs[2] === 'correct-latest' && req.method === 'PATCH') return await emotionsCorrectLatest(req, res, s1);
      if (s1 && req.method === 'GET')   return await readingsList(req, res, s1);
    }

    if (s0 === 'sleep-alarm' && s1 && req.method === 'GET') return await sleepAlarm(req, res, s1);

    if (s0 === 'ai-coach' && req.method === 'POST') return await aiCoachPost(req, res);

    if (s0 === 'irt-session' && req.method === 'POST') return await irtSessionPost(req, res);

    if (s0 === 'exercise-history' && s1 && segs[2] === 'prs' && req.method === 'GET') {
      return await exerciseHistoryPrs(req, res, s1);
    }

    if (s0 === 'dream-frames' && req.method === 'POST') return await dreamFramesPost(req, res);
    if (s0 === 'dream-session-complete' && req.method === 'POST') return await dreamSessionComplete(req, res);
    if (s0 === 'morning-briefing' && req.method === 'POST') return await morningBriefing(req, res);
    if (s0 === 'voice-emotion' && req.method === 'POST') return await voiceEmotionPost(req, res);
    // /api/food-log (offline sync alias for /api/food/log)
    if (s0 === 'food-log' && req.method === 'POST') return await foodLog(req, res);

    if (s0 === 'community') {
      if (s1 === 'mood-feed' && req.method === 'GET') return await communityMoodFeed(req, res);
      if (s1 === 'share-mood' && req.method === 'POST') return await communityShareMood(req, res);
    }

    if (s0 === 'dream-weekly-synthesis' && s1 && req.method === 'POST') return await dreamWeeklySynthesis(req, res, s1);
    if (s0 === 'dream-patterns' && s1 && req.method === 'GET') return await dreamPatternsGet(req, res, s1);
    if (s0 === 'dream-symbols' && s1 && req.method === 'GET') return await dreamSymbolsGet(req, res, s1);
    if (s0 === 'dream-quality-trend' && s1 && req.method === 'GET') return await dreamQualityTrend(req, res, s1);
    if (s0 === 'nightmare-recurrence' && s1 && req.method === 'GET') return await nightmareRecurrence(req, res, s1);

    if (s0 === 'exercises' && req.method === 'GET') {
      const db = getDb();
      const rows = await db.select().from(schema.exercises).orderBy(asc(schema.exercises.category), asc(schema.exercises.name)).limit(500);
      return success(res, rows);
    }

    if (s0 === 'research' && s1 === 'correlation' && segs[2] && req.method === 'GET') {
      return await researchCorrelation(req, res, segs[2]);
    }

    if (s0 === 'body-metrics' && s1) {
      if (segs[2] === 'latest' && req.method === 'GET') return await bodyMetricsLatest(req, res, s1);
      if (req.method === 'GET') return await bodyMetricsList(req, res, s1);
    }

    if (s0 === 'health' && s1 && segs[2]) {
      const metric = s1; // e.g. "steps" or "heart-rate"
      const userId = segs[2];
      return await healthSamplesByMetric(req, res, userId, metric);
    }

    if (s0 === 'device-connections' && s1 && req.method === 'GET') return await deviceConnectionsList(req, res, s1);
    if (s0 === 'devices') {
      const ALLOWED_PROVIDERS = ['apple', 'google', 'garmin', 'fitbit', 'polar', 'whoop', 'oura'];
      if (s1 === 'connect' && segs[2] && req.method === 'POST') {
        if (!ALLOWED_PROVIDERS.includes(segs[2])) return badRequest(res, 'Unknown device provider');
        return await devicesConnect(req, res, segs[2]);
      }
      if (s1 === 'sync' && segs[2] && req.method === 'POST') {
        if (!ALLOWED_PROVIDERS.includes(segs[2])) return badRequest(res, 'Unknown device provider');
        return await devicesSync(req, res, segs[2]);
      }
      if (s1 && req.method === 'DELETE') {
        if (!ALLOWED_PROVIDERS.includes(s1)) return badRequest(res, 'Unknown device provider');
        return await devicesDisconnect(req, res, s1);
      }
      if (s1 && req.method === 'GET') return await devicesList(req, res, s1);
    }

    if (s0 === 'health' && s1 === 'connect' && req.method === 'POST') {
      return success(res, { message: 'HealthKit connection acknowledged' });
    }

    if (s0 === 'workouts') {
      if (!s1 && req.method === 'POST') return await workoutsPost(req, res);
      if (s1 && segs[2] === 'sets' && req.method === 'POST') return await workoutSetsPost(req, res, s1);
      if (s1 && req.method === 'PUT') return await workoutsPut(req, res, s1);
      if (s1 && req.method === 'GET') return await workoutsGet(req, res, s1);
    }

    if (s0 === 'workout-templates') {
      if (!s1 && req.method === 'POST') return await workoutTemplatesPost(req, res);
      if (s1 && req.method === 'DELETE') return await workoutTemplatesDelete(req, res, s1);
      if (s1 && req.method === 'GET') return await workoutTemplatesGet(req, res, s1);
    }

    if (s0 === 'habits') {
      if (!s1 && req.method === 'POST') return await habitsPost(req, res);
      if (s1 && req.method === 'GET') return await habitsGet(req, res, s1);
      if (s1 && req.method === 'DELETE') return await habitsDelete(req, res, s1);
    }

    if (s0 === 'habit-logs') {
      if (!s1 && req.method === 'POST') return await habitLogsPost(req, res);
      if (s1 && segs[2] === 'streaks' && req.method === 'GET') return await habitLogsStreaks(req, res, s1);
      if (s1 && req.method === 'GET') return await habitLogsGet(req, res, s1);
    }

    if (s0 === 'reality-test') {
      if (!s1 && req.method === 'POST') return await realityTestPost(req, res);
      if (s1 && req.method === 'GET') return await realityTestGet(req, res, s1);
    }

    if (s0 === 'cycle') {
      if (!s1 && req.method === 'POST') return await cyclePost(req, res);
      if (s1 && segs[2] === 'phase' && req.method === 'GET') return await cyclePhase(req, res, s1);
      if (s1 && req.method === 'GET') return await cycleGet(req, res, s1);
    }

    if (s0 === 'mood') {
      if (!s1 && req.method === 'POST') return await moodLogPost(req, res);
      if (s1 && req.method === 'GET') return await moodLogGet(req, res, s1);
    }

    if (s0 === 'meal-history') {
      if (s1 && segs[2] === 'favorite' && req.method === 'PATCH') return await mealHistoryToggleFavorite(req, res, s1);
      if (s1 && req.method === 'GET') return await mealHistoryList(req, res, s1);
    }

    if (s0 === 'seed-demo') return await seedDemo(req, res);

    // ── ML backend proxy: /api/ml/* → ML_API_URL/* ─────────────────────
    // Mirrors the Express proxy in server/routes.ts.  Covers all FastAPI
    // route prefixes: voice-watch, streaks, supplements, voice-biomarkers,
    // voice-ensemble, voice-health, voice-journal, breathing, nutrition,
    // brain (timeline/export), sessions, health, biometrics, gamification,
    // wearables, community, brain-report, emotion-coach, workplace-ei,
    // social-emotion, on-device, camera-rppg, and every other ML route.
    if (s0 === 'ml') {
      const ML_API_URL =
        process.env.VITE_ML_API_URL ||
        process.env.ML_API_URL ||
        'https://neural-dream-ml.onrender.com';
      // Strip leading /api/ml to get the downstream path
      const mlPath = (req.url || '').split('?')[0].replace(/^\/api\/ml/, '');
      const queryStr = new URLSearchParams(
        (req.query || {}) as Record<string, string>,
      ).toString();
      const targetUrl = `${ML_API_URL}${mlPath}${queryStr ? `?${queryStr}` : ''}`;

      try {
        const hasBody = req.method !== 'GET' && req.method !== 'HEAD';
        const mlRes = await fetch(targetUrl, {
          method: req.method || 'GET',
          headers: { 'Content-Type': 'application/json' },
          body: hasBody ? JSON.stringify(req.body) : undefined,
        });

        const contentType = mlRes.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
          const data = await mlRes.json();
          return res.status(mlRes.status).json(data);
        }
        const text = await mlRes.text();
        return res
          .status(mlRes.status)
          .setHeader('Content-Type', contentType || 'text/plain')
          .send(text);
      } catch (err) {
        return res
          .status(503)
          .json({ message: 'ML backend unavailable', error: String(err) });
      }
    }

    return error(res, 'Not found', 404);
  } catch (err: any) {
    console.error(`[api/[...path]] ${req.method} ${req.url}:`, err);
    return error(res, 'Internal server error');
  }
}

// ── Inner Score handlers ─────────────────────────────────────────────────────

async function innerScoreGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (!requireOwner(req, res, userId)) return;
  const db = getDb();
  const fourHoursAgo = new Date(Date.now() - 4 * 3600_000);
  const [cached] = await db.select().from(schema.innerScores)
    .where(and(eq(schema.innerScores.userId, userId), gte(schema.innerScores.createdAt, fourHoursAgo)))
    .orderBy(desc(schema.innerScores.createdAt)).limit(1);
  if (!cached) return success(res, { score: null, state: 'building', cta: 'Do a voice check-in to get your Inner Score' });
  const sevenDaysAgo = new Date(Date.now() - 7 * 86400_000);
  const history = await db.select().from(schema.innerScores)
    .where(and(eq(schema.innerScores.userId, userId), gte(schema.innerScores.createdAt, sevenDaysAgo)))
    .orderBy(asc(schema.innerScores.createdAt));
  const trend = history.map((h: { score: number }) => h.score);
  const yesterday = history.length >= 2 ? history[history.length - 2].score : null;
  const delta = yesterday != null ? cached.score - yesterday : null;
  const label = cached.score >= 80 ? 'Thriving' : cached.score >= 60 ? 'Good' : cached.score >= 40 ? 'Steady' : 'Low';
  const confidence = cached.tier === 'eeg_health_voice' ? 'Based on your brain, body, and mood' : cached.tier === 'health_voice' ? 'Based on your sleep, body, and mood' : 'Based on how you sound today';
  return success(res, { score: cached.score, label, tier: cached.tier, confidence, factors: cached.factors, narrative: cached.narrative, delta, trend });
}

const innerScorePostSchema = z.object({
  score: z.number().min(0).max(100),
  tier: z.enum(['eeg_health_voice', 'health_voice', 'voice_only']),
  factors: z.record(z.unknown()).optional(),
  narrative: z.string().max(1000).nullable().optional(),
});

async function innerScorePost(req: VercelRequest, res: VercelResponse, userId: string) {
  if (!requireOwner(req, res, userId)) return;
  const body = await parseRequestBody(req) as unknown;
  const parsed = innerScorePostSchema.safeParse(body);
  if (!parsed.success) return badRequest(res, parsed.error.issues[0]?.message ?? 'Invalid inner score data');
  const db = getDb();
  const [row] = await db.insert(schema.innerScores).values({ userId, score: parsed.data.score, tier: parsed.data.tier, factors: parsed.data.factors ?? {}, narrative: parsed.data.narrative ?? null }).returning();
  return success(res, row, 201);
}

async function innerScoreHistory(req: VercelRequest, res: VercelResponse, userId: string) {
  if (!requireOwner(req, res, userId)) return;
  const url = new URL(req.url!, `http://${req.headers.host}`);
  const days = Math.min(Math.max(parseInt(url.searchParams.get('days') ?? '30', 10), 1), 365);
  const since = new Date(Date.now() - days * 86400_000);
  const db = getDb();
  const rows = await db.select().from(schema.innerScores)
    .where(and(eq(schema.innerScores.userId, userId), gte(schema.innerScores.createdAt, since)))
    .orderBy(desc(schema.innerScores.createdAt));
  return success(res, { scores: rows.map((r: { createdAt: Date; score: number; tier: string }) => ({ date: r.createdAt.toISOString().slice(0, 10), score: r.score, tier: r.tier })) });
}

// ── Streak handlers ──────────────────────────────────────────────────────────

const STREAK_MILESTONES = [3, 7, 14, 30, 60, 90, 100];

function computeCurrentStreak(dates: string[], today: string): number {
  const sorted = [...new Set(dates)].sort().reverse();
  if (sorted.length === 0) return 0;
  let streak = 0;
  let expected = today;
  for (const d of sorted) {
    if (d === expected) {
      streak++;
      const prev = new Date(expected + "T00:00:00Z");
      prev.setUTCDate(prev.getUTCDate() - 1);
      expected = prev.toISOString().slice(0, 10);
    } else if (d < expected) {
      break;
    }
  }
  return streak;
}

async function streaksCheckin(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const authPayload = requireAuth(req, res);
  if (!authPayload) return;
  await parseRequestBody(req); // consume body (no fields required — uid from JWT)
  const uid = authPayload.userId;

  const today = new Date().toISOString().slice(0, 10);
  const db = getDb();
  const rl = await checkRateLimit(db, `streaks-checkin:${uid}`, 10, 60);
  if (!rl.allowed) return tooManyRequests(res, rl.retryAfterSeconds!);

  const [existing] = await db.select().from(schema.streaks)
    .where(eq(schema.streaks.userId, uid)).limit(1);

  if (existing) {
    const dates = (existing.checkinDates as string[]) || [];
    if (!dates.includes(today)) {
      dates.push(today);
    }
    const currentStreak = computeCurrentStreak(dates, today);
    const longestStreak = Math.max(existing.longestStreak, currentStreak);

    await db.update(schema.streaks)
      .set({
        checkinDates: dates,
        currentStreak,
        longestStreak,
        lastCheckinDate: today,
        updatedAt: new Date(),
      })
      .where(eq(schema.streaks.id, existing.id));

    const milestonesAchieved = STREAK_MILESTONES.filter(m => longestStreak >= m);
    const nextMilestone = STREAK_MILESTONES.find(m => m > currentStreak) ?? null;

    return success(res, {
      currentStreak,
      longestStreak,
      lastCheckinDate: today,
      milestones: milestonesAchieved,
      nextMilestone,
      todayCheckedIn: true,
    });
  } else {
    await db.insert(schema.streaks).values({
      userId: uid,
      currentStreak: 1,
      longestStreak: 1,
      lastCheckinDate: today,
      checkinDates: [today],
    });

    return success(res, {
      currentStreak: 1,
      longestStreak: 1,
      lastCheckinDate: today,
      milestones: [],
      nextMilestone: 3,
      todayCheckedIn: true,
    }, 201);
  }
}

async function streaksGet(req: VercelRequest, res: VercelResponse, userId: string) {
  if (!requireOwner(req, res, userId)) return;
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const today = new Date().toISOString().slice(0, 10);
  const db = getDb();

  const [existing] = await db.select().from(schema.streaks)
    .where(eq(schema.streaks.userId, userId)).limit(1);

  if (!existing) {
    return success(res, {
      currentStreak: 0,
      longestStreak: 0,
      lastCheckinDate: null,
      milestones: [],
      nextMilestone: 3,
      todayCheckedIn: false,
    });
  }

  const dates = (existing.checkinDates as string[]) || [];
  const currentStreak = computeCurrentStreak(dates, today);
  const longestStreak = Math.max(existing.longestStreak, currentStreak);
  const todayCheckedIn = dates.includes(today);
  const milestonesAchieved = STREAK_MILESTONES.filter(m => longestStreak >= m);
  const nextMilestone = STREAK_MILESTONES.find(m => m > currentStreak) ?? null;

  return success(res, {
    currentStreak,
    longestStreak,
    lastCheckinDate: existing.lastCheckinDate,
    milestones: milestonesAchieved,
    nextMilestone,
    todayCheckedIn,
  });
}
