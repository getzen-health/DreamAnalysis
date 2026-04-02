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
    if (!password || typeof password !== 'string' || password.length < 6)
      return badRequest(res, 'Password must be at least 6 characters');
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
    const [user] = await db.select().from(schema.users).where(eq(schema.users.username, username.trim()));
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
  const { dreamText, userId, tags, sleepQuality, sleepDuration } = req.body;
  if (!dreamText || !userId) return badRequest(res, 'dreamText and userId are required');
  const db = getDb();
  const openai = getOpenAIClient();
  const recentDreams = await db.select({ dreamText: schema.dreamAnalysis.dreamText, symbols: schema.dreamAnalysis.symbols })
    .from(schema.dreamAnalysis).where(eq(schema.dreamAnalysis.userId, userId))
    .orderBy(desc(schema.dreamAnalysis.timestamp)).limit(5);
  const historyCtx = recentDreams.length > 0
    ? `\n\nRecent dream themes: ${recentDreams.map(d => (d.symbols as string[] | null)?.join(', ') || 'unknown').join('; ')}`
    : '';
  const resp = await openai.chat.completions.create({
    model: 'llama-3.3-70b-versatile',
    messages: [
      { role: 'system', content: `You are an expert dream analyst combining Jungian, Freudian, and neuroscience perspectives. Respond with JSON: {"symbols":[],"emotions":[{"emotion":"","intensity":0}],"analysis":"","lucidityScore":1,"themes":[],"wakingLifeConnections":"","recurringPatterns":""}${historyCtx}` },
      { role: 'user', content: `Analyze this dream: ${dreamText}` },
    ],
    response_format: { type: 'json_object' },
  });
  const analysis = JSON.parse(resp.choices[0].message.content || '{}');
  const [entry] = await db.insert(schema.dreamAnalysis).values({
    userId, dreamText, symbols: analysis.symbols || [], emotions: analysis.emotions || [],
    aiAnalysis: analysis.analysis || '', lucidityScore: analysis.lucidityScore || null,
    sleepQuality: sleepQuality || null, sleepDuration: sleepDuration || null, tags: tags || [],
  }).returning();
  if (analysis.symbols) {
    for (const sym of analysis.symbols as string[]) {
      await db.insert(schema.dreamSymbols).values({ userId, symbol: sym, meaning: null, frequency: 1 }).onConflictDoNothing();
    }
  }
  return success(res, { ...entry, themes: analysis.themes, wakingLifeConnections: analysis.wakingLifeConnections }, 201);
}

async function dreamsList(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const userId = req.query.userId as string;
  if (!userId) return error(res, 'userId required', 400);
  const page = parseInt(req.query.page as string) || 1;
  const limit = parseInt(req.query.limit as string) || 20;
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
  const db = getDb();
  const [dreams, symbols] = await Promise.all([
    db.select().from(schema.dreamAnalysis).where(eq(schema.dreamAnalysis.userId, userId)).orderBy(desc(schema.dreamAnalysis.timestamp)),
    db.select().from(schema.dreamSymbols).where(eq(schema.dreamSymbols.userId, userId)),
  ]);
  const total = dreams.length;
  const tagCounts: Record<string, number> = {};
  const emotionCounts: Record<string, number> = {};
  dreams.forEach(d => {
    (d.tags as string[] | null)?.forEach(t => { tagCounts[t] = (tagCounts[t] || 0) + 1; });
    (d.emotions as Array<{ emotion: string }> | null)?.forEach(e => { emotionCounts[e.emotion] = (emotionCounts[e.emotion] || 0) + 1; });
  });
  return success(res, {
    totalDreams: total,
    avgSleepQuality: Math.round(dreams.reduce((s, d) => s + (d.sleepQuality || 0), 0) / Math.max(total, 1) * 10) / 10,
    avgLucidity: Math.round(dreams.reduce((s, d) => s + (d.lucidityScore || 0), 0) / Math.max(total, 1) * 10) / 10,
    tagDistribution: tagCounts, emotionDistribution: emotionCounts,
    topSymbols: symbols.sort((a, b) => (b.frequency || 0) - (a.frequency || 0)).slice(0, 10),
  });
}

async function dreamsGenerateImage(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { dreamId } = req.body;
  if (!dreamId) return badRequest(res, 'dreamId required');
  const db = getDb();
  const [dream] = await db.select().from(schema.dreamAnalysis).where(eq(schema.dreamAnalysis.id, dreamId));
  if (!dream) return error(res, 'Dream not found', 404);
  // Pollinations AI — free, no API key, returns a stable URL for the prompt
  const prompt = `Surreal dreamlike digital art: ${dream.dreamText.substring(0, 400)}. Style: ethereal, mystical, glowing colors, cosmic atmosphere. No text.`;
  const imageUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(prompt)}?width=1024&height=1024&nologo=true&seed=${Date.now()}`;
  await db.update(schema.dreamAnalysis).set({ imageUrl }).where(eq(schema.dreamAnalysis.id, dreamId));
  return success(res, { imageUrl });
}

async function dreamAnalysisPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { dreamText, userId } = req.body;
  if (!dreamText || !userId) return badRequest(res, 'Missing dreamText or userId');
  const openai = getOpenAIClient();
  const resp = await openai.chat.completions.create({
    model: 'llama-3.3-70b-versatile',
    messages: [
      { role: 'system', content: 'You are a dream analysis expert. Respond with JSON: {"symbols":[],"emotions":[{"emotion":"","intensity":0}],"analysis":""}' },
      { role: 'user', content: `Analyze this dream: ${dreamText}` },
    ],
    response_format: { type: 'json_object' },
  });
  const analysis = JSON.parse(resp.choices[0].message.content || '{}');
  const db = getDb();
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

  const openai = getOpenAIClient();
  const recentCtx = Array.isArray(recentThemes) && recentThemes.length > 0
    ? `\nRecent dream themes for continuity: ${recentThemes.join(', ')}`
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
    } catch {
      return error(res, 'Dream analysis failed', 500);
    }
  }
}

async function aiChatPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { message, userId } = req.body;
  if (!message || !userId) return badRequest(res, 'Missing message or userId');
  const db = getDb();
  await db.insert(schema.aiChats).values({ userId, message, isUser: true });
  const recentMetrics = await db.select().from(schema.healthMetrics)
    .where(eq(schema.healthMetrics.userId, userId)).orderBy(desc(schema.healthMetrics.timestamp)).limit(5);
  const ctx = recentMetrics.length > 0
    ? `Recent health data: HR ${recentMetrics[0].heartRate}, stress ${recentMetrics[0].stressLevel}, sleep quality ${recentMetrics[0].sleepQuality}.`
    : '';
  const openai = getOpenAIClient();
  const resp = await openai.chat.completions.create({
    model: 'llama-3.3-70b-versatile',
    messages: [
      { role: 'system', content: `You are an AI wellness companion for a Brain-Computer Interface system. ${ctx} Be supportive and concise.` },
      { role: 'user', content: message },
    ],
  });
  const aiMsg = resp.choices[0].message.content || "I'm here to help you with your wellness journey.";
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
  const { userId, stress, happiness, focus, energy, dominantEmotion, valence, arousal, eegSnapshot } = req.body;
  if (!userId || stress === undefined || happiness === undefined || focus === undefined || energy === undefined || !dominantEmotion)
    return badRequest(res, 'Missing required fields');
  const db = getDb();
  const [reading] = await db.insert(schema.emotionReadings).values({
    userId, stress, happiness, focus, energy, dominantEmotion,
    valence: valence ?? null, arousal: arousal ?? null, eegSnapshot: eegSnapshot ?? null,
  }).returning();
  return success(res, reading, 201);
}

async function emotionsCorrect(req: VercelRequest, res: VercelResponse, id: string) {
  if (req.method !== 'PATCH') return methodNotAllowed(res, ['PATCH']);
  const { userCorrectedEmotion, userId } = req.body;
  const validEmotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral', 'stress', 'focus', 'relaxed', 'excited'];
  if (!userCorrectedEmotion || !validEmotions.includes(userCorrectedEmotion))
    return badRequest(res, `userCorrectedEmotion must be one of: ${validEmotions.join(', ')}`);
  const db = getDb();
  const [existing] = await db.select({ id: schema.emotionReadings.id, userId: schema.emotionReadings.userId })
    .from(schema.emotionReadings).where(eq(schema.emotionReadings.id, id));
  if (!existing) return error(res, 'Reading not found', 404);
  if (userId && existing.userId !== userId) return unauthorized(res, 'Not your reading');
  const [updated] = await db.update(schema.emotionReadings)
    .set({ userCorrectedEmotion, userCorrectedAt: new Date() })
    .where(eq(schema.emotionReadings.id, id))
    .returning();
  return success(res, updated);
}

async function emotionsCorrectLatest(req: VercelRequest, res: VercelResponse, userId: string) {
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
  const limit = parseInt(req.query.limit as string) || 50;
  const db = getDb();
  const rows = await db.select().from(schema.emotionReadings)
    .where(eq(schema.emotionReadings.userId, userId)).orderBy(desc(schema.emotionReadings.timestamp)).limit(limit);
  return success(res, rows);
}

async function healthMetricsPost(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  try {
    const data = schema.insertHealthMetricsSchema.parse(req.body);
    const db = getDb();
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
    const rows = samples
      .filter((s) => typeof s.value === 'number' && !isNaN(s.value))
      .map((s) => ({
        userId: user_id,
        source: s.source,
        metric: s.metric,
        value: s.value,
        unit: s.unit ?? null,
        metadata: (s.metadata ?? null) as Record<string, unknown> | null,
        recordedAt: new Date(s.recorded_at),
      }));
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
  try {
    const db = getDb();
    const metric = (req.query.metric as string) || 'heart_rate';
    const days = parseInt((req.query.days as string) || '7', 10);
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
      .where(eq(schema.healthMetrics.userId, userId)).orderBy(asc(schema.healthMetrics.timestamp));
    const records = metrics.flatMap(m => {
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
      .where(eq(schema.dreamAnalysis.userId, userId)).orderBy(desc(schema.dreamAnalysis.timestamp));
    if (dreams.length === 0) { res.setHeader('Content-Type', 'text/csv'); return res.send('timestamp,dreamText,symbols,aiAnalysis,lucidityScore\nNo dreams recorded yet'); }
    const escape = (s: unknown) => `"${String(s ?? '').replace(/"/g, '""')}"`;
    const rows = dreams.map(d => [
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
      .where(eq(schema.emotionReadings.userId, userId)).orderBy(desc(schema.emotionReadings.timestamp));
    if (readings.length === 0) { res.setHeader('Content-Type', 'text/csv'); return res.send('No emotion data yet'); }
    const rows = readings.map(r => [
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
      db.select().from(schema.healthMetrics).where(eq(schema.healthMetrics.userId, userId)).orderBy(desc(schema.healthMetrics.timestamp)),
      db.select().from(schema.dreamAnalysis).where(eq(schema.dreamAnalysis.userId, userId)).orderBy(desc(schema.dreamAnalysis.timestamp)),
      db.select().from(schema.emotionReadings).where(eq(schema.emotionReadings.userId, userId)).orderBy(desc(schema.emotionReadings.timestamp)),
    ]);
    const escape = (s: unknown) => `"${String(s ?? '').replace(/"/g, '""')}"`;
    const sections = [
      '# HEALTH METRICS',
      ['timestamp,heartRate,stressLevel,sleepQuality,neuralActivity,dailySteps,sleepDuration',
        ...metrics.map(m => [m.timestamp, m.heartRate, m.stressLevel, m.sleepQuality, m.neuralActivity, m.dailySteps, m.sleepDuration].join(','))].join('\n'),
      '\n# DREAM ANALYSIS',
      ['timestamp,dreamText,symbols,aiAnalysis',
        ...dreams.map(d => [d.timestamp, escape(d.dreamText), escape((d.symbols as string[] | null)?.join('; ') ?? ''), escape(d.aiAnalysis ?? '')].join(','))].join('\n'),
      '\n# EMOTION READINGS',
      ['timestamp,dominantEmotion,stress,happiness,focus,energy,userCorrectedEmotion',
        ...readings.map(r => [r.timestamp, r.dominantEmotion, r.stress, r.happiness, r.focus, r.energy, r.userCorrectedEmotion ?? ''].join(','))].join('\n'),
    ];
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename=neural_dream_export_${new Date().toISOString().slice(0, 10)}.csv`);
    return res.send(sections.join('\n'));
  }

  // ── Health metrics CSV (default) ────────────────────────────────────────
  const metrics = await db.select().from(schema.healthMetrics)
    .where(eq(schema.healthMetrics.userId, userId)).orderBy(desc(schema.healthMetrics.timestamp));
  if (metrics.length === 0) { res.setHeader('Content-Type', 'text/csv'); return res.send('No data available'); }
  const rows = metrics.map(m => [m.timestamp, m.heartRate, m.stressLevel, m.sleepQuality, m.neuralActivity, m.dailySteps, m.sleepDuration].join(','));
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', `attachment; filename=health_metrics_${new Date().toISOString().slice(0, 10)}.csv`);
  return res.send(['timestamp,heartRate,stressLevel,sleepQuality,neuralActivity,dailySteps,sleepDuration', ...rows].join('\n'));
}

async function insightsWeekly(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const userId = req.query.userId as string;
  if (!userId) return error(res, 'userId required', 400);
  const db = getDb();
  const [dreams, emotions, metrics] = await Promise.all([
    db.select().from(schema.dreamAnalysis).where(eq(schema.dreamAnalysis.userId, userId)).orderBy(desc(schema.dreamAnalysis.timestamp)).limit(10),
    db.select().from(schema.emotionReadings).where(eq(schema.emotionReadings.userId, userId)).orderBy(desc(schema.emotionReadings.timestamp)).limit(50),
    db.select().from(schema.healthMetrics).where(eq(schema.healthMetrics.userId, userId)).orderBy(desc(schema.healthMetrics.timestamp)).limit(50),
  ]);
  const openai = getOpenAIClient();
  const ctx = {
    dreamCount: dreams.length,
    dreamSymbols: dreams.flatMap(d => (d.symbols as string[]) || []),
    avgStress: emotions.length ? emotions.reduce((s, e) => s + e.stress, 0) / emotions.length : null,
    avgFocus: emotions.length ? emotions.reduce((s, e) => s + e.focus, 0) / emotions.length : null,
    avgSleepQuality: metrics.length ? metrics.reduce((s, m) => s + m.sleepQuality, 0) / metrics.length : null,
    dominantEmotions: emotions.slice(0, 10).map(e => e.dominantEmotion),
  };
  const resp = await openai.chat.completions.create({
    model: 'llama-3.3-70b-versatile',
    messages: [
      { role: 'system', content: 'You are an AI neuroscience wellness advisor. Generate 4 personalized weekly insights. Respond with JSON: {"insights":[{"title":"","description":"","type":"success|warning|info|secondary","icon":"brain|heart|moon|lightbulb"}],"weeklyScore":0,"recommendation":""}' },
      { role: 'user', content: `Generate weekly insights for: ${JSON.stringify(ctx)}` },
    ],
    response_format: { type: 'json_object' },
  });
  return success(res, JSON.parse(resp.choices[0].message.content || '{}'));
}

async function notificationsSubscribe(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { userId, endpoint, keys } = req.body;
  if (!userId || !endpoint || !keys) return badRequest(res, 'userId, endpoint, and keys are required');
  const db = getDb();
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
  const vapidPublic = process.env.VAPID_PUBLIC_KEY;
  const vapidPrivate = process.env.VAPID_PRIVATE_KEY;
  if (!vapidPublic || !vapidPrivate) return error(res, 'VAPID keys not configured', 503);

  const { userId, title = 'AntarAI', body = 'Good morning! Your brain report is ready.', url = '/brain-report' } = req.body;

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

  // Fetch subscriptions — optionally scoped to one user
  const subs = userId
    ? await db.select().from(schema.pushSubscriptions).where(eq(schema.pushSubscriptions.userId, userId))
    : await db.select().from(schema.pushSubscriptions);

  if (subs.length === 0) return success(res, { sent: 0, message: 'No subscriptions found' });

  const payload = JSON.stringify({ title, body, url });
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
  const { text } = req.body;
  if (!text) return badRequest(res, 'Missing text to analyze');
  const openai = getOpenAIClient();
  const resp = await openai.chat.completions.create({
    model: 'llama-3.3-70b-versatile',
    messages: [
      { role: 'system', content: 'Analyze the mood from text. Respond with JSON: {"mood":"","stressLevel":0,"emotions":[],"recommendations":[]}' },
      { role: 'user', content: text },
    ],
    response_format: { type: 'json_object' },
  });
  return success(res, JSON.parse(resp.choices[0].message.content || '{}'));
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
  const body = req.body ?? {};
  const { userId, imageBase64, textDescription, mealType, moodBefore, notes } = body;
  if (!userId) return badRequest(res, 'userId required');
  if (!textDescription && !imageBase64) return badRequest(res, 'Provide a photo or describe what you ate');

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
  // No auth gate — data is scoped by userId (same pattern as brain/history)
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

async function emotionReadingsBatch(req: VercelRequest, res: VercelResponse) {
  const body = await parseRequestBody(req) as any;
  const readings = body?.readings;
  if (!Array.isArray(readings) || readings.length === 0) return badRequest(res, 'readings array required');
  const db = getDb();
  for (const r of readings.slice(0, 50)) {
    await db.insert(schema.emotionReadings).values({
      userId: r.userId,
      sessionId: r.sessionId ?? null,
      stress: r.stress ?? 0,
      happiness: r.happiness ?? 0,
      focus: r.focus ?? 0,
      energy: r.energy ?? 0,
      dominantEmotion: r.dominantEmotion ?? 'neutral',
      valence: r.valence ?? null,
      arousal: r.arousal ?? null,
    }).catch(() => {});
  }
  return success(res, { saved: readings.length }, 201);
}

async function userReadingsPost(req: VercelRequest, res: VercelResponse) {
  const body = await parseRequestBody(req) as any;
  if (!body?.userId) return badRequest(res, 'userId required');
  const db = getDb();
  const [row] = await db.insert(schema.userReadings).values({
    userId: body.userId,
    source: body.source ?? 'voice',
    emotion: body.emotion ?? 'neutral',
    valence: body.valence ?? null,
    arousal: body.arousal ?? null,
    stress: body.stress ?? null,
    confidence: body.confidence ?? null,
    modelType: body.modelType ?? 'voice',
  }).returning();
  return success(res, row, 201);
}

async function foodLog(req: VercelRequest, res: VercelResponse) {
  const body = await parseRequestBody(req) as any;
  if (!body?.userId) return badRequest(res, 'userId required');
  const db = getDb();
  const [row] = await db.insert(schema.foodLogs).values({
    userId: body.userId,
    mealType: body.mealType ?? 'meal',
    summary: body.summary ?? null,
    totalCalories: body.totalCalories ?? null,
    dominantMacro: body.dominantMacro ?? null,
    foodItems: body.foodItems ?? null,
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
    consentFullName: consentFullName ?? null,
    consentInitials: consentInitials ?? null,
    overnightEegConsent: overnightEegConsent ?? false,
    preferredMorningTime, preferredDaytimeTime, preferredEveningTime,
  }).returning();
  return success(res, { studyCode: participant.studyCode, enrolledAt: participant.enrolledAt }, 201);
}

async function studyStatus(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
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
  const participant = await getActiveParticipant(userId);
  if (!participant) return error(res, 'Not enrolled in an active study', 404);
  const session = await getOrCreateTodaySession(participant);
  if (session.morningCompleted) return res.status(409).json({ message: 'Morning entry already submitted today' });
  const db = getDb();
  await db.insert(schema.studyMorningEntries).values({
    sessionId: session.id,
    studyCode: participant.studyCode,
    dreamText: noRecall ? null : (dreamText ?? null),
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
  if (!participant_code) return badRequest(res, 'participant_code is required');
  const db = getDb();
  await db.insert(schema.pilotParticipants).values({
    participantCode:  String(participant_code),
    age:              age != null ? Number(age) : null,
    dietType:         diet_type ?? null,
    hasAppleWatch:    has_apple_watch ? true : false,
    consentText:      consent_text ?? null,
    consentTimestamp: new Date(),
  }).onConflictDoNothing();
  return success(res, { success: true, participant_code });
}

async function pilotSessionStart(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { participant_code, block_type } = req.body;
  if (!participant_code || !block_type) return badRequest(res, 'participant_code and block_type are required');
  const db = getDb();
  const [row] = await db.insert(schema.pilotSessions).values({
    participantCode:       String(participant_code),
    blockType:             String(block_type),
    interventionTriggered: false,
  }).returning({ id: schema.pilotSessions.id });
  return success(res, { session_id: row.id });
}

async function pilotSessionComplete(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { session_id, pre_eeg_json, post_eeg_json, eeg_features_json, survey_json, intervention_triggered } = req.body;
  if (session_id == null) return badRequest(res, 'session_id is required');
  const db = getDb();
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
  const db = getDb();
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

    const db = getDb();
    const [row] = await db.insert(schema.userReadings).values({
      userId: userId ?? null,
      source,
      emotion: emotion ?? null,
      valence: typeof valence === 'number' ? valence : null,
      arousal: typeof arousal === 'number' ? arousal : null,
      stress: typeof stress === 'number' ? stress : null,
      confidence: typeof confidence === 'number' ? confidence : null,
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
    const limit = Math.min(parseInt(req.query.limit as string) || 1000, 5000);

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

    if (s0 === 'ping' || s0 === 'health') return res.status(200).json({ ok: true, ts: Date.now() });

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

    if (s0 === 'settings' && s1) return await settingsHandler(req, res, s1);
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
      if (s1 === 'intent' && req.method === 'GET')   return await userIntentGet(req, res);
      if (s1 === 'intent' && req.method === 'PATCH')  return await userIntentPatch(req, res);
    }

    if (s0 === 'readings') {
      if (!s1 && req.method === 'POST') return await readingsCreate(req, res);
      if (s1 && req.method === 'GET')   return await readingsList(req, res, s1);
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

async function innerScorePost(req: VercelRequest, res: VercelResponse, userId: string) {
  const body = await parseRequestBody(req) as any;
  if (body?.score == null || !body?.tier) return badRequest(res, 'score and tier required');
  const db = getDb();
  const [row] = await db.insert(schema.innerScores).values({ userId, score: body.score, tier: body.tier, factors: body.factors ?? {}, narrative: body.narrative ?? null }).returning();
  return success(res, row, 201);
}

async function innerScoreHistory(req: VercelRequest, res: VercelResponse, userId: string) {
  const url = new URL(req.url!, `http://${req.headers.host}`);
  const days = parseInt(url.searchParams.get('days') ?? '30');
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
  const body = await parseRequestBody(req) as any;
  const uid = authPayload.userId;

  const today = new Date().toISOString().slice(0, 10);
  const db = getDb();

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
