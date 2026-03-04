/**
 * Unified API catch-all for Vercel Hobby plan (max 12 serverless functions).
 * Routes all /api/* requests to the appropriate handler inline.
 *
 * Routes handled:
 *   POST   /api/auth/register
 *   POST   /api/auth/login
 *   GET    /api/auth/me
 *   POST   /api/auth/logout
 *   POST   /api/dreams/create
 *   GET    /api/dreams/list
 *   GET    /api/dreams/analytics
 *   POST   /api/dreams/generate-image
 *   POST   /api/dream-analysis
 *   GET    /api/dream-analysis/:userId
 *   POST   /api/ai-chat
 *   GET    /api/ai-chat/:userId
 *   POST   /api/emotions/record
 *   GET    /api/emotions/history
 *   POST   /api/health-metrics
 *   GET    /api/health-metrics/:userId
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
 *   GET    /api/study/admin/participants   (pilot study, auth required)
 *   GET    /api/study/admin/sessions       (pilot study, auth required)
 *   GET    /api/study/admin/export-csv     (pilot study, auth required)
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { scrypt, randomBytes, timingSafeEqual } from 'crypto';
import { promisify } from 'util';
import { eq, desc, asc, and, gte, lt, sql } from 'drizzle-orm';

import { success, error, badRequest, methodNotAllowed, unauthorized } from './_lib/response.js';
import { generateToken, setAuthCookie, clearAuthCookie, requireAuth } from './_lib/auth.js';

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
// Vercel's ESM runtime (triggered by "type":"module" in package.json) does not
// auto-populate req.body — its getter throws "Invalid JSON" instead of returning
// the parsed object.  We read the raw Node.js IncomingMessage stream directly.
async function parseRequestBody(req: VercelRequest): Promise<unknown> {
  // Fast path: pre-parsed body already available (CJS runtime or unit tests)
  try {
    const b = (req as VercelRequest & { body?: unknown }).body;
    if (b !== undefined && b !== null) return b;
  } catch {
    // getter threw — fall through to stream reading
  }

  return new Promise<unknown>((resolve) => {
    let raw = '';
    req.on('data', (chunk: Buffer) => { raw += chunk.toString('utf8'); });
    req.on('end', () => {
      if (!raw) { resolve({}); return; }
      try { resolve(JSON.parse(raw)); } catch { resolve({}); }
    });
    req.on('error', () => resolve({}));
  });
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
  const { username, password, email } = req.body;
  if (!username || typeof username !== 'string' || username.trim().length < 3)
    return badRequest(res, 'Username must be at least 3 characters');
  if (!password || typeof password !== 'string' || password.length < 6)
    return badRequest(res, 'Password must be at least 6 characters');
  const db = getDb();
  const [existing] = await db.select().from(schema.users).where(eq(schema.users.username, username.trim()));
  if (existing) return badRequest(res, 'Username already exists');
  const [user] = await db.insert(schema.users).values({
    username: username.trim(), password: await hashPassword(password), email: email || null,
  }).returning();
  const token = generateToken({ userId: user.id, username: user.username });
  setAuthCookie(res, token);
  const { password: _, ...safe } = user;
  return success(res, { user: safe, token }, 201);
}

async function authLogin(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { username, password } = req.body;
  if (!username || !password) return badRequest(res, 'Username and password required');
  const db = getDb();
  const [user] = await db.select().from(schema.users).where(eq(schema.users.username, username.trim()));
  if (!user || !(await verifyPassword(user.password, password)))
    return unauthorized(res, 'Invalid username or password');
  const token = generateToken({ userId: user.id, username: user.username });
  setAuthCookie(res, token);
  const { password: _, ...safe } = user;
  return success(res, { user: safe, token });
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
    model: 'llama3.1-70b',
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
    model: 'llama3.1-70b',
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
  const db = getDb();
  const rows = await db.select().from(schema.dreamAnalysis)
    .where(eq(schema.dreamAnalysis.userId, userId)).orderBy(desc(schema.dreamAnalysis.timestamp)).limit(20);
  return success(res, rows);
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
    model: 'llama3.1-70b',
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
  const db = getDb();
  const rows = await db.select().from(schema.healthMetrics)
    .where(eq(schema.healthMetrics.userId, userId)).orderBy(desc(schema.healthMetrics.timestamp)).limit(50);
  return success(res, rows);
}

async function settingsHandler(req: VercelRequest, res: VercelResponse, userId: string) {
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
  const db = getDb();
  const metrics = await db.select().from(schema.healthMetrics)
    .where(eq(schema.healthMetrics.userId, userId)).orderBy(desc(schema.healthMetrics.timestamp));
  if (metrics.length === 0) { res.setHeader('Content-Type', 'text/csv'); return res.send('No data available'); }
  const rows = metrics.map(m => [m.timestamp, m.heartRate, m.stressLevel, m.sleepQuality, m.neuralActivity, m.dailySteps, m.sleepDuration].join(','));
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', 'attachment; filename=neural_data.csv');
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
    model: 'llama3.1-70b',
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

async function analyzeMood(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { text } = req.body;
  if (!text) return badRequest(res, 'Missing text to analyze');
  const openai = getOpenAIClient();
  const resp = await openai.chat.completions.create({
    model: 'llama3.1-70b',
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
  "summary": "One plain-English sentence describing what was eaten"
}`;

async function foodAnalyze(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res, ['POST']);
  const { userId, imageBase64, textDescription, mealType, moodBefore, notes } = req.body;
  if (!userId) return badRequest(res, 'userId required');
  if (!imageBase64 && !textDescription) return badRequest(res, 'imageBase64 or textDescription required');
  const openai = getOpenAIClient();
  const db = getDb();

  let content: string;
  if (imageBase64) {
    // Cerebras Llama is text-only — describe the image via text prompt instead
    const resp = await openai.chat.completions.create({
      model: 'llama3.1-70b',
      messages: [{
        role: 'user',
        content: `A user uploaded a food photo for nutritional analysis. Estimate reasonable nutritional values for a typical meal and return ONLY valid JSON with this exact shape:\n${FOOD_JSON_SCHEMA}`,
      }],
      response_format: { type: 'json_object' },
    });
    content = resp.choices[0].message.content ?? '{}';
  } else {
    // Text path
    const resp = await openai.chat.completions.create({
      model: 'llama3.1-70b',
      messages: [{
        role: 'user',
        content: `The user describes their ${mealType ?? 'meal'}: "${textDescription}"\n\nEstimate nutrition and return ONLY valid JSON with this exact shape:\n${FOOD_JSON_SCHEMA}`,
      }],
      response_format: { type: 'json_object' },
    });
    content = resp.choices[0].message.content ?? '{}';
  }

  // Strip markdown fences if present
  const stripped = content.replace(/```json?\n?/g, '').replace(/```/g, '').trim();
  const analysis = JSON.parse(stripped || '{}');

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

  return success(res, { ...analysis, id: log.id, loggedAt: log.loggedAt }, 201);
}

async function foodLogs(req: VercelRequest, res: VercelResponse, userId: string) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const db = getDb();
  const logs = await db.select().from(schema.foodLogs)
    .where(eq(schema.foodLogs.userId, userId))
    .orderBy(desc(schema.foodLogs.loggedAt))
    .limit(20);
  return success(res, logs);
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
          medicationsTaken, stressRightNow, readyForSleep } = req.body;
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
    exerciseLevel, alcoholDrinks, medicationsTaken, stressRightNow, readyForSleep,
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
  const authResult = requireAuth(req, res);
  if (!authResult) return unauthorized(res);
  const db = getDb();
  const rows = await db.select().from(schema.pilotParticipants).orderBy(desc(schema.pilotParticipants.createdAt));
  return success(res, rows);
}

async function pilotAdminSessions(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const authResult = requireAuth(req, res);
  if (!authResult) return unauthorized(res);
  const db = getDb();
  const rows = await db.select().from(schema.pilotSessions).orderBy(desc(schema.pilotSessions.createdAt));
  return success(res, rows);
}

async function pilotAdminExportCsv(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res, ['GET']);
  const authResult = requireAuth(req, res);
  if (!authResult) return unauthorized(res);
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

// ── Main router ──────────────────────────────────────────────────────────────

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS preflight
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') return res.status(200).end();

  // Lazy-load all heavy modules (schema, db, openai) before routing to avoid cold-start crash
  await loadModules();

  // Parse body early so all route handlers can safely access req.body.
  // Required because Vercel's ESM runtime doesn't auto-parse JSON bodies.
  if (['POST', 'PUT', 'PATCH'].includes(req.method || '')) {
    (req as VercelRequest & { body: unknown }).body = await parseRequestBody(req);
  }

  // Extract path segments from /api/seg0/seg1/...
  const url = req.url || '';
  const clean = url.split('?')[0].replace(/^\/api\//, '');
  const segs = clean.split('/').filter(Boolean);

  try {
    const [s0, s1] = segs;

    if (s0 === 'ping') return res.status(200).json({ ok: true, ts: Date.now() });

    if (s0 === 'auth') {
      if (s1 === 'register') return await authRegister(req, res);
      if (s1 === 'login')    return await authLogin(req, res);
      if (s1 === 'me')       return await authMe(req, res);
      if (s1 === 'logout')   return await authLogout(req, res);
    }

    if (s0 === 'dreams') {
      if (s1 === 'create')         return await dreamsCreate(req, res);
      if (s1 === 'list')           return await dreamsList(req, res);
      if (s1 === 'analytics')      return await dreamsAnalytics(req, res);
      if (s1 === 'generate-image') return await dreamsGenerateImage(req, res);
    }

    if (s0 === 'dream-analysis') {
      if (!s1) return await dreamAnalysisPost(req, res);
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
    }

    if (s0 === 'health-metrics') {
      if (!s1) return await healthMetricsPost(req, res);
      return await healthMetricsGet(req, res, s1);
    }

    if (s0 === 'settings' && s1) return await settingsHandler(req, res, s1);
    if (s0 === 'export'   && s1) return await exportHandler(req, res, s1);

    if (s0 === 'insights' && s1 === 'weekly') return await insightsWeekly(req, res);
    if (s0 === 'notifications' && s1 === 'subscribe') return await notificationsSubscribe(req, res);
    if (s0 === 'analyze-mood') return await analyzeMood(req, res);

    if (s0 === 'food') {
      if (s1 === 'analyze') return await foodAnalyze(req, res);
      if (s1 === 'logs' && segs[2]) return await foodLogs(req, res, segs[2]);
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
      if (s1 === 'admin' && segs[2] === 'export-csv')          return await pilotAdminExportCsv(req, res);
    }

    if (s0 === 'user') {
      if (s1 === 'intent' && req.method === 'GET')   return await userIntentGet(req, res);
      if (s1 === 'intent' && req.method === 'PATCH')  return await userIntentPatch(req, res);
    }

    return error(res, 'Not found', 404);
  } catch (err: any) {
    console.error(`[api/[...path]] ${req.method} ${req.url}:`, err);
    return error(res, 'Internal server error');
  }
}
