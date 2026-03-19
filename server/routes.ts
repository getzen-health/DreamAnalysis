import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { logger } from "./logger";
import OpenAI from "openai";
import webpush from "web-push";
import cron from "node-cron";
import bcrypt from "bcryptjs";
import crypto from "crypto";
import jwt from "jsonwebtoken";
import rateLimit from "express-rate-limit";

const JWT_SECRET = process.env.SESSION_SECRET || (() => {
  if (process.env.NODE_ENV === "production") throw new Error("SESSION_SECRET must be set in production");
  return "neural-dream-dev-only-secret";
})();
import session from "express-session";
import connectPg from "connect-pg-simple";
import pg from "pg";
const { Pool: PgPool } = pg;
import { createClient as createSupabaseClient } from "@supabase/supabase-js";
import SpotifyWebApi from "spotify-web-api-node";
import nodemailer from "nodemailer";
import {
  insertHealthMetricsSchema,
  insertUserSettingsSchema, insertEmotionReadingSchema,
  studyParticipants, studySessions, studyMorningEntries,
  studyDaytimeEntries, studyEveningEntries, foodLogs,
  users, pushSubscriptions,
  pilotParticipants, pilotSessions,
  passwordResetTokens,
  healthMetrics, healthSamples,
  dreamAnalysis, dreamSymbols, eegSessions, userSettings,
  aiChats, brainReadings, emotionReadings,
  exercises, workouts, workoutSets, workoutTemplates,
  bodyMetrics, exerciseHistory,
  habits, habitLogs, cycleTracking, moodLogs,
  deviceConnections,
  circadianProfiles,
} from "@shared/schema";
import { wearableAdapters } from "../lib/wearables";
import { computeCardioLoad } from "@shared/cardio";
import { db } from "./db";
import { eq, gte, lt, lte, and, or, asc, desc, sql, ilike, arrayContains } from "drizzle-orm";

// ── VAPID setup (web push) ─────────────────────────────────────────────────
const VAPID_PUBLIC  = process.env.VAPID_PUBLIC_KEY  ?? "";
const VAPID_PRIVATE = process.env.VAPID_PRIVATE_KEY ?? "";
const VAPID_EMAIL   = process.env.VAPID_EMAIL ?? "mailto:admin@example.com";

if (VAPID_PUBLIC && VAPID_PRIVATE) {
  webpush.setVapidDetails(VAPID_EMAIL, VAPID_PUBLIC, VAPID_PRIVATE);
}

const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  : null;

// ── Auth helpers (Express) ────────────────────────────────────────────────

const ADMIN_USERNAMES = new Set(["sravya", "admin"]);

// Supabase Admin client for server-side auth verification
const supabaseAdmin = (process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY)
  ? createSupabaseClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY)
  : null;

/** Extract the authenticated user's ID from session, legacy JWT, or Supabase JWT. */
function getAuthUserId(req: import("express").Request): string | null {
  // 1. Express session (legacy)
  const sessionUserId = (req.session as any)?.userId;
  if (sessionUserId) return sessionUserId;

  const auth = req.headers.authorization;
  if (auth?.startsWith("Bearer ")) {
    const token = auth.slice(7);
    // 2. Legacy JWT (SESSION_SECRET signed)
    try {
      const decoded = jwt.verify(token, JWT_SECRET) as { userId: string };
      return decoded.userId;
    } catch { /* not a legacy JWT — try Supabase */ }

    // 3. Supabase JWT — extract user ID from token payload
    if (token.startsWith("sb_") || token.startsWith("eyJ")) {
      try {
        // Decode JWT payload without verification (Supabase handles verification via RLS)
        const payload = JSON.parse(Buffer.from(token.split('.')[1], 'base64').toString());
        if (payload.sub) return payload.sub;
      } catch { /* not a valid JWT */ }
    }
  }
  return null;
}

/** Require auth + verify the authenticated user matches the :userId param. */
function requireOwnerExpress(
  req: import("express").Request,
  res: import("express").Response,
): string | null {
  const authUserId = getAuthUserId(req);
  if (!authUserId) {
    res.status(401).json({ error: "Unauthorized" });
    return null;
  }
  const requestedUserId = req.params.userId;
  if (authUserId !== requestedUserId) {
    res.status(403).json({ error: "Forbidden — you can only access your own data" });
    return null;
  }
  return authUserId;
}

// ── Research module helpers ────────────────────────────────────────────────

async function getActiveParticipant(userId: string) {
  const [p] = await db.select().from(studyParticipants)
    .where(and(
      eq(studyParticipants.userId, userId),
      eq(studyParticipants.status, "active")
    )).limit(1);
  return p ?? null;
}

async function getOrCreateTodaySession(participant: typeof studyParticipants.$inferSelect) {
  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);
  const tomorrowStart = new Date(todayStart);
  tomorrowStart.setDate(tomorrowStart.getDate() + 1);

  const [existing] = await db.select().from(studySessions)
    .where(and(
      eq(studySessions.participantId, participant.id),
      gte(studySessions.sessionDate, todayStart),
      lt(studySessions.sessionDate, tomorrowStart)
    )).limit(1);

  if (existing) return existing;

  // Count all sessions so far → day number (handles skipped/invalid days correctly)
  const [{ n }] = await db.select({ n: sql<number>`count(*)::int` })
    .from(studySessions)
    .where(eq(studySessions.participantId, participant.id));
  const dayNumber = (n ?? 0) + 1;

  const [created] = await db.insert(studySessions).values({
    participantId: participant.id,
    studyCode: participant.studyCode,
    dayNumber,
    sessionDate: todayStart,
  }).returning();
  return created;
}

async function checkAndMarkValidDay(sessionId: string): Promise<boolean> {
  const [session] = await db.select().from(studySessions)
    .where(eq(studySessions.id, sessionId));
  if (!session) return false;
  const done = [session.morningCompleted, session.daytimeCompleted, session.eveningCompleted]
    .filter(Boolean).length;
  const isValid = done >= 2;
  if (isValid && !session.validDay) {
    await db.update(studySessions).set({ validDay: true }).where(eq(studySessions.id, sessionId));
  }
  return isValid;
}

// ── Rate limiters ─────────────────────────────────────────────────────────

/** General API limiter: 100 req/min per IP (applied to all /api/* routes). */
const generalApiLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 100,
  standardHeaders: "draft-7",
  legacyHeaders: false,
  message: { error: "Too many requests. Please slow down and try again in a minute." },
});

/** Login: 5 attempts per 15 minutes per IP. */
const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  standardHeaders: "draft-7",
  legacyHeaders: false,
  message: { error: "Too many login attempts. Please try again in 15 minutes." },
});

/** Register: 3 attempts per hour per IP. */
const registerLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: 3,
  standardHeaders: "draft-7",
  legacyHeaders: false,
  message: { error: "Too many registration attempts. Please try again in an hour." },
});

/** Forgot-password: 3 requests per hour per IP. */
const forgotPasswordLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: 3,
  standardHeaders: "draft-7",
  legacyHeaders: false,
  message: { error: "Too many password reset requests. Please try again in an hour." },
});

/** Reset-password confirm: 5 attempts per 15 minutes per IP. */
const resetPasswordLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  standardHeaders: "draft-7",
  legacyHeaders: false,
  message: { error: "Too many password reset attempts. Please try again in 15 minutes." },
});

/** LLM endpoints: 10 requests per user per minute, keyed by session userId or IP. */
const llmLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 10,
  standardHeaders: "draft-7",
  legacyHeaders: false,
  keyGenerator: (req) => {
    const userId = getAuthUserId(req);
    if (userId) return userId;
    return req.ip ?? "unknown";
  },
  message: { error: "Too many AI requests. Please wait a moment." },
});

export async function registerRoutes(app: Express): Promise<Server> {
  // ── General API rate limit (must be first, before route handlers) ─────────
  app.use("/api", generalApiLimiter);

  // ── Session middleware (PostgreSQL-backed — persists across Vercel instances) ──
  const PgSession = connectPg(session);
  const useMemorySessionStore = process.env.NODE_ENV === "development";
  const sessionPool = useMemorySessionStore
    ? null
    : new PgPool({ connectionString: process.env.DATABASE_URL!, ssl: { rejectUnauthorized: false } });

  const store = useMemorySessionStore
    ? new session.MemoryStore()
    : new PgSession({
        pool: sessionPool as any,
        createTableIfMissing: true,
        tableName: "user_sessions",
      });

  app.use(session({
    store,
    secret: process.env.SESSION_SECRET ?? (() => {
      if (process.env.NODE_ENV === "production") throw new Error("SESSION_SECRET must be set in production");
      return "svapnastra-dev-only-secret";
    })(),
    resave: false,
    saveUninitialized: false,
    cookie: {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: process.env.NODE_ENV === "production" ? "none" : "lax",
      maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
    },
  }));

  // ── Auth routes ───────────────────────────────────────────────────────────

  // POST /api/auth/register
  app.post("/api/auth/register", registerLimiter, async (req, res) => {
    try {
      const { username, password, email, age, deviceType } = req.body;
      if (!username || typeof username !== "string" || username.trim().length < 3)
        return res.status(400).json({ error: "Username must be at least 3 characters." });
      if (!password || typeof password !== "string" || password.length < 6)
        return res.status(400).json({ error: "Password must be at least 6 characters." });

      const existing = await db.select().from(users)
        .where(eq(users.username, username.trim().toLowerCase())).limit(1);
      if (existing.length > 0)
        return res.status(409).json({ error: "Username already taken." });

      if (email && typeof email === "string" && email.trim()) {
        const existingEmail = await db.select().from(users)
          .where(eq(users.email, email.trim().toLowerCase())).limit(1);
        if (existingEmail.length > 0)
          return res.status(409).json({ error: "An account with this email already exists." });
      }

      const hashed = await bcrypt.hash(password, 12);
      const [newUser] = await db.insert(users).values({
        username: username.trim().toLowerCase(),
        password: hashed,
        email: email?.trim().toLowerCase() || null,
        age: age ? Number(age) : null,
        deviceType: deviceType || null,
      }).returning({ id: users.id, username: users.username, email: users.email,
                     age: users.age, deviceType: users.deviceType, createdAt: users.createdAt });

      (req.session as any).userId = newUser.id;
      await new Promise<void>((resolve, reject) =>
        req.session.save(err => (err ? reject(err) : resolve()))
      );
      const token = jwt.sign({ userId: newUser.id }, JWT_SECRET, { expiresIn: "30d" });
      return res.status(201).json({ user: newUser, token });
    } catch (err: any) {
      logger.error({ error: err instanceof Error ? err.message : String(err) }, "Register failed");
      return res.status(500).json({ error: "Registration failed." });
    }
  });

  // POST /api/auth/login
  app.post("/api/auth/login", loginLimiter, async (req, res) => {
    try {
      const { username, password } = req.body;
      if (!username || !password)
        return res.status(400).json({ error: "Username and password are required." });

      const [user] = await db.select().from(users)
        .where(eq(users.username, username.trim().toLowerCase())).limit(1);
      if (!user)
        return res.status(401).json({ error: "Invalid username or password." });

      const match = await bcrypt.compare(password, user.password);
      if (!match)
        return res.status(401).json({ error: "Invalid username or password." });

      (req.session as any).userId = user.id;
      await new Promise<void>((resolve, reject) =>
        req.session.save(err => (err ? reject(err) : resolve()))
      );
      const { password: _pw, ...safeUser } = user;
      const token = jwt.sign({ userId: user.id }, JWT_SECRET, { expiresIn: "30d" });
      return res.json({ user: safeUser, token });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err) }, "Login failed");
      return res.status(500).json({ error: "Login failed." });
    }
  });

  // GET /api/auth/me
  app.get("/api/auth/me", async (req, res) => {
    let userId = (req.session as any).userId;
    // Fall back to JWT token for native apps (no cookies)
    if (!userId) {
      const auth = req.headers.authorization;
      if (auth?.startsWith("Bearer ")) {
        try {
          const decoded = jwt.verify(auth.slice(7), JWT_SECRET) as { userId: string };
          userId = decoded.userId;
        } catch { /* invalid token */ }
      }
    }
    if (!userId) return res.status(401).json(null);
    try {
      const [user] = await db.select().from(users)
        .where(eq(users.id, userId)).limit(1);
      if (!user) return res.status(401).json(null);
      const { password: _pw, ...safeUser } = user;
      return res.json(safeUser);
    } catch {
      return res.status(401).json(null);
    }
  });

  // POST /api/auth/logout
  app.post("/api/auth/logout", (req, res) => {
    req.session.destroy(() => {
      res.clearCookie("connect.sid");
      res.json({ ok: true });
    });
  });

  // POST /api/auth/forgot-password
  app.post("/api/auth/forgot-password", forgotPasswordLimiter, async (req, res) => {
    try {
      const { email } = req.body;
      if (!email) return res.json({ message: "If that email exists, a reset link was sent" });

      const [user] = await db.select().from(users)
        .where(eq(users.email, email.trim().toLowerCase())).limit(1);

      if (user) {
        const token = crypto.randomBytes(32).toString("hex");
        const expiresAt = new Date(Date.now() + 60 * 60 * 1000); // 1 hour
        await db.insert(passwordResetTokens).values({
          userId: user.id,
          token,
          expiresAt,
        });

        const appUrl = process.env.APP_URL ?? "https://dream-analysis.vercel.app";
        const resetUrl = `${appUrl}/reset-password?token=${token}`;
        if (process.env.GMAIL_APP_PASSWORD) {
          const transporter = nodemailer.createTransport({
            service: "gmail",
            auth: {
              user: process.env.GMAIL_USER ?? "lakshmisravya.vedantham@gmail.com",
              pass: process.env.GMAIL_APP_PASSWORD,
            },
          });

          try {
            await transporter.sendMail({
              from: `Neural Dream <${process.env.GMAIL_USER ?? "lakshmisravya.vedantham@gmail.com"}>`,
              to: email.trim(),
              subject: "Reset your password — Neural Dream",
              html: `
                <div style="font-family:sans-serif;max-width:480px;margin:auto">
                  <h2>Reset your password</h2>
                  <p>Click the button below to set a new password. This link expires in 1 hour.</p>
                  <a href="${resetUrl}" style="display:inline-block;padding:12px 24px;background:#7c3aed;color:#fff;text-decoration:none;border-radius:6px;font-weight:600">
                    Reset Password
                  </a>
                  <p style="margin-top:16px;font-size:12px;color:#888">
                    If you didn't request this, you can ignore this email.
                  </p>
                </div>
              `,
            });
          } catch (emailErr) {
            logger.error({ error: emailErr instanceof Error ? emailErr.message : String(emailErr) }, "Gmail SMTP error");
            return res.status(500).json({ message: "Failed to send reset email. Please try again later." });
          }
        } else {
          // No email provider configured — log URL for local dev/testing
          logger.info({ resetUrl }, "[dev] Password reset link");
        }
      }

      return res.json({ message: "If that email exists, a reset link was sent" });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err) }, "Forgot password failed");
      return res.json({ message: "If that email exists, a reset link was sent" });
    }
  });

  // POST /api/auth/reset-password
  app.post("/api/auth/reset-password", resetPasswordLimiter, async (req, res) => {
    try {
      const { token, newPassword } = req.body;
      if (!token || !newPassword)
        return res.status(400).json({ message: "Token and new password required" });

      const now = new Date();
      const [row] = await db.select().from(passwordResetTokens)
        .where(
          and(
            eq(passwordResetTokens.token, token),
            gte(passwordResetTokens.expiresAt, now),
            sql`${passwordResetTokens.usedAt} IS NULL`
          )
        ).limit(1);

      if (!row) return res.status(400).json({ message: "Invalid or expired reset token" });

      const hashed = await bcrypt.hash(newPassword, 12);
      await db.update(users).set({ password: hashed }).where(eq(users.id, row.userId));
      await db.update(passwordResetTokens)
        .set({ usedAt: now })
        .where(eq(passwordResetTokens.id, row.id));

      return res.json({ message: "Password updated successfully" });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err) }, "Reset password failed");
      return res.status(500).json({ message: "Reset failed" });
    }
  });

  // Health metrics endpoints
  // No auth gate — data scoped by userId. Native APK uses localStorage IDs without JWT.
  app.get("/api/health-metrics/:userId", async (req, res) => {
    try {
      const metrics = await storage.getHealthMetrics(req.params.userId);
      res.json(metrics);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch health metrics" });
    }
  });

  app.post("/api/health-metrics", async (req, res) => {
    try {
      const validatedData = insertHealthMetricsSchema.parse(req.body);
      const metrics = await storage.createHealthMetrics(validatedData);
      res.json(metrics);
    } catch (error) {
      res.status(400).json({ message: "Invalid health metrics data" });
    }
  });

  // Dream analysis endpoints
  // No auth gate — data scoped by userId. Native APK uses localStorage IDs without JWT.
  app.get("/api/dream-analysis/:userId", async (req, res) => {
    try {
      const analyses = await storage.getDreamAnalyses(req.params.userId);
      res.json(analyses);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch dream analyses" });
    }
  });

  app.post("/api/dream-analysis", llmLimiter, async (req, res) => {
    try {
      const { dreamText, userId } = req.body;

      if (!dreamText || typeof dreamText !== "string") {
        return res.status(400).json({ message: "dreamText is required" });
      }
      if (dreamText.length > 10000) {
        return res.status(400).json({ message: "dreamText exceeds max length (10000 chars)" });
      }

      // Analyze dream with OpenAI
      if (!openai) return res.status(503).json({ message: "OPENAI_API_KEY not configured" });
      // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          {
            role: "system",
            content: "You are a dream analysis expert. Analyze the dream text and provide insights about symbols, emotions, and psychological meanings. Respond with JSON in this format: { 'symbols': string[], 'emotions': { 'emotion': string, 'intensity': number }[], 'analysis': string }"
          },
          {
            role: "user",
            content: `Analyze this dream: ${dreamText}`
          }
        ],
        response_format: { type: "json_object" }
      });

      let analysis: Record<string, unknown>;
      try {
        analysis = JSON.parse(response.choices[0].message.content || "{}");
      } catch {
        analysis = {};
      }

      const dreamAnalysis = await storage.createDreamAnalysis({
        userId,
        dreamText,
        symbols: (analysis.symbols as string[]) || [],
        emotions: (analysis.emotions as Array<{emotion: string; intensity: number}>) || [],
        aiAnalysis: (analysis.analysis as string) || ""
      });

      res.json(dreamAnalysis);
    } catch (error) {
      res.status(500).json({ message: "Failed to analyze dream" });
    }
  });

  // AI chat endpoints — no auth gate (data scoped by userId, APK uses localStorage IDs)
  app.get("/api/ai-chat/:userId", async (req, res) => {
    try {
      const chats = await storage.getAiChats(req.params.userId);
      res.json(chats);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch chat history" });
    }
  });

  app.post("/api/ai-chat", llmLimiter, async (req, res) => {
    try {
      const { message, userId, history } = req.body;

      if (!message || typeof message !== "string") {
        return res.status(400).json({ message: "message is required" });
      }
      if (message.length > 5000) {
        return res.status(400).json({ message: "message exceeds max length (5000 chars)" });
      }

      // Store user message
      await storage.createAiChat({
        userId,
        message,
        isUser: true
      });

      // Get recent health metrics for context
      const recentMetrics = await storage.getHealthMetrics(userId, 5);
      const healthContext = recentMetrics.length > 0 ?
        `Recent health data: Heart rate ${recentMetrics[0].heartRate}, Stress level ${recentMetrics[0].stressLevel}, Sleep quality ${recentMetrics[0].sleepQuality}` :
        "";

      // Generate AI response
      if (!openai) return res.status(503).json({ message: "OPENAI_API_KEY not configured" });

      // Build conversation history for context (last 20 messages)
      const historyMessages = Array.isArray(history)
        ? (history as Array<{ message: string; isUser: boolean }>)
            .slice(-20)
            .map((h) => ({
              role: (h.isUser ? "user" : "assistant") as "user" | "assistant",
              content: h.message,
            }))
        : [];

      // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          {
            role: "system",
            content: `You are an AI wellness companion for a Brain-Computer Interface system. You help users with mood analysis, stress relief, meditation, focus, sleep, and general wellness guidance. ${healthContext} Be warm, supportive, and provide actionable advice. You can engage in general conversation too. Keep responses clear and helpful.`
          },
          ...historyMessages,
          {
            role: "user",
            content: message
          }
        ]
      });

      const aiResponse = response.choices[0].message.content || "I'm here to help you with your wellness journey.";

      // Store AI response
      const aiChat = await storage.createAiChat({
        userId,
        message: aiResponse,
        isUser: false
      });

      res.json(aiChat);
    } catch (error) {
      res.status(500).json({ message: "Failed to process chat message" });
    }
  });

  // ── AI Coach endpoint (context-rich, memory-aware) ───────────────────────
  app.post("/api/ai-coach", llmLimiter, async (req, res) => {
    try {
      const { message, userId, history, context, memories, tone } = req.body as {
        message: string;
        userId: string;
        history?: Array<{ message: string; isUser: boolean }>;
        context?: {
          sleep_quality?: number | null;
          voice_valence?: number | null;
          hrv_avg?: number | null;
          dream_count?: number | null;
          stress_index?: number | null;
        };
        memories?: string[];
        tone?: "supportive" | "accountability";
      };

      if (!message || typeof message !== "string") {
        return res.status(400).json({ message: "message is required" });
      }
      if (message.length > 5000) {
        return res.status(400).json({ message: "message exceeds max length (5000 chars)" });
      }

      if (!openai) return res.status(503).json({ message: "OPENAI_API_KEY not configured" });

      // ── Build context block ──────────────────────────────────────────────
      const ctxParts: string[] = [];

      if (context) {
        const c = context;
        if (c.sleep_quality != null)
          ctxParts.push(`Sleep quality (last 7 days avg): ${(c.sleep_quality * 100).toFixed(0)}%`);
        if (c.voice_valence != null) {
          const valenceLabel =
            c.voice_valence > 0.2 ? "positive" : c.voice_valence < -0.2 ? "negative" : "neutral";
          ctxParts.push(`Voice valence (last 7 days avg): ${valenceLabel} (${c.voice_valence.toFixed(2)})`);
        }
        if (c.hrv_avg != null)
          ctxParts.push(`HRV average (last 7 days): ${c.hrv_avg.toFixed(1)} ms`);
        if (c.dream_count != null)
          ctxParts.push(`Dreams recorded in last 7 days: ${c.dream_count}`);
        if (c.stress_index != null)
          ctxParts.push(`Stress index (last 7 days avg): ${(c.stress_index * 100).toFixed(0)}%`);
      }

      const contextBlock =
        ctxParts.length > 0
          ? `\n\nUser health context (last 7 days):\n${ctxParts.map((p) => `- ${p}`).join("\n")}`
          : "";

      // ── Persistent memories block ────────────────────────────────────────
      const memoriesBlock =
        Array.isArray(memories) && memories.length > 0
          ? `\n\nThings the user has confirmed about themselves:\n${memories.map((m) => `- ${m}`).join("\n")}`
          : "";

      // ── Tone instruction ─────────────────────────────────────────────────
      const toneInstruction =
        tone === "accountability"
          ? "Be direct and hold the user accountable — celebrate wins, name patterns that need attention, and give clear action steps."
          : "Be warm, empathetic, and supportive — meet the user where they are and offer gentle encouragement.";

      const systemPrompt = `You are an expert AI health coach for a Brain-Computer Interface system. You have access to the user's biometric and brainwave data.${contextBlock}${memoriesBlock}

Coaching style: ${toneInstruction}

Your role: give personalised, longitudinal coaching based on the user's actual data trends — not generic advice. Reference their specific numbers when relevant. Ask follow-up questions to deepen understanding. Keep responses concise (under 200 words unless the user asks for more).`;

      // ── Build conversation history ────────────────────────────────────────
      const historyMessages = Array.isArray(history)
        ? (history as Array<{ message: string; isUser: boolean }>)
            .slice(-20)
            .map((h) => ({
              role: (h.isUser ? "user" : "assistant") as "user" | "assistant",
              content: h.message,
            }))
        : [];

      // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          { role: "system", content: systemPrompt },
          ...historyMessages,
          { role: "user", content: message },
        ],
      });

      const aiResponse =
        response.choices[0].message.content ??
        "I'm here to help you with your wellness journey.";

      // Store both sides of the conversation
      await storage.createAiChat({ userId, message, isUser: true });
      const aiChat = await storage.createAiChat({ userId, message: aiResponse, isUser: false });

      res.json(aiChat);
    } catch (error) {
      res.status(500).json({ message: "Failed to process coach message" });
    }
  });

  // Mood analysis endpoint
  app.post("/api/analyze-mood", llmLimiter, async (req, res) => {
    try {
      const { text } = req.body;

      if (!text || typeof text !== "string") {
        return res.status(400).json({ message: "text is required" });
      }
      if (text.length > 5000) {
        return res.status(400).json({ message: "text exceeds max length (5000 chars)" });
      }

      if (!openai) return res.status(503).json({ message: "OPENAI_API_KEY not configured" });
      // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          {
            role: "system",
            content: "Analyze the mood and emotional state from the text. Provide insights about stress levels, emotional patterns, and wellness recommendations. Respond with JSON in this format: { 'mood': string, 'stressLevel': number, 'emotions': string[], 'recommendations': string[] }"
          },
          {
            role: "user",
            content: text
          }
        ],
        response_format: { type: "json_object" }
      });

      let analysis: Record<string, unknown>;
      try {
        analysis = JSON.parse(response.choices[0].message.content || "{}");
      } catch {
        analysis = {};
      }
      res.json(analysis);
    } catch (error) {
      res.status(500).json({ message: "Failed to analyze mood" });
    }
  });

  // User settings endpoints — no auth gate (data scoped by userId, APK uses localStorage IDs)
  app.get("/api/settings/:userId", async (req, res) => {
    try {
      const settings = await storage.getUserSettings(req.params.userId);
      res.json(settings);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch settings" });
    }
  });

  app.post("/api/settings/:userId", async (req, res) => {
    try {
      const validatedData = insertUserSettingsSchema.parse(req.body);
      const settings = await storage.updateUserSettings(req.params.userId, validatedData);
      res.json(settings);
    } catch (error) {
      res.status(400).json({ message: "Invalid settings data" });
    }
  });

  // Data export endpoint — no auth gate (data scoped by userId)
  app.get("/api/export/:userId", async (req, res) => {
    try {
      const metrics = await storage.getHealthMetrics(req.params.userId);

      // Convert to CSV format
      const csvData = metrics.map(m => ({
        timestamp: m.timestamp,
        heartRate: m.heartRate,
        stressLevel: m.stressLevel,
        sleepQuality: m.sleepQuality,
        neuralActivity: m.neuralActivity,
        dailySteps: m.dailySteps,
        sleepDuration: m.sleepDuration
      }));

      res.setHeader('Content-Type', 'text/csv');
      res.setHeader('Content-Disposition', 'attachment; filename=neural_data.csv');
      
      // Simple CSV conversion
      const csvHeader = Object.keys(csvData[0] || {}).join(',');
      const csvRows = csvData.map(row => Object.values(row).join(','));
      const csvContent = [csvHeader, ...csvRows].join('\n');
      
      res.send(csvContent);
    } catch (error) {
      res.status(500).json({ message: "Failed to export data" });
    }
  });

  // ── Brain history endpoints ────────────────────────────────────────

  // GET /api/brain/history/:userId?days=1  — time-range emotion history (cap 2000 rows)
  app.get("/api/brain/history/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const days = Math.min(Math.max(parseInt((req.query.days as string) || "1", 10), 1), 30);
      const fromTs = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
      const readings = await storage.getEmotionReadings(userId, 2000, fromTs);
      // Return oldest-first for chart rendering
      res.json(readings.reverse());
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch emotion history" });
    }
  });

  // GET /api/brain/today-totals/:userId — avg stress/focus/emotion since midnight
  app.get("/api/brain/today-totals/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const midnight = new Date();
      midnight.setHours(0, 0, 0, 0);

      const readings = await storage.getEmotionReadings(userId, 2000, midnight);
      if (readings.length === 0) {
        return res.json({ userId, count: 0, avgStress: null, avgFocus: null, avgHappiness: null, avgEnergy: null, dominantEmotion: null });
      }

      const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
      const emotionCounts: Record<string, number> = {};
      readings.forEach(r => {
        emotionCounts[r.dominantEmotion] = (emotionCounts[r.dominantEmotion] || 0) + 1;
      });
      const dominantEmotion = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;

      res.json({
        userId,
        count: readings.length,
        avgStress: avg(readings.map(r => r.stress)),
        avgFocus: avg(readings.map(r => r.focus)),
        avgHappiness: avg(readings.map(r => r.happiness)),
        avgEnergy: avg(readings.map(r => r.energy)),
        avgValence: avg(readings.filter(r => r.valence != null).map(r => r.valence!)),
        avgArousal: avg(readings.filter(r => r.arousal != null).map(r => r.arousal!)),
        dominantEmotion,
      });
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch today totals" });
    }
  });

  // GET /api/brain/at-this-time-yesterday/:userId — ±30 min window same time yesterday
  app.get("/api/brain/at-this-time-yesterday/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const now = Date.now();
      const oneDayMs = 24 * 60 * 60 * 1000;
      const windowMs = 30 * 60 * 1000; // ±30 min
      const fromTs = new Date(now - oneDayMs - windowMs);
      const toTs = new Date(now - oneDayMs + windowMs);

      const readings = await storage.getEmotionReadings(userId, 200, fromTs, toTs);
      if (readings.length === 0) {
        return res.json({ userId, count: 0, avgStress: null, avgFocus: null, avgHappiness: null });
      }

      const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
      res.json({
        userId,
        count: readings.length,
        windowStart: fromTs.toISOString(),
        windowEnd: toTs.toISOString(),
        avgStress: avg(readings.map(r => r.stress)),
        avgFocus: avg(readings.map(r => r.focus)),
        avgHappiness: avg(readings.map(r => r.happiness)),
        avgEnergy: avg(readings.map(r => r.energy)),
        avgValence: avg(readings.filter(r => r.valence != null).map(r => r.valence!)),
      });
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch yesterday comparison" });
    }
  });

  // GET /api/brain/yesterday-insights/:userId
  // Cross-correlates yesterday's emotion readings with EEG session events to produce
  // activity-specific insights: "Focus was 31% higher after your 3pm breathing session."
  app.get("/api/brain/yesterday-insights/:userId", async (req, res) => {
    try {
      const { userId } = req.params;

      // Fetch 48 h of readings (yesterday + today for comparison)
      const from48h = new Date(Date.now() - 48 * 60 * 60 * 1000);
      const allReadings = await storage.getEmotionReadings(userId, 3000, from48h);

      if (allReadings.length < 3) {
        return res.json({ userId, insights: [] });
      }

      // Partition into yesterday vs today
      const todayMidnight = new Date();
      todayMidnight.setHours(0, 0, 0, 0);
      const yesterdayMidnight = new Date(todayMidnight.getTime() - 86_400_000);

      const yesterday = allReadings.filter(r => {
        const t = new Date(r.timestamp).getTime();
        return t >= yesterdayMidnight.getTime() && t < todayMidnight.getTime();
      });
      const today = allReadings.filter(r => new Date(r.timestamp).getTime() >= todayMidnight.getTime());

      // Helper: average over a slice
      const avg = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
      const fmtHour = (date: Date) => {
        const h = date.getHours();
        return `${h % 12 || 12}${h < 12 ? "am" : "pm"}`;
      };
      const pctDelta = (a: number, b: number) =>
        Math.round(Math.abs(a - b) / Math.max(Math.abs(b), 0.001) * 100);

      const insights: Array<{ type: string; text: string; delta?: number }> = [];

      // ── Insight 1: peak focus hour vs rest of yesterday ────────────────
      if (yesterday.length >= 4) {
        const byHour: Record<number, number[]> = {};
        yesterday.forEach(r => {
          const h = new Date(r.timestamp).getHours();
          if (!byHour[h]) byHour[h] = [];
          byHour[h].push(r.focus);
        });
        const hourAvgs = Object.entries(byHour).map(([h, vals]) => ({
          hour: parseInt(h),
          avg: avg(vals) ?? 0,
          count: vals.length,
        })).filter(e => e.count >= 2);

        if (hourAvgs.length >= 2) {
          hourAvgs.sort((a, b) => b.avg - a.avg);
          const peak = hourAvgs[0];
          const rest = hourAvgs.slice(1);
          const restAvg = avg(rest.map(e => e.avg)) ?? 0;
          const delta = pctDelta(peak.avg, restAvg);
          if (delta >= 12) {
            const peakDate = new Date(yesterdayMidnight.getTime() + peak.hour * 3600_000);
            insights.push({
              type: "peak_focus",
              text: `Focus peaked at ${fmtHour(peakDate)}, ${delta}% above the rest of yesterday.`,
              delta,
            });
          }
        }
      }

      // ── Insight 2: activity cross-correlation (biofeedback sessions) ───
      // Fetch sessions from ML backend (via internal HTTP)
      const ML_API = process.env.ML_API_URL || "http://localhost:8000";
      try {
        const sessRes = await fetch(`${ML_API}/api/sessions`, { signal: AbortSignal.timeout(3000) });
        if (sessRes.ok) {
          const sessions = (await sessRes.json()) as Array<{
            session_id: string; session_type: string; start_time: number;
          }>;

          // Yesterday's biofeedback sessions
          const bfSessions = sessions.filter(s =>
            s.session_type === "biofeedback" &&
            s.start_time * 1000 >= yesterdayMidnight.getTime() &&
            s.start_time * 1000 < todayMidnight.getTime()
          );

          for (const sess of bfSessions.slice(0, 2)) {
            const sessionMs = sess.start_time * 1000;
            const windowMs = 60 * 60 * 1000; // 60-minute window

            const before = allReadings.filter(r => {
              const t = new Date(r.timestamp).getTime();
              return t >= sessionMs - windowMs && t < sessionMs;
            });
            const after = allReadings.filter(r => {
              const t = new Date(r.timestamp).getTime();
              return t > sessionMs && t <= sessionMs + windowMs;
            });

            if (before.length >= 2 && after.length >= 2) {
              const focusBefore = avg(before.map(r => r.focus)) ?? 0;
              const focusAfter  = avg(after.map(r => r.focus))  ?? 0;
              const stressBefore = avg(before.map(r => r.stress)) ?? 0;
              const stressAfter  = avg(after.map(r => r.stress))  ?? 0;

              const focusDelta  = pctDelta(focusAfter, focusBefore);
              const stressDelta = pctDelta(stressBefore, stressAfter);
              const sessHour = fmtHour(new Date(sessionMs));

              if (focusAfter > focusBefore && focusDelta >= 10) {
                insights.push({
                  type: "activity_focus",
                  text: `Focus was ${focusDelta}% higher in the hour after your ${sessHour} breathing session.`,
                  delta: focusDelta,
                });
              } else if (stressAfter < stressBefore && stressDelta >= 10) {
                insights.push({
                  type: "activity_stress",
                  text: `Stress dropped ${stressDelta}% in the hour after your ${sessHour} breathing session.`,
                  delta: stressDelta,
                });
              }
            }
          }
        }
      } catch {
        // ML backend offline — skip session cross-correlation, continue with other insights
      }

      // ── Insight 3: yesterday vs day-before comparison ──────────────────
      if (yesterday.length >= 3) {
        const dayBeforeReadings = allReadings.filter(r => {
          const t = new Date(r.timestamp).getTime();
          return t >= yesterdayMidnight.getTime() - 86_400_000 && t < yesterdayMidnight.getTime();
        });

        const ydayFocus  = avg(yesterday.map(r => r.focus));
        const ydayStress = avg(yesterday.map(r => r.stress));
        const prevFocus  = avg(dayBeforeReadings.map(r => r.focus));
        const prevStress = avg(dayBeforeReadings.map(r => r.stress));

        if (ydayFocus != null && prevFocus != null) {
          const delta = pctDelta(ydayFocus, prevFocus);
          if (delta >= 10) {
            insights.push({
              type: "day_comparison",
              text: ydayFocus > prevFocus
                ? `Yesterday's focus (${Math.round(ydayFocus * 100)}%) was ${delta}% stronger than the day before.`
                : `Yesterday's focus was ${delta}% lower than the day before — today is a reset.`,
              delta,
            });
          }
        } else if (ydayStress != null && prevStress != null) {
          const delta = pctDelta(ydayStress, prevStress);
          if (delta >= 15) {
            insights.push({
              type: "day_comparison",
              text: ydayStress < prevStress
                ? `Stress was ${delta}% lower yesterday than the day before.`
                : `Stress ran ${delta}% higher yesterday — watch for that pattern today.`,
              delta,
            });
          }
        }
      }

      // ── Insight 4: today vs yesterday baseline (morning preview) ──────
      if (today.length >= 2 && yesterday.length >= 2) {
        const todayFocus = avg(today.map(r => r.focus));
        const ydayFocus  = avg(yesterday.map(r => r.focus));
        if (todayFocus != null && ydayFocus != null) {
          const delta = pctDelta(todayFocus, ydayFocus);
          if (delta >= 8 && today.length >= 3) {
            insights.push({
              type: "today_vs_yesterday",
              text: todayFocus > ydayFocus
                ? `You're already ${delta}% sharper than this time yesterday.`
                : `Focus is ${delta}% below yesterday so far — give it time to warm up.`,
              delta,
            });
          }
        }
      }

      // Return top 3 most interesting insights (highest delta first)
      const ranked = insights
        .sort((a, b) => (b.delta ?? 0) - (a.delta ?? 0))
        .slice(0, 3);

      res.json({ userId, insights: ranked });
    } catch (error) {
      res.status(500).json({ message: "Failed to compute yesterday insights" });
    }
  });

  // GET /api/brain/circadian-profile/:userId — personalized circadian rhythm profile
  // Fetches emotion readings from the last N days, extracts time-stamped features,
  // sends them to the ML backend for cosinor fitting, and caches the result.
  app.get("/api/brain/circadian-profile/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const days = Math.min(90, Math.max(7, parseInt(req.query.days as string) || 14));
      const fromTs = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

      // Fetch emotion readings for the window
      const readings = await storage.getEmotionReadings(userId, 2000, fromTs);
      if (!readings || readings.length < 6) {
        return res.json({
          available: false,
          message: `Need at least 6 data points across ${days} days. Currently have ${readings?.length ?? 0}.`,
          minimum_days: 7,
        });
      }

      // Extract time-stamped feature streams from emotion readings
      const streams: Record<string, { time_h: number; value: number }[]> = {
        valence: [],
        arousal: [],
        stress: [],
        focus: [],
      };

      for (const r of readings) {
        const ts = new Date(r.timestamp);
        const time_h = ts.getHours() + ts.getMinutes() / 60;

        if (r.valence != null) streams.valence.push({ time_h, value: r.valence });
        if (r.arousal != null) streams.arousal.push({ time_h, value: r.arousal });
        if (r.stress != null && r.stress > 0) streams.stress.push({ time_h, value: r.stress });
        if (r.focus != null && r.focus > 0) streams.focus.push({ time_h, value: r.focus });
      }

      // Also try to get HRV data from health_samples
      const hrvSamples = await db.select()
        .from(healthSamples)
        .where(and(
          eq(healthSamples.userId, userId),
          eq(healthSamples.metric, "hrv"),
          gte(healthSamples.recordedAt, fromTs),
        ))
        .orderBy(asc(healthSamples.recordedAt))
        .limit(500);

      if (hrvSamples.length > 0) {
        streams["hrv"] = hrvSamples.map(s => ({
          time_h: new Date(s.recordedAt).getHours() + new Date(s.recordedAt).getMinutes() / 60,
          value: parseFloat(String(s.value)),
        }));
      }

      // Compute current hour
      const now = new Date();
      const current_hour = now.getHours() + now.getMinutes() / 60;

      // Check for previous profile (for phase-shift detection)
      const [prevProfile] = await db.select()
        .from(circadianProfiles)
        .where(eq(circadianProfiles.userId, userId))
        .orderBy(desc(circadianProfiles.computedAt))
        .limit(1);

      const baseline_acrophase = prevProfile?.acrophaseH ?? null;

      // Call ML backend
      const mlUrl = process.env.VITE_ML_API_URL || "http://localhost:8080";
      const mlRes = await fetch(`${mlUrl}/api/circadian/compute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          feature_streams: streams,
          current_hour,
          baseline_acrophase,
        }),
      });

      if (!mlRes.ok) {
        const errText = await mlRes.text();
        logger.warn(`Circadian ML call failed (${mlRes.status}): ${errText}`);
        return res.status(mlRes.status).json({ error: errText });
      }

      const profile = await mlRes.json();

      // Cache in DB (upsert-like: just insert, old profiles remain for history)
      try {
        await db.insert(circadianProfiles).values({
          userId,
          chronotype: profile.chronotype,
          chronotypeConfidence: profile.chronotype_confidence,
          acrophaseH: profile.acrophase_h,
          amplitude: profile.amplitude,
          periodH: profile.period_h,
          phaseStability: profile.phase_stability,
          predictedFocusWindow: profile.predicted_focus_window,
          predictedSlumpWindow: profile.predicted_slump_window,
          phaseShiftHours: profile.phase_shift_hours,
          fits: profile.fits,
          dataDays: profile.data_days,
        });
      } catch (dbErr) {
        logger.warn("Failed to cache circadian profile:", dbErr);
        // Non-fatal — still return the computed profile
      }

      res.json(profile);
    } catch (error) {
      logger.error("Circadian profile error:", error);
      res.status(500).json({ message: "Failed to compute circadian profile" });
    }
  });

  // GET /api/brain/weekly-summary/:userId — 7-day wellness summary (no GPT, structured data only)
  app.get("/api/brain/weekly-summary/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const fromTs = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
      const readings = await storage.getEmotionReadings(userId, 500, fromTs);

      if (!readings || readings.length === 0) {
        return res.json({ available: false, message: "Not enough data for a weekly summary" });
      }

      const avg = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
      const stressVals = readings.map(r => r.stress).filter(v => v > 0);
      const focusVals = readings.map(r => r.focus).filter(v => v > 0);
      const happinessVals = readings.map(r => r.happiness).filter(v => v > 0);
      const energyVals = readings.map(r => r.energy).filter(v => v > 0);

      // Dominant emotion
      const emotionCounts: Record<string, number> = {};
      readings.forEach(r => {
        if (r.dominantEmotion) {
          emotionCounts[r.dominantEmotion] = (emotionCounts[r.dominantEmotion] || 0) + 1;
        }
      });
      const sortedEmotions = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]);
      const dominantEmotion = sortedEmotions[0]?.[0] || "neutral";

      // Stress trend: first half vs second half
      const mid = Math.floor(stressVals.length / 2);
      const firstHalf = stressVals.slice(0, mid);
      const secondHalf = stressVals.slice(mid);
      const stressTrend = avg(secondHalf) - avg(firstHalf); // positive = increasing stress

      // Unique check-in days
      const checkinDays = new Set(readings.map(r => new Date(r.timestamp).toISOString().slice(0, 10))).size;

      // Generate insight text
      let insight = "";
      if (avg(stressVals) > 0.6) {
        insight = "Your stress has been elevated this week. Consider adding breathing exercises to your routine.";
      } else if (stressTrend < -0.1) {
        insight = "Great news — your stress levels are trending down. Keep up whatever you're doing!";
      } else if (avg(focusVals) > 0.6) {
        insight = "Your focus has been strong this week. You're in a productive flow.";
      } else if (avg(happinessVals) > 0.6) {
        insight = "You've had a positive week overall. Savor this emotional momentum.";
      } else {
        insight = "Consistency is key — keep checking in daily to build awareness of your patterns.";
      }

      res.json({
        available: true,
        period: {
          start: fromTs.toISOString().slice(0, 10),
          end: new Date().toISOString().slice(0, 10),
        },
        summary: {
          total_readings: readings.length,
          checkin_days: checkinDays,
          avg_stress: Math.round(avg(stressVals) * 100),
          avg_focus: Math.round(avg(focusVals) * 100),
          avg_happiness: Math.round(avg(happinessVals) * 100),
          avg_energy: Math.round(avg(energyVals) * 100),
          dominant_emotion: dominantEmotion,
          emotion_distribution: Object.fromEntries(sortedEmotions),
          stress_trend: stressTrend > 0.05 ? "increasing" : stressTrend < -0.05 ? "decreasing" : "stable",
        },
        insight,
      });
    } catch (error) {
      console.error("Weekly summary error:", error);
      res.status(500).json({ message: "Failed to generate weekly summary" });
    }
  });

  // Emotional Fitness Score
  app.get("/api/brain/emotional-fitness/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const force = req.query.force === "true";
      const days = parseInt(req.query.days as string) || 14;
      const { computeEmotionalFitness } = await import("./efs-compute");
      const mlBaseUrl = process.env.VITE_ML_API_URL || "http://localhost:8080";
      const result = await computeEmotionalFitness(userId, mlBaseUrl, force, days);
      res.json(result);
    } catch (error) {
      console.error("Emotional fitness error:", error);
      res.status(500).json({ message: "Failed to compute emotional fitness score" });
    }
  });

  // GET /api/brain/patterns/:userId
  // Long-term pattern engine: correlates 30 days of emotion readings with time-of-day,
  // day-of-week, sleep quality, and biofeedback sessions to produce actionable patterns.
  // Patterns: focus_peak_hour, stress_peak_hour, best_day_of_week, sleep_focus_correlation,
  //           biofeedback_effect.
  app.get("/api/brain/patterns/:userId", async (req, res) => {
    try {
      const { userId } = req.params;
      const from30d = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);

      const [readings, health] = await Promise.all([
        storage.getEmotionReadings(userId, 5000, from30d),
        storage.getHealthMetrics(userId, 500),
      ]);

      if (readings.length < 10) {
        return res.json({ userId, dataPoints: readings.length, patterns: [] });
      }

      const avg = (arr: number[]): number =>
        arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
      const pctDelta = (a: number, b: number): number =>
        b > 0 ? Math.round(Math.abs(a - b) / b * 100) : 0;
      const DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
      const fmt12h = (h: number) => `${h % 12 || 12}${h < 12 ? "am" : "pm"}`;

      type Pattern = {
        type: string;
        title: string;
        description: string;
        recommendation: string;
        confidence: number;
        data: Record<string, unknown>;
      };
      const patterns: Pattern[] = [];

      // ── 1. Focus peak hour (5am–10pm) ─────────────────────────────────
      {
        const byHour: Record<number, number[]> = {};
        for (const r of readings) {
          const h = new Date(r.timestamp).getHours();
          if (h >= 5 && h <= 22) {
            if (!byHour[h]) byHour[h] = [];
            byHour[h].push(r.focus);
          }
        }
        const hourEntries = Object.entries(byHour)
          .map(([h, vals]) => ({ hour: parseInt(h), avg: avg(vals), count: vals.length }))
          .filter(e => e.count >= 3);

        if (hourEntries.length >= 4) {
          hourEntries.sort((a, b) => b.avg - a.avg);
          const peak = hourEntries[0];
          const restAvg = avg(hourEntries.slice(1).map(e => e.avg));
          const delta = pctDelta(peak.avg, restAvg);
          if (delta >= 15) {
            patterns.push({
              type: "focus_peak_hour",
              title: `Focus peaks at ${fmt12h(peak.hour)}`,
              description: `Your focus is ${delta}% higher at ${fmt12h(peak.hour)} than the rest of the day`,
              recommendation: `Protect ${fmt12h(peak.hour - 1 < 5 ? peak.hour : peak.hour - 1)}–${fmt12h(peak.hour + 2)} for deep work`,
              confidence: Math.min(0.95, 0.5 + peak.count / 60),
              data: { hour: peak.hour, deltaPercent: delta, sampleCount: peak.count },
            });
          }
        }
      }

      // ── 2. Stress peak hour (afternoon 11am–7pm) ─────────────────────
      {
        const byHour: Record<number, number[]> = {};
        for (const r of readings) {
          const h = new Date(r.timestamp).getHours();
          if (h >= 11 && h <= 19) {
            if (!byHour[h]) byHour[h] = [];
            byHour[h].push(r.stress);
          }
        }
        const hourEntries = Object.entries(byHour)
          .map(([h, vals]) => ({ hour: parseInt(h), avg: avg(vals), count: vals.length }))
          .filter(e => e.count >= 3);

        if (hourEntries.length >= 3) {
          hourEntries.sort((a, b) => b.avg - a.avg);
          const peak = hourEntries[0];
          const overallStress = avg(readings.map(r => r.stress));
          const delta = pctDelta(peak.avg, overallStress);
          if (delta >= 15) {
            patterns.push({
              type: "stress_peak_hour",
              title: `Stress spikes at ${fmt12h(peak.hour)}`,
              description: `Stress is ${delta}% above your daily average around ${fmt12h(peak.hour)}`,
              recommendation: `Schedule a 5-min breathing break at ${fmt12h(peak.hour - 1 < 11 ? peak.hour : peak.hour - 1)}`,
              confidence: Math.min(0.92, 0.5 + peak.count / 60),
              data: { hour: peak.hour, deltaPercent: delta, sampleCount: peak.count },
            });
          }
        }
      }

      // ── 3. Best day of week ──────────────────────────────────────────
      {
        const byDay: Record<number, number[]> = {};
        for (const r of readings) {
          const d = new Date(r.timestamp).getDay();
          if (!byDay[d]) byDay[d] = [];
          byDay[d].push(r.focus);
        }
        const dayEntries = Object.entries(byDay)
          .map(([d, vals]) => ({ day: parseInt(d), avg: avg(vals), count: vals.length }))
          .filter(e => e.count >= 5);

        if (dayEntries.length >= 4) {
          dayEntries.sort((a, b) => b.avg - a.avg);
          const best = dayEntries[0];
          const restAvg = avg(dayEntries.slice(1).map(e => e.avg));
          const delta = pctDelta(best.avg, restAvg);
          if (delta >= 10) {
            patterns.push({
              type: "best_day_of_week",
              title: `${DAY_NAMES[best.day]}s are your strongest`,
              description: `Focus averages ${delta}% higher on ${DAY_NAMES[best.day]}s than other days`,
              recommendation: `Schedule your hardest work on ${DAY_NAMES[best.day]}s`,
              confidence: Math.min(0.9, 0.5 + best.count / 40),
              data: { day: DAY_NAMES[best.day], dayIndex: best.day, deltaPercent: delta },
            });
          }
        }
      }

      // ── 4. Sleep-focus correlation ───────────────────────────────────
      {
        const recentHealth = health.filter(
          h => new Date(h.timestamp).getTime() >= from30d.getTime()
        );
        if (recentHealth.length >= 5) {
          const goodSleep: number[] = [];  // focus next morning after good sleep (>6)
          const poorSleep: number[] = [];  // focus next morning after poor sleep (≤4)

          for (const hm of recentHealth) {
            const sleepQ = hm.sleepQuality ?? 5;
            const hmTs = new Date(hm.timestamp).getTime();
            const nextMorningStart = hmTs + 6 * 3600_000;
            const nextMorningEnd = hmTs + 14 * 3600_000;
            const nextMorningReadings = readings.filter(r => {
              const t = new Date(r.timestamp).getTime();
              return t >= nextMorningStart && t <= nextMorningEnd;
            });
            if (nextMorningReadings.length === 0) continue;
            const morningFocus = avg(nextMorningReadings.map(r => r.focus));
            if (sleepQ > 6) goodSleep.push(morningFocus);
            else if (sleepQ <= 4) poorSleep.push(morningFocus);
          }

          if (goodSleep.length >= 3 && poorSleep.length >= 3) {
            const goodAvg = avg(goodSleep);
            const poorAvg = avg(poorSleep);
            const delta = pctDelta(goodAvg, poorAvg);
            if (delta >= 10 && goodAvg > poorAvg) {
              patterns.push({
                type: "sleep_focus_correlation",
                title: "Sleep quality predicts focus",
                description: `On mornings after good sleep, your focus is ${delta}% higher`,
                recommendation: "Prioritise 7h+ sleep before important days",
                confidence: Math.min(0.88, 0.5 + (goodSleep.length + poorSleep.length) / 30),
                data: { goodSleepSamples: goodSleep.length, poorSleepSamples: poorSleep.length, deltaPercent: delta },
              });
            }
          }
        }
      }

      // ── 5. Biofeedback effect (cross-session average) ─────────────────
      {
        const ML_API = process.env.ML_API_URL || "http://localhost:8000";
        try {
          const sessRes = await fetch(`${ML_API}/api/sessions?user_id=${userId}`, { signal: AbortSignal.timeout(3000) });
          if (sessRes.ok) {
            const sessions = (await sessRes.json()) as Array<{
              session_id: string; session_type: string; start_time: number; end_time?: number;
            }>;
            const bfSessions = sessions.filter(s =>
              s.session_type === "biofeedback" &&
              s.start_time * 1000 >= from30d.getTime()
            );
            const windowMs = 45 * 60 * 1000; // 45-minute window
            const beforeStress: number[] = [];
            const afterStress: number[] = [];

            for (const sess of bfSessions) {
              const sessionMs = sess.start_time * 1000;
              const before = readings.filter(r => {
                const t = new Date(r.timestamp).getTime();
                return t >= sessionMs - windowMs && t < sessionMs;
              });
              const after = readings.filter(r => {
                const t = new Date(r.timestamp).getTime();
                return t > sessionMs && t <= sessionMs + windowMs;
              });
              if (before.length >= 2 && after.length >= 2) {
                beforeStress.push(avg(before.map(r => r.stress)));
                afterStress.push(avg(after.map(r => r.stress)));
              }
            }

            if (beforeStress.length >= 2) {
              const avgBefore = avg(beforeStress);
              const avgAfter = avg(afterStress);
              if (avgBefore > avgAfter) {
                const delta = pctDelta(avgBefore, avgAfter);
                if (delta >= 10) {
                  patterns.push({
                    type: "biofeedback_effect",
                    title: `Biofeedback cuts stress by ${delta}%`,
                    description: `Across ${beforeStress.length} sessions, stress drops ${delta}% in the 45 min after breathing exercises`,
                    recommendation: "Your data proves biofeedback works — keep the habit",
                    confidence: Math.min(0.96, 0.6 + beforeStress.length / 20),
                    data: { sessionCount: beforeStress.length, deltaPercent: delta },
                  });
                }
              }
            }
          }
        } catch {
          // ML backend offline — skip biofeedback cross-correlation
        }
      }

      // Sort by confidence descending, return top 4
      patterns.sort((a, b) => b.confidence - a.confidence);

      res.json({ userId, dataPoints: readings.length, patterns: patterns.slice(0, 4) });
    } catch (error) {
      res.status(500).json({ message: "Failed to compute brain patterns" });
    }
  });

  // POST /api/emotion-readings/batch — bulk insert from ML backend on session stop
  app.post("/api/emotion-readings/batch", async (req, res) => {
    try {
      const { readings } = req.body;
      if (!Array.isArray(readings) || readings.length === 0) {
        return res.status(400).json({ message: "readings array is required" });
      }
      if (readings.length > 5000) {
        return res.status(400).json({ message: "Too many readings (max 5000 per batch)" });
      }

      const parsed = readings.map((r: unknown) => insertEmotionReadingSchema.parse(r));
      const inserted = await storage.batchCreateEmotionReadings(parsed);
      res.json({ inserted: inserted.length });
    } catch (error: any) {
      const detail = error?.issues ? error.issues.map((i: any) => `${i.path?.join('.')}: ${i.message}`).join('; ') : error?.message;
      res.status(400).json({ message: "Invalid emotion readings data", detail });
    }
  });

  // POST /api/datadog/webhook — Datadog monitor webhook receiver + auto-remediation
  app.post("/api/datadog/webhook", async (req, res) => {
    try {
      const payload = req.body as Record<string, unknown>;
      const alertType = (payload.alert_type as string) || "";
      const monitorId = String(payload.id ?? "");
      const monitorName = String(payload.monitor_name ?? payload.title ?? "");

      // Log the webhook event
      logger.info({ alertType, monitorName, monitorId }, "Datadog webhook received");

      let remediationAction: string | null = null;
      let remediationStatus = "skipped";
      let remediationDetail: string | null = null;

      // Auto-remediation: model inference errors → trigger model reload
      if (alertType === "trigger" && monitorName.toLowerCase().includes("model_inference_error")) {
        remediationAction = "POST /api/models/reload";
        try {
          const ML_API_URL = process.env.ML_API_URL || "http://localhost:8000";
          const reloadRes = await fetch(`${ML_API_URL}/api/models/reload`, { method: "POST" });
          remediationStatus = reloadRes.ok ? "success" : "failed";
          remediationDetail = `HTTP ${reloadRes.status}`;
        } catch (err) {
          remediationStatus = "failed";
          remediationDetail = String(err);
        }
      }

      // Persist audit log if DB available
      if (process.env.DATABASE_URL) {
        try {
          const { datadogErrorLog } = await import("@shared/schema");
          await db.insert(datadogErrorLog).values({
            monitorId,
            monitorName,
            alertType,
            errorType: (payload.error_type as string) ?? null,
            payload,
            remediationAction,
            remediationStatus,
            remediationDetail,
          });
        } catch {
          // Non-fatal: DB log failure shouldn't break webhook response
        }
      }

      res.json({ received: true, remediationAction, remediationStatus });
    } catch (error) {
      res.status(500).json({ message: "Failed to process Datadog webhook" });
    }
  });

  // ── ML backend proxy: /api/ml/* → http://localhost:8000/* ────────
  // Forwards all /api/ml/... requests to the FastAPI ML service so
  // browser-side code (e.g. ContinuousBrainTimeline) can use relative
  // URLs instead of hard-coding port 8000.
  app.all("/api/ml/*", async (req, res) => {
    const ML_API_URL = process.env.ML_API_URL || "http://localhost:8000";
    const mlPath = req.path.replace(/^\/api\/ml/, "");
    const queryStr = new URLSearchParams(
      req.query as Record<string, string>
    ).toString();
    const targetUrl = `${ML_API_URL}${mlPath}${queryStr ? `?${queryStr}` : ""}`;

    try {
      const hasBody = req.method !== "GET" && req.method !== "HEAD";
      const mlRes = await fetch(targetUrl, {
        method: req.method,
        headers: { "Content-Type": "application/json" },
        body: hasBody ? JSON.stringify(req.body) : undefined,
      });

      const contentType = mlRes.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        const data = await mlRes.json();
        return res.status(mlRes.status).json(data);
      }
      const text = await mlRes.text();
      return res.status(mlRes.status).type(contentType || "text/plain").send(text);
    } catch (error) {
      return res
        .status(503)
        .json({ message: "ML backend unavailable", error: String(error) });
    }
  });

  // ── Research Enrollment Module ────────────────────────────────────────────

  // POST /api/study/enroll — consent wizard final step
  app.post("/api/study/enroll", async (req, res) => {
    try {
      const { userId, studyId, consentVersion, overnightEegConsent,
              preferredMorningTime, preferredDaytimeTime, preferredEveningTime,
              consentFullName, consentInitials } = req.body;

      if (!userId || !studyId || !consentVersion) {
        return res.status(400).json({ message: "userId, studyId, and consentVersion are required" });
      }

      // Block duplicate enrollment
      const already = await getActiveParticipant(userId);
      if (already) {
        return res.status(409).json({ message: "Already enrolled in an active study", studyCode: already.studyCode });
      }

      // Generate unique 6-char code (no O/0/I/1 — visually ambiguous)
      const chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
      const makeCode = () =>
        Array.from({ length: 6 }, () => chars[Math.floor(Math.random() * chars.length)]).join("");

      let studyCode = makeCode();
      let collision = await db.select({ id: studyParticipants.id })
        .from(studyParticipants).where(eq(studyParticipants.studyCode, studyCode)).limit(1);
      while (collision.length > 0) {
        studyCode = makeCode();
        collision = await db.select({ id: studyParticipants.id })
          .from(studyParticipants).where(eq(studyParticipants.studyCode, studyCode)).limit(1);
      }

      const [participant] = await db.insert(studyParticipants).values({
        userId, studyId, studyCode, consentVersion,
        consentSignedAt: new Date(),
        consentFullName: consentFullName ?? null,
        consentInitials: consentInitials ?? null,
        overnightEegConsent: overnightEegConsent ?? false,
        preferredMorningTime, preferredDaytimeTime, preferredEveningTime,
      }).returning();

      res.json({ studyCode: participant.studyCode, enrolledAt: participant.enrolledAt });
    } catch (error) {
      res.status(500).json({ message: "Failed to enroll in study" });
    }
  });

  // GET /api/study/status/:userId — enrollment state + today's session progress
  app.get("/api/study/status/:userId", async (req, res) => {
    try {
      const participant = await getActiveParticipant(req.params.userId);
      if (!participant) return res.json({ enrolled: false });

      const today = await (async () => {
        const todayStart = new Date();
        todayStart.setHours(0, 0, 0, 0);
        const tomorrowStart = new Date(todayStart);
        tomorrowStart.setDate(tomorrowStart.getDate() + 1);
        const [s] = await db.select().from(studySessions)
          .where(and(
            eq(studySessions.participantId, participant.id),
            gte(studySessions.sessionDate, todayStart),
            lt(studySessions.sessionDate, tomorrowStart)
          )).limit(1);
        return s ?? null;
      })();

      res.json({
        enrolled: true,
        studyCode: participant.studyCode,
        completedDays: participant.completedDays,
        targetDays: participant.targetDays,
        todaySession: today,
        preferredTimes: {
          morning: participant.preferredMorningTime,
          daytime: participant.preferredDaytimeTime,
          evening: participant.preferredEveningTime,
        },
      });
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch study status" });
    }
  });

  // POST /api/study/morning — dream journal + welfare check
  app.post("/api/study/morning", async (req, res) => {
    try {
      const { userId, dreamText, noRecall, dreamValence, dreamArousal,
              nightmareFlag, sleepQuality, sleepHours,
              minutesFromWaking, currentMoodRating } = req.body;

      const participant = await getActiveParticipant(userId);
      if (!participant) return res.status(404).json({ message: "Not enrolled in an active study" });

      const session = await getOrCreateTodaySession(participant);

      if (session.morningCompleted) {
        return res.status(409).json({ message: "Morning entry already submitted today" });
      }

      await db.insert(studyMorningEntries).values({
        sessionId: session.id,
        studyCode: participant.studyCode,
        dreamText: noRecall ? null : (dreamText ?? null),
        noRecall: noRecall ?? false,
        dreamValence, dreamArousal, nightmareFlag,
        sleepQuality, sleepHours, minutesFromWaking, currentMoodRating,
      });

      await db.update(studySessions)
        .set({ morningCompleted: true })
        .where(eq(studySessions.id, session.id));

      await checkAndMarkValidDay(session.id);

      // Welfare flag: if mood ≤ 2, prompt resources in response
      const needsSupport = typeof currentMoodRating === "number" && currentMoodRating <= 2;

      res.json({ success: true, dayNumber: session.dayNumber, needsSupport });
    } catch (error) {
      res.status(500).json({ message: "Failed to submit morning entry" });
    }
  });

  // POST /api/study/daytime — EEG session + PANAS + mood ratings
  app.post("/api/study/daytime", async (req, res) => {
    try {
      const { userId, eegFeatures, faa, highBeta, fmt, sqiMean, eegDurationSec,
              samValence, samArousal, samStress, panasItems,
              sleepHoursReported, caffeineServings, significantEventYN } = req.body;

      const participant = await getActiveParticipant(userId);
      if (!participant) return res.status(404).json({ message: "Not enrolled in an active study" });

      const session = await getOrCreateTodaySession(participant);

      if (session.daytimeCompleted) {
        return res.status(409).json({ message: "Daytime entry already submitted today" });
      }

      await db.insert(studyDaytimeEntries).values({
        sessionId: session.id,
        studyCode: participant.studyCode,
        eegFeatures, faa, highBeta, fmt, sqiMean, eegDurationSec,
        samValence, samArousal, samStress, panasItems,
        sleepHoursReported, caffeineServings, significantEventYN,
      });

      await db.update(studySessions)
        .set({ daytimeCompleted: true })
        .where(eq(studySessions.id, session.id));

      await checkAndMarkValidDay(session.id);

      res.json({ success: true, dayNumber: session.dayNumber });
    } catch (error) {
      res.status(500).json({ message: "Failed to submit daytime entry" });
    }
  });

  // POST /api/study/evening — eating behavior + day summary
  app.post("/api/study/evening", async (req, res) => {
    try {
      const { userId, dayValence, dayArousal, peakEmotionIntensity,
              peakEmotionDirection, meals, emotionalEatingDay,
              cravingsToday, cravingTypes, exerciseLevel, alcoholDrinks,
              supplementsTaken, medicationsTaken, medicationsDetails,
              stressRightNow, readyForSleep } = req.body;

      const participant = await getActiveParticipant(userId);
      if (!participant) return res.status(404).json({ message: "Not enrolled in an active study" });

      const session = await getOrCreateTodaySession(participant);

      if (session.eveningCompleted) {
        return res.status(409).json({ message: "Evening entry already submitted today" });
      }

      await db.insert(studyEveningEntries).values({
        sessionId: session.id,
        studyCode: participant.studyCode,
        dayValence, dayArousal, peakEmotionIntensity, peakEmotionDirection,
        meals, emotionalEatingDay, cravingsToday, cravingTypes,
        exerciseLevel, alcoholDrinks, supplementsTaken, medicationsTaken,
        medicationsDetails, stressRightNow, readyForSleep,
      });

      await db.update(studySessions)
        .set({ eveningCompleted: true })
        .where(eq(studySessions.id, session.id));

      const isValid = await checkAndMarkValidDay(session.id);

      // Evening closes the day — increment completedDays when validDay is reached
      if (isValid) {
        await db.update(studyParticipants)
          .set({ completedDays: sql`${studyParticipants.completedDays} + 1` })
          .where(eq(studyParticipants.id, participant.id));
      }

      const finalDays = (participant.completedDays ?? 0) + (isValid ? 1 : 0);
      res.json({
        success: true,
        validDay: isValid,
        completedDays: finalDays,
      });
    } catch (error) {
      res.status(500).json({ message: "Failed to submit evening entry" });
    }
  });

  // GET /api/study/history/:userId — all sessions for calendar view
  app.get("/api/study/history/:userId", async (req, res) => {
    try {
      const participant = await getActiveParticipant(req.params.userId);
      if (!participant) return res.json([]);

      const sessions = await db.select().from(studySessions)
        .where(eq(studySessions.participantId, participant.id))
        .orderBy(asc(studySessions.dayNumber));

      res.json(sessions);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch study history" });
    }
  });

  // POST /api/study/withdraw — participant exits the study
  app.post("/api/study/withdraw", async (req, res) => {
    try {
      const { userId } = req.body;
      if (!userId) return res.status(400).json({ message: "userId is required" });

      const participant = await getActiveParticipant(userId);
      if (!participant) return res.status(404).json({ message: "No active study enrollment found" });

      await db.update(studyParticipants)
        .set({ status: "withdrawn", withdrawnAt: new Date() })
        .where(eq(studyParticipants.id, participant.id));

      res.json({
        daysCompleted: participant.completedDays ?? 0,
        message: "You have been withdrawn from the study. Thank you for your contribution.",
      });
    } catch (error) {
      res.status(500).json({ message: "Failed to process withdrawal" });
    }
  });

  // ── Food photo log ─────────────────────────────────────────────────────────

  // POST /api/food/analyze — analyze a meal (photo OR text description) with GPT-5, store the log
  app.post("/api/food/analyze", async (req, res) => {
    try {
      const { userId, imageBase64, textDescription, mealType, moodBefore, notes } = req.body;
      if (!userId) return res.status(400).json({ message: "userId required" });
      if (!imageBase64 && !textDescription) return res.status(400).json({ message: "imageBase64 or textDescription required" });
      if (!openai) return res.status(503).json({ message: "OpenAI not configured" });

      const JSON_SCHEMA = `{
  "foodItems": [{"name":"...","portion":"...","calories":0,"carbs_g":0,"protein_g":0,"fat_g":0}],
  "totalCalories": 0,
  "dominantMacro": "carbs|protein|fat|balanced",
  "glycemicImpact": "low|medium|high",
  "moodImpact": "2-sentence prediction: how this meal typically affects mood/energy 2-4 hours later",
  "dreamRelevance": "2-sentence note: how this nutrition may affect tonight's sleep depth or dream vividness",
  "summary": "One plain-English sentence describing what was eaten",
  "vitamins": {"vitamin_d_mcg":0,"vitamin_b12_mcg":0,"vitamin_c_mg":0,"iron_mg":0,"magnesium_mg":0,"zinc_mg":0,"omega3_g":0}
}`;

      // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      let response;
      if (imageBase64) {
        // Vision path — analyze a photo (include user description if provided for better accuracy)
        const textPrompt = textDescription
          ? `Analyze this food photo. The user describes it as: "${textDescription}". Return ONLY valid JSON (no markdown fences) with this exact shape:\n${JSON_SCHEMA}`
          : `Analyze this food photo. Return ONLY valid JSON (no markdown fences) with this exact shape:\n${JSON_SCHEMA}`;
        response = await openai.chat.completions.create({
          model: "gpt-5",
          messages: [{
            role: "user",
            content: [
              { type: "text", text: textPrompt },
              { type: "image_url", image_url: { url: `data:image/jpeg;base64,${imageBase64}`, detail: "low" } },
            ],
          }],
          max_completion_tokens: 8000,
        });
      } else {
        // Text path — analyze a written description
        response = await openai.chat.completions.create({
          model: "gpt-5",
          messages: [{
            role: "user",
            content: `The user describes their ${mealType ?? "meal"}: "${textDescription}"\n\nEstimate nutrition and return ONLY valid JSON (no markdown fences) with this exact shape:\n${JSON_SCHEMA}`,
          }],
          max_completion_tokens: 8000,
        });
      }

      logger.info({ finish_reason: response.choices[0].finish_reason, refusal: response.choices[0].message.refusal }, "GPT response received");
      const raw = response.choices[0].message.content ?? "{}";
      let analysis: Record<string, unknown>;
      try {
        analysis = JSON.parse(raw);
      } catch {
        // GPT sometimes wraps in markdown despite instructions — strip fences
        const stripped = raw.replace(/```json?\n?/g, "").replace(/```/g, "").trim();
        try {
          analysis = JSON.parse(stripped);
        } catch (parseErr) {
          logger.error({ raw }, "Food analyze — GPT response parse failed");
          throw parseErr;
        }
      }

      const [log] = await db.insert(foodLogs).values({
        userId,
        mealType: mealType ?? "snack",
        foodItems: analysis.foodItems as object[],
        totalCalories: typeof analysis.totalCalories === "number" ? analysis.totalCalories : null,
        dominantMacro: typeof analysis.dominantMacro === "string" ? analysis.dominantMacro : null,
        glycemicImpact: typeof analysis.glycemicImpact === "string" ? analysis.glycemicImpact : null,
        aiMoodImpact: typeof analysis.moodImpact === "string" ? analysis.moodImpact : null,
        aiDreamRelevance: typeof analysis.dreamRelevance === "string" ? analysis.dreamRelevance : null,
        summary: typeof analysis.summary === "string" ? analysis.summary : null,
        moodBefore: moodBefore ?? null,
        notes: notes ?? null,
      }).returning();

      // Push nutrition metrics to health pipeline (best-effort)
      const totalCalories = typeof analysis.totalCalories === "number" ? analysis.totalCalories : 0;
      const items = Array.isArray(analysis.foodItems) ? analysis.foodItems : [];
      const totalProtein = items.reduce((s: number, f: any) => s + (f?.protein_g ?? 0), 0);
      const totalCarbs = items.reduce((s: number, f: any) => s + (f?.carbs_g ?? 0), 0);
      const totalFat = items.reduce((s: number, f: any) => s + (f?.fat_g ?? 0), 0);
      const supabaseUrl = process.env.SUPABASE_URL;
      if (supabaseUrl && totalCalories > 0) {
        try {
          const now = new Date().toISOString();
          const samples: { source: string; metric: string; value: number; unit: string; recorded_at: string }[] = [
            { source: "manual", metric: "total_calories", value: totalCalories, unit: "kcal", recorded_at: now },
          ];
          if (totalProtein > 0) samples.push({ source: "manual", metric: "total_protein_g", value: totalProtein, unit: "g", recorded_at: now });
          if (totalCarbs > 0) samples.push({ source: "manual", metric: "total_carbs_g", value: totalCarbs, unit: "g", recorded_at: now });
          if (totalFat > 0) samples.push({ source: "manual", metric: "total_fat_g", value: totalFat, unit: "g", recorded_at: now });
          await fetch(`${supabaseUrl}/functions/v1/ingest-health-data`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: userId, samples }),
          });
        } catch (e) { logger.error({ error: e instanceof Error ? e.message : String(e) }, "Pipeline push failed (non-fatal)"); }
      }

      // Also insert into health_samples table directly (dual-write for reliability)
      if (totalCalories > 0) {
        try {
          const now = new Date();
          await db.insert(healthSamples).values({
            userId, source: "manual", metric: "total_calories", value: totalCalories, unit: "kcal", recordedAt: now,
          }).onConflictDoNothing();
          if (totalProtein > 0) {
            await db.insert(healthSamples).values({
              userId, source: "manual", metric: "total_protein_g", value: totalProtein, unit: "g", recordedAt: now,
            }).onConflictDoNothing();
          }
          if (totalCarbs > 0) {
            await db.insert(healthSamples).values({
              userId, source: "manual", metric: "total_carbs_g", value: totalCarbs, unit: "g", recordedAt: now,
            }).onConflictDoNothing();
          }
          if (totalFat > 0) {
            await db.insert(healthSamples).values({
              userId, source: "manual", metric: "total_fat_g", value: totalFat, unit: "g", recordedAt: now,
            }).onConflictDoNothing();
          }
        } catch (e) { logger.error({ error: e instanceof Error ? e.message : String(e) }, "health_samples insert failed (non-fatal)"); }
      }

      res.json({ ...analysis, id: log.id, loggedAt: log.loggedAt });
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : "Unknown error";
      logger.error({ error: msg }, "Food analyze error");
      if (msg.includes("API key") || msg.includes("OPENAI_API_KEY")) {
        res.status(503).json({ message: "AI-powered meal suggestions require an active API key. Please check your configuration." });
      } else if (msg.includes("parse") || msg.includes("JSON")) {
        res.status(502).json({ message: "AI returned invalid response. Please try again." });
      } else {
        res.status(500).json({ message: `Food analysis failed: ${msg}` });
      }
    }
  });

  // POST /api/food/log — directly log a meal with pre-computed nutrition data (no GPT needed)
  // Used as fallback when /api/food/analyze fails (e.g. no API key)
  app.post("/api/food/log", async (req, res) => {
    try {
      const { userId, mealType, summary, totalCalories, dominantMacro, foodItems } = req.body;
      if (!userId) return res.status(400).json({ message: "userId required" });
      if (!foodItems || !Array.isArray(foodItems)) return res.status(400).json({ message: "foodItems required" });

      const [log] = await db.insert(foodLogs).values({
        userId,
        mealType: mealType ?? "snack",
        foodItems: foodItems as object[],
        totalCalories: typeof totalCalories === "number" ? totalCalories : null,
        dominantMacro: typeof dominantMacro === "string" ? dominantMacro : null,
        summary: typeof summary === "string" ? summary : null,
      }).returning();

      res.json({ id: log.id, loggedAt: log.loggedAt, totalCalories, summary });
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : "Unknown error";
      logger.error({ error: msg }, "Food log error");
      res.status(500).json({ message: `Food log failed: ${msg}` });
    }
  });

  // GET /api/food/logs/:userId — food logs with pagination (default 50, max 200)
  // No auth gate — data is scoped by userId (same pattern as /api/brain/history).
  // Native APK uses localStorage participant IDs without JWT sessions.
  app.get("/api/food/logs/:userId", async (req, res) => {
    try {
      const limit = Math.min(Math.max(parseInt((req.query.limit as string) || "50", 10), 1), 200);
      const offset = Math.max(parseInt((req.query.offset as string) || "0", 10), 0);
      const logs = await db.select().from(foodLogs)
        .where(eq(foodLogs.userId, req.params.userId))
        .orderBy(desc(foodLogs.loggedAt))
        .limit(limit)
        .offset(offset);
      res.json(logs);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch food logs" });
    }
  });

  // GET /api/meal-history/:userId — meal history shaped for MealHistory component
  app.get("/api/meal-history/:userId", async (req, res) => {
    try {
      const logs = await db.select().from(foodLogs)
        .where(eq(foodLogs.userId, req.params.userId))
        .orderBy(desc(foodLogs.loggedAt))
        .limit(50);
      const entries = logs.map(log => ({
        id: log.id,
        userId: log.userId,
        images: null,
        foodItems: log.foodItems as any[] | null,
        totalCalories: log.totalCalories,
        totalProtein: (log.foodItems as any[] | null)?.reduce((s: number, f: any) => s + (f?.protein_g ?? 0), 0) ?? null,
        totalCarbs: (log.foodItems as any[] | null)?.reduce((s: number, f: any) => s + (f?.carbs_g ?? 0), 0) ?? null,
        totalFat: (log.foodItems as any[] | null)?.reduce((s: number, f: any) => s + (f?.fat_g ?? 0), 0) ?? null,
        totalFiber: (log.foodItems as any[] | null)?.reduce((s: number, f: any) => s + (f?.fiber_g ?? 0), 0) ?? null,
        mealType: log.mealType,
        isFavorite: false,
        createdAt: log.loggedAt?.toISOString() ?? new Date().toISOString(),
      }));
      res.json(entries);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch meal history" });
    }
  });

  // PATCH /api/meal-history/:id/favorite — toggle favorite (no-op for now, stored client-side)
  app.patch("/api/meal-history/:id/favorite", async (_req, res) => {
    res.json({ ok: true });
  });

  // GET /api/research/correlation/:userId — last 7 days with food + EEG mood + dream data joined
  app.get("/api/research/correlation/:userId", async (req, res) => {
    if (!requireOwnerExpress(req, res)) return;
    try {
      const userId = req.params.userId;

      const [participant] = await db.select().from(studyParticipants)
        .where(and(eq(studyParticipants.userId, userId), eq(studyParticipants.status, "active")))
        .limit(1);

      if (!participant) return res.json([]);

      const sessions = await db.select().from(studySessions)
        .where(eq(studySessions.participantId, participant.id))
        .orderBy(desc(studySessions.dayNumber))
        .limit(7);

      const results = await Promise.all(sessions.map(async (session) => {
        const dayStart = new Date(session.sessionDate);
        dayStart.setHours(0, 0, 0, 0);
        const dayEnd = new Date(dayStart);
        dayEnd.setDate(dayEnd.getDate() + 1);

        const [morning] = await db.select().from(studyMorningEntries)
          .where(eq(studyMorningEntries.sessionId, session.id)).limit(1);

        const [daytime] = await db.select().from(studyDaytimeEntries)
          .where(eq(studyDaytimeEntries.sessionId, session.id)).limit(1);

        const foods = await db.select().from(foodLogs)
          .where(and(
            eq(foodLogs.userId, userId),
            gte(foodLogs.loggedAt, dayStart),
            lt(foodLogs.loggedAt, dayEnd),
          ))
          .orderBy(asc(foodLogs.loggedAt));

        return {
          dayNumber: session.dayNumber,
          sessionDate: session.sessionDate,
          validDay: session.validDay,
          morning: morning ? {
            dreamValence: morning.dreamValence,
            noRecall: morning.noRecall,
            nightmareFlag: morning.nightmareFlag,
            dreamSnippet: morning.dreamText
              ? morning.dreamText.slice(0, 120) + (morning.dreamText.length > 120 ? "…" : "")
              : null,
            welfareScore: morning.currentMoodRating,
          } : null,
          daytime: daytime ? {
            samValence: daytime.samValence,
            samStress: daytime.samStress,
            faa: daytime.faa,
          } : null,
          foods: foods.map(f => ({
            id: f.id,
            summary: f.summary,
            mealType: f.mealType,
            totalCalories: f.totalCalories,
            dominantMacro: f.dominantMacro,
            glycemicImpact: f.glycemicImpact,
            aiMoodImpact: f.aiMoodImpact,
            aiDreamRelevance: f.aiDreamRelevance,
            loggedAt: f.loggedAt,
          })),
        };
      }));

      res.json(results);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch correlation data" });
    }
  });

  // ── Push notifications ────────────────────────────────────────────────────

  /** Return VAPID public key so client can subscribe */
  app.get("/api/notifications/vapid-public-key", (_req, res) => {
    if (!VAPID_PUBLIC) return res.status(503).json({ error: "VAPID not configured" });
    res.json({ publicKey: VAPID_PUBLIC });
  });

  /** Send a push notification to all subscribers (or a specific userId) */
  app.post("/api/notifications/send", async (req, res) => {
    if (!VAPID_PUBLIC || !VAPID_PRIVATE) {
      return res.status(503).json({ error: "VAPID not configured" });
    }
    const { userId, title, body, url } = req.body as {
      userId?: string; title?: string; body?: string; url?: string;
    };
    try {
      const query = userId
        ? db.select().from(pushSubscriptions).where(eq(pushSubscriptions.userId, userId))
        : db.select().from(pushSubscriptions);
      const subs = await query;

      const payload = JSON.stringify({
        title: title ?? "AntarAI",
        body:  body  ?? "Your morning brain report is ready.",
        url:   url   ?? "/brain-report",
        tag:   "ndw-morning",
      });

      const results = await Promise.allSettled(
        subs.map((s) =>
          webpush.sendNotification(
            { endpoint: s.endpoint, keys: s.keys as { p256dh: string; auth: string } },
            payload
          )
        )
      );
      const sent   = results.filter((r) => r.status === "fulfilled").length;
      const failed = results.filter((r) => r.status === "rejected").length;
      res.json({ sent, failed });
    } catch (err) {
      res.status(500).json({ error: "Failed to send notifications" });
    }
  });

  // ── Just-in-time brain-state push trigger ─────────────────────────────────
  // Per-user cooldown map: userId → last fire timestamp (ms)
  const _pushCooldown = new Map<string, number>();
  const PUSH_COOLDOWN_MS = 15 * 60 * 1000; // 15 minutes per user

  /**
   * POST /api/notifications/brain-state-trigger
   * Called by the frontend after each EEG batch (every ~15s when streaming).
   * Fires a push notification if:
   *   stress >= 0.70  → "High stress detected — try a breathing exercise"
   *   focus  <= 0.25  → "Focus is fading — a short walk might help"
   * Cooldown: 15 minutes per user so notifications don't spam.
   */
  app.post("/api/notifications/brain-state-trigger", async (req, res) => {
    if (!VAPID_PUBLIC || !VAPID_PRIVATE) {
      return res.status(503).json({ error: "VAPID not configured" });
    }
    const { userId, stress, focus } = req.body as {
      userId?: string; stress?: number; focus?: number;
    };
    if (!userId || stress == null || focus == null) {
      return res.status(400).json({ error: "userId, stress, and focus are required" });
    }

    // Rate limit
    const lastFire = _pushCooldown.get(userId) ?? 0;
    if (Date.now() - lastFire < PUSH_COOLDOWN_MS) {
      return res.json({ triggered: false, reason: "cooldown" });
    }

    // Determine if a notification is warranted
    let title: string | null = null;
    let body: string | null  = null;
    let url  = "/biofeedback";

    if (stress >= 0.70) {
      title = "High stress detected";
      body  = "Your stress is elevated. A 4-minute breathing exercise can help.";
      url   = "/biofeedback?protocol=coherence&auto=true";
    } else if (focus <= 0.25) {
      title = "Focus is fading";
      body  = "Your concentration is dropping. A short walk or music break may help.";
      url   = "/biofeedback?tab=music&mood=focus";
    }

    if (!title) {
      return res.json({ triggered: false, reason: "thresholds not met" });
    }

    try {
      const subs = await db
        .select()
        .from(pushSubscriptions)
        .where(eq(pushSubscriptions.userId, userId));

      if (subs.length === 0) {
        return res.json({ triggered: false, reason: "no subscriptions" });
      }

      const payload = JSON.stringify({ title, body, url, tag: "ndw-brain-state" });
      await Promise.allSettled(
        subs.map((s) =>
          webpush.sendNotification(
            { endpoint: s.endpoint, keys: s.keys as { p256dh: string; auth: string } },
            payload
          )
        )
      );
      _pushCooldown.set(userId, Date.now());
      res.json({ triggered: true, title, body, url });
    } catch (err) {
      res.status(500).json({ error: "Failed to send push notification" });
    }
  });

  // ── Native push token registration (APNs / FCM) ────────────────────────────
  // Store the FCM/APNs device token per user so the server can send
  // native pushes when the app is backgrounded.

  /** In-memory token store (process restart clears it — acceptable for MVP) */
  const _nativeTokens = new Map<string, { token: string; platform: string }>();

  app.post("/api/notifications/native-token", (req, res) => {
    const { userId, token, platform } = req.body as {
      userId?: string; token?: string; platform?: string;
    };
    if (!userId || !token) {
      return res.status(400).json({ error: "userId and token required" });
    }
    _nativeTokens.set(userId, { token, platform: platform ?? "unknown" });
    res.json({ registered: true });
  });

  /**
   * Send a native push notification to a specific user via FCM.
   * Requires FIREBASE_SERVICE_ACCOUNT_KEY env var (JSON string).
   * Falls back gracefully if Firebase is not configured.
   */
  app.post("/api/notifications/send-native", async (req, res) => {
    const { userId, title, body, data } = req.body as {
      userId?: string; title?: string; body?: string; data?: Record<string, string>;
    };
    if (!userId || !title || !body) {
      return res.status(400).json({ error: "userId, title, body required" });
    }

    const tokenInfo = _nativeTokens.get(userId);
    if (!tokenInfo) {
      return res.status(404).json({ error: "No native token registered for user" });
    }

    const serviceAccountJson = process.env.FIREBASE_SERVICE_ACCOUNT_KEY;
    if (!serviceAccountJson) {
      // Firebase not configured — log and return graceful response
      logger.warn({ userId, title }, "[native-push] FIREBASE_SERVICE_ACCOUNT_KEY not set");
      return res.json({ sent: false, reason: "firebase_not_configured" });
    }

    try {
      // Dynamic import to avoid requiring firebase-admin when not needed
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const adminImport: Promise<any> = new Function('m', 'return import(m)')("firebase-admin");
      const admin = await adminImport.catch(() => null);
      if (!admin) {
        return res.json({ sent: false, reason: "firebase-admin not installed" });
      }

      // Initialize only once
      if (!admin.default.apps.length) {
        const serviceAccount = JSON.parse(serviceAccountJson);
        admin.default.initializeApp({
          credential: admin.default.credential.cert(serviceAccount),
        });
      }

      const message = {
        token: tokenInfo.token,
        notification: { title, body },
        data: data ?? {},
        apns: { payload: { aps: { badge: 1, sound: "default" } } },
        android: { priority: "high" as const, notification: { sound: "default" } },
      };

      const result = await admin.default.messaging().send(message);
      res.json({ sent: true, messageId: result });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), userId }, "[native-push] FCM error");
      res.status(500).json({ error: "Failed to send native push" });
    }
  });

  // ── Emotion corrections (user label feedback → online learner) ────────────

  app.post("/api/emotions/correct", async (req, res) => {
    const { userId, detectedEmotion, correctedEmotion, confidence } = req.body as {
      userId: string; detectedEmotion: string; correctedEmotion: string; confidence?: number;
    };
    if (!userId || !correctedEmotion) {
      return res.status(400).json({ error: "userId and correctedEmotion are required" });
    }
    // Store as a new emotion reading with the corrected label
    try {
      await db.insert(emotionReadings).values({
        userId,
        dominantEmotion: correctedEmotion,
        stress: 0,
        focus: 0,
        happiness: 0,
        energy: 0,
        valence: 0,
        arousal: 0,
        timestamp: new Date(),
        eegSnapshot: { userCorrected: true, detectedEmotion, confidence: confidence ?? 0 },
      });
    } catch { /* best-effort — non-blocking */ }

    // Count this user's corrections; trigger fine-tuning every 5th
    try {
      const [{ count }] = await db
        .select({ count: sql<number>`count(*)::int` })
        .from(emotionReadings)
        .where(
          and(
            eq(emotionReadings.userId, userId),
            sql`(eeg_snapshot->>'userCorrected')::text = 'true'`
          )
        );
      if (count > 0 && count % 5 === 0) {
        const mlBase = process.env.ML_BACKEND_URL ?? "http://localhost:8000";
        fetch(`${mlBase}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id: userId, detected: detectedEmotion, corrected: correctedEmotion }),
          signal: AbortSignal.timeout(3000),
        }).catch(() => {});
      }
    } catch { /* best-effort — non-blocking */ }

    res.json({ ok: true });
  });

  // PATCH /api/emotions/correct-latest/:userId
  // Called by emotion-lab.tsx correction UI — updates the most recent emotion
  // reading for the user and forwards the label to the ML backend for learning.
  app.patch("/api/emotions/correct-latest/:userId", async (req, res) => {
    const { userId } = req.params;
    const { userCorrectedEmotion } = req.body as { userCorrectedEmotion: string };
    if (!userId || !userCorrectedEmotion) {
      return res.status(400).json({ error: "userId and userCorrectedEmotion are required" });
    }

    // Find the most recent reading for this user
    let detectedEmotion = "unknown";
    try {
      const [latest] = await db
        .select({ id: emotionReadings.id, dominantEmotion: emotionReadings.dominantEmotion })
        .from(emotionReadings)
        .where(eq(emotionReadings.userId, userId))
        .orderBy(sql`timestamp desc`)
        .limit(1);

      if (latest) {
        detectedEmotion = latest.dominantEmotion;
        // Update in place
        await db
          .update(emotionReadings)
          .set({
            userCorrectedEmotion,
            userCorrectedAt: new Date(),
          })
          .where(eq(emotionReadings.id, latest.id));
      } else {
        // No prior reading — insert a stub so correction is at least recorded
        await db.insert(emotionReadings).values({
          userId,
          dominantEmotion: userCorrectedEmotion,
          stress: 0, focus: 0, happiness: 0, energy: 0, valence: 0, arousal: 0,
          userCorrectedEmotion,
          userCorrectedAt: new Date(),
          timestamp: new Date(),
          eegSnapshot: { userCorrected: true },
        });
      }
    } catch { /* best-effort */ }

    // Forward to ML backend for online learning (fire-and-forget)
    try {
      const mlBase = process.env.ML_BACKEND_URL ?? "http://localhost:8000";
      fetch(`${mlBase}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          predicted_label: detectedEmotion,
          correct_label: userCorrectedEmotion,
        }),
        signal: AbortSignal.timeout(3000),
      }).catch(() => {});
    } catch { /* best-effort */ }

    res.json({ ok: true });
  });

  // ── Daily 8 am morning brain report push ─────────────────────────────────

  if (VAPID_PUBLIC && VAPID_PRIVATE) {
    // "0 8 * * *" = every day at 08:00 server local time
    cron.schedule("0 8 * * *", async () => {
      try {
        const subs = await db.select().from(pushSubscriptions);
        const payload = JSON.stringify({
          title: "Morning Brain Report",
          body:  "Your sleep summary, focus forecast, and recommended action are ready.",
          url:   "/brain-report",
          tag:   "ndw-morning",
        });
        await Promise.allSettled(
          subs.map((s) =>
            webpush.sendNotification(
              { endpoint: s.endpoint, keys: s.keys as { p256dh: string; auth: string } },
              payload
            )
          )
        );
      } catch { /* log silently — cron must not crash the server */ }
    });
  }

  // ── Spotify integration ──────────────────────────────────────────────────
  //
  // Requires env vars: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI
  // Gracefully disabled when vars are missing.
  //
  // Endpoints:
  //   GET  /api/spotify/auth            → redirect to Spotify login page
  //   GET  /api/spotify/callback        → OAuth callback, stores token in session
  //   GET  /api/spotify/status          → { connected: bool, username? }
  //   POST /api/spotify/play            → play a URI or mood (calm/focus/sleep) on active device
  //   POST /api/spotify/disconnect      → revoke stored token

  const SPOTIFY_CLIENT_ID     = process.env.SPOTIFY_CLIENT_ID ?? "";
  const SPOTIFY_CLIENT_SECRET = process.env.SPOTIFY_CLIENT_SECRET ?? "";
  const SPOTIFY_REDIRECT_URI  = process.env.SPOTIFY_REDIRECT_URI ?? "http://localhost:5000/api/spotify/callback";
  const SPOTIFY_ENABLED       = !!(SPOTIFY_CLIENT_ID && SPOTIFY_CLIENT_SECRET);

  // Curated playlist URIs (same as biofeedback music tab, stored as Spotify URIs)
  const MOOD_PLAYLISTS: Record<string, string[]> = {
    calm: [
      "spotify:playlist:37i9dQZF1DX4sWSpwq3LiO",  // Peaceful Piano
      "spotify:playlist:37i9dQZF1DWN1Y7lu9lY4v",  // Deep Sleep
      "spotify:playlist:37i9dQZF1DWWQRwui0ExPn",  // Calming Acoustic
    ],
    focus: [
      "spotify:playlist:37i9dQZF1DX0SM0LYsmbMT",  // Focus Flow
      "spotify:playlist:37i9dQZF1DWXLeA8Omikj7",  // Brain Food
      "spotify:playlist:37i9dQZF1DWUZ5bk6qqDSy",  // Brown Noise
    ],
    sleep: [
      "spotify:playlist:37i9dQZF1DWN1Y7lu9lY4v",  // Deep Sleep
      "spotify:playlist:37i9dQZF1DX4sWSpwq3LiO",  // Peaceful Piano
    ],
  };

  // Per-session Spotify token store (userId → SpotifyWebApi instance)
  const _spotifyClients = new Map<string, SpotifyWebApi>();

  function makeSpotifyApi(accessToken?: string, refreshToken?: string): SpotifyWebApi {
    const api = new SpotifyWebApi({
      clientId: SPOTIFY_CLIENT_ID,
      clientSecret: SPOTIFY_CLIENT_SECRET,
      redirectUri: SPOTIFY_REDIRECT_URI,
    });
    if (accessToken)  api.setAccessToken(accessToken);
    if (refreshToken) api.setRefreshToken(refreshToken);
    return api;
  }

  async function refreshIfNeeded(api: SpotifyWebApi): Promise<void> {
    if (!api.getRefreshToken()) return;
    try {
      const data = await api.refreshAccessToken();
      api.setAccessToken(data.body.access_token);
    } catch { /* token may still be valid — ignore */ }
  }

  // GET /api/spotify/auth — redirect user to Spotify OAuth page
  app.get("/api/spotify/auth", (req, res) => {
    if (!SPOTIFY_ENABLED) {
      return res.status(503).json({ error: "Spotify not configured. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET." });
    }
    const api = makeSpotifyApi();
    const scopes = [
      "user-read-playback-state",
      "user-modify-playback-state",
      "streaming",
      "user-read-email",
      "user-read-private",
    ];
    const url = api.createAuthorizeURL(scopes, "neural-dream-state");
    res.redirect(url);
  });

  // GET /api/spotify/callback — Spotify redirects here with ?code=...
  app.get("/api/spotify/callback", async (req, res) => {
    const { code, error } = req.query as { code?: string; error?: string };

    if (error || !code) {
      return res.redirect("/?spotify=error");
    }

    try {
      const api = makeSpotifyApi();
      const data = await api.authorizationCodeGrant(code);
      api.setAccessToken(data.body.access_token);
      api.setRefreshToken(data.body.refresh_token);

      // Get user profile to identify them
      const me = await api.getMe();
      const spotifyUserId = me.body.id;

      // Map session user → Spotify client
      const sessionUserId = (req.session as { userId?: string }).userId ?? spotifyUserId;
      _spotifyClients.set(sessionUserId, api);

      // Store tokens in session so they survive server restart (lightweight)
      (req.session as any).spotifyAccessToken  = data.body.access_token;
      (req.session as any).spotifyRefreshToken = data.body.refresh_token;
      (req.session as any).spotifyUsername     = me.body.display_name ?? me.body.id;

      res.redirect("/biofeedback?tab=music&spotify=connected");
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err) }, "Spotify OAuth error");
      res.redirect("/?spotify=error");
    }
  });

  // GET /api/spotify/status
  app.get("/api/spotify/status", (req, res) => {
    const sess = req.session as any;
    const connected = !!(sess.spotifyAccessToken);
    res.json({
      connected,
      enabled: SPOTIFY_ENABLED,
      username: connected ? sess.spotifyUsername : null,
    });
  });

  // POST /api/spotify/play — play a mood or specific URI
  // Body: { userId?, mood?: "calm"|"focus"|"sleep", uri?: string }
  app.post("/api/spotify/play", async (req, res) => {
    const sess = req.session as any;
    if (!sess.spotifyAccessToken) {
      return res.status(401).json({ error: "not_connected", authUrl: "/api/spotify/auth" });
    }

    const { mood, uri } = req.body as { mood?: string; uri?: string };

    // Restore or create API client from session tokens
    const sessionUserId = (req.session as { userId?: string }).userId;
    if (!sessionUserId) {
      return res.status(401).json({ error: "not_connected", authUrl: "/api/spotify/auth" });
    }
    let api = _spotifyClients.get(sessionUserId);
    if (!api) {
      api = makeSpotifyApi(
        sess.spotifyAccessToken as string,
        sess.spotifyRefreshToken as string
      );
      _spotifyClients.set(sessionUserId, api);
    }

    try {
      await refreshIfNeeded(api);

      // Resolve URI to play
      let targetUri = uri;
      if (!targetUri && mood) {
        const options = MOOD_PLAYLISTS[mood] ?? MOOD_PLAYLISTS.calm;
        targetUri = options[Math.floor(Math.random() * options.length)];
      }
      if (!targetUri) targetUri = MOOD_PLAYLISTS.calm[0];

      // Start playback on active device (shuffle on for playlists)
      await api.setShuffle(true);
      await api.play({ context_uri: targetUri });

      res.json({ ok: true, playing: targetUri, mood: mood ?? "calm" });
    } catch (err: unknown) {
      // Common error: no active device (user must have Spotify open somewhere)
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("NO_ACTIVE_DEVICE") || msg.includes("404")) {
        return res.status(409).json({
          error: "no_active_device",
          message: "Open Spotify on any device first, then try again.",
        });
      }
      logger.error({ error: msg }, "Spotify play error");
      res.status(500).json({ error: "playback_failed", message: msg });
    }
  });

  // POST /api/spotify/disconnect
  app.post("/api/spotify/disconnect", (req, res) => {
    const sess = req.session as any;
    const sessionUserId = (req.session as { userId?: string }).userId;
    if (sessionUserId) {
      _spotifyClients.delete(sessionUserId);
    }
    delete sess.spotifyAccessToken;
    delete sess.spotifyRefreshToken;
    delete sess.spotifyUsername;
    res.json({ ok: true });
  });

  // ── Pilot study routes (US-002) ───────────────────────────────────────────
  // These routes are independent of the 30-day longitudinal study routes above.
  // They operate on pilot_participants and pilot_sessions tables.

  // Auth guard reused for admin routes — checks login AND admin role
  async function requireStudyAdmin(
    req: import("express").Request,
    res: import("express").Response,
    next: import("express").NextFunction,
  ) {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });
    try {
      const [user] = await db.select({ username: users.username }).from(users)
        .where(eq(users.id, userId)).limit(1);
      if (!user || !ADMIN_USERNAMES.has(user.username?.toLowerCase())) {
        return res.status(403).json({ error: "Forbidden — admin access required" });
      }
      next();
    } catch {
      return res.status(500).json({ error: "Authorization check failed" });
    }
  }

  // GET /api/study/check-code — check if a participant code is available
  app.get("/api/study/check-code", async (req, res) => {
    try {
      const code = req.query.code as string | undefined;
      if (!code) {
        return res.status(400).json({ error: "code query parameter is required" });
      }
      const [existing] = await db
        .select({ id: pilotParticipants.id })
        .from(pilotParticipants)
        .where(eq(pilotParticipants.participantCode, code))
        .limit(1);
      return res.json({ available: !existing });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), path: "/api/study/check-code" }, "Study check-code failed");
      return res.status(500).json({ error: "Failed to check code availability" });
    }
  });

  // POST /api/study/consent
  app.post("/api/study/consent", async (req, res) => {
    try {
      const { participant_code, age, diet_type, has_apple_watch, consent_text } = req.body;
      if (!participant_code) {
        return res.status(400).json({ error: "participant_code is required" });
      }
      await db.insert(pilotParticipants).values({
        participantCode:  String(participant_code),
        age:              age != null ? Number(age) : null,
        dietType:         diet_type ?? null,
        hasAppleWatch:    has_apple_watch ? true : false,
        consentText:      consent_text ?? null,
        consentTimestamp: new Date(),
      }).onConflictDoUpdate({
        target: pilotParticipants.participantCode,
        set: {
          ...(age != null && { age: Number(age) }),
          ...(diet_type != null && { dietType: diet_type }),
          ...(has_apple_watch !== undefined && { hasAppleWatch: has_apple_watch ? true : false }),
        },
      });
      return res.json({ success: true, participant_code });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), path: "/api/study/consent" }, "Study consent failed");
      return res.status(500).json({ error: "Failed to save consent" });
    }
  });

  // POST /api/study/session/start
  app.post("/api/study/session/start", async (req, res) => {
    try {
      const { participant_code, block_type } = req.body;
      if (!participant_code || !block_type) {
        return res.status(400).json({ error: "participant_code and block_type are required" });
      }
      const [row] = await db.insert(pilotSessions).values({
        participantCode:       String(participant_code),
        blockType:             String(block_type),
        interventionTriggered: false,
        startedAt:             new Date(),
      }).returning({ id: pilotSessions.id });
      return res.json({ session_id: row.id });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), path: "/api/study/session/start" }, "Study session start failed");
      return res.status(500).json({ error: "Failed to start session" });
    }
  });

  // POST /api/study/session/complete
  app.post("/api/study/session/complete", async (req, res) => {
    try {
      const {
        session_id,
        pre_eeg_json,
        post_eeg_json,
        eeg_features_json,
        survey_json,
        intervention_triggered,
        phase_log,
        data_quality_score,
        voice_emotion_json,
        watch_biometrics_json,
      } = req.body;
      if (session_id == null || Number(session_id) <= 0) {
        return res.status(400).json({ error: "session_id is required" });
      }

      // Compute duration from startedAt
      let durationSeconds: number | null = null;
      const [existing] = await db.select({ startedAt: pilotSessions.startedAt })
        .from(pilotSessions)
        .where(eq(pilotSessions.id, Number(session_id)));
      if (existing?.startedAt) {
        durationSeconds = Math.floor((Date.now() - existing.startedAt.getTime()) / 1000);
      }

      await db.update(pilotSessions)
        .set({
          preEegJson:            pre_eeg_json ?? null,
          postEegJson:           post_eeg_json ?? null,
          eegFeaturesJson:       eeg_features_json ?? null,
          surveyJson:            survey_json ?? null,
          interventionTriggered: intervention_triggered ? true : false,
          partial:               false,
          ...(phase_log !== undefined && { phaseLog: phase_log }),
          ...(data_quality_score != null && { dataQualityScore: Number(data_quality_score) }),
          ...(durationSeconds != null && { durationSeconds }),
          ...(voice_emotion_json !== undefined && { voiceEmotionJson: voice_emotion_json }),
          ...(watch_biometrics_json !== undefined && { watchBiometricsJson: watch_biometrics_json }),
        })
        .where(eq(pilotSessions.id, Number(session_id)));
      return res.json({ success: true });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), path: "/api/study/session/complete" }, "Study session complete failed");
      return res.status(500).json({ error: "Failed to complete session" });
    }
  });

  // PATCH /api/study/session/:id/checkpoint
  app.patch("/api/study/session/:id/checkpoint", async (req, res) => {
    try {
      const sessionId = Number(req.params.id);
      if (isNaN(sessionId)) return res.status(400).json({ error: "invalid session id" });
      const { pre_eeg_json, post_eeg_json, eeg_features_json, intervention_triggered, partial, phase_log, voice_emotion_json, watch_biometrics_json } = req.body;
      await db.update(pilotSessions)
        .set({
          ...(pre_eeg_json !== undefined   && { preEegJson: pre_eeg_json }),
          ...(post_eeg_json !== undefined  && { postEegJson: post_eeg_json }),
          ...(eeg_features_json !== undefined && { eegFeaturesJson: eeg_features_json }),
          ...(intervention_triggered !== undefined && { interventionTriggered: !!intervention_triggered }),
          ...(partial !== undefined        && { partial: !!partial }),
          ...(phase_log !== undefined      && { phaseLog: phase_log }),
          ...(voice_emotion_json !== undefined && { voiceEmotionJson: voice_emotion_json }),
          ...(watch_biometrics_json !== undefined && { watchBiometricsJson: watch_biometrics_json }),
          checkpointAt: new Date(),
        })
        .where(eq(pilotSessions.id, sessionId));
      return res.json({ success: true });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), path: "/api/study/session/:id/checkpoint" }, "Study session checkpoint failed");
      return res.status(500).json({ error: "checkpoint failed" });
    }
  });

  // GET /api/user/intent
  app.get("/api/user/intent", async (req, res) => {
    const userId = (req.session as { userId?: string }).userId;
    if (!userId) return res.status(401).json({ error: "Unauthorized" });
    const [u] = await db.select({ intent: users.intent }).from(users).where(eq(users.id, userId));
    return res.json({ intent: u?.intent ?? null });
  });

  // PATCH /api/user/intent
  app.patch("/api/user/intent", async (req, res) => {
    const userId = (req.session as { userId?: string }).userId;
    if (!userId) return res.status(401).json({ error: "Unauthorized" });
    const { intent } = req.body;
    if (!["study", "explore"].includes(intent)) return res.status(400).json({ error: "invalid intent" });
    await db.update(users).set({ intent }).where(eq(users.id, userId));
    return res.json({ success: true, intent });
  });

  // GET /api/study/admin/participants
  app.get("/api/study/admin/participants", requireStudyAdmin, async (_req, res) => {
    try {
      const rows = await db.select().from(pilotParticipants).orderBy(desc(pilotParticipants.createdAt));
      return res.json(rows);
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), path: "/api/study/admin/participants" }, "Fetch participants failed");
      return res.status(500).json({ error: "Failed to fetch participants" });
    }
  });

  // PATCH /api/study/admin/participant/:code/notes
  app.patch("/api/study/admin/participant/:code/notes", requireStudyAdmin, async (req, res) => {
    try {
      const { notes } = req.body as { notes?: string };
      if (typeof notes !== "string") {
        return res.status(400).json({ error: "notes (string) is required" });
      }
      const [updated] = await db
        .update(pilotParticipants)
        .set({ researcherNotes: notes })
        .where(eq(pilotParticipants.participantCode, req.params.code))
        .returning({ id: pilotParticipants.id });
      if (!updated) {
        return res.status(404).json({ error: "Participant not found" });
      }
      return res.json({ ok: true });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), path: "/api/study/admin/participant/:code/notes" }, "Update researcher notes failed");
      return res.status(500).json({ error: "Failed to update researcher notes" });
    }
  });

  // GET /api/study/admin/sessions
  app.get("/api/study/admin/sessions", requireStudyAdmin, async (_req, res) => {
    try {
      const rows = await db.select().from(pilotSessions).orderBy(desc(pilotSessions.createdAt));
      return res.json(rows);
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), path: "/api/study/admin/sessions" }, "Fetch sessions failed");
      return res.status(500).json({ error: "Failed to fetch sessions" });
    }
  });

  // DELETE /api/study/admin/session/:id
  app.delete("/api/study/admin/session/:id", requireStudyAdmin, async (req, res) => {
    try {
      const id = Number(req.params.id);
      if (!Number.isFinite(id)) {
        return res.status(400).json({ error: "Invalid session id" });
      }
      const deleted = await db
        .delete(pilotSessions)
        .where(eq(pilotSessions.id, id))
        .returning({ id: pilotSessions.id });
      if (deleted.length === 0) {
        return res.status(404).json({ error: "Session not found" });
      }
      return res.json({ success: true, deleted_id: deleted[0].id });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), path: "/api/study/admin/session/:id" }, "Delete session failed");
      return res.status(500).json({ error: "Failed to delete session" });
    }
  });

  // GET /api/study/admin/stats
  app.get("/api/study/admin/stats", requireStudyAdmin, async (_req, res) => {
    try {
      const participants = await db.select().from(pilotParticipants);
      const sessions = await db.select().from(pilotSessions);

      const total_participants = participants.length;
      const total_sessions = sessions.length;
      const stress_sessions = sessions.filter((s) => s.blockType === "stress").length;
      const food_sessions = sessions.filter((s) => s.blockType === "food").length;
      const complete_sessions = sessions.filter((s) => !s.partial && s.surveyJson !== null).length;
      const partial_sessions = sessions.filter((s) => s.partial === true).length;

      const qualityScores = sessions
        .map((s) => s.dataQualityScore)
        .filter((v): v is number => v != null);
      const avg_quality_score =
        qualityScores.length > 0
          ? qualityScores.reduce((a, b) => a + b, 0) / qualityScores.length
          : 0;

      const durations = sessions
        .map((s) => s.durationSeconds)
        .filter((v): v is number => v != null);
      const avg_duration_seconds =
        durations.length > 0
          ? durations.reduce((a, b) => a + b, 0) / durations.length
          : 0;

      // avg_stress_reduction: for stress sessions with both pre and post EEG,
      // compute average of (pre.stress_level - post.stress_level)
      const stressReductions: number[] = [];
      for (const s of sessions) {
        if (s.blockType !== "stress") continue;
        const pre = s.preEegJson as Record<string, unknown> | null;
        const post = s.postEegJson as Record<string, unknown> | null;
        if (
          pre && typeof pre === "object" && typeof pre.stress_level === "number" &&
          post && typeof post === "object" && typeof post.stress_level === "number"
        ) {
          stressReductions.push(pre.stress_level - post.stress_level);
        }
      }
      const avg_stress_reduction =
        stressReductions.length > 0
          ? stressReductions.reduce((a, b) => a + b, 0) / stressReductions.length
          : 0;

      return res.json({
        total_participants,
        total_sessions,
        stress_sessions,
        food_sessions,
        complete_sessions,
        partial_sessions,
        avg_quality_score,
        avg_duration_seconds,
        avg_stress_reduction,
      });
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), path: "/api/study/admin/stats" }, "Compute stats failed");
      return res.status(500).json({ error: "Failed to compute stats" });
    }
  });

  // GET /api/study/admin/export-csv
  app.get("/api/study/admin/export-csv", requireStudyAdmin, async (_req, res) => {
    try {
      // Join sessions with participant demographics
      const rows = await db
        .select({
          participantCode:       pilotSessions.participantCode,
          blockType:             pilotSessions.blockType,
          age:                   pilotParticipants.age,
          dietType:              pilotParticipants.dietType,
          hasAppleWatch:         pilotParticipants.hasAppleWatch,
          interventionTriggered: pilotSessions.interventionTriggered,
          preEegJson:            pilotSessions.preEegJson,
          postEegJson:           pilotSessions.postEegJson,
          surveyJson:            pilotSessions.surveyJson,
          dataQualityScore:      pilotSessions.dataQualityScore,
          durationSeconds:       pilotSessions.durationSeconds,
          phaseLog:              pilotSessions.phaseLog,
        })
        .from(pilotSessions)
        .leftJoin(
          pilotParticipants,
          eq(pilotSessions.participantCode, pilotParticipants.participantCode),
        )
        .orderBy(desc(pilotSessions.createdAt));

      const EEG_BANDS = ["alpha", "beta", "theta", "delta", "gamma"] as const;

      // Collect all survey keys that appear across all rows (numeric values only)
      const surveyKeys = new Set<string>();
      for (const row of rows) {
        const s = row.surveyJson as Record<string, unknown> | null;
        if (s && typeof s === "object") {
          for (const [k, v] of Object.entries(s)) {
            if (typeof v === "number") surveyKeys.add(k);
          }
        }
      }
      const sortedSurveyKeys = Array.from(surveyKeys).sort();

      // Build CSV header
      const baseHeaders = [
        "participant_code", "block_type", "age", "diet_type", "has_apple_watch",
        "intervention_triggered",
        ...EEG_BANDS.map((b) => `pre_${b}`),
        ...EEG_BANDS.map((b) => `post_${b}`),
        ...sortedSurveyKeys,
        "data_quality_score",
        "duration_seconds",
        "phase_log",
      ];

      const csvLines: string[] = [baseHeaders.join(",")];

      for (const row of rows) {
        const pre = row.preEegJson as Record<string, unknown> | null;
        const post = row.postEegJson as Record<string, unknown> | null;
        const survey = row.surveyJson as Record<string, unknown> | null;

        const cell = (v: unknown): string => {
          if (v == null) return "";
          return String(v).replace(/,/g, ";");
        };

        const csvRow = [
          cell(row.participantCode),
          cell(row.blockType),
          cell(row.age),
          cell(row.dietType),
          cell(row.hasAppleWatch),
          cell(row.interventionTriggered),
          ...EEG_BANDS.map((b) => cell(pre?.[b])),
          ...EEG_BANDS.map((b) => cell(post?.[b])),
          ...sortedSurveyKeys.map((k) => cell(survey?.[k])),
          cell(row.dataQualityScore),
          cell(row.durationSeconds),
          cell(JSON.stringify(row.phaseLog ?? null)),
        ];
        csvLines.push(csvRow.join(","));
      }

      const csvString = csvLines.join("\n");
      const dateStr = new Date().toISOString().slice(0, 10);

      res.setHeader("Content-Type", "text/csv");
      res.setHeader("Content-Disposition", `attachment; filename="study-data-${dateStr}.csv"`);
      return res.send(csvString);
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err), path: "/api/study/admin/export-csv" }, "Export CSV failed");
      return res.status(500).json({ error: "Failed to export CSV" });
    }
  });

  // ── PHIA-style LLM health insights ───────────────────────────────────────
  // POST /api/health-insights
  // Queries last 7 days of EEG, emotion, and health data for the user,
  // builds a structured context string, and calls GPT-5 to generate
  // 3–5 personalized, cross-modal health insights.

  app.post("/api/health-insights", async (req, res) => {
    const { userId } = req.body as { userId?: string };
    if (!userId) {
      return res.status(400).json({ message: "userId is required" });
    }

    if (!openai) {
      return res.status(503).json({ message: "OPENAI_API_KEY not configured" });
    }

    const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    const avg = (arr: number[]): number | null =>
      arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
    const round2 = (n: number | null): number | null =>
      n != null ? Math.round(n * 100) / 100 : null;

    // ── 1. EEG emotion readings (last 7 days) ────────────────────────────
    let avgStress: number | null = null;
    let avgFocus: number | null = null;
    let avgValence: number | null = null;
    let dominantEmotions: string[] = [];
    let bestDay: string | null = null;
    let worstDay: string | null = null;
    let emotionReadingCount = 0;

    try {
      const readings = await db
        .select({
          stress: emotionReadings.stress,
          focus: emotionReadings.focus,
          valence: emotionReadings.valence,
          dominantEmotion: emotionReadings.dominantEmotion,
          timestamp: emotionReadings.timestamp,
        })
        .from(emotionReadings)
        .where(and(eq(emotionReadings.userId, userId), gte(emotionReadings.timestamp, sevenDaysAgo)))
        .orderBy(asc(emotionReadings.timestamp))
        .limit(2000);

      emotionReadingCount = readings.length;

      if (readings.length > 0) {
        avgStress = round2(avg(readings.map((r) => r.stress)));
        avgFocus  = round2(avg(readings.map((r) => r.focus)));
        avgValence = round2(avg(readings.filter((r) => r.valence != null).map((r) => r.valence!)));

        // Dominant emotion counts
        const emotionCounts: Record<string, number> = {};
        readings.forEach((r) => {
          emotionCounts[r.dominantEmotion] = (emotionCounts[r.dominantEmotion] ?? 0) + 1;
        });
        dominantEmotions = Object.entries(emotionCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([e]) => e);

        // Best / worst day by avg focus per calendar day
        const byDay: Record<string, { focus: number[]; stress: number[] }> = {};
        readings.forEach((r) => {
          const day = new Date(r.timestamp).toLocaleDateString("en-US", { weekday: "long" });
          if (!byDay[day]) byDay[day] = { focus: [], stress: [] };
          byDay[day].focus.push(r.focus);
          byDay[day].stress.push(r.stress);
        });
        const dayAvgs = Object.entries(byDay).map(([day, data]) => ({
          day,
          avgFocus: avg(data.focus) ?? 0,
          avgStress: avg(data.stress) ?? 0,
        }));
        if (dayAvgs.length > 0) {
          dayAvgs.sort((a, b) => b.avgFocus - a.avgFocus);
          bestDay  = dayAvgs[0].day;
          worstDay = dayAvgs[dayAvgs.length - 1].day;
        }
      }
    } catch {
      // Non-fatal: continue with nulls
    }

    // ── 2. Health metrics (last 7 days) ──────────────────────────────────
    let avgSleepQuality: number | null = null;
    let avgDailySteps: number | null = null;
    let avgHeartRate: number | null = null;

    try {
      const metrics = await db
        .select({
          sleepQuality: healthMetrics.sleepQuality,
          dailySteps: healthMetrics.dailySteps,
          heartRate: healthMetrics.heartRate,
          timestamp: healthMetrics.timestamp,
        })
        .from(healthMetrics)
        .where(and(eq(healthMetrics.userId, userId), gte(healthMetrics.timestamp, sevenDaysAgo)))
        .limit(500);

      if (metrics.length > 0) {
        avgSleepQuality = round2(avg(metrics.map((m) => m.sleepQuality)));
        avgHeartRate    = round2(avg(metrics.map((m) => m.heartRate)));
        const steps = metrics.filter((m) => m.dailySteps != null).map((m) => m.dailySteps!);
        if (steps.length > 0) avgDailySteps = round2(avg(steps));
      }
    } catch {
      // Non-fatal
    }

    // ── 3. Health samples (HRV, sleep efficiency from Apple Health / Fit) ─
    let avgHrv: number | null = null;
    let avgSleepEfficiency: number | null = null;

    try {
      const hrvRows = await db
        .select({ value: healthSamples.value })
        .from(healthSamples)
        .where(and(
          eq(healthSamples.userId, userId),
          eq(healthSamples.metric, "hrv"),
          gte(healthSamples.recordedAt, sevenDaysAgo),
        ))
        .limit(200);
      if (hrvRows.length > 0) avgHrv = round2(avg(hrvRows.map((r) => r.value)));

      const sleepEffRows = await db
        .select({ value: healthSamples.value })
        .from(healthSamples)
        .where(and(
          eq(healthSamples.userId, userId),
          eq(healthSamples.metric, "sleep_efficiency"),
          gte(healthSamples.recordedAt, sevenDaysAgo),
        ))
        .limit(200);
      if (sleepEffRows.length > 0) avgSleepEfficiency = round2(avg(sleepEffRows.map((r) => r.value)));
    } catch {
      // Non-fatal
    }

    // ── 4. Check minimum data threshold ──────────────────────────────────
    if (emotionReadingCount < 3 && avgSleepQuality == null && avgHrv == null) {
      return res.json({
        insights: [],
        message: "Record at least 3 sessions to see personalized insights",
      });
    }

    // ── 5. Build context string ───────────────────────────────────────────
    const contextLines: string[] = ["User health data (last 7 days):"];
    if (avgStress    != null) contextLines.push(`- Average stress index: ${avgStress} (0–1 scale, higher = more stressed)`);
    if (avgFocus     != null) contextLines.push(`- Average focus index: ${avgFocus} (0–1 scale, higher = more focused)`);
    if (avgValence   != null) contextLines.push(`- Average emotional valence: ${avgValence} (−1 to 1; positive = pleasant)`);
    if (dominantEmotions.length) contextLines.push(`- Top emotions detected by EEG: ${dominantEmotions.join(", ")}`);
    if (bestDay)               contextLines.push(`- Best focus day of the week: ${bestDay}`);
    if (worstDay && worstDay !== bestDay) contextLines.push(`- Lowest focus day of the week: ${worstDay}`);
    if (avgSleepQuality != null) contextLines.push(`- Average sleep quality (1–10 scale): ${avgSleepQuality}`);
    if (avgDailySteps   != null) contextLines.push(`- Average daily steps: ${Math.round(avgDailySteps)}`);
    if (avgHeartRate    != null) contextLines.push(`- Average resting heart rate: ${Math.round(avgHeartRate)} bpm`);
    if (avgHrv          != null) contextLines.push(`- Average HRV (ms): ${Math.round(avgHrv)}`);
    if (avgSleepEfficiency != null) contextLines.push(`- Average sleep efficiency: ${Math.round(avgSleepEfficiency)}%`);
    contextLines.push(`- EEG readings used: ${emotionReadingCount}`);

    const context = contextLines.join("\n");

    // ── 6. Call GPT-5 ─────────────────────────────────────────────────────
    const systemPrompt = `You are a personal health insights AI analyzing EEG and biometric data. \
Generate 3 to 5 specific, actionable insights about patterns in the user's data. \
Focus on: correlations between sleep and focus, stress patterns, emotional trends, \
heart rate or HRV relationship with stress, and daily step impact on mood. \
Be specific with the numbers provided. \
Respond ONLY with valid JSON in this exact format: { "insights": [{ "title": string, "detail": string, "recommendation": string }] }`;

    let insights: Array<{ title: string; detail: string; recommendation: string }> = [];
    try {
      // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user",   content: context },
        ],
        response_format: { type: "json_object" },
      });

      const raw = response.choices[0].message.content ?? "{}";
      const parsed = JSON.parse(raw) as { insights?: unknown };
      if (Array.isArray(parsed.insights)) {
        insights = parsed.insights.filter(
          (i): i is { title: string; detail: string; recommendation: string } =>
            i != null &&
            typeof (i as Record<string, unknown>).title === "string" &&
            typeof (i as Record<string, unknown>).detail === "string" &&
            typeof (i as Record<string, unknown>).recommendation === "string"
        );
      }
    } catch (err) {
      logger.error({ error: err instanceof Error ? err.message : String(err) }, "[health-insights] GPT-5 error");
      return res.status(500).json({ message: "Failed to generate insights" });
    }

    res.json({ insights });
  });

  // ── GDPR / Privacy endpoints ──────────────────────────────────────────────

  // GET /api/user/:userId/export-all — full GDPR data export (Art. 20)
  app.get("/api/user/:userId/export-all", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const exportedAt = new Date().toISOString();

      const [
        userRows,
        healthMetricsRows,
        dreamAnalysisRows,
        dreamSymbolsRows,
        emotionReadingsRows,
        eegSessionsRows,
        userSettingsRows,
        pushSubscriptionsRows,
        aiChatsRows,
        foodLogsRows,
        brainReadingsRows,
        healthSamplesRows,
        studyParticipantsRows,
      ] = await Promise.all([
        db.select().from(users).where(eq(users.id, userId)),
        db.select().from(healthMetrics).where(eq(healthMetrics.userId, userId)),
        db.select().from(dreamAnalysis).where(eq(dreamAnalysis.userId, userId)),
        db.select().from(dreamSymbols).where(eq(dreamSymbols.userId, userId)),
        db.select().from(emotionReadings).where(eq(emotionReadings.userId, userId)),
        db.select().from(eegSessions).where(eq(eegSessions.userId, userId)),
        db.select().from(userSettings).where(eq(userSettings.userId, userId)),
        db.select().from(pushSubscriptions).where(eq(pushSubscriptions.userId, userId)),
        db.select().from(aiChats).where(eq(aiChats.userId, userId)),
        db.select().from(foodLogs).where(eq(foodLogs.userId as any, userId)),
        db.select().from(brainReadings).where(eq(brainReadings.userId, userId)),
        db.select().from(healthSamples).where(eq(healthSamples.userId, userId)),
        db.select().from(studyParticipants).where(eq(studyParticipants.userId, userId)),
      ]);

      // Omit password hash from user export
      const sanitisedUser = userRows.map(({ password: _pw, ...u }) => u);

      const payload = {
        metadata: {
          exportedAt,
          userId,
          dataCategories: [
            "account", "healthMetrics", "dreamAnalysis", "dreamSymbols",
            "emotionReadings", "eegSessions", "userSettings",
            "pushSubscriptions", "aiChats", "foodLogs", "brainReadings",
            "healthSamples", "studyParticipants",
          ],
          rowCounts: {
            account:            sanitisedUser.length,
            healthMetrics:      healthMetricsRows.length,
            dreamAnalysis:      dreamAnalysisRows.length,
            dreamSymbols:       dreamSymbolsRows.length,
            emotionReadings:    emotionReadingsRows.length,
            eegSessions:        eegSessionsRows.length,
            userSettings:       userSettingsRows.length,
            pushSubscriptions:  pushSubscriptionsRows.length,
            aiChats:            aiChatsRows.length,
            foodLogs:           foodLogsRows.length,
            brainReadings:      brainReadingsRows.length,
            healthSamples:      healthSamplesRows.length,
            studyParticipants:  studyParticipantsRows.length,
          },
        },
        data: {
          account:            sanitisedUser,
          healthMetrics:      healthMetricsRows,
          dreamAnalysis:      dreamAnalysisRows,
          dreamSymbols:       dreamSymbolsRows,
          emotionReadings:    emotionReadingsRows,
          eegSessions:        eegSessionsRows,
          userSettings:       userSettingsRows,
          pushSubscriptions:  pushSubscriptionsRows,
          aiChats:            aiChatsRows,
          foodLogs:           foodLogsRows,
          brainReadings:      brainReadingsRows,
          healthSamples:      healthSamplesRows,
          studyParticipants:  studyParticipantsRows,
        },
      };

      // Record export timestamp in user settings (best-effort)
      try {
        const [existing] = await db.select().from(userSettings).where(eq(userSettings.userId, userId));
        if (existing) {
          const thresholds = (existing.alertThresholds as Record<string, unknown> | null) ?? {};
          const exportHistory: string[] = Array.isArray((thresholds as any).__exportHistory)
            ? (thresholds as any).__exportHistory
            : [];
          exportHistory.push(exportedAt);
          await db.update(userSettings)
            .set({ alertThresholds: { ...thresholds, __exportHistory: exportHistory.slice(-20) } })
            .where(eq(userSettings.userId, userId));
        } else {
          await db.insert(userSettings).values({
            userId,
            alertThresholds: { __exportHistory: [exportedAt] },
          });
        }
      } catch { /* non-fatal */ }

      res.json(payload);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "GDPR export failed");
      res.status(500).json({ error: "Failed to export data" });
    }
  });

  // DELETE /api/user/:userId — soft-delete (GDPR Art. 17 right-to-erasure request)
  app.delete("/api/user/:userId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    const { confirm } = req.body as { confirm?: boolean };
    if (confirm !== true) {
      return res.status(400).json({
        error: "Deletion requires { confirm: true } in request body",
      });
    }

    try {
      const [updated] = await db
        .update(users)
        .set({ deletionRequestedAt: new Date() })
        .where(eq(users.id, userId))
        .returning({ id: users.id, deletionRequestedAt: users.deletionRequestedAt });

      if (!updated) {
        return res.status(404).json({ error: "User not found" });
      }

      const gracePeriodDays = 30;
      const scheduledDeletionDate = new Date(updated.deletionRequestedAt!);
      scheduledDeletionDate.setDate(scheduledDeletionDate.getDate() + gracePeriodDays);

      logger.info({ userId, scheduledDeletion: scheduledDeletionDate.toISOString() }, "GDPR deletion requested");

      return res.json({
        message: "Deletion request recorded. Your account and all data will be permanently deleted after the grace period.",
        gracePeriodDays,
        requestedAt: updated.deletionRequestedAt,
        scheduledDeletionDate: scheduledDeletionDate.toISOString(),
      });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "GDPR deletion request failed");
      return res.status(500).json({ error: "Failed to record deletion request" });
    }
  });

  // GET /api/user/:userId/export-history — list of past export timestamps
  app.get("/api/user/:userId/export-history", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const [settings] = await db
        .select({ alertThresholds: userSettings.alertThresholds })
        .from(userSettings)
        .where(eq(userSettings.userId, userId));

      const thresholds = (settings?.alertThresholds as Record<string, unknown> | null) ?? {};
      const exportHistory: string[] = Array.isArray((thresholds as any).__exportHistory)
        ? (thresholds as any).__exportHistory
        : [];

      return res.json({ userId, exportHistory });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Export history fetch failed");
      return res.status(500).json({ error: "Failed to fetch export history" });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // EXERCISE LIBRARY
  // ══════════════════════════════════════════════════════════════════════════

  // GET /api/exercises — list all exercises with optional filters
  app.get("/api/exercises", async (req, res) => {
    try {
      const { category, equipment, muscle } = req.query as Record<string, string | undefined>;
      const conditions = [];

      if (category) conditions.push(ilike(exercises.category, category));
      if (equipment) conditions.push(ilike(exercises.equipment, equipment));
      if (muscle) conditions.push(arrayContains(exercises.muscleGroups, [muscle]));

      const rows = conditions.length > 0
        ? await db.select().from(exercises).where(and(...conditions))
        : await db.select().from(exercises);

      return res.json(rows);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to list exercises");
      return res.status(500).json({ error: "Failed to list exercises" });
    }
  });

  // GET /api/exercises/:id — get single exercise
  app.get("/api/exercises/:id", async (req, res) => {
    try {
      const [exercise] = await db.select().from(exercises)
        .where(eq(exercises.id, req.params.id)).limit(1);

      if (!exercise) return res.status(404).json({ error: "Exercise not found" });
      return res.json(exercise);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to fetch exercise");
      return res.status(500).json({ error: "Failed to fetch exercise" });
    }
  });

  // POST /api/exercises — create custom exercise (requires auth)
  app.post("/api/exercises", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const { name, category, muscleGroups, equipment, instructions, videoUrl } = req.body;
      if (!name || !category || !Array.isArray(muscleGroups) || muscleGroups.length === 0) {
        return res.status(400).json({ error: "name, category, and muscleGroups[] are required" });
      }

      const [created] = await db.insert(exercises).values({
        name,
        category,
        muscleGroups,
        equipment: equipment ?? null,
        instructions: instructions ?? null,
        videoUrl: videoUrl ?? null,
        isCustom: true,
        createdBy: userId,
      }).returning();

      return res.status(201).json(created);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to create exercise");
      return res.status(500).json({ error: "Failed to create exercise" });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // WORKOUTS
  // ══════════════════════════════════════════════════════════════════════════

  // POST /api/workouts — create new workout
  app.post("/api/workouts", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const { name, workoutType, startedAt, endedAt, durationMin, caloriesBurned, avgHr, maxHr, source, notes, eegSessionId } = req.body;
      if (!workoutType) return res.status(400).json({ error: "workoutType is required" });

      const [created] = await db.insert(workouts).values({
        userId,
        name: name ?? null,
        workoutType,
        startedAt: startedAt ? new Date(startedAt) : new Date(),
        endedAt: endedAt ? new Date(endedAt) : null,
        durationMin: durationMin != null ? String(durationMin) : null,
        caloriesBurned: caloriesBurned != null ? String(caloriesBurned) : null,
        avgHr: avgHr != null ? String(avgHr) : null,
        maxHr: maxHr != null ? String(maxHr) : null,
        source: source ?? "manual",
        notes: notes ?? null,
        eegSessionId: eegSessionId ?? null,
      }).returning();

      return res.status(201).json(created);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to create workout");
      return res.status(500).json({ error: "Failed to create workout" });
    }
  });

  // GET /api/workouts/:userId — list user's workouts (requires owner auth, paginated)
  app.get("/api/workouts/:userId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const limit = Math.min(Math.max(parseInt((req.query.limit as string) || "20", 10), 1), 100);
      const offset = Math.max(parseInt((req.query.offset as string) || "0", 10), 0);

      const rows = await db.select().from(workouts)
        .where(eq(workouts.userId, userId))
        .orderBy(desc(workouts.startedAt))
        .limit(limit)
        .offset(offset);

      return res.json(rows);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to list workouts");
      return res.status(500).json({ error: "Failed to list workouts" });
    }
  });

  // GET /api/workouts/:userId/:workoutId — get workout with sets
  app.get("/api/workouts/:userId/:workoutId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const [workout] = await db.select().from(workouts)
        .where(and(eq(workouts.id, req.params.workoutId), eq(workouts.userId, userId)))
        .limit(1);

      if (!workout) return res.status(404).json({ error: "Workout not found" });

      const sets = await db.select().from(workoutSets)
        .where(eq(workoutSets.workoutId, workout.id))
        .orderBy(asc(workoutSets.setNumber));

      return res.json({ ...workout, sets });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to fetch workout");
      return res.status(500).json({ error: "Failed to fetch workout" });
    }
  });

  // PUT /api/workouts/:workoutId — update workout (endedAt, notes, etc.)
  app.put("/api/workouts/:workoutId", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      // Verify ownership
      const [existing] = await db.select().from(workouts)
        .where(and(eq(workouts.id, req.params.workoutId), eq(workouts.userId, userId)))
        .limit(1);

      if (!existing) return res.status(404).json({ error: "Workout not found" });

      const { endedAt, notes, avgHr, maxHr, caloriesBurned, hrZones, hrRecovery } = req.body;

      const updates: Record<string, unknown> = {};
      if (endedAt !== undefined) updates.endedAt = new Date(endedAt);
      if (notes !== undefined) updates.notes = notes;
      if (avgHr !== undefined) updates.avgHr = avgHr;
      if (maxHr !== undefined) updates.maxHr = maxHr;
      if (caloriesBurned !== undefined) updates.caloriesBurned = caloriesBurned;
      if (hrZones !== undefined) updates.hrZones = hrZones;
      if (hrRecovery !== undefined) updates.hrRecovery = hrRecovery;

      // Compute duration and strain when completing a workout
      if (endedAt) {
        const endDate = new Date(endedAt);
        const durationMin = (endDate.getTime() - existing.startedAt.getTime()) / 60000;
        updates.durationMin = durationMin.toFixed(1);

        // TRIMP-based strain if HR data available
        if (avgHr && maxHr) {
          const restingHr = 60; // population default
          const avgHrNum = parseFloat(String(avgHr));
          const maxHrNum = parseFloat(String(maxHr));
          const avgHrRatio = (avgHrNum - restingHr) / (maxHrNum - restingHr);
          if (avgHrRatio > 0 && avgHrRatio < 1) {
            const trimp = durationMin * avgHrRatio * Math.exp(1.92 * avgHrRatio);
            const strain = 14.3 * Math.log(1 + trimp);
            updates.totalStrain = strain.toFixed(2);
          }
        }
      }

      const [updated] = await db.update(workouts)
        .set(updates)
        .where(eq(workouts.id, req.params.workoutId))
        .returning();

      // Update exercise history with best lifts when workout completes
      if (endedAt) {
        try {
          const sets = await db.select().from(workoutSets)
            .where(eq(workoutSets.workoutId, existing.id));

          const today = new Date().toISOString().slice(0, 10);

          // Group sets by exerciseId and compute bests
          const byExercise = new Map<string, { bestWeightKg: number; bestReps: number; estimated1rm: number; totalVolume: number }>();
          for (const s of sets) {
            if (!s.exerciseId) continue;
            const w = parseFloat(String(s.weightKg ?? 0));
            const r = s.reps ?? 0;
            const e1rm = w * (1 + r / 30); // Epley formula
            const volume = w * r;

            const prev = byExercise.get(s.exerciseId);
            if (!prev) {
              byExercise.set(s.exerciseId, { bestWeightKg: w, bestReps: r, estimated1rm: e1rm, totalVolume: volume });
            } else {
              if (w > prev.bestWeightKg) prev.bestWeightKg = w;
              if (r > prev.bestReps) prev.bestReps = r;
              if (e1rm > prev.estimated1rm) prev.estimated1rm = e1rm;
              prev.totalVolume += volume;
            }
          }

          for (const [exerciseId, stats] of Array.from(byExercise.entries())) {
            await db.insert(exerciseHistory).values({
              userId,
              exerciseId,
              date: today,
              bestWeightKg: stats.bestWeightKg.toFixed(2),
              bestReps: stats.bestReps,
              estimated1rm: stats.estimated1rm.toFixed(2),
              totalVolume: stats.totalVolume.toFixed(2),
            }).onConflictDoUpdate({
              target: [exerciseHistory.userId, exerciseHistory.exerciseId, exerciseHistory.date],
              set: {
                bestWeightKg: sql`GREATEST(exercise_history.best_weight_kg::numeric, EXCLUDED.best_weight_kg::numeric)::text`,
                bestReps: sql`GREATEST(exercise_history.best_reps, EXCLUDED.best_reps)`,
                estimated1rm: sql`GREATEST(exercise_history.estimated_1rm::numeric, EXCLUDED.estimated_1rm::numeric)::text`,
                totalVolume: sql`(exercise_history.total_volume::numeric + EXCLUDED.total_volume::numeric)::text`,
              },
            });
          }
        } catch (historyErr) {
          logger.error({ error: historyErr instanceof Error ? historyErr.message : String(historyErr) }, "Failed to update exercise history (non-fatal)");
        }

        // Pipeline integration: insert workout_strain into health_samples
        if (updates.totalStrain) {
          try {
            await db.insert(healthSamples).values({
              userId,
              source: "workout",
              metric: "workout_strain",
              value: parseFloat(String(updates.totalStrain)),
              unit: "strain",
              recordedAt: new Date(endedAt),
            }).onConflictDoNothing();
          } catch (pipelineErr) {
            logger.error({ error: pipelineErr instanceof Error ? pipelineErr.message : String(pipelineErr) }, "Failed to insert workout strain into health_samples (non-fatal)");
          }
        }
      }

      return res.json(updated);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to update workout");
      return res.status(500).json({ error: "Failed to update workout" });
    }
  });

  // DELETE /api/workouts/:workoutId — delete workout + cascade sets
  app.delete("/api/workouts/:workoutId", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const [existing] = await db.select().from(workouts)
        .where(and(eq(workouts.id, req.params.workoutId), eq(workouts.userId, userId)))
        .limit(1);

      if (!existing) return res.status(404).json({ error: "Workout not found" });

      // Sets cascade via FK onDelete
      await db.delete(workouts).where(eq(workouts.id, req.params.workoutId));
      return res.json({ deleted: true });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to delete workout");
      return res.status(500).json({ error: "Failed to delete workout" });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // WORKOUT SETS
  // ══════════════════════════════════════════════════════════════════════════

  // POST /api/workouts/:workoutId/sets — add set to workout
  app.post("/api/workouts/:workoutId/sets", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      // Verify workout ownership
      const [workout] = await db.select().from(workouts)
        .where(and(eq(workouts.id, req.params.workoutId), eq(workouts.userId, userId)))
        .limit(1);

      if (!workout) return res.status(404).json({ error: "Workout not found" });

      const { exerciseId, setNumber, reps, weightKg, setType, durationSec, restSec, rpe } = req.body;
      if (!exerciseId || !setNumber) {
        return res.status(400).json({ error: "exerciseId and setNumber are required" });
      }

      const [created] = await db.insert(workoutSets).values({
        workoutId: req.params.workoutId,
        exerciseId,
        setNumber,
        setType: setType ?? "normal",
        reps: reps ?? null,
        weightKg: weightKg != null ? String(weightKg) : null,
        durationSec: durationSec ?? null,
        restSec: restSec ?? null,
        rpe: rpe != null ? String(rpe) : null,
      }).returning();

      return res.status(201).json(created);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to add workout set");
      return res.status(500).json({ error: "Failed to add workout set" });
    }
  });

  // PUT /api/workout-sets/:setId — update set
  app.put("/api/workout-sets/:setId", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      // Verify set ownership via workout
      const [set] = await db.select().from(workoutSets).where(eq(workoutSets.id, req.params.setId)).limit(1);
      if (!set) return res.status(404).json({ error: "Set not found" });

      const [workout] = await db.select().from(workouts)
        .where(and(eq(workouts.id, set.workoutId!), eq(workouts.userId, userId)))
        .limit(1);

      if (!workout) return res.status(403).json({ error: "Forbidden" });

      const { reps, weightKg, setType, durationSec, restSec, rpe, completed } = req.body;
      const updates: Record<string, unknown> = {};
      if (reps !== undefined) updates.reps = reps;
      if (weightKg !== undefined) updates.weightKg = weightKg != null ? String(weightKg) : null;
      if (setType !== undefined) updates.setType = setType;
      if (durationSec !== undefined) updates.durationSec = durationSec;
      if (restSec !== undefined) updates.restSec = restSec;
      if (rpe !== undefined) updates.rpe = rpe != null ? String(rpe) : null;
      if (completed !== undefined) updates.completed = completed;

      const [updated] = await db.update(workoutSets)
        .set(updates)
        .where(eq(workoutSets.id, req.params.setId))
        .returning();

      return res.json(updated);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to update workout set");
      return res.status(500).json({ error: "Failed to update workout set" });
    }
  });

  // DELETE /api/workout-sets/:setId — delete set
  app.delete("/api/workout-sets/:setId", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const [set] = await db.select().from(workoutSets).where(eq(workoutSets.id, req.params.setId)).limit(1);
      if (!set) return res.status(404).json({ error: "Set not found" });

      const [workout] = await db.select().from(workouts)
        .where(and(eq(workouts.id, set.workoutId!), eq(workouts.userId, userId)))
        .limit(1);

      if (!workout) return res.status(403).json({ error: "Forbidden" });

      await db.delete(workoutSets).where(eq(workoutSets.id, req.params.setId));
      return res.json({ deleted: true });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to delete workout set");
      return res.status(500).json({ error: "Failed to delete workout set" });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // BODY METRICS
  // ══════════════════════════════════════════════════════════════════════════

  // POST /api/body-metrics — log weight/body fat
  app.post("/api/body-metrics", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const { weightKg, bodyFatPct, heightCm, source, recordedAt } = req.body;
      if (!source) return res.status(400).json({ error: "source is required" });

      // Compute BMI if both weight and height provided
      let bmi: string | null = null;
      const wKg = weightKg != null ? parseFloat(String(weightKg)) : null;
      const hCm = heightCm != null ? parseFloat(String(heightCm)) : null;
      if (wKg && hCm && hCm > 0) {
        bmi = (wKg / ((hCm / 100) ** 2)).toFixed(2);
      }

      // Compute lean mass if weight and body fat provided
      let leanMassKg: string | null = null;
      const bfPct = bodyFatPct != null ? parseFloat(String(bodyFatPct)) : null;
      if (wKg && bfPct != null && bfPct >= 0 && bfPct <= 100) {
        leanMassKg = (wKg * (1 - bfPct / 100)).toFixed(2);
      }

      const [created] = await db.insert(bodyMetrics).values({
        userId,
        weightKg: weightKg != null ? String(weightKg) : null,
        bodyFatPct: bodyFatPct != null ? String(bodyFatPct) : null,
        heightCm: heightCm != null ? String(heightCm) : null,
        bmi,
        leanMassKg,
        source,
        recordedAt: recordedAt ? new Date(recordedAt) : new Date(),
      }).returning();

      // Pipeline integration: feed weight into health_samples
      if (wKg) {
        try {
          await db.insert(healthSamples).values({
            userId,
            source: "manual",
            metric: "weight_kg",
            value: wKg,
            unit: "kg",
            recordedAt: created.recordedAt,
          }).onConflictDoNothing();
        } catch (pipelineErr) {
          logger.error({ error: pipelineErr instanceof Error ? pipelineErr.message : String(pipelineErr) }, "Failed to insert weight into health_samples (non-fatal)");
        }
      }

      return res.status(201).json(created);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to log body metrics");
      return res.status(500).json({ error: "Failed to log body metrics" });
    }
  });

  // GET /api/body-metrics/:userId — get body metrics history (last 90 days)
  app.get("/api/body-metrics/:userId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const days = Math.min(Math.max(parseInt((req.query.days as string) || "90", 10), 1), 365);
      const fromDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

      const rows = await db.select().from(bodyMetrics)
        .where(and(eq(bodyMetrics.userId, userId), gte(bodyMetrics.recordedAt, fromDate)))
        .orderBy(desc(bodyMetrics.recordedAt));

      return res.json(rows);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to fetch body metrics");
      return res.status(500).json({ error: "Failed to fetch body metrics" });
    }
  });

  // GET /api/body-metrics/:userId/latest — get most recent body metrics
  app.get("/api/body-metrics/:userId/latest", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const [latest] = await db.select().from(bodyMetrics)
        .where(eq(bodyMetrics.userId, userId))
        .orderBy(desc(bodyMetrics.recordedAt))
        .limit(1);

      if (!latest) return res.status(404).json({ error: "No body metrics found" });
      return res.json(latest);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to fetch latest body metrics");
      return res.status(500).json({ error: "Failed to fetch latest body metrics" });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // WORKOUT TEMPLATES
  // ══════════════════════════════════════════════════════════════════════════

  // POST /api/workout-templates — create template
  app.post("/api/workout-templates", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const { name, description, exercises: templateExercises } = req.body;
      if (!name || !templateExercises) {
        return res.status(400).json({ error: "name and exercises are required" });
      }

      const [created] = await db.insert(workoutTemplates).values({
        userId,
        name,
        description: description ?? null,
        exercises: templateExercises,
      }).returning();

      return res.status(201).json(created);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to create workout template");
      return res.status(500).json({ error: "Failed to create workout template" });
    }
  });

  // GET /api/workout-templates/:userId — list user's templates
  app.get("/api/workout-templates/:userId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const rows = await db.select().from(workoutTemplates)
        .where(eq(workoutTemplates.userId, userId))
        .orderBy(desc(workoutTemplates.createdAt));

      return res.json(rows);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to list workout templates");
      return res.status(500).json({ error: "Failed to list workout templates" });
    }
  });

  // DELETE /api/workout-templates/:id — delete template
  app.delete("/api/workout-templates/:id", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const [existing] = await db.select().from(workoutTemplates)
        .where(and(eq(workoutTemplates.id, req.params.id), eq(workoutTemplates.userId, userId)))
        .limit(1);

      if (!existing) return res.status(404).json({ error: "Template not found" });

      await db.delete(workoutTemplates).where(eq(workoutTemplates.id, req.params.id));
      return res.json({ deleted: true });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to delete workout template");
      return res.status(500).json({ error: "Failed to delete workout template" });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // EXERCISE PROGRESSION & PERSONAL RECORDS
  // ══════════════════════════════════════════════════════════════════════════

  // GET /api/exercise-history/:userId/prs — personal records across all exercises
  // NOTE: this route MUST be registered before /:exerciseId to avoid "prs" matching as an exerciseId
  app.get("/api/exercise-history/:userId/prs", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      // For each exercise, find the row with the highest estimated 1RM
      const rows = await db
        .select({
          exerciseId: exerciseHistory.exerciseId,
          exerciseName: exercises.name,
          estimated1rm: sql<string>`MAX(exercise_history.estimated_1rm::numeric)`,
          bestWeightKg: sql<string>`MAX(exercise_history.best_weight_kg::numeric)`,
          bestReps: sql<number>`MAX(exercise_history.best_reps)`,
          date: sql<string>`(ARRAY_AGG(exercise_history.date ORDER BY exercise_history.estimated_1rm::numeric DESC))[1]`,
        })
        .from(exerciseHistory)
        .innerJoin(exercises, eq(exerciseHistory.exerciseId, exercises.id))
        .where(eq(exerciseHistory.userId, userId))
        .groupBy(exerciseHistory.exerciseId, exercises.name);

      return res.json(rows.map(r => ({
        exerciseId: r.exerciseId,
        exerciseName: r.exerciseName,
        estimated1rm: r.estimated1rm ? parseFloat(String(r.estimated1rm)) : null,
        bestWeightKg: r.bestWeightKg ? parseFloat(String(r.bestWeightKg)) : null,
        bestReps: r.bestReps,
        date: r.date,
      })));
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to fetch personal records");
      return res.status(500).json({ error: "Failed to fetch personal records" });
    }
  });

  // GET /api/exercise-history/:userId/:exerciseId — progression data for a specific exercise
  app.get("/api/exercise-history/:userId/:exerciseId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const rows = await db.select().from(exerciseHistory)
        .where(and(
          eq(exerciseHistory.userId, userId),
          eq(exerciseHistory.exerciseId, req.params.exerciseId),
        ))
        .orderBy(asc(exerciseHistory.date));

      return res.json(rows.map(r => ({
        date: r.date,
        bestWeightKg: r.bestWeightKg ? parseFloat(String(r.bestWeightKg)) : null,
        bestReps: r.bestReps,
        estimated1rm: r.estimated1rm ? parseFloat(String(r.estimated1rm)) : null,
        totalVolume: r.totalVolume ? parseFloat(String(r.totalVolume)) : null,
      })));
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to fetch exercise progression");
      return res.status(500).json({ error: "Failed to fetch exercise progression" });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // CARDIO LOAD (ATL / CTL / TSB)
  // ══════════════════════════════════════════════════════════════════════════

  // GET /api/cardio-load/:userId — current Acute/Chronic Training Load and status
  app.get("/api/cardio-load/:userId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      // Pull all workouts with strain from the last 42 days
      const cutoff = new Date();
      cutoff.setDate(cutoff.getDate() - 42);

      const recentWorkouts = await db.select({
        startedAt: workouts.startedAt,
        totalStrain: workouts.totalStrain,
      })
        .from(workouts)
        .where(and(
          eq(workouts.userId, userId),
          gte(workouts.startedAt, cutoff),
        ))
        .orderBy(asc(workouts.startedAt));

      // Aggregate strain by date (YYYY-MM-DD) to get daily TRIMP-equivalent values
      const dailyMap = new Map<string, number>();
      for (const w of recentWorkouts) {
        const strain = w.totalStrain ? parseFloat(String(w.totalStrain)) : 0;
        if (strain <= 0) continue;
        const dateKey = w.startedAt.toISOString().slice(0, 10);
        dailyMap.set(dateKey, (dailyMap.get(dateKey) ?? 0) + strain);
      }

      const dailyTrimpValues = Array.from(dailyMap.entries()).map(([date, trimp]) => ({ date, trimp }));

      const result = computeCardioLoad(dailyTrimpValues);
      return res.json(result);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to compute cardio load");
      return res.status(500).json({ error: "Failed to compute cardio load" });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // HABITS
  // ══════════════════════════════════════════════════════════════════════════

  // POST /api/habits — create a new habit
  app.post("/api/habits", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const { name, category, icon, targetValue, unit } = req.body;
      if (!name) return res.status(400).json({ error: "name is required" });

      const [created] = await db.insert(habits).values({
        userId,
        name,
        category: category ?? null,
        icon: icon ?? null,
        targetValue: targetValue != null ? String(targetValue) : null,
        unit: unit ?? null,
      }).returning();

      return res.status(201).json(created);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to create habit");
      return res.status(500).json({ error: "Failed to create habit" });
    }
  });

  // GET /api/habits/:userId — list user's active habits
  app.get("/api/habits/:userId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const rows = await db.select().from(habits)
        .where(and(eq(habits.userId, userId), eq(habits.isActive, true)))
        .orderBy(asc(habits.createdAt));

      return res.json(rows);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to list habits");
      return res.status(500).json({ error: "Failed to list habits" });
    }
  });

  // PUT /api/habits/:id — update habit
  app.put("/api/habits/:id", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const { name, targetValue, unit, isActive, category, icon } = req.body;

      // Verify ownership
      const [existing] = await db.select().from(habits)
        .where(and(eq(habits.id, req.params.id), eq(habits.userId, userId)))
        .limit(1);
      if (!existing) return res.status(404).json({ error: "Habit not found" });

      const updates: Record<string, unknown> = {};
      if (name !== undefined) updates.name = name;
      if (targetValue !== undefined) updates.targetValue = targetValue != null ? String(targetValue) : null;
      if (unit !== undefined) updates.unit = unit;
      if (isActive !== undefined) updates.isActive = isActive;
      if (category !== undefined) updates.category = category;
      if (icon !== undefined) updates.icon = icon;

      const [updated] = await db.update(habits)
        .set(updates)
        .where(eq(habits.id, req.params.id))
        .returning();

      return res.json(updated);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to update habit");
      return res.status(500).json({ error: "Failed to update habit" });
    }
  });

  // DELETE /api/habits/:id — soft delete (set isActive=false)
  app.delete("/api/habits/:id", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const [existing] = await db.select().from(habits)
        .where(and(eq(habits.id, req.params.id), eq(habits.userId, userId)))
        .limit(1);
      if (!existing) return res.status(404).json({ error: "Habit not found" });

      await db.update(habits)
        .set({ isActive: false })
        .where(eq(habits.id, req.params.id));

      return res.json({ deleted: true });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to delete habit");
      return res.status(500).json({ error: "Failed to delete habit" });
    }
  });

  // POST /api/habit-logs — log a habit entry
  app.post("/api/habit-logs", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const { habitId, value, note } = req.body;
      if (!habitId || value == null) return res.status(400).json({ error: "habitId and value are required" });

      // Verify habit ownership
      const [habit] = await db.select().from(habits)
        .where(and(eq(habits.id, habitId), eq(habits.userId, userId)))
        .limit(1);
      if (!habit) return res.status(404).json({ error: "Habit not found" });

      const [created] = await db.insert(habitLogs).values({
        userId,
        habitId,
        value: String(value),
        note: note ?? null,
      }).returning();

      return res.status(201).json(created);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to log habit");
      return res.status(500).json({ error: "Failed to log habit" });
    }
  });

  // GET /api/habit-logs/:userId — get logs for last 30 days
  app.get("/api/habit-logs/:userId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const days = Math.min(Math.max(parseInt((req.query.days as string) || "30", 10), 1), 365);
      const fromDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

      const rows = await db.select().from(habitLogs)
        .where(and(eq(habitLogs.userId, userId), gte(habitLogs.loggedAt, fromDate)))
        .orderBy(desc(habitLogs.loggedAt));

      return res.json(rows);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to fetch habit logs");
      return res.status(500).json({ error: "Failed to fetch habit logs" });
    }
  });

  // GET /api/habit-logs/:userId/streaks — compute current streaks per habit
  app.get("/api/habit-logs/:userId/streaks", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      // Get all active habits
      const userHabits = await db.select().from(habits)
        .where(and(eq(habits.userId, userId), eq(habits.isActive, true)));

      // Get all habit logs for the last 365 days
      const fromDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000);
      const logs = await db.select().from(habitLogs)
        .where(and(eq(habitLogs.userId, userId), gte(habitLogs.loggedAt, fromDate)))
        .orderBy(desc(habitLogs.loggedAt));

      // Group logs by habitId -> set of date strings (YYYY-MM-DD)
      const logsByHabit = new Map<string, Set<string>>();
      for (const log of logs) {
        if (!log.habitId) continue;
        if (!logsByHabit.has(log.habitId)) logsByHabit.set(log.habitId, new Set());
        const dateStr = log.loggedAt ? new Date(log.loggedAt).toISOString().slice(0, 10) : null;
        if (dateStr) logsByHabit.get(log.habitId)!.add(dateStr);
      }

      // Compute streaks: count consecutive days going backwards from today
      const streaks: Record<string, number> = {};
      const today = new Date();

      for (const habit of userHabits) {
        const dates = logsByHabit.get(habit.id) ?? new Set();
        let streak = 0;
        const checkDate = new Date(today);

        while (true) {
          const dateStr = checkDate.toISOString().slice(0, 10);
          if (dates.has(dateStr)) {
            streak++;
            checkDate.setDate(checkDate.getDate() - 1);
          } else {
            break;
          }
        }

        streaks[habit.id] = streak;
      }

      return res.json(streaks);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to compute habit streaks");
      return res.status(500).json({ error: "Failed to compute habit streaks" });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // CYCLE TRACKING
  // ══════════════════════════════════════════════════════════════════════════

  // POST /api/cycle — log cycle data
  app.post("/api/cycle", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const { date, flowLevel, symptoms, contraception, basalTemp, notes } = req.body;
      if (!date) return res.status(400).json({ error: "date is required" });

      // Upsert: if entry for this date already exists, update it
      const [existing] = await db.select().from(cycleTracking)
        .where(and(eq(cycleTracking.userId, userId), eq(cycleTracking.date, date)))
        .limit(1);

      if (existing) {
        const [updated] = await db.update(cycleTracking)
          .set({
            flowLevel: flowLevel ?? existing.flowLevel,
            symptoms: symptoms ?? existing.symptoms,
            contraception: contraception ?? existing.contraception,
            basalTemp: basalTemp != null ? String(basalTemp) : existing.basalTemp,
            notes: notes ?? existing.notes,
          })
          .where(eq(cycleTracking.id, existing.id))
          .returning();

        return res.json(updated);
      }

      const [created] = await db.insert(cycleTracking).values({
        userId,
        date,
        flowLevel: flowLevel ?? null,
        symptoms: symptoms ?? null,
        contraception: contraception ?? null,
        basalTemp: basalTemp != null ? String(basalTemp) : null,
        notes: notes ?? null,
      }).returning();

      return res.status(201).json(created);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to log cycle data");
      return res.status(500).json({ error: "Failed to log cycle data" });
    }
  });

  // GET /api/cycle/:userId — get cycle data (last 90 days)
  app.get("/api/cycle/:userId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const days = Math.min(Math.max(parseInt((req.query.days as string) || "90", 10), 1), 365);
      const fromDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);

      const rows = await db.select().from(cycleTracking)
        .where(and(eq(cycleTracking.userId, userId), gte(cycleTracking.date, fromDate)))
        .orderBy(desc(cycleTracking.date));

      return res.json(rows);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to fetch cycle data");
      return res.status(500).json({ error: "Failed to fetch cycle data" });
    }
  });

  // GET /api/cycle/:userId/phase — predict current phase
  app.get("/api/cycle/:userId/phase", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      // Get all cycle data (last 365 days for averaging)
      const fromDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);

      const rows = await db.select().from(cycleTracking)
        .where(and(
          eq(cycleTracking.userId, userId),
          gte(cycleTracking.date, fromDate),
        ))
        .orderBy(asc(cycleTracking.date));

      // Find period starts (days where flowLevel is not 'none' and not null,
      // preceded by a day with 'none' or no entry)
      const periodStarts: string[] = [];
      for (let i = 0; i < rows.length; i++) {
        const flow = rows[i].flowLevel;
        if (flow && flow !== "none") {
          // Check if previous day was none or no entry
          if (i === 0) {
            periodStarts.push(rows[i].date);
          } else {
            const prevDate = new Date(rows[i].date);
            prevDate.setDate(prevDate.getDate() - 1);
            const prevDateStr = prevDate.toISOString().slice(0, 10);
            const prevRow = rows.find(r => r.date === prevDateStr);
            if (!prevRow || prevRow.flowLevel === "none" || !prevRow.flowLevel) {
              periodStarts.push(rows[i].date);
            }
          }
        }
      }

      // Average cycle length from past 3 cycles
      let avgCycleLength = 28;
      if (periodStarts.length >= 2) {
        const cycleLengths: number[] = [];
        const count = Math.min(periodStarts.length - 1, 3);
        for (let i = periodStarts.length - count; i < periodStarts.length; i++) {
          const prev = new Date(periodStarts[i - 1]);
          const curr = new Date(periodStarts[i]);
          const diffDays = Math.round((curr.getTime() - prev.getTime()) / (24 * 60 * 60 * 1000));
          if (diffDays > 15 && diffDays < 60) {
            cycleLengths.push(diffDays);
          }
        }
        if (cycleLengths.length > 0) {
          avgCycleLength = Math.round(cycleLengths.reduce((a, b) => a + b, 0) / cycleLengths.length);
        }
      }

      // Predict current phase
      const lastPeriodStart = periodStarts.length > 0 ? periodStarts[periodStarts.length - 1] : null;
      let currentPhase = "unknown";
      let dayOfCycle = 0;
      let nextPeriodDate: string | null = null;

      if (lastPeriodStart) {
        const lastStart = new Date(lastPeriodStart);
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        dayOfCycle = Math.round((today.getTime() - lastStart.getTime()) / (24 * 60 * 60 * 1000)) + 1;

        if (dayOfCycle >= 1 && dayOfCycle <= 5) {
          currentPhase = "menstrual";
        } else if (dayOfCycle >= 6 && dayOfCycle <= 13) {
          currentPhase = "follicular";
        } else if (dayOfCycle >= 14 && dayOfCycle <= 16) {
          currentPhase = "ovulatory";
        } else if (dayOfCycle >= 17 && dayOfCycle <= avgCycleLength) {
          currentPhase = "luteal";
        } else {
          // Past expected cycle length — period may be late
          currentPhase = "late";
        }

        const nextStart = new Date(lastStart);
        nextStart.setDate(nextStart.getDate() + avgCycleLength);
        nextPeriodDate = nextStart.toISOString().slice(0, 10);
      }

      return res.json({
        currentPhase,
        dayOfCycle,
        avgCycleLength,
        lastPeriodStart,
        nextPeriodDate,
        periodStartCount: periodStarts.length,
      });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to predict cycle phase");
      return res.status(500).json({ error: "Failed to predict cycle phase" });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // MOOD LOGS
  // ══════════════════════════════════════════════════════════════════════════

  // POST /api/mood — log mood
  app.post("/api/mood", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    try {
      const { moodScore, energyLevel, notes } = req.body;
      if (moodScore == null) return res.status(400).json({ error: "moodScore is required" });

      const score = parseFloat(String(moodScore));
      if (score < 1 || score > 10) return res.status(400).json({ error: "moodScore must be 1-10" });

      const [created] = await db.insert(moodLogs).values({
        userId,
        moodScore: String(moodScore),
        energyLevel: energyLevel != null ? String(energyLevel) : null,
        notes: notes ?? null,
      }).returning();

      // Pipeline integration: feed mood into health_samples
      try {
        await db.insert(healthSamples).values({
          userId,
          source: "manual",
          metric: "mood_score",
          value: score,
          unit: "score_1_10",
          recordedAt: created.loggedAt ?? new Date(),
        }).onConflictDoNothing();
      } catch (pipelineErr) {
        logger.error({ error: pipelineErr instanceof Error ? pipelineErr.message : String(pipelineErr) }, "Failed to insert mood into health_samples (non-fatal)");
      }

      return res.status(201).json(created);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to log mood");
      return res.status(500).json({ error: "Failed to log mood" });
    }
  });

  // GET /api/mood/:userId — get mood logs (last 30 days)
  app.get("/api/mood/:userId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const days = Math.min(Math.max(parseInt((req.query.days as string) || "30", 10), 1), 365);
      const fromDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

      const rows = await db.select().from(moodLogs)
        .where(and(eq(moodLogs.userId, userId), gte(moodLogs.loggedAt, fromDate)))
        .orderBy(desc(moodLogs.loggedAt));

      return res.json(rows);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to fetch mood logs");
      return res.status(500).json({ error: "Failed to fetch mood logs" });
    }
  });

  // ── Wearable Device Connection Routes ──────────────────────────────────────

  const INGEST_EDGE_FN = 'https://tpiyavugafhplsmwvrel.supabase.co/functions/v1/ingest-health-data';
  const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY ?? process.env.SUPABASE_SERVICE_ROLE_KEY ?? '';

  /** List connected devices for the authenticated user */
  app.get("/api/devices/:userId", async (req, res) => {
    const userId = requireOwnerExpress(req, res);
    if (!userId) return;

    try {
      const connections = await db.select({
        id: deviceConnections.id,
        provider: deviceConnections.provider,
        lastSyncAt: deviceConnections.lastSyncAt,
        syncStatus: deviceConnections.syncStatus,
        errorMessage: deviceConnections.errorMessage,
        connectedAt: deviceConnections.connectedAt,
        scopes: deviceConnections.scopes,
      })
        .from(deviceConnections)
        .where(eq(deviceConnections.userId, userId));

      return res.json({ devices: connections });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error) }, "Failed to list devices");
      return res.status(500).json({ error: "Failed to list devices" });
    }
  });

  /** Initiate OAuth flow for a wearable provider */
  app.post("/api/devices/connect/:provider", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    const { provider } = req.params;
    const adapter = wearableAdapters[provider];
    if (!adapter) return res.status(400).json({ error: `Unknown provider: ${provider}` });

    try {
      const state = crypto.randomBytes(16).toString('hex');
      // Store state in session for CSRF protection
      (req.session as any)[`oauth_state_${provider}`] = state;
      (req.session as any)[`oauth_user_${provider}`] = userId;

      const protocol = req.headers['x-forwarded-proto'] || req.protocol;
      const host = req.headers['x-forwarded-host'] || req.get('host');
      const redirectUri = `${protocol}://${host}/api/devices/callback/${provider}`;

      const authUrl = adapter.getAuthUrl(redirectUri, state);
      return res.json({ authUrl, state });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error), provider }, "Failed to initiate OAuth");
      return res.status(500).json({ error: "Failed to initiate OAuth flow" });
    }
  });

  /** Handle OAuth callback — exchange code for tokens, save connection, trigger initial sync */
  app.get("/api/devices/callback/:provider", async (req, res) => {
    const { provider } = req.params;
    const { code, state } = req.query;
    const adapter = wearableAdapters[provider];

    if (!adapter) return res.status(400).send("Unknown provider");
    if (!code || typeof code !== 'string') return res.status(400).send("Missing authorization code");

    // Verify state for CSRF protection
    const expectedState = (req.session as any)?.[`oauth_state_${provider}`];
    const userId = (req.session as any)?.[`oauth_user_${provider}`];

    if (!userId) return res.status(400).send("Session expired — please try connecting again from settings.");
    if (expectedState && state !== expectedState) return res.status(400).send("Invalid state parameter — possible CSRF attack.");

    try {
      const protocol = req.headers['x-forwarded-proto'] || req.protocol;
      const host = req.headers['x-forwarded-host'] || req.get('host');
      const redirectUri = `${protocol}://${host}/api/devices/callback/${provider}`;

      const tokens = await adapter.exchangeCode(code, redirectUri);

      // Upsert device connection (replace if already exists for this user+provider)
      await db.insert(deviceConnections)
        .values({
          userId,
          provider,
          accessToken: tokens.accessToken,
          refreshToken: tokens.refreshToken ?? null,
          tokenExpiresAt: tokens.expiresAt ?? null,
          scopes: tokens.scopes ?? null,
          syncStatus: 'active',
          errorMessage: null,
        })
        .onConflictDoUpdate({
          target: [deviceConnections.userId, deviceConnections.provider],
          set: {
            accessToken: tokens.accessToken,
            refreshToken: tokens.refreshToken ?? null,
            tokenExpiresAt: tokens.expiresAt ?? null,
            scopes: tokens.scopes ?? null,
            syncStatus: 'active',
            errorMessage: null,
            connectedAt: sql`now()`,
          },
        });

      // Clean up session state
      delete (req.session as any)[`oauth_state_${provider}`];
      delete (req.session as any)[`oauth_user_${provider}`];

      // Trigger initial sync (last 7 days) in the background
      const since = new Date();
      since.setDate(since.getDate() - 7);

      adapter.sync(tokens.accessToken, since).then(async (samples) => {
        if (samples.length > 0) {
          // Forward to Supabase Edge Function for ingestion
          try {
            await fetch(INGEST_EDGE_FN, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
              },
              body: JSON.stringify({ userId, samples }),
            });
          } catch (e) {
            logger.error({ error: e instanceof Error ? e.message : String(e), provider }, "Failed to ingest initial sync data");
          }
        }

        // Update last sync time
        await db.update(deviceConnections)
          .set({ lastSyncAt: new Date(), syncStatus: 'active', errorMessage: null })
          .where(and(eq(deviceConnections.userId, userId), eq(deviceConnections.provider, provider)));

        logger.info({ provider, userId, sampleCount: samples.length }, "Initial device sync completed");
      }).catch(async (error) => {
        logger.error({ error: error instanceof Error ? error.message : String(error), provider }, "Initial sync failed");
        await db.update(deviceConnections)
          .set({ syncStatus: 'error', errorMessage: error instanceof Error ? error.message : String(error) })
          .where(and(eq(deviceConnections.userId, userId), eq(deviceConnections.provider, provider)));
      });

      // Return success HTML that closes the popup window
      return res.send(`
        <!DOCTYPE html>
        <html><head><title>Connected</title></head>
        <body>
          <script>
            if (window.opener) {
              window.opener.postMessage({ type: 'device-connected', provider: '${provider}' }, '*');
              window.close();
            } else {
              document.body.innerHTML = '<h2>Device connected successfully. You can close this window.</h2>';
            }
          </script>
          <h2>Device connected successfully. Closing...</h2>
        </body></html>
      `);
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error), provider }, "OAuth callback failed");
      return res.status(500).send(`
        <!DOCTYPE html>
        <html><head><title>Connection Failed</title></head>
        <body>
          <h2>Failed to connect ${provider}.</h2>
          <p>${error instanceof Error ? error.message : 'Unknown error'}</p>
          <p>Please close this window and try again.</p>
        </body></html>
      `);
    }
  });

  /** Trigger manual sync for a connected provider */
  app.post("/api/devices/sync/:provider", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    const { provider } = req.params;
    const adapter = wearableAdapters[provider];
    if (!adapter) return res.status(400).json({ error: `Unknown provider: ${provider}` });

    try {
      // Get stored connection
      const [connection] = await db.select()
        .from(deviceConnections)
        .where(and(eq(deviceConnections.userId, userId), eq(deviceConnections.provider, provider)));

      if (!connection) return res.status(404).json({ error: "Device not connected" });

      let accessToken = connection.accessToken;

      // Refresh token if expired
      if (connection.tokenExpiresAt && new Date(connection.tokenExpiresAt) < new Date()) {
        if (!connection.refreshToken) {
          await db.update(deviceConnections)
            .set({ syncStatus: 'error', errorMessage: 'Token expired and no refresh token available' })
            .where(eq(deviceConnections.id, connection.id));
          return res.status(401).json({ error: "Token expired — please reconnect the device" });
        }

        const newTokens = await adapter.refreshToken(connection.refreshToken);
        accessToken = newTokens.accessToken;

        await db.update(deviceConnections)
          .set({
            accessToken: newTokens.accessToken,
            refreshToken: newTokens.refreshToken ?? connection.refreshToken,
            tokenExpiresAt: newTokens.expiresAt ?? null,
          })
          .where(eq(deviceConnections.id, connection.id));
      }

      // Sync from last sync time or last 24 hours
      const since = connection.lastSyncAt
        ? new Date(connection.lastSyncAt)
        : new Date(Date.now() - 24 * 60 * 60 * 1000);

      const samples = await adapter.sync(accessToken, since);

      if (samples.length > 0) {
        await fetch(INGEST_EDGE_FN, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
          },
          body: JSON.stringify({ userId, samples }),
        });
      }

      await db.update(deviceConnections)
        .set({ lastSyncAt: new Date(), syncStatus: 'active', errorMessage: null })
        .where(eq(deviceConnections.id, connection.id));

      return res.json({ synced: samples.length, lastSyncAt: new Date().toISOString() });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error), provider }, "Manual sync failed");

      // Update sync status to error
      await db.update(deviceConnections)
        .set({ syncStatus: 'error', errorMessage: error instanceof Error ? error.message : String(error) })
        .where(and(eq(deviceConnections.userId, userId), eq(deviceConnections.provider, provider)));

      return res.status(500).json({ error: "Sync failed" });
    }
  });

  /** Disconnect a device (delete from device_connections) */
  app.delete("/api/devices/:provider", async (req, res) => {
    const userId = getAuthUserId(req);
    if (!userId) return res.status(401).json({ error: "Unauthorized" });

    const { provider } = req.params;

    try {
      const result = await db.delete(deviceConnections)
        .where(and(eq(deviceConnections.userId, userId), eq(deviceConnections.provider, provider)));

      return res.json({ disconnected: true, provider });
    } catch (error) {
      logger.error({ error: error instanceof Error ? error.message : String(error), provider }, "Failed to disconnect device");
      return res.status(500).json({ error: "Failed to disconnect device" });
    }
  });

  // ── Anonymous Community Mood ──────────────────────────────────────────────
  // In-memory store — no personal data, just emotion counts
  const communityMoods: Array<{ emotion: string; timestamp: number }> = [];

  app.post("/api/community/share-mood", (req, res) => {
    try {
      const { emotion } = req.body;
      const VALID_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"];
      if (!emotion || !VALID_EMOTIONS.includes(emotion)) {
        return res.status(400).json({ message: "Invalid emotion" });
      }
      communityMoods.push({ emotion, timestamp: Date.now() });
      // Keep only last 24 hours
      const cutoff = Date.now() - 24 * 60 * 60 * 1000;
      while (communityMoods.length > 0 && communityMoods[0].timestamp < cutoff) {
        communityMoods.shift();
      }
      res.json({ ok: true });
    } catch (error) {
      res.status(500).json({ message: "Failed to share mood" });
    }
  });

  app.get("/api/community/mood-feed", (_req, res) => {
    try {
      const cutoff = Date.now() - 24 * 60 * 60 * 1000;
      const recent = communityMoods.filter(m => m.timestamp >= cutoff);
      // Aggregate by emotion
      const counts: Record<string, number> = {};
      recent.forEach(m => {
        counts[m.emotion] = (counts[m.emotion] || 0) + 1;
      });
      const total = recent.length;
      res.json({ counts, total });
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch mood feed" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
