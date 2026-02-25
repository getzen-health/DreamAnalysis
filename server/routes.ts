import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import OpenAI from "openai";
import {
  insertHealthMetricsSchema, insertDreamAnalysisSchema, insertAiChatSchema,
  insertUserSettingsSchema, insertEmotionReadingSchema,
  studyParticipants, studySessions, studyMorningEntries,
  studyDaytimeEntries, studyEveningEntries, foodLogs,
  users,
} from "@shared/schema";
import { db } from "./db";
import { emotionReadings } from "@shared/schema";
import { eq, gte, lt, and, asc, desc, sql } from "drizzle-orm";

const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  : null;

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

// Ensure the hardcoded "default" user exists — runs once on startup.
// All research + food features use USER_ID = "default" on the client.
async function ensureDefaultUser() {
  await db.insert(users)
    .values({ id: "default", username: "default", password: "n/a" })
    .onConflictDoNothing();
}

export async function registerRoutes(app: Express): Promise<Server> {
  // Seed the default user so FK constraints never block self-study usage
  await ensureDefaultUser();

  // Health metrics endpoints
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
  app.get("/api/dream-analysis/:userId", async (req, res) => {
    try {
      const analyses = await storage.getDreamAnalyses(req.params.userId);
      res.json(analyses);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch dream analyses" });
    }
  });

  app.post("/api/dream-analysis", async (req, res) => {
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

  // AI chat endpoints
  app.get("/api/ai-chat/:userId", async (req, res) => {
    try {
      const chats = await storage.getAiChats(req.params.userId);
      res.json(chats);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch chat history" });
    }
  });

  app.post("/api/ai-chat", async (req, res) => {
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

  // Mood analysis endpoint
  app.post("/api/analyze-mood", async (req, res) => {
    try {
      const { text, userId } = req.body;

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

  // User settings endpoints
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

  // Data export endpoint
  app.get("/api/export/:userId", async (req, res) => {
    try {
      const metrics = await storage.getHealthMetrics(req.params.userId);
      const dreams = await storage.getDreamAnalyses(req.params.userId);
      
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
    } catch (error) {
      res.status(400).json({ message: "Invalid emotion readings data" });
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
      console.log(`[datadog-webhook] ${alertType} — ${monitorName} (${monitorId})`);

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
              medicationsTaken, stressRightNow, readyForSleep } = req.body;

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
        exerciseLevel, alcoholDrinks, medicationsTaken, stressRightNow, readyForSleep,
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
  "summary": "One plain-English sentence describing what was eaten"
}`;

      // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      let response;
      if (imageBase64) {
        // Vision path — analyze a photo
        response = await openai.chat.completions.create({
          model: "gpt-5",
          messages: [{
            role: "user",
            content: [
              { type: "text", text: `Analyze this food photo. Return ONLY valid JSON (no markdown fences) with this exact shape:\n${JSON_SCHEMA}` },
              { type: "image_url", image_url: { url: `data:image/jpeg;base64,${imageBase64}`, detail: "low" } },
            ],
          }],
          max_tokens: 700,
        });
      } else {
        // Text path — analyze a written description
        response = await openai.chat.completions.create({
          model: "gpt-5",
          messages: [{
            role: "user",
            content: `The user describes their ${mealType ?? "meal"}: "${textDescription}"\n\nEstimate nutrition and return ONLY valid JSON (no markdown fences) with this exact shape:\n${JSON_SCHEMA}`,
          }],
          max_tokens: 700,
        });
      }

      const raw = response.choices[0].message.content ?? "{}";
      let analysis: Record<string, unknown>;
      try {
        analysis = JSON.parse(raw);
      } catch {
        // GPT sometimes wraps in markdown despite instructions — strip fences
        const stripped = raw.replace(/```json?\n?/g, "").replace(/```/g, "").trim();
        analysis = JSON.parse(stripped);
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

      res.json({ ...analysis, id: log.id, loggedAt: log.loggedAt });
    } catch (error) {
      console.error("Food analyze error:", error);
      res.status(500).json({ message: "Food analysis failed" });
    }
  });

  // GET /api/food/logs/:userId — recent food logs (last 20)
  app.get("/api/food/logs/:userId", async (req, res) => {
    try {
      const logs = await db.select().from(foodLogs)
        .where(eq(foodLogs.userId, req.params.userId))
        .orderBy(desc(foodLogs.loggedAt))
        .limit(20);
      res.json(logs);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch food logs" });
    }
  });

  // GET /api/research/correlation/:userId — last 7 days with food + EEG mood + dream data joined
  app.get("/api/research/correlation/:userId", async (req, res) => {
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

  const httpServer = createServer(app);
  return httpServer;
}
