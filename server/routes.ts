import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import OpenAI from "openai";
import { insertHealthMetricsSchema, insertDreamAnalysisSchema, insertAiChatSchema, insertUserSettingsSchema, insertEmotionReadingSchema } from "@shared/schema";
import { db } from "./db";
import { emotionReadings } from "@shared/schema";
import { eq, gte, lte, avg, and } from "drizzle-orm";

const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  : null;

export async function registerRoutes(app: Express): Promise<Server> {
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
      const { message, userId } = req.body;

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
      // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
      const response = await openai.chat.completions.create({
        model: "gpt-5",
        messages: [
          {
            role: "system",
            content: `You are an AI wellness companion for a Brain-Computer Interface system. You help users with mood analysis, stress relief, and wellness guidance. ${healthContext} Be supportive, insightful, and provide actionable advice. Keep responses concise but meaningful.`
          },
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

  const httpServer = createServer(app);
  return httpServer;
}
