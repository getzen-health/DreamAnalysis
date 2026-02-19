import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import OpenAI from "openai";
import { insertHealthMetricsSchema, insertDreamAnalysisSchema, insertAiChatSchema, insertUserSettingsSchema } from "@shared/schema";

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

  const httpServer = createServer(app);
  return httpServer;
}
