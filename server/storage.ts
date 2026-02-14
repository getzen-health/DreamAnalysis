import { eq, desc, asc, sql } from "drizzle-orm";
import { type User, type InsertUser, type HealthMetrics, type InsertHealthMetrics, type DreamAnalysis, type InsertDreamAnalysis, type AiChat, type InsertAiChat, type UserSettings, type InsertUserSettings, type EmotionReading, type InsertEmotionReading, type DreamSymbol, type InsertDreamSymbol, users, healthMetrics, dreamAnalysis, aiChats, userSettings, emotionReadings, dreamSymbols } from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  getHealthMetrics(userId: string, limit?: number): Promise<HealthMetrics[]>;
  createHealthMetrics(metrics: InsertHealthMetrics): Promise<HealthMetrics>;
  getDreamAnalyses(userId: string, limit?: number): Promise<DreamAnalysis[]>;
  getDreamAnalysisById(id: string): Promise<DreamAnalysis | undefined>;
  createDreamAnalysis(analysis: InsertDreamAnalysis): Promise<DreamAnalysis>;
  updateDreamAnalysis(id: string, updates: Partial<DreamAnalysis>): Promise<DreamAnalysis | undefined>;
  getAiChats(userId: string, limit?: number): Promise<AiChat[]>;
  createAiChat(chat: InsertAiChat): Promise<AiChat>;
  getUserSettings(userId: string): Promise<UserSettings | undefined>;
  updateUserSettings(userId: string, settings: Partial<InsertUserSettings>): Promise<UserSettings>;
  getEmotionReadings(userId: string, limit?: number): Promise<EmotionReading[]>;
  createEmotionReading(reading: InsertEmotionReading): Promise<EmotionReading>;
  getDreamSymbols(userId: string): Promise<DreamSymbol[]>;
  upsertDreamSymbol(symbol: InsertDreamSymbol): Promise<DreamSymbol>;
  clearUserData(userId: string): Promise<void>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User>;
  private healthMetricsMap: Map<string, HealthMetrics>;
  private dreamAnalysesMap: Map<string, DreamAnalysis>;
  private aiChatsMap: Map<string, AiChat>;
  private userSettingsMap: Map<string, UserSettings>;
  private emotionReadingsMap: Map<string, EmotionReading>;
  private dreamSymbolsMap: Map<string, DreamSymbol>;

  constructor() {
    this.users = new Map();
    this.healthMetricsMap = new Map();
    this.dreamAnalysesMap = new Map();
    this.aiChatsMap = new Map();
    this.userSettingsMap = new Map();
    this.emotionReadingsMap = new Map();
    this.dreamSymbolsMap = new Map();
  }

  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { ...insertUser, id, email: insertUser.email || null, createdAt: new Date() };
    this.users.set(id, user);
    return user;
  }

  async getHealthMetrics(userId: string, limit = 50): Promise<HealthMetrics[]> {
    return Array.from(this.healthMetricsMap.values())
      .filter(metric => metric.userId === userId)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);
  }

  async createHealthMetrics(insertMetrics: InsertHealthMetrics): Promise<HealthMetrics> {
    const id = randomUUID();
    const metrics: HealthMetrics = {
      ...insertMetrics,
      id,
      timestamp: new Date(),
      userId: insertMetrics.userId || null,
      dailySteps: insertMetrics.dailySteps || null,
      sleepDuration: insertMetrics.sleepDuration || null
    };
    this.healthMetricsMap.set(id, metrics);
    return metrics;
  }

  async getDreamAnalyses(userId: string, limit = 20): Promise<DreamAnalysis[]> {
    return Array.from(this.dreamAnalysesMap.values())
      .filter(analysis => analysis.userId === userId)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);
  }

  async getDreamAnalysisById(id: string): Promise<DreamAnalysis | undefined> {
    return this.dreamAnalysesMap.get(id);
  }

  async createDreamAnalysis(insertAnalysis: InsertDreamAnalysis): Promise<DreamAnalysis> {
    const id = randomUUID();
    const analysis: DreamAnalysis = {
      ...insertAnalysis,
      id,
      timestamp: new Date(),
      userId: insertAnalysis.userId || null,
      symbols: insertAnalysis.symbols || [],
      emotions: insertAnalysis.emotions || [],
      aiAnalysis: insertAnalysis.aiAnalysis || null,
      imageUrl: insertAnalysis.imageUrl || null,
      lucidityScore: insertAnalysis.lucidityScore || null,
      sleepQuality: insertAnalysis.sleepQuality || null,
      voiceRecordingUrl: insertAnalysis.voiceRecordingUrl || null,
      tags: insertAnalysis.tags || null,
      sleepDuration: insertAnalysis.sleepDuration || null,
    };
    this.dreamAnalysesMap.set(id, analysis);
    return analysis;
  }

  async updateDreamAnalysis(id: string, updates: Partial<DreamAnalysis>): Promise<DreamAnalysis | undefined> {
    const existing = this.dreamAnalysesMap.get(id);
    if (!existing) return undefined;
    const updated = { ...existing, ...updates };
    this.dreamAnalysesMap.set(id, updated);
    return updated;
  }

  async getAiChats(userId: string, limit = 50): Promise<AiChat[]> {
    return Array.from(this.aiChatsMap.values())
      .filter(chat => chat.userId === userId)
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
      .slice(-limit);
  }

  async createAiChat(insertChat: InsertAiChat): Promise<AiChat> {
    const id = randomUUID();
    const chat: AiChat = {
      ...insertChat,
      id,
      timestamp: new Date(),
      userId: insertChat.userId || null
    };
    this.aiChatsMap.set(id, chat);
    return chat;
  }

  async getUserSettings(userId: string): Promise<UserSettings | undefined> {
    return Array.from(this.userSettingsMap.values()).find(settings => settings.userId === userId);
  }

  async updateUserSettings(userId: string, partialSettings: Partial<InsertUserSettings>): Promise<UserSettings> {
    const existing = await this.getUserSettings(userId);
    const id = existing?.id || randomUUID();
    const settings: UserSettings = {
      id,
      userId,
      theme: "dark",
      electrodeCount: 64,
      samplingRate: 500,
      alertThresholds: {},
      animationsEnabled: true,
      ...existing,
      ...partialSettings,
    };
    this.userSettingsMap.set(id, settings);
    return settings;
  }

  async getEmotionReadings(userId: string, limit = 50): Promise<EmotionReading[]> {
    return Array.from(this.emotionReadingsMap.values())
      .filter(r => r.userId === userId)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);
  }

  async createEmotionReading(insertReading: InsertEmotionReading): Promise<EmotionReading> {
    const id = randomUUID();
    const reading: EmotionReading = {
      ...insertReading,
      id,
      timestamp: new Date(),
      userId: insertReading.userId || null,
      valence: insertReading.valence || null,
      arousal: insertReading.arousal || null,
      eegSnapshot: insertReading.eegSnapshot || null,
    };
    this.emotionReadingsMap.set(id, reading);
    return reading;
  }

  async getDreamSymbols(userId: string): Promise<DreamSymbol[]> {
    return Array.from(this.dreamSymbolsMap.values())
      .filter(s => s.userId === userId);
  }

  async upsertDreamSymbol(insertSymbol: InsertDreamSymbol): Promise<DreamSymbol> {
    const existing = Array.from(this.dreamSymbolsMap.values()).find(
      s => s.userId === insertSymbol.userId && s.symbol === insertSymbol.symbol
    );
    if (existing) {
      const updated: DreamSymbol = {
        ...existing,
        frequency: (existing.frequency || 0) + 1,
        lastSeen: new Date(),
        meaning: insertSymbol.meaning || existing.meaning,
      };
      this.dreamSymbolsMap.set(existing.id, updated);
      return updated;
    }
    const id = randomUUID();
    const symbol: DreamSymbol = {
      id,
      userId: insertSymbol.userId || null,
      symbol: insertSymbol.symbol,
      meaning: insertSymbol.meaning || null,
      frequency: 1,
      firstSeen: new Date(),
      lastSeen: new Date(),
    };
    this.dreamSymbolsMap.set(id, symbol);
    return symbol;
  }

  async clearUserData(userId: string): Promise<void> {
    Array.from(this.healthMetricsMap.entries()).forEach(([key, val]) => {
      if (val.userId === userId) this.healthMetricsMap.delete(key);
    });
    Array.from(this.dreamAnalysesMap.entries()).forEach(([key, val]) => {
      if (val.userId === userId) this.dreamAnalysesMap.delete(key);
    });
    Array.from(this.aiChatsMap.entries()).forEach(([key, val]) => {
      if (val.userId === userId) this.aiChatsMap.delete(key);
    });
    Array.from(this.emotionReadingsMap.entries()).forEach(([key, val]) => {
      if (val.userId === userId) this.emotionReadingsMap.delete(key);
    });
    Array.from(this.dreamSymbolsMap.entries()).forEach(([key, val]) => {
      if (val.userId === userId) this.dreamSymbolsMap.delete(key);
    });
  }
}

export const storage: IStorage = new MemStorage();
