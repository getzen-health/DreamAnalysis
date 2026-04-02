import { eq, desc, asc, sql, and, gte, lte } from "drizzle-orm";
import { type User, type InsertUser, type HealthMetrics, type InsertHealthMetrics, type DreamAnalysis, type InsertDreamAnalysis, type AiChat, type InsertAiChat, type UserSettings, type InsertUserSettings, type EmotionReading, type InsertEmotionReading, type DreamSymbol, type InsertDreamSymbol, type IrtSession, type InsertIrtSession, users, healthMetrics, dreamAnalysis, aiChats, userSettings, emotionReadings, dreamSymbols, irtSessions } from "@shared/schema";
import { randomUUID } from "crypto";
import { db } from "./db";

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
  getEmotionReadings(userId: string, limit?: number, fromTs?: Date, toTs?: Date): Promise<EmotionReading[]>;
  createEmotionReading(reading: InsertEmotionReading): Promise<EmotionReading>;
  batchCreateEmotionReadings(readings: InsertEmotionReading[]): Promise<EmotionReading[]>;
  getDreamSymbols(userId: string): Promise<DreamSymbol[]>;
  upsertDreamSymbol(symbol: InsertDreamSymbol): Promise<DreamSymbol>;
  createIrtSession(session: InsertIrtSession): Promise<IrtSession>;
  getIrtSessions(userId: string, limit?: number): Promise<IrtSession[]>;
  clearUserData(userId: string): Promise<void>;
}

export class DbStorage implements IStorage {
  async getUser(id: string): Promise<User | undefined> {
    const rows = await db.select().from(users).where(eq(users.id, id)).limit(1);
    return rows[0];
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const rows = await db.select().from(users).where(eq(users.username, username)).limit(1);
    return rows[0];
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const rows = await db.insert(users).values(insertUser).returning();
    return rows[0];
  }

  async getHealthMetrics(userId: string, limit = 50): Promise<HealthMetrics[]> {
    return db
      .select()
      .from(healthMetrics)
      .where(eq(healthMetrics.userId, userId))
      .orderBy(desc(healthMetrics.timestamp))
      .limit(limit);
  }

  async createHealthMetrics(insertMetrics: InsertHealthMetrics): Promise<HealthMetrics> {
    const rows = await db.insert(healthMetrics).values(insertMetrics).returning();
    return rows[0];
  }

  async getDreamAnalyses(userId: string, limit = 20): Promise<DreamAnalysis[]> {
    return db
      .select()
      .from(dreamAnalysis)
      .where(eq(dreamAnalysis.userId, userId))
      .orderBy(desc(dreamAnalysis.timestamp))
      .limit(limit);
  }

  async getDreamAnalysisById(id: string): Promise<DreamAnalysis | undefined> {
    const rows = await db.select().from(dreamAnalysis).where(eq(dreamAnalysis.id, id)).limit(1);
    return rows[0];
  }

  async createDreamAnalysis(insertAnalysis: InsertDreamAnalysis): Promise<DreamAnalysis> {
    const payload = {
      ...insertAnalysis,
      themes: Array.isArray(insertAnalysis.themes) ? (insertAnalysis.themes as string[]) : null,
    };
    const rows = await db.insert(dreamAnalysis).values(payload).returning();
    return rows[0];
  }

  async updateDreamAnalysis(id: string, updates: Partial<DreamAnalysis>): Promise<DreamAnalysis | undefined> {
    const rows = await db
      .update(dreamAnalysis)
      .set(updates)
      .where(eq(dreamAnalysis.id, id))
      .returning();
    return rows[0];
  }

  async getAiChats(userId: string, limit = 50): Promise<AiChat[]> {
    return db
      .select()
      .from(aiChats)
      .where(eq(aiChats.userId, userId))
      .orderBy(asc(aiChats.timestamp))
      .limit(limit);
  }

  async createAiChat(insertChat: InsertAiChat): Promise<AiChat> {
    const rows = await db.insert(aiChats).values(insertChat).returning();
    return rows[0];
  }

  async getUserSettings(userId: string): Promise<UserSettings | undefined> {
    const rows = await db
      .select()
      .from(userSettings)
      .where(eq(userSettings.userId, userId))
      .limit(1);
    return rows[0];
  }

  async updateUserSettings(userId: string, partialSettings: Partial<InsertUserSettings>): Promise<UserSettings> {
    const existing = await this.getUserSettings(userId);
    if (existing) {
      const rows = await db
        .update(userSettings)
        .set(partialSettings)
        .where(eq(userSettings.userId, userId))
        .returning();
      return rows[0];
    }
    const rows = await db
      .insert(userSettings)
      .values({
        userId,
        theme: "dark",
        electrodeCount: 64,
        samplingRate: 500,
        alertThresholds: {},
        animationsEnabled: true,
        ...partialSettings,
      })
      .returning();
    return rows[0];
  }

  async getEmotionReadings(userId: string, limit = 50, fromTs?: Date, toTs?: Date): Promise<EmotionReading[]> {
    const conditions = [eq(emotionReadings.userId, userId)];
    if (fromTs) conditions.push(gte(emotionReadings.timestamp, fromTs));
    if (toTs) conditions.push(lte(emotionReadings.timestamp, toTs));

    return db
      .select()
      .from(emotionReadings)
      .where(and(...conditions))
      .orderBy(desc(emotionReadings.timestamp))
      .limit(limit);
  }

  async createEmotionReading(insertReading: InsertEmotionReading): Promise<EmotionReading> {
    const rows = await db.insert(emotionReadings).values(insertReading).returning();
    return rows[0];
  }

  async batchCreateEmotionReadings(readings: InsertEmotionReading[]): Promise<EmotionReading[]> {
    if (readings.length === 0) return [];
    return db.insert(emotionReadings).values(readings).returning();
  }

  async getDreamSymbols(userId: string): Promise<DreamSymbol[]> {
    return db
      .select()
      .from(dreamSymbols)
      .where(eq(dreamSymbols.userId, userId));
  }

  async upsertDreamSymbol(insertSymbol: InsertDreamSymbol): Promise<DreamSymbol> {
    const rows = await db
      .insert(dreamSymbols)
      .values({
        ...insertSymbol,
        frequency: 1,
        firstSeen: new Date(),
        lastSeen: new Date(),
      })
      .onConflictDoUpdate({
        target: [dreamSymbols.userId, dreamSymbols.symbol],
        set: {
          frequency: sql`${dreamSymbols.frequency} + 1`,
          lastSeen: new Date(),
          meaning: insertSymbol.meaning ?? sql`${dreamSymbols.meaning}`,
        },
      })
      .returning();
    return rows[0];
  }

  async createIrtSession(session: InsertIrtSession): Promise<IrtSession> {
    const rows = await db.insert(irtSessions).values(session).returning();
    return rows[0];
  }

  async getIrtSessions(userId: string, limit = 20): Promise<IrtSession[]> {
    return db
      .select()
      .from(irtSessions)
      .where(eq(irtSessions.userId, userId))
      .orderBy(desc(irtSessions.createdAt))
      .limit(limit);
  }

  async clearUserData(userId: string): Promise<void> {
    await Promise.all([
      db.delete(healthMetrics).where(eq(healthMetrics.userId, userId)),
      db.delete(dreamAnalysis).where(eq(dreamAnalysis.userId, userId)),
      db.delete(aiChats).where(eq(aiChats.userId, userId)),
      db.delete(emotionReadings).where(eq(emotionReadings.userId, userId)),
      db.delete(dreamSymbols).where(eq(dreamSymbols.userId, userId)),
      db.delete(irtSessions).where(eq(irtSessions.userId, userId)),
    ]);
  }
}

export class MemStorage implements IStorage {
  private users: Map<string, User>;
  private healthMetricsMap: Map<string, HealthMetrics>;
  private dreamAnalysesMap: Map<string, DreamAnalysis>;
  private aiChatsMap: Map<string, AiChat>;
  private userSettingsMap: Map<string, UserSettings>;
  private emotionReadingsMap: Map<string, EmotionReading>;
  private dreamSymbolsMap: Map<string, DreamSymbol>;
  private irtSessionsMap: Map<string, IrtSession>;

  constructor() {
    this.users = new Map();
    this.healthMetricsMap = new Map();
    this.dreamAnalysesMap = new Map();
    this.aiChatsMap = new Map();
    this.userSettingsMap = new Map();
    this.emotionReadingsMap = new Map();
    this.dreamSymbolsMap = new Map();
    this.irtSessionsMap = new Map();
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
    const user: User = { ...insertUser, id, email: insertUser.email || null, age: insertUser.age ?? null, deviceType: insertUser.deviceType ?? null, intent: null, role: 'user', createdAt: new Date(), deletionRequestedAt: null };
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
      themes: (Array.isArray(insertAnalysis.themes) ? insertAnalysis.themes as string[] : null),
      emotionalArc: insertAnalysis.emotionalArc || null,
      keyInsight: insertAnalysis.keyInsight || null,
      threatSimulationIndex: insertAnalysis.threatSimulationIndex ?? null,
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

  async getEmotionReadings(userId: string, limit = 50, fromTs?: Date, toTs?: Date): Promise<EmotionReading[]> {
    return Array.from(this.emotionReadingsMap.values())
      .filter(r => {
        if (r.userId !== userId) return false;
        const ts = new Date(r.timestamp).getTime();
        if (fromTs && ts < fromTs.getTime()) return false;
        if (toTs && ts > toTs.getTime()) return false;
        return true;
      })
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
      sessionId: insertReading.sessionId || null,
      valence: insertReading.valence || null,
      arousal: insertReading.arousal || null,
      eegSnapshot: insertReading.eegSnapshot || null,
      userCorrectedEmotion: insertReading.userCorrectedEmotion ?? null,
      userCorrectedAt: insertReading.userCorrectedAt ?? null,
    };
    this.emotionReadingsMap.set(id, reading);
    return reading;
  }

  async batchCreateEmotionReadings(readings: InsertEmotionReading[]): Promise<EmotionReading[]> {
    return Promise.all(readings.map(r => this.createEmotionReading(r)));
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

  async createIrtSession(session: InsertIrtSession): Promise<IrtSession> {
    const id = randomUUID();
    const row: IrtSession = {
      id,
      userId: session.userId || null,
      originalDreamText: session.originalDreamText,
      rewrittenEnding: session.rewrittenEnding,
      rehearsalNote: session.rehearsalNote || null,
      createdAt: new Date(),
    };
    this.irtSessionsMap.set(id, row);
    return row;
  }

  async getIrtSessions(userId: string, limit = 20): Promise<IrtSession[]> {
    return Array.from(this.irtSessionsMap.values())
      .filter(s => s.userId === userId)
      .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
      .slice(0, limit);
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
    Array.from(this.irtSessionsMap.entries()).forEach(([key, val]) => {
      if (val.userId === userId) this.irtSessionsMap.delete(key);
    });
  }
}

export const storage: IStorage = process.env.DATABASE_URL
  ? new DbStorage()
  : new MemStorage();
