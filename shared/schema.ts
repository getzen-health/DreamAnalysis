import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, jsonb, timestamp, real, boolean, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(), // bcrypt hashed
  email: text("email"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const healthMetrics = pgTable("health_metrics", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id),
  heartRate: integer("heart_rate").notNull(),
  stressLevel: integer("stress_level").notNull(),
  sleepQuality: integer("sleep_quality").notNull(),
  neuralActivity: integer("neural_activity").notNull(),
  dailySteps: integer("daily_steps"),
  sleepDuration: real("sleep_duration"),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
}, (table) => [
  index("health_metrics_user_ts_idx").on(table.userId, table.timestamp),
]);

export const dreamAnalysis = pgTable("dream_analysis", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id),
  dreamText: text("dream_text").notNull(),
  symbols: jsonb("symbols"),
  emotions: jsonb("emotions"),
  aiAnalysis: text("ai_analysis"),
  imageUrl: text("image_url"),
  lucidityScore: integer("lucidity_score"),
  sleepQuality: integer("sleep_quality"),
  voiceRecordingUrl: text("voice_recording_url"),
  tags: jsonb("tags"), // ['lucid', 'nightmare', 'recurring', 'vivid']
  sleepDuration: real("sleep_duration"),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
}, (table) => [
  index("dream_analysis_user_ts_idx").on(table.userId, table.timestamp),
]);

export const dreamSymbols = pgTable("dream_symbols", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id),
  symbol: text("symbol").notNull(),
  meaning: text("meaning"),
  frequency: integer("frequency").default(1),
  firstSeen: timestamp("first_seen").defaultNow().notNull(),
  lastSeen: timestamp("last_seen").defaultNow().notNull(),
}, (table) => [
  index("dream_symbols_user_idx").on(table.userId),
]);

export const emotionReadings = pgTable("emotion_readings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id),
  stress: real("stress").notNull(),
  happiness: real("happiness").notNull(),
  focus: real("focus").notNull(),
  energy: real("energy").notNull(),
  dominantEmotion: text("dominant_emotion").notNull(),
  valence: real("valence"), // -1 to 1 (negative to positive)
  arousal: real("arousal"), // 0 to 1 (low to high)
  eegSnapshot: jsonb("eeg_snapshot"),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
}, (table) => [
  index("emotion_readings_user_ts_idx").on(table.userId, table.timestamp),
]);

export const aiChats = pgTable("ai_chats", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id),
  message: text("message").notNull(),
  isUser: boolean("is_user").notNull(),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
}, (table) => [
  index("ai_chats_user_ts_idx").on(table.userId, table.timestamp),
]);

export const userSettings = pgTable("user_settings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id).unique(),
  theme: text("theme").default("dark"),
  electrodeCount: integer("electrode_count").default(64),
  samplingRate: integer("sampling_rate").default(500),
  alertThresholds: jsonb("alert_thresholds"),
  animationsEnabled: boolean("animations_enabled").default(true),
});

export const pushSubscriptions = pgTable("push_subscriptions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id),
  endpoint: text("endpoint").notNull(),
  keys: jsonb("keys").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// Insert schemas
export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
  email: true,
});

export const insertHealthMetricsSchema = createInsertSchema(healthMetrics).omit({
  id: true,
  timestamp: true,
});

export const insertDreamAnalysisSchema = createInsertSchema(dreamAnalysis).omit({
  id: true,
  timestamp: true,
});

export const insertDreamSymbolSchema = createInsertSchema(dreamSymbols).omit({
  id: true,
});

export const insertEmotionReadingSchema = createInsertSchema(emotionReadings).omit({
  id: true,
  timestamp: true,
});

export const insertAiChatSchema = createInsertSchema(aiChats).omit({
  id: true,
  timestamp: true,
});

export const insertUserSettingsSchema = createInsertSchema(userSettings).omit({
  id: true,
});

// Types
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type HealthMetrics = typeof healthMetrics.$inferSelect;
export type InsertHealthMetrics = z.infer<typeof insertHealthMetricsSchema>;
export type DreamAnalysis = typeof dreamAnalysis.$inferSelect;
export type InsertDreamAnalysis = z.infer<typeof insertDreamAnalysisSchema>;
export type DreamSymbol = typeof dreamSymbols.$inferSelect;
export type InsertDreamSymbol = z.infer<typeof insertDreamSymbolSchema>;
export type EmotionReading = typeof emotionReadings.$inferSelect;
export type InsertEmotionReading = z.infer<typeof insertEmotionReadingSchema>;
export type AiChat = typeof aiChats.$inferSelect;
export type InsertAiChat = z.infer<typeof insertAiChatSchema>;
export type UserSettings = typeof userSettings.$inferSelect;
export type InsertUserSettings = z.infer<typeof insertUserSettingsSchema>;
