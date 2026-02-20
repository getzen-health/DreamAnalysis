import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, jsonb, timestamp, real, boolean, index, uniqueIndex } from "drizzle-orm/pg-core";
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
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
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
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
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
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  symbol: text("symbol").notNull(),
  meaning: text("meaning"),
  frequency: integer("frequency").default(1),
  firstSeen: timestamp("first_seen").defaultNow().notNull(),
  lastSeen: timestamp("last_seen").defaultNow().notNull(),
}, (table) => [
  index("dream_symbols_user_idx").on(table.userId),
  uniqueIndex("dream_symbols_user_symbol_uidx").on(table.userId, table.symbol),
]);

export const emotionReadings = pgTable("emotion_readings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  sessionId: varchar("session_id"), // FK to eeg_sessions.session_id (no constraint to avoid forward-ref)
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
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  message: text("message").notNull(),
  isUser: boolean("is_user").notNull(),
  timestamp: timestamp("timestamp").defaultNow().notNull(),
}, (table) => [
  index("ai_chats_user_ts_idx").on(table.userId, table.timestamp),
]);

export const userSettings = pgTable("user_settings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }).unique(),
  theme: text("theme").default("dark"),
  electrodeCount: integer("electrode_count").default(64),
  samplingRate: integer("sampling_rate").default(500),
  alertThresholds: jsonb("alert_thresholds"),
  animationsEnabled: boolean("animations_enabled").default(true),
});

export const eegSessions = pgTable("eeg_sessions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  sessionId: varchar("session_id").notNull().unique(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }).notNull(),
  sessionType: text("session_type").default("general"),
  status: text("status").default("recording"), // 'recording' | 'completed'
  startTime: real("start_time"),               // unix epoch float
  endTime: real("end_time"),
  summary: jsonb("summary"),                    // duration, avg_focus, avg_stress, etc.
  signalR2Key: text("signal_r2_key"),           // e.g. users/{userId}/{sessionId}.npz
  createdAt: timestamp("created_at").defaultNow().notNull(),
}, (table) => [
  index("eeg_sessions_user_ts_idx").on(table.userId, table.startTime),
]);

export const pushSubscriptions = pgTable("push_subscriptions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  endpoint: text("endpoint").notNull(),
  keys: jsonb("keys").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// ── New tables added by Phase 2 ────────────────────────────────────────────

// Raw brain readings synced from Python timescale_writer
export const brainReadings = pgTable("brain_readings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  sessionId: varchar("session_id"),
  stress: real("stress"),
  focus: real("focus"),
  relaxation: real("relaxation"),
  flow: real("flow"),
  creativity: real("creativity"),
  valence: real("valence"),
  arousal: real("arousal"),
  dominantEmotion: text("dominant_emotion"),
  bandPowers: jsonb("band_powers"), // { delta, theta, alpha, beta, gamma }
  timestamp: timestamp("timestamp").defaultNow().notNull(),
}, (table) => [
  index("brain_readings_user_ts_idx").on(table.userId, table.timestamp),
]);

// Apple Health / Google Fit persistent samples
export const healthSamples = pgTable("health_samples", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  source: text("source").notNull(), // 'apple_health' | 'google_fit' | 'health_connect'
  metric: text("metric").notNull(), // 'heart_rate' | 'hrv' | 'steps' | 'sleep_duration' | etc.
  value: real("value").notNull(),
  unit: text("unit"),
  metadata: jsonb("metadata"),
  recordedAt: timestamp("recorded_at").notNull(),
  ingestedAt: timestamp("ingested_at").defaultNow().notNull(),
}, (table) => [
  index("health_samples_user_metric_ts_idx").on(table.userId, table.metric, table.recordedAt),
]);

// Datadog auto-remediation audit trail
export const datadogErrorLog = pgTable("datadog_error_log", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  monitorId: text("monitor_id"),
  monitorName: text("monitor_name"),
  alertType: text("alert_type"), // 'trigger' | 'recover'
  errorType: text("error_type"),
  payload: jsonb("payload"),
  remediationAction: text("remediation_action"),
  remediationStatus: text("remediation_status"), // 'success' | 'failed' | 'skipped'
  remediationDetail: text("remediation_detail"),
  receivedAt: timestamp("received_at").defaultNow().notNull(),
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

export const insertEegSessionSchema = createInsertSchema(eegSessions).omit({
  id: true,
  createdAt: true,
});
export type EegSession = typeof eegSessions.$inferSelect;
export type InsertEegSession = z.infer<typeof insertEegSessionSchema>;

export const insertBrainReadingSchema = createInsertSchema(brainReadings).omit({
  id: true,
  timestamp: true,
});
export type BrainReading = typeof brainReadings.$inferSelect;
export type InsertBrainReading = z.infer<typeof insertBrainReadingSchema>;

export const insertHealthSampleSchema = createInsertSchema(healthSamples).omit({
  id: true,
  ingestedAt: true,
});
export type HealthSample = typeof healthSamples.$inferSelect;
export type InsertHealthSample = z.infer<typeof insertHealthSampleSchema>;

export type DatadogErrorLog = typeof datadogErrorLog.$inferSelect;
