import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, serial, jsonb, timestamp, real, boolean, index, uniqueIndex } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(), // bcrypt hashed
  email: text("email").unique(),
  age: integer("age"),                  // for research demographics
  deviceType: text("device_type"),      // "muse_2" | "openbci_cyton" | "none"
  intent: varchar("intent", { length: 10 }), // 'study' | 'explore' | null (not yet chosen)
  createdAt: timestamp("created_at").defaultNow().notNull(),
  deletionRequestedAt: timestamp("deletion_requested_at"), // GDPR soft-delete; null = active
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
  userCorrectedEmotion: text("user_corrected_emotion"), // null = not yet corrected
  userCorrectedAt: timestamp("user_corrected_at"),
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

// ── Research Enrollment Module (30-day longitudinal study) ─────────────────

export const studyParticipants = pgTable("study_participants", {
  id:                    varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId:                varchar("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  studyId:               text("study_id").notNull(),              // "emotional-day-night-v1"
  studyCode:             varchar("study_code", { length: 6 }).notNull().unique(), // "NX4T82"
  enrolledAt:            timestamp("enrolled_at").defaultNow(),
  consentVersion:        text("consent_version").notNull(),        // "2.0"
  consentSignedAt:       timestamp("consent_signed_at").notNull(),
  consentFullName:       text("consent_full_name"),                // typed full name = digital signature
  consentInitials:       jsonb("consent_initials"),                // { [sectionId]: "SL" } per-section initials
  overnightEegConsent:   boolean("overnight_eeg_consent").default(false),
  status:                text("status").default("active"),         // "active" | "completed" | "withdrawn"
  targetDays:            integer("target_days").default(30),
  completedDays:         integer("completed_days").default(0),
  startDate:             timestamp("start_date").defaultNow(),
  withdrawnAt:           timestamp("withdrawn_at"),
  preferredMorningTime:  text("preferred_morning_time"),           // "07:00"
  preferredDaytimeTime:  text("preferred_daytime_time"),           // "10:00"
  preferredEveningTime:  text("preferred_evening_time"),           // "21:00"
}, (table) => [
  index("study_participants_user_idx").on(table.userId),
  index("study_participants_code_idx").on(table.studyCode),
]);

export const studySessions = pgTable("study_sessions", {
  id:                varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  participantId:     varchar("participant_id").notNull().references(() => studyParticipants.id, { onDelete: "cascade" }),
  studyCode:         varchar("study_code", { length: 6 }).notNull(),
  dayNumber:         integer("day_number").notNull(),               // 1–30
  sessionDate:       timestamp("session_date").notNull(),           // stored as midnight UTC
  morningCompleted:  boolean("morning_completed").default(false),
  daytimeCompleted:  boolean("daytime_completed").default(false),
  eveningCompleted:  boolean("evening_completed").default(false),
  validDay:          boolean("valid_day").default(false),           // true if ≥ 2 of 3 completed
  createdAt:         timestamp("created_at").defaultNow(),
}, (table) => [
  uniqueIndex("study_session_day_uidx").on(table.participantId, table.dayNumber),
  index("study_sessions_code_idx").on(table.studyCode),
]);

export const studyMorningEntries = pgTable("study_morning_entries", {
  id:                    varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  sessionId:             varchar("session_id").notNull().references(() => studySessions.id, { onDelete: "cascade" }),
  studyCode:             varchar("study_code", { length: 6 }).notNull(),
  dreamText:             text("dream_text"),                        // null if noRecall or skipped
  noRecall:              boolean("no_recall").default(false),
  dreamValence:          integer("dream_valence"),                  // SAM 1–9
  dreamArousal:          integer("dream_arousal"),                  // SAM 1–9
  nightmareFlag:         text("nightmare_flag"),                    // "yes" | "no" | "unsure"
  sleepQuality:          integer("sleep_quality"),                  // 1–9
  sleepHours:            real("sleep_hours"),
  minutesFromWaking:     integer("minutes_from_waking"),            // data quality metric
  currentMoodRating:     integer("current_mood_rating"),            // welfare check 1–9
  submittedAt:           timestamp("submitted_at").defaultNow(),
}, (table) => [
  index("study_morning_session_idx").on(table.sessionId),
]);

export const studyDaytimeEntries = pgTable("study_daytime_entries", {
  id:                   varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  sessionId:            varchar("session_id").notNull().references(() => studySessions.id, { onDelete: "cascade" }),
  studyCode:            varchar("study_code", { length: 6 }).notNull(),
  eegFeatures:          jsonb("eeg_features"),                      // 85-dim feature vector
  faa:                  real("faa"),                                // frontal alpha asymmetry
  highBeta:             real("high_beta"),                          // stress/anxiety power
  fmt:                  real("fmt"),                                // frontal midline theta
  sqiMean:              real("sqi_mean"),                           // signal quality index
  eegDurationSec:       integer("eeg_duration_sec"),
  samValence:           integer("sam_valence"),                     // 1–9
  samArousal:           integer("sam_arousal"),                     // 1–9
  samStress:            integer("sam_stress"),                      // 1–9
  panasItems:           jsonb("panas_items"),                       // {pa: number, na: number}
  sleepHoursReported:   real("sleep_hours_reported"),
  caffeineServings:     integer("caffeine_servings"),
  significantEventYN:   boolean("significant_event_yn"),
  submittedAt:          timestamp("submitted_at").defaultNow(),
}, (table) => [
  index("study_daytime_session_idx").on(table.sessionId),
]);

export const studyEveningEntries = pgTable("study_evening_entries", {
  id:                    varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  sessionId:             varchar("session_id").notNull().references(() => studySessions.id, { onDelete: "cascade" }),
  studyCode:             varchar("study_code", { length: 6 }).notNull(),
  dayValence:            integer("day_valence"),                    // 1–9
  dayArousal:            integer("day_arousal"),                    // 1–9
  peakEmotionIntensity:  integer("peak_emotion_intensity"),         // 1–9
  peakEmotionDirection:  text("peak_emotion_direction"),            // "positive" | "negative"
  meals:                 jsonb("meals"),                            // [{description, motivation, fullness, mindfulness}]
  emotionalEatingDay:    text("emotional_eating_day"),              // "yes" | "no" | "unsure"
  cravingsToday:         boolean("cravings_today"),
  cravingTypes:          jsonb("craving_types"),                    // ["sweet", "salty", ...]
  exerciseLevel:         text("exercise_level"),                    // "none"|"light"|"moderate"|"vigorous"
  alcoholDrinks:         integer("alcohol_drinks"),
  supplementsTaken:      jsonb("supplements_taken"),                // [{name, dosage, timeTaken}]
  medicationsTaken:      boolean("medications_taken"),
  medicationsDetails:    jsonb("medications_details"),              // [{name, dosage, timeTaken}]
  stressRightNow:        integer("stress_right_now"),               // 1–9
  readyForSleep:         boolean("ready_for_sleep"),
  submittedAt:           timestamp("submitted_at").defaultNow(),
}, (table) => [
  index("study_evening_session_idx").on(table.sessionId),
]);

// ── Insert schemas (research) ───────────────────────────────────────────────

export const insertStudyParticipantSchema = createInsertSchema(studyParticipants).omit({
  id: true,
  enrolledAt: true,
  completedDays: true,
  startDate: true,
  withdrawnAt: true,
});

export const insertStudySessionSchema = createInsertSchema(studySessions).omit({
  id: true,
  createdAt: true,
  morningCompleted: true,
  daytimeCompleted: true,
  eveningCompleted: true,
  validDay: true,
});

export const insertStudyMorningEntrySchema = createInsertSchema(studyMorningEntries).omit({
  id: true,
  submittedAt: true,
});

export const insertStudyDaytimeEntrySchema = createInsertSchema(studyDaytimeEntries).omit({
  id: true,
  submittedAt: true,
});

export const insertStudyEveningEntrySchema = createInsertSchema(studyEveningEntries).omit({
  id: true,
  submittedAt: true,
});

// ── Types (research — longitudinal 30-day study) ───────────────────────────

export type LongStudyParticipant = typeof studyParticipants.$inferSelect;
export type InsertLongStudyParticipant = z.infer<typeof insertStudyParticipantSchema>;
export type LongStudySession = typeof studySessions.$inferSelect;
export type InsertLongStudySession = z.infer<typeof insertStudySessionSchema>;
export type StudyMorningEntry = typeof studyMorningEntries.$inferSelect;
export type InsertStudyMorningEntry = z.infer<typeof insertStudyMorningEntrySchema>;
export type StudyDaytimeEntry = typeof studyDaytimeEntries.$inferSelect;
export type InsertStudyDaytimeEntry = z.infer<typeof insertStudyDaytimeEntrySchema>;
export type StudyEveningEntry = typeof studyEveningEntries.$inferSelect;
export type InsertStudyEveningEntry = z.infer<typeof insertStudyEveningEntrySchema>;

// ── Insert schemas (existing) ───────────────────────────────────────────────

// Insert schemas
export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
  email: true,
  age: true,
  deviceType: true,
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

// ── Rate limiting ────────────────────────────────────────────────────────────

export const rateLimitEntries = pgTable("rate_limit_entries", {
  key: text("key").primaryKey(),
  count: integer("count").notNull().default(0),
  windowStart: timestamp("window_start").notNull().defaultNow(),
});

// ── Password reset tokens ───────────────────────────────────────────────────

export const passwordResetTokens = pgTable("password_reset_tokens", {
  id:        serial("id").primaryKey(),
  userId:    varchar("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  token:     text("token").notNull().unique(),
  expiresAt: timestamp("expires_at").notNull(),
  usedAt:    timestamp("used_at"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
}, (table) => [
  index("prt_user_idx").on(table.userId),
  index("prt_token_idx").on(table.token),
]);

export type PasswordResetToken = typeof passwordResetTokens.$inferSelect;

// ── Food photo log ──────────────────────────────────────────────────────────

export const foodLogs = pgTable("food_logs", {
  id:               varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId:           varchar("user_id"),
  loggedAt:         timestamp("logged_at").defaultNow(),
  mealType:         text("meal_type"),              // "breakfast"|"lunch"|"dinner"|"snack"
  foodItems:        jsonb("food_items"),            // [{name, portion, calories, carbs_g, protein_g, fat_g}]
  totalCalories:    integer("total_calories"),
  dominantMacro:    text("dominant_macro"),         // "carbs"|"protein"|"fat"|"balanced"
  glycemicImpact:   text("glycemic_impact"),        // "low"|"medium"|"high"
  aiMoodImpact:     text("ai_mood_impact"),         // GPT prediction of mood effect
  aiDreamRelevance: text("ai_dream_relevance"),     // GPT prediction of sleep/dream effect
  summary:          text("summary"),               // one-sentence description
  moodBefore:       integer("mood_before"),         // 1-9 optional user rating
  notes:            text("notes"),
}, (table) => [
  index("food_logs_user_ts_idx").on(table.userId, table.loggedAt),
]);

export const insertFoodLogSchema = createInsertSchema(foodLogs).omit({ id: true, loggedAt: true });
export type FoodLog = typeof foodLogs.$inferSelect;
export type InsertFoodLog = z.infer<typeof insertFoodLogSchema>;

// ── Meal history (issues #367 + #378) ──────────────────────────────────────
// Stores multi-image meals with aggregated nutrition and favorite/re-log support.

export const mealHistory = pgTable("meal_history", {
  id:             varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId:         varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  images:         jsonb("images"),           // string[] — base64 or storage URLs
  foodItems:      jsonb("food_items"),        // FoodItem[] — full per-item breakdown
  totalCalories:  integer("total_calories"),
  totalProtein:   real("total_protein"),
  totalCarbs:     real("total_carbs"),
  totalFat:       real("total_fat"),
  totalFiber:     real("total_fiber"),
  mealType:       text("meal_type"),          // "breakfast"|"lunch"|"dinner"|"snack"
  isFavorite:     boolean("is_favorite").default(false),
  createdAt:      timestamp("created_at").defaultNow().notNull(),
}, (table) => [
  index("meal_history_user_ts_idx").on(table.userId, table.createdAt),
]);

export const insertMealHistorySchema = createInsertSchema(mealHistory).omit({
  id: true,
  createdAt: true,
});
export type MealHistory = typeof mealHistory.$inferSelect;
export type InsertMealHistory = z.infer<typeof insertMealHistorySchema>;

// ── Pilot study tables (US-001) ─────────────────────────────────────────────
// Anonymous consent + EEG session records for the 2-week human pilot study.
// Uses integer serial PKs and a participant_code slug (e.g. "P001") as the
// natural key, keeping these tables fully independent of the existing
// longitudinal study_participants / study_sessions tables above.

export const pilotParticipants = pgTable("pilot_participants", {
  id:                 serial("id").primaryKey(),
  participantCode:    varchar("participant_code", { length: 20 }).notNull().unique(),
  age:                integer("age"),
  dietType:           varchar("diet_type", { length: 20 }),  // "omnivore" | "vegetarian" | "vegan" | "other"
  hasAppleWatch:      boolean("has_apple_watch").default(false),
  consentText:        text("consent_text"),
  consentTimestamp:   timestamp("consent_timestamp"),
  researcherNotes:   text("researcher_notes"),
  createdAt:          timestamp("created_at").defaultNow(),
});

export const pilotSessions = pgTable("pilot_sessions", {
  id:                    serial("id").primaryKey(),
  participantCode:       varchar("participant_code", { length: 20 }).notNull(),
  blockType:             varchar("block_type", { length: 20 }).notNull(), // "stress" | "food" | "sleep"
  preEegJson:            jsonb("pre_eeg_json"),
  postEegJson:           jsonb("post_eeg_json"),
  eegFeaturesJson:       jsonb("eeg_features_json"),
  surveyJson:            jsonb("survey_json"),
  interventionTriggered: boolean("intervention_triggered").default(false),
  partial:               boolean("partial").default(false),
  phaseLog:              jsonb("phase_log"),
  checkpointAt:          timestamp("checkpoint_at"),
  dataQualityScore:      integer("data_quality_score"),
  durationSeconds:       integer("duration_seconds"),
  voiceEmotionJson:      jsonb("voice_emotion_json"),
  watchBiometricsJson:   jsonb("watch_biometrics_json"),
  startedAt:             timestamp("started_at"),
  createdAt:             timestamp("created_at").defaultNow(),
});

export const insertPilotParticipantSchema = createInsertSchema(pilotParticipants).omit({
  id: true,
  createdAt: true,
});

export const insertPilotSessionSchema = createInsertSchema(pilotSessions).omit({
  id: true,
  createdAt: true,
});

export type StudyParticipant = typeof pilotParticipants.$inferSelect;
export type InsertStudyParticipant = z.infer<typeof insertPilotParticipantSchema>;
export type StudySession = typeof pilotSessions.$inferSelect;
export type InsertStudySession = z.infer<typeof insertPilotSessionSchema>;
