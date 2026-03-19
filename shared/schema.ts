import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, serial, jsonb, timestamp, real, boolean, index, uniqueIndex, uuid, date, numeric, primaryKey } from "drizzle-orm/pg-core";
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

// ── User readings (voice / food / health / EEG) — training data accumulator ──
// Every analysis result from voice check-ins, food emotion, health emotion
// estimates, and EEG is persisted here so the ML pipeline can retrain on real
// user data over time.  The `features` column stores the raw feature vector
// used by the model so retraining does not need to re-compute features.

export const userReadings = pgTable("user_readings", {
  id:            varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId:        varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  source:        varchar("source", { length: 20 }).notNull(), // "voice" | "food" | "health" | "eeg"
  emotion:       varchar("emotion", { length: 30 }),
  valence:       real("valence"),
  arousal:       real("arousal"),
  stress:        real("stress"),
  confidence:    real("confidence"),
  modelType:     varchar("model_type", { length: 50 }),
  features:      jsonb("features"),                            // raw feature vector for retraining
  userCorrected: varchar("user_corrected", { length: 30 }),   // if user corrected the label
  createdAt:     timestamp("created_at").defaultNow().notNull(),
}, (table) => [
  index("user_readings_user_source_ts_idx").on(table.userId, table.source, table.createdAt),
]);

export const insertUserReadingSchema = createInsertSchema(userReadings).omit({
  id: true,
  createdAt: true,
});
export type UserReading = typeof userReadings.$inferSelect;
export type InsertUserReading = z.infer<typeof insertUserReadingSchema>;

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

// ============ PIPELINE TABLES ============

export const dailyAggregates = pgTable("daily_aggregates", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  date: date("date").notNull(),
  metric: text("metric").notNull(),
  avgValue: numeric("avg_value"),
  minValue: numeric("min_value"),
  maxValue: numeric("max_value"),
  sumValue: numeric("sum_value"),
  sampleCount: integer("sample_count").default(0),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => [
  uniqueIndex("daily_aggregates_user_date_metric_uidx").on(table.userId, table.date, table.metric),
]);

export const userBaselines = pgTable("user_baselines", {
  userId: varchar("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  metric: text("metric").notNull(),
  baselineAvg: numeric("baseline_avg"),
  baselineStddev: numeric("baseline_stddev"),
  sampleCount: integer("sample_count").default(0),
  lastUpdated: timestamp("last_updated").defaultNow(),
}, (table) => [
  primaryKey({ columns: [table.userId, table.metric] }),
]);

export const userScores = pgTable("user_scores", {
  userId: varchar("user_id").primaryKey().references(() => users.id, { onDelete: "cascade" }),
  recoveryScore: numeric("recovery_score"),
  sleepScore: numeric("sleep_score"),
  strainScore: numeric("strain_score"),
  stressScore: numeric("stress_score"),
  nutritionScore: numeric("nutrition_score"),
  energyBank: numeric("energy_bank"),
  recoveryInputs: jsonb("recovery_inputs"),
  sleepInputs: jsonb("sleep_inputs"),
  strainInputs: jsonb("strain_inputs"),
  stressInputs: jsonb("stress_inputs"),
  nutritionInputs: jsonb("nutrition_inputs"),
  computedAt: timestamp("computed_at").defaultNow(),
});

export const scoreHistory = pgTable("score_history", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  date: date("date").notNull(),
  recoveryScore: numeric("recovery_score"),
  sleepScore: numeric("sleep_score"),
  strainScore: numeric("strain_score"),
  stressScore: numeric("stress_score"),
  nutritionScore: numeric("nutrition_score"),
  energyBank: numeric("energy_bank"),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => [
  uniqueIndex("score_history_user_date_uidx").on(table.userId, table.date),
]);

export const trendAlerts = pgTable("trend_alerts", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  alertType: text("alert_type").notNull(),
  metric: text("metric").notNull(),
  severity: text("severity").notNull(),
  message: text("message").notNull(),
  value: numeric("value"),
  baseline: numeric("baseline"),
  acknowledged: boolean("acknowledged").default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

export const deviceConnections = pgTable("device_connections", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  provider: text("provider").notNull(),
  accessToken: text("access_token").notNull(),
  refreshToken: text("refresh_token"),
  tokenExpiresAt: timestamp("token_expires_at"),
  scopes: text("scopes").array(),
  lastSyncAt: timestamp("last_sync_at"),
  syncStatus: text("sync_status").default("active"),
  errorMessage: text("error_message"),
  connectedAt: timestamp("connected_at").defaultNow(),
}, (table) => [
  uniqueIndex("device_connections_user_provider_uidx").on(table.userId, table.provider),
]);

// ============ EXERCISE & BODY TABLES ============

export const exercises = pgTable("exercises", {
  id: uuid("id").primaryKey().defaultRandom(),
  name: text("name").notNull(),
  category: text("category").notNull(),
  muscleGroups: text("muscle_groups").array().notNull(),
  equipment: text("equipment"),
  instructions: text("instructions"),
  videoUrl: text("video_url"),
  isCustom: boolean("is_custom").default(false),
  createdBy: varchar("created_by").references(() => users.id),
});

export const workouts = pgTable("workouts", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  name: text("name"),
  workoutType: text("workout_type").notNull(),
  startedAt: timestamp("started_at").notNull(),
  endedAt: timestamp("ended_at"),
  durationMin: numeric("duration_min"),
  totalStrain: numeric("total_strain"),
  avgHr: numeric("avg_hr"),
  maxHr: numeric("max_hr"),
  caloriesBurned: numeric("calories_burned"),
  hrZones: jsonb("hr_zones"),
  hrRecovery: numeric("hr_recovery"),
  source: text("source").notNull(),
  eegSessionId: varchar("eeg_session_id"),
  notes: text("notes"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const workoutSets = pgTable("workout_sets", {
  id: uuid("id").primaryKey().defaultRandom(),
  workoutId: uuid("workout_id").references(() => workouts.id, { onDelete: "cascade" }),
  exerciseId: uuid("exercise_id").references(() => exercises.id),
  setNumber: integer("set_number").notNull(),
  setType: text("set_type").default("normal"),
  reps: integer("reps"),
  weightKg: numeric("weight_kg"),
  durationSec: integer("duration_sec"),
  restSec: integer("rest_sec"),
  rpe: numeric("rpe"),
  completed: boolean("completed").default(true),
  createdAt: timestamp("created_at").defaultNow(),
});

export const workoutTemplates = pgTable("workout_templates", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  name: text("name").notNull(),
  description: text("description"),
  exercises: jsonb("exercises").notNull(),
  isAiGenerated: boolean("is_ai_generated").default(false),
  timesUsed: integer("times_used").default(0),
  createdAt: timestamp("created_at").defaultNow(),
});

export const bodyMetrics = pgTable("body_metrics", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  weightKg: numeric("weight_kg"),
  bodyFatPct: numeric("body_fat_pct"),
  leanMassKg: numeric("lean_mass_kg"),
  bmi: numeric("bmi"),
  heightCm: numeric("height_cm"),
  source: text("source").notNull(),
  recordedAt: timestamp("recorded_at").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const exerciseHistory = pgTable("exercise_history", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  exerciseId: uuid("exercise_id").references(() => exercises.id),
  date: date("date").notNull(),
  bestWeightKg: numeric("best_weight_kg"),
  bestReps: integer("best_reps"),
  estimated1rm: numeric("estimated_1rm"),
  totalVolume: numeric("total_volume"),
}, (table) => [
  uniqueIndex("exercise_history_user_exercise_date_uidx").on(table.userId, table.exerciseId, table.date),
]);

// ============ LIFESTYLE TABLES ============

export const habits = pgTable("habits", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  name: text("name").notNull(),
  category: text("category"),
  icon: text("icon"),
  targetValue: numeric("target_value"),
  unit: text("unit"),
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
});

export const habitLogs = pgTable("habit_logs", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  habitId: uuid("habit_id").references(() => habits.id, { onDelete: "cascade" }),
  value: numeric("value").notNull(),
  note: text("note"),
  loggedAt: timestamp("logged_at").defaultNow(),
});

export const cycleTracking = pgTable("cycle_tracking", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  date: date("date").notNull(),
  flowLevel: text("flow_level"),
  symptoms: text("symptoms").array(),
  phase: text("phase"),
  contraception: text("contraception"),
  basalTemp: numeric("basal_temp"),
  notes: text("notes"),
}, (table) => [
  uniqueIndex("cycle_tracking_user_date_uidx").on(table.userId, table.date),
]);

// ── Circadian profiles (issue #410) ──────────────────────────────────────────
// Stores computed circadian rhythm parameters per user, updated daily.

export const circadianProfiles = pgTable("circadian_profiles", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  chronotype: text("chronotype").notNull(),                // "early_bird" | "night_owl" | "intermediate"
  chronotypeConfidence: real("chronotype_confidence"),      // 0–1
  acrophaseH: real("acrophase_h").notNull(),               // peak alertness hour (0–24)
  amplitude: real("amplitude"),                             // rhythm strength
  periodH: real("period_h").default(24.0),                 // fitted period
  phaseStability: real("phase_stability"),                  // 0–1
  predictedFocusWindow: text("predicted_focus_window"),     // e.g. "9:30am – 12:00pm"
  predictedSlumpWindow: text("predicted_slump_window"),     // e.g. "2:00pm – 3:30pm"
  phaseShiftHours: real("phase_shift_hours").default(0),   // drift from baseline
  fits: jsonb("fits"),                                      // per-stream cosinor fit details
  dataDays: integer("data_days").default(0),               // days of data used
  computedAt: timestamp("computed_at").defaultNow().notNull(),
}, (table) => [
  index("circadian_profiles_user_ts_idx").on(table.userId, table.computedAt),
]);

// ── Emotion calibration (issue #411) ─────────────────────────────────────────
// Stores paired observations of reported vs measured emotion for self-report calibration.

export const emotionCalibration = pgTable("emotion_calibration", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  reportedValence: real("reported_valence"),       // user self-report (-1 to 1)
  reportedArousal: real("reported_arousal"),        // user self-report (0 to 1)
  measuredValence: real("measured_valence"),        // EEG/voice measured (-1 to 1)
  measuredArousal: real("measured_arousal"),        // EEG/voice measured (0 to 1)
  eegValence: real("eeg_valence"),                 // EEG-only measurement
  voiceValence: real("voice_valence"),             // voice-only measurement
  channelAgreement: real("channel_agreement"),     // cosine similarity EEG<>voice
  reporterType: text("reporter_type"),             // computed: accurate|suppressor|amplifier|inconsistent
  awarenessScore: real("awareness_score"),          // computed: 0–100
  context: text("context"),                        // optional tag (e.g. "after_meeting")
  sessionId: varchar("session_id"),
  recordedAt: timestamp("recorded_at").defaultNow().notNull(),
}, (table) => [
  index("emotion_calibration_user_ts_idx").on(table.userId, table.recordedAt),
]);

export const moodLogs = pgTable("mood_logs", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: varchar("user_id").references(() => users.id, { onDelete: "cascade" }),
  moodScore: numeric("mood_score").notNull(),
  energyLevel: numeric("energy_level"),
  notes: text("notes"),
  loggedAt: timestamp("logged_at").defaultNow(),
});

// ── Insert schemas (pipeline + exercise + lifestyle) ─────────────────────────

export const insertDailyAggregateSchema = createInsertSchema(dailyAggregates).omit({
  id: true,
  updatedAt: true,
});

export const insertUserBaselineSchema = createInsertSchema(userBaselines).omit({
  lastUpdated: true,
});

export const insertUserScoreSchema = createInsertSchema(userScores).omit({
  computedAt: true,
});

export const insertScoreHistorySchema = createInsertSchema(scoreHistory).omit({
  id: true,
  createdAt: true,
});

export const insertTrendAlertSchema = createInsertSchema(trendAlerts).omit({
  id: true,
  acknowledged: true,
  createdAt: true,
});

export const insertDeviceConnectionSchema = createInsertSchema(deviceConnections).omit({
  id: true,
  connectedAt: true,
});

export const insertExerciseSchema = createInsertSchema(exercises).omit({
  id: true,
});

export const insertWorkoutSchema = createInsertSchema(workouts).omit({
  id: true,
  createdAt: true,
});

export const insertWorkoutSetSchema = createInsertSchema(workoutSets).omit({
  id: true,
  createdAt: true,
});

export const insertWorkoutTemplateSchema = createInsertSchema(workoutTemplates).omit({
  id: true,
  createdAt: true,
});

export const insertBodyMetricSchema = createInsertSchema(bodyMetrics).omit({
  id: true,
  createdAt: true,
});

export const insertExerciseHistorySchema = createInsertSchema(exerciseHistory).omit({
  id: true,
});

export const insertHabitSchema = createInsertSchema(habits).omit({
  id: true,
  createdAt: true,
});

export const insertHabitLogSchema = createInsertSchema(habitLogs).omit({
  id: true,
  loggedAt: true,
});

export const insertCycleTrackingSchema = createInsertSchema(cycleTracking).omit({
  id: true,
});

export const insertCircadianProfileSchema = createInsertSchema(circadianProfiles).omit({
  id: true,
  computedAt: true,
});
export type CircadianProfile = typeof circadianProfiles.$inferSelect;
export type InsertCircadianProfile = z.infer<typeof insertCircadianProfileSchema>;

export const insertEmotionCalibrationSchema = createInsertSchema(emotionCalibration).omit({
  id: true,
  recordedAt: true,
});
export type EmotionCalibrationEntry = typeof emotionCalibration.$inferSelect;
export type InsertEmotionCalibration = z.infer<typeof insertEmotionCalibrationSchema>;

export const insertMoodLogSchema = createInsertSchema(moodLogs).omit({
  id: true,
  loggedAt: true,
});

// ── Types (pipeline + exercise + lifestyle) ──────────────────────────────────

export type DailyAggregate = typeof dailyAggregates.$inferSelect;
export type InsertDailyAggregate = z.infer<typeof insertDailyAggregateSchema>;
export type UserBaseline = typeof userBaselines.$inferSelect;
export type InsertUserBaseline = z.infer<typeof insertUserBaselineSchema>;
export type UserScore = typeof userScores.$inferSelect;
export type InsertUserScore = z.infer<typeof insertUserScoreSchema>;
export type ScoreHistory = typeof scoreHistory.$inferSelect;
export type InsertScoreHistory = z.infer<typeof insertScoreHistorySchema>;
export type TrendAlert = typeof trendAlerts.$inferSelect;
export type InsertTrendAlert = z.infer<typeof insertTrendAlertSchema>;
export type DeviceConnection = typeof deviceConnections.$inferSelect;
export type InsertDeviceConnection = z.infer<typeof insertDeviceConnectionSchema>;
export type Exercise = typeof exercises.$inferSelect;
export type InsertExercise = z.infer<typeof insertExerciseSchema>;
export type Workout = typeof workouts.$inferSelect;
export type InsertWorkout = z.infer<typeof insertWorkoutSchema>;
export type WorkoutSet = typeof workoutSets.$inferSelect;
export type InsertWorkoutSet = z.infer<typeof insertWorkoutSetSchema>;
export type WorkoutTemplate = typeof workoutTemplates.$inferSelect;
export type InsertWorkoutTemplate = z.infer<typeof insertWorkoutTemplateSchema>;
export type BodyMetric = typeof bodyMetrics.$inferSelect;
export type InsertBodyMetric = z.infer<typeof insertBodyMetricSchema>;
export type ExerciseHistoryRecord = typeof exerciseHistory.$inferSelect;
export type InsertExerciseHistory = z.infer<typeof insertExerciseHistorySchema>;
export type Habit = typeof habits.$inferSelect;
export type InsertHabit = z.infer<typeof insertHabitSchema>;
export type HabitLog = typeof habitLogs.$inferSelect;
export type InsertHabitLog = z.infer<typeof insertHabitLogSchema>;
export type CycleTrackingEntry = typeof cycleTracking.$inferSelect;
export type InsertCycleTracking = z.infer<typeof insertCycleTrackingSchema>;
export type MoodLog = typeof moodLogs.$inferSelect;
export type InsertMoodLog = z.infer<typeof insertMoodLogSchema>;
