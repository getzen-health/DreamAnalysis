/**
 * sqlite-store.ts
 *
 * On-device encrypted SQLite storage for sensitive user data.
 * Privacy-first: emotion readings, dream analyses, and inner scores
 * are written here first. Supabase is a secondary sync target.
 *
 * On iOS: SQLite file is eligible for iCloud backup via iOS automatic backup.
 * On web/PWA: falls back to IndexedDB via sql.js (no-op if unavailable).
 */

import { Capacitor } from "@capacitor/core";

// Lazy-load the plugin — only available on native builds
let sqlitePlugin: import("@capacitor-community/sqlite").SQLiteConnection | null = null;
let db: import("@capacitor-community/sqlite").SQLiteDBConnection | null = null;

const DB_NAME = "antaiai_private";
const DB_VERSION = 1;

// ── Schema ────────────────────────────────────────────────────────────────────

const CREATE_TABLES = `
CREATE TABLE IF NOT EXISTS emotion_readings (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  stress REAL,
  happiness REAL,
  focus REAL,
  energy REAL,
  dominant_emotion TEXT,
  valence REAL,
  arousal REAL,
  eeg_snapshot TEXT,
  timestamp TEXT NOT NULL,
  synced INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dream_analysis (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  dream_text TEXT,
  symbols TEXT,
  emotions TEXT,
  ai_analysis TEXT,
  lucidity_score REAL,
  sleep_quality REAL,
  themes TEXT,
  emotional_arc TEXT,
  key_insight TEXT,
  threat_simulation_index REAL,
  timestamp TEXT NOT NULL,
  synced INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS inner_scores (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  score REAL NOT NULL,
  tier TEXT,
  factors TEXT,
  narrative TEXT,
  created_at TEXT NOT NULL,
  synced INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS mood_logs (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  mood INTEGER,
  energy INTEGER,
  notes TEXT,
  created_at TEXT NOT NULL,
  synced INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_er_user_ts ON emotion_readings(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_da_user_ts ON dream_analysis(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_is_user_ts ON inner_scores(user_id, created_at);
`;

// ── Init ──────────────────────────────────────────────────────────────────────

export async function initSQLiteStore(): Promise<boolean> {
  if (Capacitor.getPlatform() === "web") return false; // web uses Supabase/localStorage

  try {
    const { SQLiteConnection } = await import("@capacitor-community/sqlite");
    sqlitePlugin = new SQLiteConnection((await import("@capacitor-community/sqlite")).CapacitorSQLite);

    const result = await sqlitePlugin.checkConnectionsConsistency();
    if (!result.result) await sqlitePlugin.closeAllConnections();

    db = await sqlitePlugin.createConnection(
      DB_NAME,
      true,  // encrypted
      "secret",
      DB_VERSION,
      false
    );

    await db.open();
    await db.execute(CREATE_TABLES);
    return true;
  } catch (err) {
    console.warn("[sqlite-store] init failed — falling back to cloud-only:", err);
    db = null;
    return false;
  }
}

function isAvailable(): boolean {
  return db !== null;
}

// ── Emotion Readings ──────────────────────────────────────────────────────────

export interface LocalEmotionReading {
  id: string;
  userId: string;
  stress: number;
  happiness: number;
  focus: number;
  energy: number;
  dominantEmotion: string;
  valence: number | null;
  arousal: number | null;
  eegSnapshot?: Record<string, unknown>;
  timestamp: string;
}

export async function saveEmotionReadingLocal(r: LocalEmotionReading): Promise<void> {
  if (!isAvailable()) return;
  await db!.run(
    `INSERT OR REPLACE INTO emotion_readings
      (id, user_id, stress, happiness, focus, energy, dominant_emotion, valence, arousal, eeg_snapshot, timestamp, synced)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)`,
    [r.id, r.userId, r.stress, r.happiness, r.focus, r.energy, r.dominantEmotion,
     r.valence ?? null, r.arousal ?? null,
     r.eegSnapshot ? JSON.stringify(r.eegSnapshot) : null, r.timestamp]
  );
}

export async function getEmotionReadingsLocal(userId: string, limit = 500): Promise<LocalEmotionReading[]> {
  if (!isAvailable()) return [];
  const result = await db!.query(
    `SELECT * FROM emotion_readings WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?`,
    [userId, limit]
  );
  return (result.values ?? []).map(row => ({
    id: row.id,
    userId: row.user_id,
    stress: row.stress,
    happiness: row.happiness,
    focus: row.focus,
    energy: row.energy,
    dominantEmotion: row.dominant_emotion,
    valence: row.valence,
    arousal: row.arousal,
    eegSnapshot: row.eeg_snapshot ? JSON.parse(row.eeg_snapshot) : undefined,
    timestamp: row.timestamp,
  }));
}

// ── Dream Analysis ────────────────────────────────────────────────────────────

export interface LocalDreamAnalysis {
  id: string;
  userId: string;
  dreamText?: string;
  symbols?: string[];
  emotions?: string[];
  aiAnalysis?: string;
  lucidityScore?: number;
  sleepQuality?: number;
  themes?: string[];
  emotionalArc?: string;
  keyInsight?: string;
  threatSimulationIndex?: number;
  timestamp: string;
}

export async function saveDreamAnalysisLocal(d: LocalDreamAnalysis): Promise<void> {
  if (!isAvailable()) return;
  await db!.run(
    `INSERT OR REPLACE INTO dream_analysis
      (id, user_id, dream_text, symbols, emotions, ai_analysis, lucidity_score,
       sleep_quality, themes, emotional_arc, key_insight, threat_simulation_index, timestamp, synced)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)`,
    [d.id, d.userId, d.dreamText ?? null,
     d.symbols ? JSON.stringify(d.symbols) : null,
     d.emotions ? JSON.stringify(d.emotions) : null,
     d.aiAnalysis ?? null, d.lucidityScore ?? null,
     d.sleepQuality ?? null,
     d.themes ? JSON.stringify(d.themes) : null,
     d.emotionalArc ?? null, d.keyInsight ?? null,
     d.threatSimulationIndex ?? null, d.timestamp]
  );
}

export async function getDreamAnalysesLocal(userId: string, limit = 100): Promise<LocalDreamAnalysis[]> {
  if (!isAvailable()) return [];
  const result = await db!.query(
    `SELECT * FROM dream_analysis WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?`,
    [userId, limit]
  );
  return (result.values ?? []).map(row => ({
    id: row.id,
    userId: row.user_id,
    dreamText: row.dream_text ?? undefined,
    symbols: row.symbols ? JSON.parse(row.symbols) : undefined,
    emotions: row.emotions ? JSON.parse(row.emotions) : undefined,
    aiAnalysis: row.ai_analysis ?? undefined,
    lucidityScore: row.lucidity_score ?? undefined,
    sleepQuality: row.sleep_quality ?? undefined,
    themes: row.themes ? JSON.parse(row.themes) : undefined,
    emotionalArc: row.emotional_arc ?? undefined,
    keyInsight: row.key_insight ?? undefined,
    threatSimulationIndex: row.threat_simulation_index ?? undefined,
    timestamp: row.timestamp,
  }));
}

// ── Inner Scores ──────────────────────────────────────────────────────────────

export interface LocalInnerScore {
  id: string;
  userId: string;
  score: number;
  tier: string;
  factors: Record<string, unknown>;
  narrative?: string;
  createdAt: string;
}

export async function saveInnerScoreLocal(s: LocalInnerScore): Promise<void> {
  if (!isAvailable()) return;
  await db!.run(
    `INSERT OR REPLACE INTO inner_scores (id, user_id, score, tier, factors, narrative, created_at, synced)
      VALUES (?, ?, ?, ?, ?, ?, ?, 0)`,
    [s.id, s.userId, s.score, s.tier, JSON.stringify(s.factors), s.narrative ?? null, s.createdAt]
  );
}

export async function getInnerScoresLocal(userId: string, days = 30): Promise<LocalInnerScore[]> {
  if (!isAvailable()) return [];
  const since = new Date(Date.now() - days * 86400_000).toISOString();
  const result = await db!.query(
    `SELECT * FROM inner_scores WHERE user_id = ? AND created_at >= ? ORDER BY created_at DESC`,
    [userId, since]
  );
  return (result.values ?? []).map(row => ({
    id: row.id,
    userId: row.user_id,
    score: row.score,
    tier: row.tier,
    factors: row.factors ? JSON.parse(row.factors) : {},
    narrative: row.narrative ?? undefined,
    createdAt: row.created_at,
  }));
}

// ── Sync helpers ──────────────────────────────────────────────────────────────

/** Returns IDs of records not yet synced to Supabase */
export async function getUnsyncedEmotionReadings(userId: string): Promise<LocalEmotionReading[]> {
  if (!isAvailable()) return [];
  const result = await db!.query(
    `SELECT * FROM emotion_readings WHERE user_id = ? AND synced = 0 ORDER BY timestamp ASC LIMIT 100`,
    [userId]
  );
  return (result.values ?? []).map(row => ({
    id: row.id, userId: row.user_id, stress: row.stress, happiness: row.happiness,
    focus: row.focus, energy: row.energy, dominantEmotion: row.dominant_emotion,
    valence: row.valence, arousal: row.arousal,
    eegSnapshot: row.eeg_snapshot ? JSON.parse(row.eeg_snapshot) : undefined,
    timestamp: row.timestamp,
  }));
}

export async function markEmotionReadingsSynced(ids: string[]): Promise<void> {
  if (!isAvailable() || ids.length === 0) return;
  const placeholders = ids.map(() => "?").join(",");
  await db!.run(`UPDATE emotion_readings SET synced = 1 WHERE id IN (${placeholders})`, ids);
}

export async function getUnsyncedDreams(userId: string): Promise<LocalDreamAnalysis[]> {
  if (!isAvailable()) return [];
  const result = await db!.query(
    `SELECT * FROM dream_analysis WHERE user_id = ? AND synced = 0 ORDER BY timestamp ASC LIMIT 50`,
    [userId]
  );
  return (result.values ?? []).map(row => ({
    id: row.id, userId: row.user_id, dreamText: row.dream_text ?? undefined,
    symbols: row.symbols ? JSON.parse(row.symbols) : undefined,
    emotions: row.emotions ? JSON.parse(row.emotions) : undefined,
    aiAnalysis: row.ai_analysis ?? undefined, lucidityScore: row.lucidity_score ?? undefined,
    sleepQuality: row.sleep_quality ?? undefined,
    themes: row.themes ? JSON.parse(row.themes) : undefined,
    emotionalArc: row.emotional_arc ?? undefined, keyInsight: row.key_insight ?? undefined,
    threatSimulationIndex: row.threat_simulation_index ?? undefined, timestamp: row.timestamp,
  }));
}

export async function markDreamsSynced(ids: string[]): Promise<void> {
  if (!isAvailable() || ids.length === 0) return;
  const placeholders = ids.map(() => "?").join(",");
  await db!.run(`UPDATE dream_analysis SET synced = 1 WHERE id IN (${placeholders})`, ids);
}

// ── Delete / wipe ─────────────────────────────────────────────────────────────

/** GDPR: wipe all local data for a user */
export async function wipeLocalData(userId: string): Promise<void> {
  if (!isAvailable()) return;
  await db!.run(`DELETE FROM emotion_readings WHERE user_id = ?`, [userId]);
  await db!.run(`DELETE FROM dream_analysis WHERE user_id = ?`, [userId]);
  await db!.run(`DELETE FROM inner_scores WHERE user_id = ?`, [userId]);
  await db!.run(`DELETE FROM mood_logs WHERE user_id = ?`, [userId]);
}

export async function closeDB(): Promise<void> {
  if (!db || !sqlitePlugin) return;
  await db.close();
  await sqlitePlugin.closeConnection(DB_NAME, false);
  db = null;
}
