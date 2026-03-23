/**
 * supabase-store.ts — Unified data layer for all app data.
 *
 * Pattern for every data type:
 *   1. Try Supabase first
 *   2. If Supabase fails (offline, no auth, not configured), fall back to localStorage
 *   3. On save: write to both Supabase + localStorage (localStorage is cache)
 *   4. On first connect: sync localStorage -> Supabase (migrate existing data)
 *
 * All functions are async and never throw — they log warnings and degrade gracefully.
 */

import { getSupabase } from "./supabase-browser";

// ── Privacy Mode gate ────────────────────────────────────────────────────────

/**
 * When Privacy Mode is enabled (ndw_privacy_mode = "true"), ALL Supabase
 * sync is disabled. Data stays in localStorage only. (Issue #493)
 */
function isPrivacyModeEnabled(): boolean {
  try {
    return localStorage.getItem("ndw_privacy_mode") === "true";
  } catch {
    return false;
  }
}

/**
 * Get Supabase client only if Privacy Mode is OFF.
 * Returns null when privacy mode is active (blocks all cloud sync).
 */
async function getSupabaseIfAllowed() {
  if (isPrivacyModeEnabled()) return null;
  return getSupabase();
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function safeJsonParse<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function safeLocalGet<T>(key: string, fallback: T): T {
  try {
    return safeJsonParse(localStorage.getItem(key), fallback);
  } catch {
    return fallback;
  }
}

function safeLocalSet(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // localStorage full or unavailable
  }
}

// ── Mood Logs ────────────────────────────────────────────────────────────────

export interface MoodLogEntry {
  mood: number;
  energy?: number;
  notes?: string;
  created_at?: string;
}

export async function saveMoodLog(userId: string, entry: MoodLogEntry): Promise<void> {
  // Always write to localStorage cache
  const key = "ndw_mood_logs";
  const existing = safeLocalGet<any[]>(key, []);
  existing.unshift({
    id: `local_${Date.now()}`,
    userId,
    moodScore: String(entry.mood),
    energyLevel: entry.energy != null ? String(entry.energy) : undefined,
    notes: entry.notes ?? null,
    loggedAt: entry.created_at ?? new Date().toISOString(),
  });
  if (existing.length > 100) existing.length = 100;
  safeLocalSet(key, existing);

  // Try Supabase
  const sb = await getSupabaseIfAllowed();
  if (!sb) return;
  try {
    await sb.from("mood_logs").insert({
      user_id: userId,
      mood: entry.mood,
      energy: entry.energy ?? null,
      notes: entry.notes ?? null,
      created_at: entry.created_at ?? new Date().toISOString(),
    });
  } catch (err) {
    console.warn("[supabase-store] saveMoodLog failed:", err);
  }
}

export async function getMoodLogs(userId: string, limit = 30): Promise<any[]> {
  const sb = await getSupabaseIfAllowed();
  if (sb) {
    try {
      const { data, error } = await sb
        .from("mood_logs")
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .limit(limit);
      if (!error && data && data.length > 0) {
        // Map to the shape the UI expects
        const mapped = data.map((r: any) => ({
          id: r.id,
          userId: r.user_id,
          moodScore: String(r.mood),
          energyLevel: r.energy != null ? String(r.energy) : undefined,
          notes: r.notes,
          loggedAt: r.created_at,
        }));
        safeLocalSet("ndw_mood_logs", mapped);
        return mapped;
      }
    } catch (err) {
      console.warn("[supabase-store] getMoodLogs failed:", err);
    }
  }
  return safeLocalGet<any[]>("ndw_mood_logs", []).slice(0, limit);
}

// ── Voice History ────────────────────────────────────────────────────────────

export interface VoiceHistoryEntry {
  emotion?: string;
  stress?: number;
  focus?: number;
  valence?: number;
  arousal?: number;
  created_at?: string;
  // Aliases used by some callers (e.g. VoiceWatchCheckinResult)
  stress_index?: number;
  focus_index?: number;
}

export async function saveVoiceHistory(userId: string, entry: VoiceHistoryEntry): Promise<void> {
  // localStorage cache
  const key = "ndw_voice_history";
  const existing = safeLocalGet<any[]>(key, []);
  existing.unshift({ ...entry, timestamp: Date.now() });
  if (existing.length > 50) existing.length = 50;
  safeLocalSet(key, existing);

  const sb = await getSupabaseIfAllowed();
  if (!sb) return;
  try {
    await sb.from("voice_history").insert({
      user_id: userId,
      emotion: entry.emotion ?? null,
      stress: entry.stress ?? entry.stress_index ?? null,
      focus: entry.focus ?? entry.focus_index ?? null,
      valence: entry.valence ?? null,
      arousal: entry.arousal ?? null,
      created_at: entry.created_at ?? new Date().toISOString(),
    });
  } catch (err) {
    console.warn("[supabase-store] saveVoiceHistory failed:", err);
  }
}

export async function getVoiceHistory(userId: string, limit = 50): Promise<any[]> {
  const sb = await getSupabaseIfAllowed();
  if (sb) {
    try {
      const { data, error } = await sb
        .from("voice_history")
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .limit(limit);
      if (!error && data && data.length > 0) {
        const mapped = data.map((r: any) => ({
          ...r,
          stress_index: r.stress,
          focus_index: r.focus,
          timestamp: new Date(r.created_at).getTime(),
        }));
        safeLocalSet("ndw_voice_history", mapped);
        return mapped;
      }
    } catch (err) {
      console.warn("[supabase-store] getVoiceHistory failed:", err);
    }
  }
  return safeLocalGet<any[]>("ndw_voice_history", []).slice(0, limit);
}

// ── Emotion History ──────────────────────────────────────────────────────────

export interface EmotionHistoryEntry {
  stress: number;
  focus: number;
  mood: number;
  source?: string;
  dominantEmotion?: string;
  created_at?: string;
}

export async function saveEmotionHistory(userId: string, entry: EmotionHistoryEntry): Promise<void> {
  const key = "ndw_emotion_history";
  const existing = safeLocalGet<any[]>(key, []);
  existing.push({
    stress: entry.stress,
    happiness: entry.mood,
    focus: entry.focus,
    dominantEmotion: entry.dominantEmotion ?? "neutral",
    timestamp: entry.created_at ?? new Date().toISOString(),
  });
  // Cap at 200, prune older than 7 days
  const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;
  const pruned = existing
    .filter((e: any) => new Date(e.timestamp).getTime() > cutoff)
    .slice(-200);
  safeLocalSet(key, pruned);

  const sb = await getSupabaseIfAllowed();
  if (!sb) return;
  try {
    await sb.from("emotion_history").insert({
      user_id: userId,
      stress: entry.stress,
      focus: entry.focus,
      mood: entry.mood,
      source: entry.source ?? null,
      created_at: entry.created_at ?? new Date().toISOString(),
    });
  } catch (err) {
    console.warn("[supabase-store] saveEmotionHistory failed:", err);
  }
}

export async function getEmotionHistory(userId: string, days = 7): Promise<any[]> {
  const sb = await getSupabaseIfAllowed();
  const cutoffDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString();

  if (sb) {
    try {
      const { data, error } = await sb
        .from("emotion_history")
        .select("*")
        .eq("user_id", userId)
        .gte("created_at", cutoffDate)
        .order("created_at", { ascending: true })
        .limit(200);
      if (!error && data && data.length > 0) {
        const mapped = data.map((r: any) => ({
          stress: r.stress,
          happiness: r.mood,
          focus: r.focus,
          dominantEmotion: r.source ?? "neutral",
          timestamp: r.created_at,
        }));
        safeLocalSet("ndw_emotion_history", mapped);
        return mapped;
      }
    } catch (err) {
      console.warn("[supabase-store] getEmotionHistory failed:", err);
    }
  }

  // Fallback: localStorage
  const all = safeLocalGet<any[]>("ndw_emotion_history", []);
  const cutoffMs = Date.now() - days * 24 * 60 * 60 * 1000;
  return all.filter((e: any) => new Date(e.timestamp).getTime() > cutoffMs);
}

// ── Food Logs ────────────────────────────────────────────────────────────────

export interface FoodLogEntry {
  summary?: string;
  calories?: number;
  protein?: number;
  carbs?: number;
  fat?: number;
  food_quality_score?: number;
  created_at?: string;
  [key: string]: unknown;
}

export async function saveFoodLog(userId: string, entry: FoodLogEntry): Promise<void> {
  const key = `ndw_food_logs_${userId}`;
  const existing = safeLocalGet<any[]>(key, []);
  const localEntry = {
    id: `local_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    ...entry,
    loggedAt: entry.created_at ?? new Date().toISOString(),
  };
  existing.unshift(localEntry);
  if (existing.length > 200) existing.length = 200;
  safeLocalSet(key, existing);

  const sb = await getSupabaseIfAllowed();
  if (!sb) return;
  try {
    await sb.from("food_logs").insert({
      user_id: userId,
      summary: entry.summary ?? null,
      calories: entry.calories ?? null,
      protein: entry.protein ?? null,
      carbs: entry.carbs ?? null,
      fat: entry.fat ?? null,
      food_quality_score: entry.food_quality_score ?? null,
      created_at: entry.created_at ?? new Date().toISOString(),
    });
  } catch (err) {
    console.warn("[supabase-store] saveFoodLog failed:", err);
  }
}

export async function getFoodLogs(userId: string, limit = 50): Promise<any[]> {
  const sb = await getSupabaseIfAllowed();
  if (sb) {
    try {
      const { data, error } = await sb
        .from("food_logs")
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .limit(limit);
      if (!error && data && data.length > 0) {
        safeLocalSet(`ndw_food_logs_${userId}`, data);
        return data;
      }
    } catch (err) {
      console.warn("[supabase-store] getFoodLogs failed:", err);
    }
  }
  return safeLocalGet<any[]>(`ndw_food_logs_${userId}`, []).slice(0, limit);
}

// ── Cycle Data ───────────────────────────────────────────────────────────────

export interface CycleDataEntry {
  last_period_start?: string;
  cycle_length?: number;
  period_length?: number;
  logged_days?: any[];
}

export async function saveCycleData(userId: string, data: CycleDataEntry): Promise<void> {
  // localStorage in the format wellness.tsx expects
  safeLocalSet("ndw_cycle_data", {
    lastPeriodStart: data.last_period_start,
    cycleLength: data.cycle_length ?? 28,
    periodLength: data.period_length ?? 5,
  });

  const sb = await getSupabaseIfAllowed();
  if (!sb) return;
  try {
    // Upsert: one row per user
    const { data: existing } = await sb
      .from("cycle_data")
      .select("id")
      .eq("user_id", userId)
      .limit(1);
    if (existing && existing.length > 0) {
      await sb.from("cycle_data").update({
        last_period_start: data.last_period_start ?? null,
        cycle_length: data.cycle_length ?? 28,
        period_length: data.period_length ?? 5,
        logged_days: data.logged_days ?? [],
        updated_at: new Date().toISOString(),
      }).eq("id", existing[0].id);
    } else {
      await sb.from("cycle_data").insert({
        user_id: userId,
        last_period_start: data.last_period_start ?? null,
        cycle_length: data.cycle_length ?? 28,
        period_length: data.period_length ?? 5,
        logged_days: data.logged_days ?? [],
      });
    }
  } catch (err) {
    console.warn("[supabase-store] saveCycleData failed:", err);
  }
}

export async function getCycleData(userId: string): Promise<CycleDataEntry | null> {
  const sb = await getSupabaseIfAllowed();
  if (sb) {
    try {
      const { data, error } = await sb
        .from("cycle_data")
        .select("*")
        .eq("user_id", userId)
        .order("updated_at", { ascending: false })
        .limit(1);
      if (!error && data && data.length > 0) {
        const row = data[0];
        const entry: CycleDataEntry = {
          last_period_start: row.last_period_start,
          cycle_length: row.cycle_length,
          period_length: row.period_length,
          logged_days: row.logged_days,
        };
        // Update localStorage cache
        safeLocalSet("ndw_cycle_data", {
          lastPeriodStart: entry.last_period_start,
          cycleLength: entry.cycle_length ?? 28,
          periodLength: entry.period_length ?? 5,
        });
        return entry;
      }
    } catch (err) {
      console.warn("[supabase-store] getCycleData failed:", err);
    }
  }

  // Fallback: localStorage
  const local = safeLocalGet<any>("ndw_cycle_data", null);
  if (!local) return null;
  return {
    last_period_start: local.lastPeriodStart,
    cycle_length: local.cycleLength ?? 28,
    period_length: local.periodLength ?? 5,
  };
}

// ── Brain Age ────────────────────────────────────────────────────────────────

export interface BrainAgeEntry {
  estimated_age: number;
  actual_age: number;
  gap: number;
  created_at?: string;
}

export async function saveBrainAge(userId: string, entry: BrainAgeEntry): Promise<void> {
  safeLocalSet("ndw_brain_age", {
    estimatedAge: entry.estimated_age,
    brainAgeGap: entry.gap,
    actualAge: entry.actual_age,
    timestamp: Date.now(),
  });

  const sb = await getSupabaseIfAllowed();
  if (!sb) return;
  try {
    await sb.from("brain_age").insert({
      user_id: userId,
      estimated_age: entry.estimated_age,
      actual_age: entry.actual_age,
      gap: entry.gap,
      created_at: entry.created_at ?? new Date().toISOString(),
    });
  } catch (err) {
    console.warn("[supabase-store] saveBrainAge failed:", err);
  }
}

export async function getBrainAge(userId: string): Promise<BrainAgeEntry | null> {
  const sb = await getSupabaseIfAllowed();
  if (sb) {
    try {
      const { data, error } = await sb
        .from("brain_age")
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .limit(1);
      if (!error && data && data.length > 0) {
        const row = data[0];
        safeLocalSet("ndw_brain_age", {
          estimatedAge: row.estimated_age,
          brainAgeGap: row.gap,
          actualAge: row.actual_age,
          timestamp: new Date(row.created_at).getTime(),
        });
        return {
          estimated_age: row.estimated_age,
          actual_age: row.actual_age,
          gap: row.gap,
          created_at: row.created_at,
        };
      }
    } catch (err) {
      console.warn("[supabase-store] getBrainAge failed:", err);
    }
  }

  const local = safeLocalGet<any>("ndw_brain_age", null);
  if (!local) return null;
  return {
    estimated_age: local.estimatedAge,
    actual_age: local.actualAge ?? 0,
    gap: local.brainAgeGap,
  };
}

// ── GLP-1 Injections ─────────────────────────────────────────────────────────

export interface Glp1InjectionEntry {
  medication: string;
  dose?: number;
  injected_at?: string;
}

export async function saveGlp1Injection(userId: string, entry: Glp1InjectionEntry): Promise<void> {
  const key = "ndw_glp1_injections";
  const existing = safeLocalGet<any[]>(key, []);
  existing.unshift({
    id: `glp1_${Date.now()}`,
    medication: entry.medication,
    dose: entry.dose,
    date: entry.injected_at ?? new Date().toISOString(),
  });
  safeLocalSet(key, existing);

  const sb = await getSupabaseIfAllowed();
  if (!sb) return;
  try {
    await sb.from("glp1_injections").insert({
      user_id: userId,
      medication: entry.medication,
      dose: entry.dose ?? null,
      injected_at: entry.injected_at ?? new Date().toISOString(),
    });
  } catch (err) {
    console.warn("[supabase-store] saveGlp1Injection failed:", err);
  }
}

export async function getGlp1Injections(userId: string): Promise<any[]> {
  const sb = await getSupabaseIfAllowed();
  if (sb) {
    try {
      const { data, error } = await sb
        .from("glp1_injections")
        .select("*")
        .eq("user_id", userId)
        .order("injected_at", { ascending: false })
        .limit(100);
      if (!error && data && data.length > 0) {
        const mapped = data.map((r: any) => ({
          id: r.id,
          medication: r.medication,
          dose: r.dose,
          date: r.injected_at,
        }));
        safeLocalSet("ndw_glp1_injections", mapped);
        return mapped;
      }
    } catch (err) {
      console.warn("[supabase-store] getGlp1Injections failed:", err);
    }
  }
  return safeLocalGet<any[]>("ndw_glp1_injections", []);
}

// ── Notifications ────────────────────────────────────────────────────────────

export interface NotificationEntry {
  type: string;
  title: string;
  body?: string;
  read?: boolean;
  created_at?: string;
}

export async function saveNotification(userId: string, entry: NotificationEntry): Promise<void> {
  const key = "ndw_notifications";
  const existing = safeLocalGet<any[]>(key, []);
  const localEntry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    type: entry.type,
    title: entry.title,
    body: entry.body ?? "",
    timestamp: Date.now(),
    read: false,
  };
  existing.unshift(localEntry);
  if (existing.length > 100) existing.length = 100;
  safeLocalSet(key, existing);

  const sb = await getSupabaseIfAllowed();
  if (!sb) return;
  try {
    await sb.from("notifications").insert({
      user_id: userId,
      type: entry.type,
      title: entry.title,
      body: entry.body ?? null,
      read: false,
      created_at: entry.created_at ?? new Date().toISOString(),
    });
  } catch (err) {
    console.warn("[supabase-store] saveNotification failed:", err);
  }
}

export async function getNotifications(userId: string): Promise<any[]> {
  const sb = await getSupabaseIfAllowed();
  if (sb) {
    try {
      const { data, error } = await sb
        .from("notifications")
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .limit(100);
      if (!error && data && data.length > 0) {
        const mapped = data.map((r: any) => ({
          id: r.id,
          type: r.type,
          title: r.title,
          body: r.body ?? "",
          timestamp: new Date(r.created_at).getTime(),
          read: r.read,
        }));
        safeLocalSet("ndw_notifications", mapped);
        return mapped;
      }
    } catch (err) {
      console.warn("[supabase-store] getNotifications failed:", err);
    }
  }
  return safeLocalGet<any[]>("ndw_notifications", []);
}

export async function markNotificationRead(userId: string, notificationId: string): Promise<void> {
  // Update localStorage
  const key = "ndw_notifications";
  const all = safeLocalGet<any[]>(key, []);
  const updated = all.map((n: any) => n.id === notificationId ? { ...n, read: true } : n);
  safeLocalSet(key, updated);

  const sb = await getSupabaseIfAllowed();
  if (!sb) return;
  try {
    await sb.from("notifications").update({ read: true }).eq("id", notificationId);
  } catch (err) {
    console.warn("[supabase-store] markNotificationRead failed:", err);
  }
}

export async function markAllNotificationsRead(userId: string): Promise<void> {
  const key = "ndw_notifications";
  const all = safeLocalGet<any[]>(key, []);
  safeLocalSet(key, all.map((n: any) => ({ ...n, read: true })));

  const sb = await getSupabaseIfAllowed();
  if (!sb) return;
  try {
    await sb.from("notifications").update({ read: true }).eq("user_id", userId).eq("read", false);
  } catch (err) {
    console.warn("[supabase-store] markAllNotificationsRead failed:", err);
  }
}

export async function clearAllNotifications(userId: string): Promise<void> {
  safeLocalSet("ndw_notifications", []);

  const sb = await getSupabaseIfAllowed();
  if (!sb) return;
  try {
    await sb.from("notifications").delete().eq("user_id", userId);
  } catch (err) {
    console.warn("[supabase-store] clearAllNotifications failed:", err);
  }
}

// ── Generic Settings (string/boolean values) ─────────────────────────────────

/**
 * Read a setting value. Sync read from localStorage for fast initial render,
 * with an async Supabase fetch that updates localStorage cache in the background.
 * Returns the localStorage value immediately (or null).
 */
export function sbGetSetting(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/**
 * Write a setting to both localStorage (sync) and Supabase user_settings (async).
 * Fire-and-forget — never throws.
 */
export function sbSaveSetting(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch { /* localStorage full or unavailable */ }

  // Async Supabase upsert (fire-and-forget)
  (async () => {
    const sb = await getSupabaseIfAllowed();
    if (!sb) return;
    try {
      const { data: { user } } = await sb.auth.getUser();
      if (!user) return;
      await sb.from("user_settings").upsert(
        { user_id: user.id, key, value, updated_at: new Date().toISOString() },
        { onConflict: "user_id,key" }
      );
    } catch (err) {
      console.warn("[supabase-store] sbSaveSetting failed:", err);
    }
  })();
}

/**
 * Remove a setting from both localStorage and Supabase.
 */
export function sbRemoveSetting(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch { /* ok */ }

  (async () => {
    const sb = await getSupabaseIfAllowed();
    if (!sb) return;
    try {
      const { data: { user } } = await sb.auth.getUser();
      if (!user) return;
      await sb.from("user_settings").delete().eq("user_id", user.id).eq("key", key);
    } catch (err) {
      console.warn("[supabase-store] sbRemoveSetting failed:", err);
    }
  })();
}

// ── Generic Store (JSON blobs) ───────────────────────────────────────────────

/**
 * Read a JSON blob. Sync read from localStorage for immediate render.
 * Returns parsed JSON or null.
 */
export function sbGetGeneric<T = any>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

/**
 * Write a JSON blob to both localStorage (sync) and Supabase generic_store (async).
 * Fire-and-forget — never throws.
 */
export function sbSaveGeneric(key: string, value: any): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch { /* localStorage full or unavailable */ }

  (async () => {
    const sb = await getSupabaseIfAllowed();
    if (!sb) return;
    try {
      const { data: { user } } = await sb.auth.getUser();
      if (!user) return;
      await sb.from("generic_store").upsert(
        { user_id: user.id, key, value, updated_at: new Date().toISOString() },
        { onConflict: "user_id,key" }
      );
    } catch (err) {
      console.warn("[supabase-store] sbSaveGeneric failed:", err);
    }
  })();
}

// ── One-time migration: localStorage -> Supabase ─────────────────────────────

const SYNC_FLAG_KEY = "ndw_supabase_synced";

export async function syncLocalToSupabase(userId: string): Promise<void> {
  if (safeLocalGet<boolean>(SYNC_FLAG_KEY, false)) return;

  const sb = await getSupabaseIfAllowed();
  if (!sb) return;

  try {
    // Mood logs
    const moodLogs = safeLocalGet<any[]>("ndw_mood_logs", []);
    if (moodLogs.length > 0) {
      const rows = moodLogs.map((m: any) => ({
        user_id: userId,
        mood: parseInt(m.moodScore, 10) || 5,
        energy: m.energyLevel != null ? parseInt(m.energyLevel, 10) : null,
        notes: m.notes ?? null,
        created_at: m.loggedAt ?? new Date().toISOString(),
      }));
      await sb.from("mood_logs").insert(rows).throwOnError();
    }

    // Voice history
    const voiceHistory = safeLocalGet<any[]>("ndw_voice_history", []);
    if (voiceHistory.length > 0) {
      const rows = voiceHistory.map((v: any) => ({
        user_id: userId,
        emotion: v.emotion ?? null,
        stress: v.stress_index ?? v.stress ?? null,
        focus: v.focus_index ?? v.focus ?? null,
        valence: v.valence ?? null,
        arousal: v.arousal ?? null,
        created_at: v.timestamp ? new Date(v.timestamp).toISOString() : new Date().toISOString(),
      }));
      await sb.from("voice_history").insert(rows).throwOnError();
    }

    // Emotion history
    const emotionHistory = safeLocalGet<any[]>("ndw_emotion_history", []);
    if (emotionHistory.length > 0) {
      const rows = emotionHistory.map((e: any) => ({
        user_id: userId,
        stress: e.stress ?? 0,
        focus: e.focus ?? 0,
        mood: e.happiness ?? e.mood ?? 0.5,
        source: e.dominantEmotion ?? null,
        created_at: e.timestamp ?? new Date().toISOString(),
      }));
      await sb.from("emotion_history").insert(rows).throwOnError();
    }

    // Food logs (user-specific key)
    const foodLogs = safeLocalGet<any[]>(`ndw_food_logs_${userId}`, []);
    if (foodLogs.length > 0) {
      const rows = foodLogs.map((f: any) => ({
        user_id: userId,
        summary: f.summary ?? null,
        calories: f.totalCalories ?? f.calories ?? null,
        protein: f.protein ?? null,
        carbs: f.carbs ?? null,
        fat: f.fat ?? null,
        food_quality_score: null,
        created_at: f.loggedAt ?? f.created_at ?? new Date().toISOString(),
      }));
      await sb.from("food_logs").insert(rows).throwOnError();
    }

    // Cycle data
    const cycleData = safeLocalGet<any>("ndw_cycle_data", null);
    if (cycleData && cycleData.lastPeriodStart) {
      await sb.from("cycle_data").insert({
        user_id: userId,
        last_period_start: cycleData.lastPeriodStart,
        cycle_length: cycleData.cycleLength ?? 28,
        period_length: cycleData.periodLength ?? 5,
        logged_days: [],
      }).throwOnError();
    }

    // Brain age
    const brainAge = safeLocalGet<any>("ndw_brain_age", null);
    if (brainAge && brainAge.estimatedAge) {
      await sb.from("brain_age").insert({
        user_id: userId,
        estimated_age: brainAge.estimatedAge,
        actual_age: brainAge.actualAge ?? 0,
        gap: brainAge.brainAgeGap ?? 0,
      }).throwOnError();
    }

    // GLP-1 injections
    const glp1 = safeLocalGet<any[]>("ndw_glp1_injections", []);
    if (glp1.length > 0) {
      const rows = glp1.map((g: any) => ({
        user_id: userId,
        medication: g.medication,
        dose: g.dose ?? null,
        injected_at: g.date ?? new Date().toISOString(),
      }));
      await sb.from("glp1_injections").insert(rows).throwOnError();
    }

    // Notifications
    const notifications = safeLocalGet<any[]>("ndw_notifications", []);
    if (notifications.length > 0) {
      const rows = notifications.map((n: any) => ({
        user_id: userId,
        type: n.type,
        title: n.title,
        body: n.body ?? "",
        read: n.read ?? false,
        created_at: n.timestamp ? new Date(n.timestamp).toISOString() : new Date().toISOString(),
      }));
      await sb.from("notifications").insert(rows).throwOnError();
    }

    // Mark as synced
    safeLocalSet(SYNC_FLAG_KEY, true);
    console.log("[supabase-store] localStorage -> Supabase sync complete");
  } catch (err) {
    console.warn("[supabase-store] syncLocalToSupabase partial failure:", err);
    // Don't mark as synced so it retries next time
  }
}
