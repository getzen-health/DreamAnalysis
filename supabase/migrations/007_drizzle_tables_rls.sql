-- 007_drizzle_tables_rls.sql
-- Enable Row-Level Security on all Drizzle-managed tables that were pushed
-- via drizzle-kit and never covered by migrations 003–006.
--
-- WHY THIS IS NEEDED:
--   drizzle-kit push creates tables but never runs ENABLE ROW LEVEL SECURITY.
--   Supabase security advisor flagged the project (AntarAI / tpiyavugafhplsmwvrel)
--   with two critical issues:
--     1. rls_disabled_in_public  — tables accessible to anyone with the project URL
--     2. sensitive_columns_exposed — password, token, email columns publicly readable
--
-- APPROACH:
--   • All user-data tables: enable RLS + per-user policy using auth.uid()
--   • Internal/backend-only tables: enable RLS + NO policies
--     (service_role key bypasses RLS; anon/authenticated keys get zero access)
--
-- NOTE: The FastAPI backend uses SUPABASE_SERVICE_ROLE_KEY for all queries,
-- which bypasses RLS. These policies protect against direct API access using
-- the anon key (e.g., from a leaked project URL).
-- ============================================================

-- ── 1. CRITICAL: Internal tables — no user-accessible policies ──────────────
-- Service role only. Anon/authenticated keys cannot read these at all.

ALTER TABLE password_reset_tokens ENABLE ROW LEVEL SECURITY;
-- No policy: zero access via API keys. Only service_role (FastAPI backend) can query.

ALTER TABLE rate_limit_entries ENABLE ROW LEVEL SECURITY;
-- No policy: internal backend table.

ALTER TABLE datadog_error_log ENABLE ROW LEVEL SECURITY;
-- No policy: internal monitoring table.

-- ── 2. users table — self-access only ───────────────────────────────────────
-- Contains: username, password (bcrypt), email — CRITICAL sensitive columns.

ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users_select_own" ON users FOR SELECT TO authenticated
  USING ((SELECT auth.uid()::text) = id);

CREATE POLICY "users_update_own" ON users FOR UPDATE TO authenticated
  USING ((SELECT auth.uid()::text) = id)
  WITH CHECK ((SELECT auth.uid()::text) = id);

-- No INSERT policy: user creation is server-side only (FastAPI /register endpoint).
-- No DELETE policy: deletion is soft-delete via deletion_requested_at (server-side).

-- ── 3. Dream & journaling tables ─────────────────────────────────────────────

ALTER TABLE dream_analysis ENABLE ROW LEVEL SECURITY;
CREATE POLICY "dream_analysis_own" ON dream_analysis FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE dream_symbols ENABLE ROW LEVEL SECURITY;
CREATE POLICY "dream_symbols_own" ON dream_symbols FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE dream_frames ENABLE ROW LEVEL SECURITY;
CREATE POLICY "dream_frames_own" ON dream_frames FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE irt_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "irt_sessions_own" ON irt_sessions FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE reality_tests ENABLE ROW LEVEL SECURITY;
CREATE POLICY "reality_tests_own" ON reality_tests FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE ai_chats ENABLE ROW LEVEL SECURITY;
CREATE POLICY "ai_chats_own" ON ai_chats FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

-- ── 4. Health & biometric tables ─────────────────────────────────────────────

ALTER TABLE health_metrics ENABLE ROW LEVEL SECURITY;
CREATE POLICY "health_metrics_own" ON health_metrics FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE emotion_readings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "emotion_readings_own" ON emotion_readings FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE eeg_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "eeg_sessions_own" ON eeg_sessions FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE brain_readings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "brain_readings_own" ON brain_readings FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE circadian_profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "circadian_profiles_own" ON circadian_profiles FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE emotion_calibration ENABLE ROW LEVEL SECURITY;
CREATE POLICY "emotion_calibration_own" ON emotion_calibration FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE emotional_fitness_scores ENABLE ROW LEVEL SECURITY;
CREATE POLICY "emotional_fitness_scores_own" ON emotional_fitness_scores FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE inner_scores ENABLE ROW LEVEL SECURITY;
CREATE POLICY "inner_scores_own" ON inner_scores FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE user_patterns ENABLE ROW LEVEL SECURITY;
CREATE POLICY "user_patterns_own" ON user_patterns FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE emotion_fingerprints ENABLE ROW LEVEL SECURITY;
CREATE POLICY "emotion_fingerprints_own" ON emotion_fingerprints FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

-- ── 5. Food & nutrition tables ────────────────────────────────────────────────

ALTER TABLE meal_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY "meal_history_own" ON meal_history FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE user_readings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "user_readings_own" ON user_readings FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

-- ── 6. Activity & gamification tables ────────────────────────────────────────

ALTER TABLE streaks ENABLE ROW LEVEL SECURITY;
CREATE POLICY "streaks_own" ON streaks FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

-- ── 7. Push subscriptions ─────────────────────────────────────────────────────
-- push_keys are sensitive (endpoint + encryption keys).

ALTER TABLE push_subscriptions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "push_subscriptions_own" ON push_subscriptions FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

-- ── 8. Research study tables ──────────────────────────────────────────────────
-- Study data is personal + PII; participants can access only their own records.

ALTER TABLE study_participants ENABLE ROW LEVEL SECURITY;
CREATE POLICY "study_participants_own" ON study_participants FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE study_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "study_sessions_own" ON study_sessions FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE study_morning_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY "study_morning_entries_own" ON study_morning_entries FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE study_daytime_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY "study_daytime_entries_own" ON study_daytime_entries FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE study_evening_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY "study_evening_entries_own" ON study_evening_entries FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE pilot_participants ENABLE ROW LEVEL SECURITY;
CREATE POLICY "pilot_participants_own" ON pilot_participants FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

ALTER TABLE pilot_sessions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "pilot_sessions_own" ON pilot_sessions FOR ALL
  USING ((SELECT auth.uid()::text) = user_id)
  WITH CHECK ((SELECT auth.uid()::text) = user_id);

-- health_samples already covered by 003_rls.sql — skip.
-- user_settings already covered by 006_user_settings.sql — skip.
-- food_logs already covered by 005_fix_rls_policies.sql — skip.
-- mood_logs, voice_history, emotion_history, etc. covered by 004/005 — skip.
