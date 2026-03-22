-- 005_fix_rls_policies.sql — Replace permissive RLS policies on app_data tables
-- with proper auth.uid() matching.
--
-- The 004_app_data.sql migration created policies with USING (true) which
-- allows ANY authenticated user (or even anon key holder) to read/write ALL
-- rows. This migration drops those policies and creates proper per-user
-- isolation using the initPlan-optimized pattern:
--
--   (SELECT auth.uid()::text) = user_id
--
-- The sub-select forces Postgres to evaluate auth.uid() once per query
-- (initPlan) rather than once per row, which is critical for high-write
-- tables like emotion_history.
--
-- NOTE: Disable Supabase Realtime on high-write tables (emotion_history,
-- voice_history) to avoid broadcasting every EEG frame to the Realtime
-- multiplexer. Only enable Realtime on tables where live UI updates matter
-- (e.g., notifications).

-- ── Drop the old permissive policies ────────────────────────────────────────

DROP POLICY IF EXISTS "Users access own data" ON mood_logs;
DROP POLICY IF EXISTS "Users access own data" ON voice_history;
DROP POLICY IF EXISTS "Users access own data" ON emotion_history;
DROP POLICY IF EXISTS "Users access own data" ON food_logs;
DROP POLICY IF EXISTS "Users access own data" ON cycle_data;
DROP POLICY IF EXISTS "Users access own data" ON brain_age;
DROP POLICY IF EXISTS "Users access own data" ON glp1_injections;
DROP POLICY IF EXISTS "Users access own data" ON supplements;
DROP POLICY IF EXISTS "Users access own data" ON notifications;

-- ── Create per-operation policies with auth.uid() matching ──────────────────
-- Using separate SELECT / INSERT / UPDATE / DELETE policies gives finer
-- control and clearer audit trail than a single FOR ALL policy.

-- mood_logs
CREATE POLICY "mood_logs_select" ON mood_logs FOR SELECT
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "mood_logs_insert" ON mood_logs FOR INSERT
  WITH CHECK ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "mood_logs_update" ON mood_logs FOR UPDATE
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "mood_logs_delete" ON mood_logs FOR DELETE
  USING ((SELECT auth.uid()::text) = user_id);

-- voice_history
CREATE POLICY "voice_history_select" ON voice_history FOR SELECT
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "voice_history_insert" ON voice_history FOR INSERT
  WITH CHECK ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "voice_history_update" ON voice_history FOR UPDATE
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "voice_history_delete" ON voice_history FOR DELETE
  USING ((SELECT auth.uid()::text) = user_id);

-- emotion_history
CREATE POLICY "emotion_history_select" ON emotion_history FOR SELECT
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "emotion_history_insert" ON emotion_history FOR INSERT
  WITH CHECK ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "emotion_history_update" ON emotion_history FOR UPDATE
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "emotion_history_delete" ON emotion_history FOR DELETE
  USING ((SELECT auth.uid()::text) = user_id);

-- food_logs
CREATE POLICY "food_logs_select" ON food_logs FOR SELECT
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "food_logs_insert" ON food_logs FOR INSERT
  WITH CHECK ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "food_logs_update" ON food_logs FOR UPDATE
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "food_logs_delete" ON food_logs FOR DELETE
  USING ((SELECT auth.uid()::text) = user_id);

-- cycle_data
CREATE POLICY "cycle_data_select" ON cycle_data FOR SELECT
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "cycle_data_insert" ON cycle_data FOR INSERT
  WITH CHECK ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "cycle_data_update" ON cycle_data FOR UPDATE
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "cycle_data_delete" ON cycle_data FOR DELETE
  USING ((SELECT auth.uid()::text) = user_id);

-- brain_age
CREATE POLICY "brain_age_select" ON brain_age FOR SELECT
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "brain_age_insert" ON brain_age FOR INSERT
  WITH CHECK ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "brain_age_update" ON brain_age FOR UPDATE
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "brain_age_delete" ON brain_age FOR DELETE
  USING ((SELECT auth.uid()::text) = user_id);

-- glp1_injections
CREATE POLICY "glp1_injections_select" ON glp1_injections FOR SELECT
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "glp1_injections_insert" ON glp1_injections FOR INSERT
  WITH CHECK ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "glp1_injections_update" ON glp1_injections FOR UPDATE
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "glp1_injections_delete" ON glp1_injections FOR DELETE
  USING ((SELECT auth.uid()::text) = user_id);

-- supplements
CREATE POLICY "supplements_select" ON supplements FOR SELECT
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "supplements_insert" ON supplements FOR INSERT
  WITH CHECK ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "supplements_update" ON supplements FOR UPDATE
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "supplements_delete" ON supplements FOR DELETE
  USING ((SELECT auth.uid()::text) = user_id);

-- notifications
CREATE POLICY "notifications_select" ON notifications FOR SELECT
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "notifications_insert" ON notifications FOR INSERT
  WITH CHECK ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "notifications_update" ON notifications FOR UPDATE
  USING ((SELECT auth.uid()::text) = user_id);
CREATE POLICY "notifications_delete" ON notifications FOR DELETE
  USING ((SELECT auth.uid()::text) = user_id);

-- ── Realtime guidance ───────────────────────────────────────────────────────
-- Do NOT add high-write tables to Supabase Realtime. EEG emotion_history and
-- voice_history can generate 1+ row/second per active user — Realtime
-- broadcasting this would overwhelm the multiplexer.
--
-- Safe for Realtime: notifications (low write frequency, user wants instant UI update)
-- NOT safe: emotion_history, voice_history, mood_logs (high write frequency during sessions)
--
-- To enable Realtime on notifications only:
--   ALTER PUBLICATION supabase_realtime ADD TABLE notifications;
