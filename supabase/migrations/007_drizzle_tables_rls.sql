-- 007_drizzle_tables_rls.sql
-- Enable Row-Level Security on all Drizzle-managed tables.
-- Handles tables with user_id, nested participant_id/session_id chains,
-- and internal tables with no user ownership.
-- ============================================================

-- ── 1. Internal tables — no policies (service_role only) ────────────────────
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename='password_reset_tokens') THEN
    ALTER TABLE password_reset_tokens ENABLE ROW LEVEL SECURITY;
    RAISE NOTICE 'RLS enabled (no policy): password_reset_tokens';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename='rate_limit_entries') THEN
    ALTER TABLE rate_limit_entries ENABLE ROW LEVEL SECURITY;
    RAISE NOTICE 'RLS enabled (no policy): rate_limit_entries';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename='datadog_error_log') THEN
    ALTER TABLE datadog_error_log ENABLE ROW LEVEL SECURITY;
    RAISE NOTICE 'RLS enabled (no policy): datadog_error_log';
  END IF;
  -- pilot tables have no user_id — anonymous research data, backend-only
  IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename='pilot_participants') THEN
    ALTER TABLE pilot_participants ENABLE ROW LEVEL SECURITY;
    RAISE NOTICE 'RLS enabled (no policy): pilot_participants';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename='pilot_sessions') THEN
    ALTER TABLE pilot_sessions ENABLE ROW LEVEL SECURITY;
    RAISE NOTICE 'RLS enabled (no policy): pilot_sessions';
  END IF;
END $$;

-- ── 2. users table (id column, not user_id) ──────────────────────────────────
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename='users') THEN
    ALTER TABLE users ENABLE ROW LEVEL SECURITY;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='users' AND policyname='users_select_own') THEN
      CREATE POLICY "users_select_own" ON users FOR SELECT TO authenticated
        USING ((SELECT auth.uid()::text) = id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='users' AND policyname='users_update_own') THEN
      CREATE POLICY "users_update_own" ON users FOR UPDATE TO authenticated
        USING ((SELECT auth.uid()::text) = id)
        WITH CHECK ((SELECT auth.uid()::text) = id);
    END IF;
    RAISE NOTICE 'RLS + policy applied: users';
  END IF;
END $$;

-- ── 3. Standard user_id tables ───────────────────────────────────────────────
DO $$
DECLARE tbl text;
BEGIN
  FOREACH tbl IN ARRAY ARRAY[
    'dream_analysis','dream_symbols','dream_frames','irt_sessions',
    'reality_tests','ai_chats','health_metrics','emotion_readings',
    'eeg_sessions','brain_readings','circadian_profiles','emotion_calibration',
    'emotional_fitness_scores','inner_scores','user_patterns','emotion_fingerprints',
    'meal_history','user_readings','streaks','push_subscriptions',
    'study_participants'
  ] LOOP
    IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename=tbl) THEN
      EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', tbl);
      IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename=tbl AND policyname=tbl||'_own') THEN
        EXECUTE format(
          'CREATE POLICY %I ON %I FOR ALL USING ((SELECT auth.uid()::text) = user_id) WITH CHECK ((SELECT auth.uid()::text) = user_id)',
          tbl||'_own', tbl
        );
      END IF;
      RAISE NOTICE 'RLS + policy applied: %', tbl;
    ELSE
      RAISE NOTICE 'SKIP (not found): %', tbl;
    END IF;
  END LOOP;
END $$;

-- ── 4. study_sessions (uses participant_id → study_participants.user_id) ─────
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename='study_sessions') THEN
    ALTER TABLE study_sessions ENABLE ROW LEVEL SECURITY;
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='study_sessions' AND policyname='study_sessions_own') THEN
      CREATE POLICY "study_sessions_own" ON study_sessions FOR ALL
        USING (EXISTS (
          SELECT 1 FROM study_participants sp
          WHERE sp.id = study_sessions.participant_id
            AND (SELECT auth.uid()::text) = sp.user_id
        ))
        WITH CHECK (EXISTS (
          SELECT 1 FROM study_participants sp
          WHERE sp.id = study_sessions.participant_id
            AND (SELECT auth.uid()::text) = sp.user_id
        ));
    END IF;
    RAISE NOTICE 'RLS + policy applied: study_sessions';
  END IF;
END $$;

-- ── 5. study_*_entries (uses session_id → study_sessions → study_participants)
DO $$
DECLARE tbl text;
BEGIN
  FOREACH tbl IN ARRAY ARRAY['study_morning_entries','study_daytime_entries','study_evening_entries'] LOOP
    IF EXISTS (SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename=tbl) THEN
      EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', tbl);
      IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename=tbl AND policyname=tbl||'_own') THEN
        EXECUTE format(
          'CREATE POLICY %I ON %I FOR ALL
            USING (EXISTS (
              SELECT 1 FROM study_sessions ss
              JOIN study_participants sp ON sp.id = ss.participant_id
              WHERE ss.id = %I.session_id
                AND (SELECT auth.uid()::text) = sp.user_id
            ))
            WITH CHECK (EXISTS (
              SELECT 1 FROM study_sessions ss
              JOIN study_participants sp ON sp.id = ss.participant_id
              WHERE ss.id = %I.session_id
                AND (SELECT auth.uid()::text) = sp.user_id
            ))',
          tbl||'_own', tbl, tbl, tbl
        );
      END IF;
      RAISE NOTICE 'RLS + policy applied: %', tbl;
    ELSE
      RAISE NOTICE 'SKIP (not found): %', tbl;
    END IF;
  END LOOP;
END $$;
