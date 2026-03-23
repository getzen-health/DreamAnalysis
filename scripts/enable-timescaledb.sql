-- enable-timescaledb.sql — Migrate emotion_history and voice_history to TimescaleDB hypertables.
--
-- WHEN TO RUN:
--   Only when emotion_history or voice_history exceed ~100K rows and time-range
--   queries become slow. For small datasets, standard PostgreSQL is sufficient.
--
-- PREREQUISITES:
--   - Supabase project with TimescaleDB extension available
--   - Backup your data before running (pg_dump or Supabase dashboard export)
--
-- HOW TO RUN:
--   Connect to your Supabase database via psql or the SQL editor in the dashboard,
--   then execute this script. Each section is idempotent — safe to re-run.
--
-- REFERENCE:
--   https://docs.timescale.com/self-hosted/latest/install/installation-supabase/

-- ══════════════════════════════════════════════════════════════════════════
-- Step 1: Enable the TimescaleDB extension
-- ══════════════════════════════════════════════════════════════════════════

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ══════════════════════════════════════════════════════════════════════════
-- Step 2: Convert emotion_history to a hypertable
--
-- Partitions data by created_at with 7-day chunks (tunable).
-- migrate_data => true moves existing rows into the new chunk structure.
-- ══════════════════════════════════════════════════════════════════════════

SELECT create_hypertable(
  'emotion_history',
  'created_at',
  chunk_time_interval => INTERVAL '7 days',
  migrate_data => true,
  if_not_exists => true
);

-- ══════════════════════════════════════════════════════════════════════════
-- Step 3: Convert voice_history to a hypertable
-- ══════════════════════════════════════════════════════════════════════════

SELECT create_hypertable(
  'voice_history',
  'created_at',
  chunk_time_interval => INTERVAL '7 days',
  migrate_data => true,
  if_not_exists => true
);

-- ══════════════════════════════════════════════════════════════════════════
-- Step 4: Enable compression on older chunks (optional, saves ~90% space)
--
-- Compresses chunks older than 30 days. Compressed chunks are read-only
-- but drastically reduce storage for historical EEG/voice data.
-- ══════════════════════════════════════════════════════════════════════════

ALTER TABLE emotion_history SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'user_id',
  timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('emotion_history', INTERVAL '30 days', if_not_exists => true);

ALTER TABLE voice_history SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'user_id',
  timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('voice_history', INTERVAL '30 days', if_not_exists => true);

-- ══════════════════════════════════════════════════════════════════════════
-- Step 5: Create continuous aggregates for dashboard queries (optional)
--
-- Pre-computes hourly averages for fast dashboard rendering.
-- Refreshes automatically as new data arrives.
-- ══════════════════════════════════════════════════════════════════════════

-- Hourly emotion averages per user
CREATE MATERIALIZED VIEW IF NOT EXISTS emotion_hourly
WITH (timescaledb.continuous) AS
SELECT
  user_id,
  time_bucket('1 hour', created_at) AS bucket,
  AVG(stress) AS avg_stress,
  AVG(focus) AS avg_focus,
  AVG(mood) AS avg_mood,
  COUNT(*) AS sample_count
FROM emotion_history
GROUP BY user_id, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('emotion_hourly',
  start_offset => INTERVAL '7 days',
  end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour',
  if_not_exists => true
);

-- Hourly voice analysis averages per user
CREATE MATERIALIZED VIEW IF NOT EXISTS voice_hourly
WITH (timescaledb.continuous) AS
SELECT
  user_id,
  time_bucket('1 hour', created_at) AS bucket,
  AVG(stress) AS avg_stress,
  AVG(focus) AS avg_focus,
  AVG(valence) AS avg_valence,
  AVG(arousal) AS avg_arousal,
  COUNT(*) AS sample_count
FROM voice_history
GROUP BY user_id, bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('voice_hourly',
  start_offset => INTERVAL '7 days',
  end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour',
  if_not_exists => true
);

-- ══════════════════════════════════════════════════════════════════════════
-- Done. Verify with:
--   SELECT * FROM timescaledb_information.hypertables;
--   SELECT * FROM timescaledb_information.compressed_hypertable_stats;
-- ══════════════════════════════════════════════════════════════════════════
