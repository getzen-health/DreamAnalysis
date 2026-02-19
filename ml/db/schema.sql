-- TimescaleDB Schema for Neural Dream Workshop
-- Run once in the Neon console or via psql

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================
-- brain_readings  (1Hz downsampled from 4Hz WebSocket stream)
-- ============================================================
CREATE TABLE IF NOT EXISTS brain_readings (
    time                TIMESTAMPTZ     NOT NULL,
    user_id             TEXT            NOT NULL DEFAULT 'default',

    -- Band powers (averaged from 4 frames)
    alpha               REAL,
    beta                REAL,
    theta               REAL,
    gamma               REAL,
    delta               REAL,

    -- Derived metrics
    emotion             TEXT,           -- dominant emotion label
    valence             REAL,           -- -1..1
    arousal             REAL,           -- -1..1
    focus_index         REAL,           -- 0..1
    stress_index        REAL,           -- 0..1
    relaxation_idx      REAL,           -- 0..1
    flow_score          REAL,           -- 0..1
    creativity_score    REAL,           -- 0..1
    attention_score     REAL,           -- 0..1
    sleep_stage         TEXT,           -- wake/n1/n2/n3/rem or NULL

    -- Signal quality
    sqi                 REAL,           -- 0..1 signal quality index

    -- Raw snapshot: last 64-sample frame per second per channel (4×64 JSONB)
    raw_snapshot        JSONB
);

-- Convert to hypertable partitioned by time
SELECT create_hypertable(
    'brain_readings',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- 90-day data retention
SELECT add_retention_policy(
    'brain_readings',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

-- Index for user queries
CREATE INDEX IF NOT EXISTS brain_readings_user_time_idx
    ON brain_readings (user_id, time DESC);

-- ============================================================
-- health_samples_ts  (replaces SQLite health_samples)
-- ============================================================
CREATE TABLE IF NOT EXISTS health_samples_ts (
    time                TIMESTAMPTZ     NOT NULL,
    user_id             TEXT            NOT NULL DEFAULT 'default',
    metric_type         TEXT            NOT NULL,   -- steps/heart_rate/hrv/sleep/etc.
    value               REAL,
    unit                TEXT,
    source              TEXT,           -- apple_health / google_fit / manual
    metadata            JSONB
);

SELECT create_hypertable(
    'health_samples_ts',
    'time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy(
    'health_samples_ts',
    INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS health_samples_user_type_time_idx
    ON health_samples_ts (user_id, metric_type, time DESC);

-- ============================================================
-- Continuous aggregate: 1-minute buckets
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS brain_readings_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time)   AS bucket,
    user_id,
    AVG(alpha)                      AS alpha,
    AVG(beta)                       AS beta,
    AVG(theta)                      AS theta,
    AVG(gamma)                      AS gamma,
    AVG(delta)                      AS delta,
    AVG(valence)                    AS valence,
    AVG(arousal)                    AS arousal,
    AVG(focus_index)                AS focus_index,
    AVG(stress_index)               AS stress_index,
    AVG(relaxation_idx)             AS relaxation_idx,
    AVG(flow_score)                 AS flow_score,
    AVG(creativity_score)           AS creativity_score,
    AVG(attention_score)            AS attention_score,
    AVG(sqi)                        AS sqi,
    -- Dominant emotion in bucket
    mode() WITHIN GROUP (ORDER BY emotion) AS dominant_emotion,
    COUNT(*)                        AS sample_count
FROM brain_readings
GROUP BY bucket, user_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'brain_readings_1min',
    start_offset    => INTERVAL '1 hour',
    end_offset      => INTERVAL '30 seconds',
    schedule_interval => INTERVAL '30 seconds',
    if_not_exists => TRUE
);

-- ============================================================
-- Continuous aggregate: 1-hour buckets  (chained from 1min)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS brain_readings_1hr
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', bucket)   AS bucket,
    user_id,
    AVG(alpha)                      AS alpha,
    AVG(beta)                       AS beta,
    AVG(theta)                      AS theta,
    AVG(gamma)                      AS gamma,
    AVG(delta)                      AS delta,
    AVG(valence)                    AS valence,
    AVG(arousal)                    AS arousal,
    AVG(focus_index)                AS focus_index,
    AVG(stress_index)               AS stress_index,
    AVG(relaxation_idx)             AS relaxation_idx,
    AVG(flow_score)                 AS flow_score,
    AVG(creativity_score)           AS creativity_score,
    AVG(attention_score)            AS attention_score,
    AVG(sqi)                        AS sqi,
    mode() WITHIN GROUP (ORDER BY dominant_emotion) AS dominant_emotion,
    SUM(sample_count)               AS sample_count
FROM brain_readings_1min
GROUP BY time_bucket('1 hour', bucket), user_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'brain_readings_1hr',
    start_offset    => INTERVAL '2 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ============================================================
-- Continuous aggregate: 1-day buckets  (chained from 1hr)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS brain_readings_1day
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', bucket)    AS bucket,
    user_id,
    AVG(alpha)                      AS alpha,
    AVG(beta)                       AS beta,
    AVG(theta)                      AS theta,
    AVG(gamma)                      AS gamma,
    AVG(delta)                      AS delta,
    AVG(valence)                    AS valence,
    AVG(arousal)                    AS arousal,
    AVG(focus_index)                AS focus_index,
    AVG(stress_index)               AS stress_index,
    AVG(relaxation_idx)             AS relaxation_idx,
    AVG(flow_score)                 AS flow_score,
    AVG(creativity_score)           AS creativity_score,
    AVG(attention_score)            AS attention_score,
    AVG(sqi)                        AS sqi,
    mode() WITHIN GROUP (ORDER BY dominant_emotion) AS dominant_emotion,
    SUM(sample_count)               AS sample_count
FROM brain_readings_1hr
GROUP BY time_bucket('1 day', bucket), user_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'brain_readings_1day',
    start_offset    => INTERVAL '30 days',
    end_offset      => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
