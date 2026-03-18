-- ============================================================
-- PG Triggers for Health Data Pipeline
-- Run after tables are created via drizzle-kit push
-- ============================================================

-- Trigger 1: Update daily_aggregates on health_samples INSERT
CREATE OR REPLACE FUNCTION update_daily_aggregates()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO daily_aggregates (id, user_id, date, metric, avg_value, min_value, max_value, sum_value, sample_count, updated_at)
  VALUES (
    gen_random_uuid(),
    NEW.user_id,
    DATE(NEW.recorded_at),
    NEW.metric,
    NEW.value,
    NEW.value,
    NEW.value,
    NEW.value,
    1,
    NOW()
  )
  ON CONFLICT (user_id, date, metric) DO UPDATE SET
    avg_value = (daily_aggregates.avg_value * daily_aggregates.sample_count + NEW.value) / (daily_aggregates.sample_count + 1),
    min_value = LEAST(daily_aggregates.min_value, NEW.value),
    max_value = GREATEST(daily_aggregates.max_value, NEW.value),
    sum_value = daily_aggregates.sum_value + NEW.value,
    sample_count = daily_aggregates.sample_count + 1,
    updated_at = NOW();

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_health_samples_aggregate
  AFTER INSERT ON health_samples
  FOR EACH ROW
  EXECUTE FUNCTION update_daily_aggregates();

-- Trigger 2: Update user_baselines (14-day rolling) on health_samples INSERT
CREATE OR REPLACE FUNCTION update_user_baselines()
RETURNS TRIGGER AS $$
DECLARE
  _avg numeric;
  _stddev numeric;
  _count integer;
BEGIN
  SELECT AVG(avg_value), STDDEV(avg_value), COUNT(*)
  INTO _avg, _stddev, _count
  FROM daily_aggregates
  WHERE user_id = NEW.user_id
    AND metric = NEW.metric
    AND date >= CURRENT_DATE - INTERVAL '14 days';

  INSERT INTO user_baselines (user_id, metric, baseline_avg, baseline_stddev, sample_count, last_updated)
  VALUES (NEW.user_id, NEW.metric, _avg, COALESCE(_stddev, 0), _count, NOW())
  ON CONFLICT (user_id, metric) DO UPDATE SET
    baseline_avg = _avg,
    baseline_stddev = COALESCE(_stddev, 0),
    sample_count = _count,
    last_updated = NOW();

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_health_samples_baselines
  AFTER INSERT ON health_samples
  FOR EACH ROW
  EXECUTE FUNCTION update_user_baselines();

-- Trigger 3: Simple threshold alerts on health_samples INSERT
CREATE OR REPLACE FUNCTION check_health_thresholds()
RETURNS TRIGGER AS $$
BEGIN
  -- Heart rate too high
  IF NEW.metric = 'heart_rate' AND NEW.value > 120 THEN
    INSERT INTO trend_alerts (id, user_id, alert_type, metric, severity, message, value, created_at)
    VALUES (gen_random_uuid(), NEW.user_id, 'threshold', 'heart_rate', 'warning',
            'Heart rate above 120 bpm: ' || ROUND(NEW.value) || ' bpm', NEW.value, NOW());
  END IF;

  -- SpO2 too low
  IF NEW.metric = 'spo2' AND NEW.value < 94 THEN
    INSERT INTO trend_alerts (id, user_id, alert_type, metric, severity, message, value, created_at)
    VALUES (gen_random_uuid(), NEW.user_id, 'threshold', 'spo2', 'critical',
            'Blood oxygen below 94%: ' || ROUND(NEW.value, 1) || '%', NEW.value, NOW());
  END IF;

  -- Resting HR too high
  IF NEW.metric = 'resting_hr' AND NEW.value > 100 THEN
    INSERT INTO trend_alerts (id, user_id, alert_type, metric, severity, message, value, created_at)
    VALUES (gen_random_uuid(), NEW.user_id, 'threshold', 'resting_hr', 'warning',
            'Resting heart rate above 100 bpm: ' || ROUND(NEW.value) || ' bpm', NEW.value, NOW());
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_health_thresholds
  AFTER INSERT ON health_samples
  FOR EACH ROW
  EXECUTE FUNCTION check_health_thresholds();
