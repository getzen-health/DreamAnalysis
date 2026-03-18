-- ============================================================
-- Row-Level Security Policies for All Pipeline Tables
-- Run after tables are created via drizzle-kit push
-- ============================================================

-- Enable RLS on all user-owned tables
ALTER TABLE health_samples ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_aggregates ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE score_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE trend_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE device_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE exercises ENABLE ROW LEVEL SECURITY;
ALTER TABLE workouts ENABLE ROW LEVEL SECURITY;
ALTER TABLE workout_sets ENABLE ROW LEVEL SECURITY;
ALTER TABLE workout_templates ENABLE ROW LEVEL SECURITY;
ALTER TABLE body_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE exercise_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE habits ENABLE ROW LEVEL SECURITY;
ALTER TABLE habit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE cycle_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE mood_logs ENABLE ROW LEVEL SECURITY;

-- Default policy: users can only access their own data
CREATE POLICY "own_data" ON health_samples FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON daily_aggregates FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON user_baselines FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON user_scores FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON score_history FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON trend_alerts FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON device_connections FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON workouts FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON workout_templates FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON body_metrics FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON exercise_history FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON habits FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON habit_logs FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON cycle_tracking FOR ALL USING (auth.uid()::text = user_id);
CREATE POLICY "own_data" ON mood_logs FOR ALL USING (auth.uid()::text = user_id);

-- workout_sets: access via parent workout ownership
CREATE POLICY "own_data" ON workout_sets FOR ALL
  USING (EXISTS (SELECT 1 FROM workouts WHERE workouts.id = workout_sets.workout_id AND workouts.user_id = auth.uid()::text));

-- Exercises: shared library readable by all authenticated users
CREATE POLICY "read_all_exercises" ON exercises FOR SELECT TO authenticated USING (true);
CREATE POLICY "manage_custom_exercises" ON exercises FOR INSERT TO authenticated WITH CHECK (is_custom = true AND created_by = auth.uid()::text);
CREATE POLICY "update_custom_exercises" ON exercises FOR UPDATE TO authenticated USING (is_custom = true AND created_by = auth.uid()::text);
CREATE POLICY "delete_custom_exercises" ON exercises FOR DELETE TO authenticated USING (is_custom = true AND created_by = auth.uid()::text);

-- Enable Supabase Realtime on user_scores for live dashboard updates
ALTER PUBLICATION supabase_realtime ADD TABLE user_scores;
