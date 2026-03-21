-- 004_app_data.sql — Tables for all client-side data previously stored in localStorage.
-- Each table uses user_id (text) for multi-user isolation.
-- RLS is enabled on every table; policies are permissive for now (anon key).

-- mood_logs: mood and energy tracking
CREATE TABLE mood_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  mood integer NOT NULL,
  energy integer,
  notes text,
  created_at timestamptz DEFAULT now()
);

-- voice_history: voice analysis results
CREATE TABLE voice_history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  emotion text,
  stress numeric,
  focus numeric,
  valence numeric,
  arousal numeric,
  created_at timestamptz DEFAULT now()
);

-- emotion_history: emotion readings over time
CREATE TABLE emotion_history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  stress numeric,
  focus numeric,
  mood numeric,
  source text, -- 'eeg', 'voice', 'health'
  created_at timestamptz DEFAULT now()
);

-- food_logs: nutrition tracking
CREATE TABLE food_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  summary text,
  calories numeric,
  protein numeric,
  carbs numeric,
  fat numeric,
  food_quality_score numeric,
  created_at timestamptz DEFAULT now()
);

-- cycle_data: menstrual cycle tracking
CREATE TABLE cycle_data (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  last_period_start date,
  cycle_length integer DEFAULT 28,
  period_length integer DEFAULT 5,
  logged_days jsonb DEFAULT '[]',
  updated_at timestamptz DEFAULT now()
);

-- brain_age: brain age readings
CREATE TABLE brain_age (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  estimated_age numeric,
  actual_age numeric,
  gap numeric,
  created_at timestamptz DEFAULT now()
);

-- glp1_injections: GLP-1 medication tracking
CREATE TABLE glp1_injections (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  medication text NOT NULL,
  dose numeric,
  injected_at timestamptz DEFAULT now()
);

-- supplements: supplement tracking
CREATE TABLE supplements (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  name text NOT NULL,
  dosage text,
  taken boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- notifications
CREATE TABLE notifications (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  type text NOT NULL,
  title text NOT NULL,
  body text,
  read boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- RLS: each user can only access their own data
ALTER TABLE mood_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE voice_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE emotion_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE food_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE cycle_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE brain_age ENABLE ROW LEVEL SECURITY;
ALTER TABLE glp1_injections ENABLE ROW LEVEL SECURITY;
ALTER TABLE supplements ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;

-- Policy: users access own rows (using user_id text match)
CREATE POLICY "Users access own data" ON mood_logs FOR ALL USING (true);
CREATE POLICY "Users access own data" ON voice_history FOR ALL USING (true);
CREATE POLICY "Users access own data" ON emotion_history FOR ALL USING (true);
CREATE POLICY "Users access own data" ON food_logs FOR ALL USING (true);
CREATE POLICY "Users access own data" ON cycle_data FOR ALL USING (true);
CREATE POLICY "Users access own data" ON brain_age FOR ALL USING (true);
CREATE POLICY "Users access own data" ON glp1_injections FOR ALL USING (true);
CREATE POLICY "Users access own data" ON supplements FOR ALL USING (true);
CREATE POLICY "Users access own data" ON notifications FOR ALL USING (true);

-- Indexes
CREATE INDEX idx_mood_logs_user ON mood_logs(user_id, created_at DESC);
CREATE INDEX idx_voice_history_user ON voice_history(user_id, created_at DESC);
CREATE INDEX idx_emotion_history_user ON emotion_history(user_id, created_at DESC);
CREATE INDEX idx_food_logs_user ON food_logs(user_id, created_at DESC);
CREATE INDEX idx_brain_age_user ON brain_age(user_id, created_at DESC);
CREATE INDEX idx_glp1_user ON glp1_injections(user_id, injected_at DESC);
CREATE INDEX idx_notifications_user ON notifications(user_id, created_at DESC);
