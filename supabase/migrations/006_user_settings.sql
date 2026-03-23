-- Generic key-value settings (string values)
CREATE TABLE IF NOT EXISTS user_settings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  key text NOT NULL,
  value text,
  updated_at timestamptz DEFAULT now(),
  UNIQUE(user_id, key)
);

-- Generic key-value store (JSON blobs)
CREATE TABLE IF NOT EXISTS generic_store (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  key text NOT NULL,
  value jsonb,
  updated_at timestamptz DEFAULT now(),
  UNIQUE(user_id, key)
);

ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE generic_store ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users access own settings" ON user_settings
  FOR ALL USING ((SELECT auth.uid()::text) = user_id);

CREATE POLICY "Users access own data" ON generic_store
  FOR ALL USING ((SELECT auth.uid()::text) = user_id);

CREATE INDEX IF NOT EXISTS idx_settings_user_key ON user_settings(user_id, key);
CREATE INDEX IF NOT EXISTS idx_generic_user_key ON generic_store(user_id, key);
