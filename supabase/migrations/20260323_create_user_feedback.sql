CREATE TABLE IF NOT EXISTS public.user_feedback (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id TEXT NOT NULL,
  predicted_emotion TEXT NOT NULL,
  corrected_emotion TEXT NOT NULL,
  source TEXT DEFAULT 'manual',
  confidence REAL,
  features JSONB,
  session_id TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_user_feedback_user_id ON public.user_feedback(user_id);
CREATE INDEX idx_user_feedback_created_at ON public.user_feedback(created_at);

ALTER TABLE public.user_feedback ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can insert own feedback"
ON public.user_feedback FOR INSERT TO authenticated
WITH CHECK ((SELECT auth.uid()::text) = user_id);

CREATE POLICY "Users can read own feedback"
ON public.user_feedback FOR SELECT TO authenticated
USING ((SELECT auth.uid()::text) = user_id);
