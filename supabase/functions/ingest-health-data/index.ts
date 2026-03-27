import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const VALID_SOURCES = ['apple_health', 'google_fit', 'oura', 'whoop', 'garmin', 'cgm', 'eeg', 'manual']
const VALID_METRICS = [
  'heart_rate', 'hrv_rmssd', 'resting_hr', 'respiratory_rate', 'spo2',
  'skin_temp', 'sleep_deep_min', 'sleep_rem_min', 'sleep_light_min',
  'sleep_awake_min', 'sleep_efficiency', 'steps', 'active_calories',
  'weight_kg', 'body_fat_pct', 'lean_mass_kg', 'height_cm', 'vo2_max',
  'workout_strain', 'glucose_mgdl', 'basal_calories', 'exercise_minutes',
  // Extended metrics from Apple Health / Google Health Connect
  'walking_distance_km', 'flights_climbed', 'standing_hours',
  'blood_pressure_systolic', 'blood_pressure_diastolic',
  'body_temperature', 'water_intake_ml',
]

interface HealthSample {
  source: string
  metric: string
  value: number
  unit: string
  recorded_at: string
  metadata?: Record<string, unknown>
}

Deno.serve(async (req: Request) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'authorization, content-type, apikey' } })
  }
  if (req.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 })
  }

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
  )

  let body: { user_id: string; samples: HealthSample[] }
  try {
    body = await req.json()
  } catch {
    return new Response(JSON.stringify({ error: 'Invalid JSON' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
  }

  if (!body.user_id || !body.samples?.length) {
    return new Response(JSON.stringify({ error: 'user_id and samples required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
  }

  const validSamples = body.samples.filter(s =>
    VALID_SOURCES.includes(s.source) &&
    VALID_METRICS.includes(s.metric) &&
    typeof s.value === 'number' && !isNaN(s.value) &&
    s.recorded_at
  ).map(s => ({
    user_id: body.user_id,
    source: s.source,
    metric: s.metric,
    value: s.value,
    unit: s.unit || 'unknown',
    metadata: s.metadata || null,
    recorded_at: s.recorded_at,
    ingested_at: new Date().toISOString(),
  }))

  if (validSamples.length === 0) {
    return new Response(JSON.stringify({ error: 'No valid samples', accepted: 0 }), { status: 400, headers: { 'Content-Type': 'application/json' } })
  }

  // Idempotent insert
  const { error } = await supabase
    .from('health_samples')
    .upsert(validSamples, { onConflict: 'user_id,source,metric,recorded_at', ignoreDuplicates: true })

  if (error) {
    return new Response(JSON.stringify({ error: error.message }), { status: 500, headers: { 'Content-Type': 'application/json' } })
  }

  // Trigger score computation
  const metricsChanged = [...new Set(validSamples.map(s => s.metric))]
  try {
    await fetch(`${Deno.env.get('SUPABASE_URL')}/functions/v1/compute-scores`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')}` },
      body: JSON.stringify({ user_id: body.user_id, metrics_changed: metricsChanged }),
    })
  } catch (e) {
    console.error('Score computation trigger failed:', e)
  }

  return new Response(JSON.stringify({ accepted: validSamples.length, rejected: body.samples.length - validSamples.length }), {
    status: 200, headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
  })
})
