import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

// Score engine stubs — full implementations come in Phase 4
function computeRecoveryScore(_agg: any[], _base: any[]): number | null { return null }
function computeSleepScore(_agg: any[], _base: any[]): number | null { return null }
function computeStrainScore(_agg: any[], _base: any[]): number | null { return null }
function computeStressScore(_agg: any[], _base: any[]): number | null { return null }
function computeNutritionScore(_agg: any[], _base: any[]): number | null { return null }
function computeEnergyBank(_scores: Record<string, number | null>): number | null { return null }

Deno.serve(async (req: Request) => {
  if (req.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 })
  }

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
  )

  const body = await req.json()
  if (!body.user_id) {
    return new Response(JSON.stringify({ error: 'user_id required' }), { status: 400, headers: { 'Content-Type': 'application/json' } })
  }

  const fourteenDaysAgo = new Date(Date.now() - 14 * 86400000).toISOString().split('T')[0]

  const [{ data: aggregates }, { data: baselines }] = await Promise.all([
    supabase.from('daily_aggregates').select('*').eq('user_id', body.user_id).gte('date', fourteenDaysAgo).order('date', { ascending: false }),
    supabase.from('user_baselines').select('*').eq('user_id', body.user_id),
  ])

  const agg = aggregates || []
  const base = baselines || []

  const scores = {
    recovery_score: computeRecoveryScore(agg, base),
    sleep_score: computeSleepScore(agg, base),
    strain_score: computeStrainScore(agg, base),
    stress_score: computeStressScore(agg, base),
    nutrition_score: computeNutritionScore(agg, base),
    energy_bank: computeEnergyBank({}),
    computed_at: new Date().toISOString(),
  }

  // Upsert current scores (triggers Supabase Realtime)
  await supabase.from('user_scores').upsert({ user_id: body.user_id, ...scores })

  // Record daily history
  const today = new Date().toISOString().split('T')[0]
  await supabase.from('score_history').upsert(
    { user_id: body.user_id, date: today, ...scores },
    { onConflict: 'user_id,date' }
  )

  return new Response(JSON.stringify({ scores, status: 'skeleton' }), {
    status: 200, headers: { 'Content-Type': 'application/json' },
  })
})
