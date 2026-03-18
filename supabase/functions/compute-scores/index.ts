import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import {
  computeRecoveryScore,
  computeSleepScore,
  computeStrainScore,
  computeStressScore,
  computeNutritionScore,
  computeEnergyBank,
} from '../../../shared/score-engines.ts'

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

  const recovery = computeRecoveryScore(agg, base)
  const sleep = computeSleepScore(agg, base)
  const strain = computeStrainScore(agg, base)
  const stress = computeStressScore(agg, base)
  const nutrition = computeNutritionScore(agg, base)
  const energy = computeEnergyBank({ recovery, sleep, strain, stress, nutrition })

  const scores = {
    recovery_score: recovery,
    sleep_score: sleep,
    strain_score: strain,
    stress_score: stress,
    nutrition_score: nutrition,
    energy_bank: energy,
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

  return new Response(JSON.stringify({ scores, status: 'live' }), {
    status: 200, headers: { 'Content-Type': 'application/json' },
  })
})
