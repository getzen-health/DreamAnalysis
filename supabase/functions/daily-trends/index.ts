import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

Deno.serve(async (_req: Request) => {
  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
  )

  // Get active users (have data in last 7 days)
  const sevenDaysAgo = new Date(Date.now() - 7 * 86400000).toISOString().split('T')[0]
  const { data: activeRows } = await supabase
    .from('daily_aggregates')
    .select('user_id')
    .gte('date', sevenDaysAgo)

  const userIds = [...new Set((activeRows || []).map((r: any) => r.user_id))]
  const alerts: any[] = []

  for (const userId of userIds) {
    // Weight trend: >2% change in 7 days
    const { data: weightData } = await supabase
      .from('body_metrics')
      .select('weight_kg, recorded_at')
      .eq('user_id', userId)
      .order('recorded_at', { ascending: false })
      .limit(14)

    if (weightData && weightData.length >= 2) {
      const latest = Number(weightData[0].weight_kg)
      const sixDaysAgo = new Date(Date.now() - 6 * 86400000)
      const weekAgo = weightData.find((w: any) => new Date(w.recorded_at) <= sixDaysAgo)
      if (weekAgo) {
        const change = Math.abs((latest - Number(weekAgo.weight_kg)) / Number(weekAgo.weight_kg))
        if (change > 0.02) {
          alerts.push({
            user_id: userId,
            alert_type: 'rapid_change',
            metric: 'weight',
            severity: change > 0.05 ? 'critical' : 'warning',
            message: `Weight changed ${(change * 100).toFixed(1)}% in the past week`,
            value: latest,
            baseline: Number(weekAgo.weight_kg),
          })
        }
      }
    }

    // HRV declining trend
    const { data: hrvData } = await supabase
      .from('daily_aggregates')
      .select('avg_value, date')
      .eq('user_id', userId)
      .eq('metric', 'hrv_rmssd')
      .gte('date', sevenDaysAgo)
      .order('date', { ascending: true })

    if (hrvData && hrvData.length >= 5) {
      const values = hrvData.map((d: any) => Number(d.avg_value))
      const mid = Math.floor(values.length / 2)
      const firstAvg = values.slice(0, mid).reduce((a: number, b: number) => a + b, 0) / mid
      const secondAvg = values.slice(mid).reduce((a: number, b: number) => a + b, 0) / (values.length - mid)
      if (secondAvg < firstAvg * 0.85) {
        alerts.push({
          user_id: userId,
          alert_type: 'trend',
          metric: 'hrv_rmssd',
          severity: 'warning',
          message: `HRV declining: avg dropped from ${firstAvg.toFixed(0)}ms to ${secondAvg.toFixed(0)}ms over 7 days`,
          value: secondAvg,
          baseline: firstAvg,
        })
      }
    }

    // 90-day data retention cleanup
    const ninetyDaysAgo = new Date(Date.now() - 90 * 86400000).toISOString()
    await supabase.from('health_samples').delete().eq('user_id', userId).lt('ingested_at', ninetyDaysAgo)
  }

  if (alerts.length > 0) {
    await supabase.from('trend_alerts').insert(alerts)
  }

  return new Response(JSON.stringify({ users_processed: userIds.length, alerts_created: alerts.length }), {
    status: 200, headers: { 'Content-Type': 'application/json' },
  })
})
