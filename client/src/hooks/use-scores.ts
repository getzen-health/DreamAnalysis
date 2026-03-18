import { useState, useEffect } from 'react'

export interface UserScores {
  recoveryScore: number | null
  sleepScore: number | null
  strainScore: number | null
  stressScore: number | null
  nutritionScore: number | null
  energyBank: number | null
  computedAt: string | null
}

export function useScores(userId: string | undefined) {
  const [scores, setScores] = useState<UserScores | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!userId) {
      setLoading(false)
      return
    }

    const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
    const supabaseKey = import.meta.env.VITE_SUPABASE_ANON_KEY

    if (!supabaseUrl || !supabaseKey) {
      console.warn('[useScores] Supabase env vars not set — scores disabled')
      setLoading(false)
      return
    }

    // Dynamic import to avoid breaking if @supabase/supabase-js not installed yet
    import('@supabase/supabase-js').then(({ createClient }) => {
      const supabase = createClient(supabaseUrl, supabaseKey)

      // Initial fetch
      supabase
        .from('user_scores')
        .select('*')
        .eq('user_id', userId)
        .single()
        .then(({ data }) => {
          if (data) {
            setScores({
              recoveryScore: data.recovery_score,
              sleepScore: data.sleep_score,
              strainScore: data.strain_score,
              stressScore: data.stress_score,
              nutritionScore: data.nutrition_score,
              energyBank: data.energy_bank,
              computedAt: data.computed_at,
            })
          }
          setLoading(false)
        })

      // Subscribe to real-time updates
      const channel = supabase
        .channel(`user-scores-${userId}`)
        .on(
          'postgres_changes',
          {
            event: 'UPDATE',
            schema: 'public',
            table: 'user_scores',
            filter: `user_id=eq.${userId}`,
          },
          (payload: any) => {
            const d = payload.new
            setScores({
              recoveryScore: d.recovery_score,
              sleepScore: d.sleep_score,
              strainScore: d.strain_score,
              stressScore: d.stress_score,
              nutritionScore: d.nutrition_score,
              energyBank: d.energy_bank,
              computedAt: d.computed_at,
            })
          }
        )
        .subscribe()

      return () => {
        supabase.removeChannel(channel)
      }
    }).catch(() => {
      console.warn('[useScores] @supabase/supabase-js not available')
      setLoading(false)
    })
  }, [userId])

  return { scores, loading }
}
