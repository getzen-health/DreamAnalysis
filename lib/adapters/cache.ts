export interface UserScores {
  recoveryScore: number | null
  sleepScore: number | null
  strainScore: number | null
  stressScore: number | null
  nutritionScore: number | null
  energyBank: number | null
  computedAt: string | null
}

export interface ScoreCache {
  get(userId: string): Promise<UserScores | null>
  set(userId: string, scores: UserScores): Promise<void>
  invalidate(userId: string): Promise<void>
}

export class PgTableCache implements ScoreCache {
  constructor(private supabaseUrl: string, private serviceKey: string) {}

  async get(userId: string): Promise<UserScores | null> {
    const response = await fetch(
      `${this.supabaseUrl}/rest/v1/user_scores?user_id=eq.${userId}&select=*`,
      {
        headers: {
          'Authorization': `Bearer ${this.serviceKey}`,
          'apikey': this.serviceKey,
        },
      }
    )
    const rows = await response.json()
    if (!rows?.length) return null
    const data = rows[0]
    return {
      recoveryScore: data.recovery_score,
      sleepScore: data.sleep_score,
      strainScore: data.strain_score,
      stressScore: data.stress_score,
      nutritionScore: data.nutrition_score,
      energyBank: data.energy_bank,
      computedAt: data.computed_at,
    }
  }

  async set(userId: string, scores: UserScores): Promise<void> {
    await fetch(`${this.supabaseUrl}/rest/v1/user_scores`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.serviceKey}`,
        'apikey': this.serviceKey,
        'Prefer': 'resolution=merge-duplicates',
      },
      body: JSON.stringify({
        user_id: userId,
        recovery_score: scores.recoveryScore,
        sleep_score: scores.sleepScore,
        strain_score: scores.strainScore,
        stress_score: scores.stressScore,
        nutrition_score: scores.nutritionScore,
        energy_bank: scores.energyBank,
        computed_at: new Date().toISOString(),
      }),
    })
  }

  async invalidate(_userId: string): Promise<void> {
    // No-op for PG cache — data is always fresh from table
  }
}
