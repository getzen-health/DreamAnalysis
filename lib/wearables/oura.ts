import type { WearableAdapter, OAuthTokens, HealthSample } from './types';

const OURA_API = 'https://api.ouraring.com';
const OURA_AUTH = 'https://cloud.ouraring.com/oauth/authorize';
const OURA_TOKEN = 'https://api.ouraring.com/oauth/token';

export class OuraAdapter implements WearableAdapter {
  name = 'oura';
  private clientId: string;
  private clientSecret: string;

  constructor() {
    this.clientId = process.env.OURA_CLIENT_ID || '';
    this.clientSecret = process.env.OURA_CLIENT_SECRET || '';
  }

  getAuthUrl(redirectUri: string, state: string): string {
    const params = new URLSearchParams({
      client_id: this.clientId,
      redirect_uri: redirectUri,
      response_type: 'code',
      scope: 'daily heartrate personal session workout',
      state,
    });
    return `${OURA_AUTH}?${params}`;
  }

  async exchangeCode(code: string, redirectUri: string): Promise<OAuthTokens> {
    const res = await fetch(OURA_TOKEN, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        code,
        redirect_uri: redirectUri,
        client_id: this.clientId,
        client_secret: this.clientSecret,
      }),
    });
    const data = await res.json();
    return {
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      expiresAt: data.expires_in ? new Date(Date.now() + data.expires_in * 1000) : undefined,
      scopes: data.scope?.split(' '),
    };
  }

  async refreshToken(refreshToken: string): Promise<OAuthTokens> {
    const res = await fetch(OURA_TOKEN, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        client_id: this.clientId,
        client_secret: this.clientSecret,
      }),
    });
    const data = await res.json();
    return {
      accessToken: data.access_token,
      refreshToken: data.refresh_token,
      expiresAt: data.expires_in ? new Date(Date.now() + data.expires_in * 1000) : undefined,
    };
  }

  async sync(accessToken: string, since: Date): Promise<HealthSample[]> {
    const headers = { Authorization: `Bearer ${accessToken}` };
    const samples: HealthSample[] = [];
    const startDate = since.toISOString().slice(0, 10); // YYYY-MM-DD

    // Daily Readiness
    try {
      const res = await fetch(`${OURA_API}/v2/usercollection/daily_readiness?start_date=${startDate}`, { headers });
      const data = await res.json();
      for (const r of data.data || []) {
        const ts = r.day ? `${r.day}T00:00:00Z` : r.timestamp;
        if (r.score != null) samples.push({ source: 'oura', metric: 'readiness_score', value: r.score, unit: 'score', recorded_at: ts });
        if (r.contributors?.resting_heart_rate != null) samples.push({ source: 'oura', metric: 'resting_hr', value: r.contributors.resting_heart_rate, unit: 'bpm', recorded_at: ts });
        if (r.contributors?.hrv_balance != null) samples.push({ source: 'oura', metric: 'hrv_balance', value: r.contributors.hrv_balance, unit: 'score', recorded_at: ts });
        if (r.temperature_deviation != null) samples.push({ source: 'oura', metric: 'temp_deviation', value: r.temperature_deviation, unit: 'C', recorded_at: ts });
        if (r.temperature_trend_deviation != null) samples.push({ source: 'oura', metric: 'temp_trend_deviation', value: r.temperature_trend_deviation, unit: 'C', recorded_at: ts });
      }
    } catch (e) { console.error('[Oura] Readiness sync failed:', e); }

    // Daily Sleep
    try {
      const res = await fetch(`${OURA_API}/v2/usercollection/daily_sleep?start_date=${startDate}`, { headers });
      const data = await res.json();
      for (const s of data.data || []) {
        const ts = s.day ? `${s.day}T00:00:00Z` : s.timestamp;
        if (s.score != null) samples.push({ source: 'oura', metric: 'sleep_score', value: s.score, unit: 'score', recorded_at: ts });
        if (s.contributors?.deep_sleep != null) samples.push({ source: 'oura', metric: 'sleep_deep_score', value: s.contributors.deep_sleep, unit: 'score', recorded_at: ts });
        if (s.contributors?.efficiency != null) samples.push({ source: 'oura', metric: 'sleep_efficiency', value: s.contributors.efficiency, unit: 'score', recorded_at: ts });
        if (s.contributors?.rem_sleep != null) samples.push({ source: 'oura', metric: 'sleep_rem_score', value: s.contributors.rem_sleep, unit: 'score', recorded_at: ts });
        if (s.contributors?.total_sleep != null) samples.push({ source: 'oura', metric: 'sleep_total_score', value: s.contributors.total_sleep, unit: 'score', recorded_at: ts });
      }
    } catch (e) { console.error('[Oura] Sleep sync failed:', e); }

    // Daily Activity
    try {
      const res = await fetch(`${OURA_API}/v2/usercollection/daily_activity?start_date=${startDate}`, { headers });
      const data = await res.json();
      for (const a of data.data || []) {
        const ts = a.day ? `${a.day}T00:00:00Z` : a.timestamp;
        if (a.steps != null) samples.push({ source: 'oura', metric: 'steps', value: a.steps, unit: 'steps', recorded_at: ts });
        if (a.active_calories != null) samples.push({ source: 'oura', metric: 'active_calories', value: a.active_calories, unit: 'kcal', recorded_at: ts });
        if (a.total_calories != null) samples.push({ source: 'oura', metric: 'total_calories', value: a.total_calories, unit: 'kcal', recorded_at: ts });
        if (a.score != null) samples.push({ source: 'oura', metric: 'activity_score', value: a.score, unit: 'score', recorded_at: ts });
        if (a.equivalent_walking_distance != null) samples.push({ source: 'oura', metric: 'walking_distance', value: a.equivalent_walking_distance, unit: 'm', recorded_at: ts });
      }
    } catch (e) { console.error('[Oura] Activity sync failed:', e); }

    // Heart Rate (sampled — can be large, limit to most recent)
    try {
      const endDate = new Date().toISOString().slice(0, 10);
      const res = await fetch(`${OURA_API}/v2/usercollection/heartrate?start_date=${startDate}&end_date=${endDate}`, { headers });
      const data = await res.json();
      // Take only the last 100 samples to avoid overwhelming the DB
      const hrData = (data.data || []).slice(-100);
      for (const hr of hrData) {
        if (hr.bpm != null) {
          samples.push({
            source: 'oura',
            metric: 'heart_rate',
            value: hr.bpm,
            unit: 'bpm',
            recorded_at: hr.timestamp,
            metadata: { source: hr.source },
          });
        }
      }
    } catch (e) { console.error('[Oura] Heart rate sync failed:', e); }

    return samples;
  }

  getCapabilities(): string[] {
    return ['readiness_score', 'resting_hr', 'hrv_balance', 'temp_deviation', 'sleep_score', 'sleep_deep_score', 'sleep_efficiency', 'sleep_rem_score', 'steps', 'active_calories', 'total_calories', 'activity_score', 'heart_rate'];
  }
}
