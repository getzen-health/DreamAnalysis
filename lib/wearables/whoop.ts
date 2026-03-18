import type { WearableAdapter, OAuthTokens, HealthSample } from './types';

const WHOOP_API = 'https://api.prod.whoop.com';
const WHOOP_AUTH = 'https://api.prod.whoop.com/oauth/oauth2';

export class WhoopAdapter implements WearableAdapter {
  name = 'whoop';
  private clientId: string;
  private clientSecret: string;

  constructor() {
    this.clientId = process.env.WHOOP_CLIENT_ID || '';
    this.clientSecret = process.env.WHOOP_CLIENT_SECRET || '';
  }

  getAuthUrl(redirectUri: string, state: string): string {
    const params = new URLSearchParams({
      client_id: this.clientId,
      redirect_uri: redirectUri,
      response_type: 'code',
      scope: 'read:recovery read:sleep read:workout read:body_measurement read:profile',
      state,
    });
    return `${WHOOP_AUTH}/auth?${params}`;
  }

  async exchangeCode(code: string, redirectUri: string): Promise<OAuthTokens> {
    const res = await fetch(`${WHOOP_AUTH}/token`, {
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
      expiresAt: new Date(Date.now() + data.expires_in * 1000),
      scopes: data.scope?.split(' '),
    };
  }

  async refreshToken(refreshToken: string): Promise<OAuthTokens> {
    const res = await fetch(`${WHOOP_AUTH}/token`, {
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
      expiresAt: new Date(Date.now() + data.expires_in * 1000),
    };
  }

  async sync(accessToken: string, since: Date): Promise<HealthSample[]> {
    const headers = { Authorization: `Bearer ${accessToken}` };
    const samples: HealthSample[] = [];
    const sinceStr = since.toISOString();

    // Recovery
    try {
      const res = await fetch(`${WHOOP_API}/developer/v1/recovery?start=${sinceStr}`, { headers });
      const data = await res.json();
      for (const r of data.records || []) {
        const ts = r.created_at || r.updated_at;
        if (r.score?.hrv?.rmssd_milli) samples.push({ source: 'whoop', metric: 'hrv_rmssd', value: r.score.hrv.rmssd_milli, unit: 'ms', recorded_at: ts });
        if (r.score?.resting_heart_rate) samples.push({ source: 'whoop', metric: 'resting_hr', value: r.score.resting_heart_rate, unit: 'bpm', recorded_at: ts });
        if (r.score?.spo2_percentage) samples.push({ source: 'whoop', metric: 'spo2', value: r.score.spo2_percentage, unit: '%', recorded_at: ts });
        if (r.score?.skin_temp_celsius) samples.push({ source: 'whoop', metric: 'skin_temp', value: r.score.skin_temp_celsius, unit: 'C', recorded_at: ts });
      }
    } catch (e) { console.error('[Whoop] Recovery sync failed:', e); }

    // Sleep
    try {
      const res = await fetch(`${WHOOP_API}/developer/v1/activity/sleep?start=${sinceStr}`, { headers });
      const data = await res.json();
      for (const s of data.records || []) {
        const ts = s.end;
        const stages = s.score?.stage_summary;
        if (stages) {
          if (stages.total_light_sleep_time_milli) samples.push({ source: 'whoop', metric: 'sleep_light_min', value: stages.total_light_sleep_time_milli / 60000, unit: 'min', recorded_at: ts });
          if (stages.total_slow_wave_sleep_time_milli) samples.push({ source: 'whoop', metric: 'sleep_deep_min', value: stages.total_slow_wave_sleep_time_milli / 60000, unit: 'min', recorded_at: ts });
          if (stages.total_rem_sleep_time_milli) samples.push({ source: 'whoop', metric: 'sleep_rem_min', value: stages.total_rem_sleep_time_milli / 60000, unit: 'min', recorded_at: ts });
          if (stages.total_awake_time_milli) samples.push({ source: 'whoop', metric: 'sleep_awake_min', value: stages.total_awake_time_milli / 60000, unit: 'min', recorded_at: ts });

          // Total sleep
          const totalMin = ((stages.total_light_sleep_time_milli || 0) + (stages.total_slow_wave_sleep_time_milli || 0) + (stages.total_rem_sleep_time_milli || 0)) / 60000;
          if (totalMin > 0) samples.push({ source: 'whoop', metric: 'sleep_total_min', value: totalMin, unit: 'min', recorded_at: ts });
        }
        if (s.score?.sleep_efficiency_percentage) samples.push({ source: 'whoop', metric: 'sleep_efficiency', value: s.score.sleep_efficiency_percentage, unit: '%', recorded_at: ts });
        if (s.score?.respiratory_rate) samples.push({ source: 'whoop', metric: 'respiratory_rate', value: s.score.respiratory_rate, unit: 'brpm', recorded_at: ts });
      }
    } catch (e) { console.error('[Whoop] Sleep sync failed:', e); }

    // Workouts
    try {
      const res = await fetch(`${WHOOP_API}/developer/v1/activity/workout?start=${sinceStr}`, { headers });
      const data = await res.json();
      for (const w of data.records || []) {
        if (w.score?.strain) samples.push({ source: 'whoop', metric: 'workout_strain', value: w.score.strain, unit: 'strain', recorded_at: w.end, metadata: { sport: w.sport_id } });
        if (w.score?.average_heart_rate) samples.push({ source: 'whoop', metric: 'heart_rate', value: w.score.average_heart_rate, unit: 'bpm', recorded_at: w.end });
        if (w.score?.kilojoule) samples.push({ source: 'whoop', metric: 'active_calories', value: w.score.kilojoule * 0.239, unit: 'kcal', recorded_at: w.end });
      }
    } catch (e) { console.error('[Whoop] Workout sync failed:', e); }

    return samples;
  }

  getCapabilities(): string[] {
    return ['hrv_rmssd', 'resting_hr', 'spo2', 'skin_temp', 'respiratory_rate', 'sleep_deep_min', 'sleep_rem_min', 'sleep_light_min', 'sleep_efficiency', 'sleep_total_min', 'workout_strain', 'heart_rate', 'active_calories'];
  }
}
