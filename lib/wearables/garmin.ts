import type { WearableAdapter, OAuthTokens, HealthSample } from './types';

/**
 * Garmin Health API adapter.
 *
 * IMPORTANT: Garmin uses OAuth 1.0a, which is fundamentally different from OAuth2.
 * The OAuth 1.0a flow requires:
 *   1. Obtain a request token
 *   2. Redirect user to Garmin authorization page
 *   3. Receive oauth_verifier callback
 *   4. Exchange request token + verifier for access token
 *
 * Additionally, Garmin's Health API uses a PUSH model — Garmin sends data to
 * your registered webhook endpoint rather than you polling their API. For the
 * sync() method, we use the Garmin Health API's backfill endpoints to pull
 * historical data on demand.
 *
 * Requires a Garmin Health API developer account and registered application.
 * Consumer key/secret are provided when you register your app at:
 * https://developerportal.garmin.com/
 */

const GARMIN_CONNECT_API = 'https://apis.garmin.com';
const GARMIN_OAUTH_REQUEST_TOKEN = 'https://connectapi.garmin.com/oauth-service/oauth/request_token';
const GARMIN_OAUTH_AUTHORIZE = 'https://connect.garmin.com/oauthConfirm';
const GARMIN_OAUTH_ACCESS_TOKEN = 'https://connectapi.garmin.com/oauth-service/oauth/access_token';

export class GarminAdapter implements WearableAdapter {
  name = 'garmin';
  private consumerKey: string;
  private consumerSecret: string;

  constructor() {
    this.consumerKey = process.env.GARMIN_CONSUMER_KEY || '';
    this.consumerSecret = process.env.GARMIN_CONSUMER_SECRET || '';
  }

  /**
   * NOTE: Garmin uses OAuth 1.0a, not OAuth2. This method returns a placeholder
   * URL. The actual flow requires first obtaining a request token via signed
   * OAuth 1.0a request, then redirecting to the authorize URL with that token.
   *
   * In production, the connect route handler should:
   *   1. Call GARMIN_OAUTH_REQUEST_TOKEN with signed OAuth1 headers
   *   2. Store the request token/secret in the session
   *   3. Return the authorization URL with the request token
   */
  getAuthUrl(redirectUri: string, state: string): string {
    // This is the authorization URL template — requires oauth_token from step 1
    const params = new URLSearchParams({
      oauth_callback: redirectUri,
      state,
    });
    return `${GARMIN_OAUTH_AUTHORIZE}?${params}`;
  }

  /**
   * OAuth 1.0a token exchange. The `code` parameter here represents the
   * oauth_verifier received in the callback. In practice, this also needs
   * the request token and request token secret stored during getAuthUrl().
   *
   * Full implementation requires an OAuth 1.0a signature library (e.g. oauth-1.0a).
   */
  async exchangeCode(code: string, _redirectUri: string): Promise<OAuthTokens> {
    // In a full implementation, this would:
    // 1. Retrieve the stored request token + request token secret
    // 2. Sign a request to GARMIN_OAUTH_ACCESS_TOKEN with oauth_verifier
    // 3. Return the permanent access token

    const res = await fetch(GARMIN_OAUTH_ACCESS_TOKEN, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        // OAuth 1.0a Authorization header would go here (signed)
      },
      body: new URLSearchParams({
        oauth_verifier: code,
      }),
    });

    if (!res.ok) {
      throw new Error(`Garmin OAuth exchange failed: ${res.status} ${res.statusText}. OAuth 1.0a requires signed request headers — see adapter documentation.`);
    }

    const text = await res.text();
    const params = new URLSearchParams(text);

    return {
      accessToken: params.get('oauth_token') || '',
      refreshToken: params.get('oauth_token_secret') || '',
      // Garmin OAuth 1.0a tokens don't expire
    };
  }

  /**
   * Garmin OAuth 1.0a tokens do not expire, so refresh is a no-op.
   * The token secret serves as the "refresh token" for signing future requests.
   */
  async refreshToken(tokenSecret: string): Promise<OAuthTokens> {
    // OAuth 1.0a tokens are permanent — no refresh needed
    return {
      accessToken: '', // Existing access token should be retained
      refreshToken: tokenSecret,
    };
  }

  /**
   * Sync data from Garmin Health API.
   *
   * NOTE: Garmin primarily uses a push model where they send data to your
   * registered webhook URL. This sync method uses the backfill/summary
   * endpoints to pull data on demand. All requests must be signed with
   * OAuth 1.0a (consumer key/secret + user token/secret).
   */
  async sync(accessToken: string, since: Date): Promise<HealthSample[]> {
    const samples: HealthSample[] = [];
    const startSec = Math.floor(since.getTime() / 1000);
    const endSec = Math.floor(Date.now() / 1000);

    // Note: All Garmin API calls require OAuth 1.0a signed headers.
    // The accessToken alone is insufficient — you also need the token secret
    // (stored as refreshToken) and consumer key/secret to generate the
    // OAuth signature. This is a simplified version showing the API shape.

    const headers = {
      Authorization: `Bearer ${accessToken}`, // Placeholder — real impl needs OAuth1 sig
    };

    // Daily Summaries
    try {
      const res = await fetch(
        `${GARMIN_CONNECT_API}/wellness-api/rest/dailies?uploadStartTimeInSeconds=${startSec}&uploadEndTimeInSeconds=${endSec}`,
        { headers }
      );
      const data = await res.json();
      for (const d of Array.isArray(data) ? data : []) {
        const ts = new Date((d.startTimeInSeconds || 0) * 1000).toISOString();
        if (d.steps != null) samples.push({ source: 'garmin', metric: 'steps', value: d.steps, unit: 'steps', recorded_at: ts });
        if (d.activeKilocalories != null) samples.push({ source: 'garmin', metric: 'active_calories', value: d.activeKilocalories, unit: 'kcal', recorded_at: ts });
        if (d.restingHeartRateInBeatsPerMinute != null) samples.push({ source: 'garmin', metric: 'resting_hr', value: d.restingHeartRateInBeatsPerMinute, unit: 'bpm', recorded_at: ts });
        if (d.averageHeartRateInBeatsPerMinute != null) samples.push({ source: 'garmin', metric: 'heart_rate', value: d.averageHeartRateInBeatsPerMinute, unit: 'bpm', recorded_at: ts });
        if (d.averageStressLevel != null) samples.push({ source: 'garmin', metric: 'stress_level', value: d.averageStressLevel, unit: 'score', recorded_at: ts });
        if (d.stepsGoal != null) samples.push({ source: 'garmin', metric: 'steps_goal', value: d.stepsGoal, unit: 'steps', recorded_at: ts, metadata: { type: 'goal' } });
        if (d.bodyBatteryChargedValue != null) samples.push({ source: 'garmin', metric: 'body_battery', value: d.bodyBatteryChargedValue, unit: 'score', recorded_at: ts });
      }
    } catch (e) { console.error('[Garmin] Daily summary sync failed:', e); }

    // Sleep Summaries
    try {
      const res = await fetch(
        `${GARMIN_CONNECT_API}/wellness-api/rest/epochs?uploadStartTimeInSeconds=${startSec}&uploadEndTimeInSeconds=${endSec}`,
        { headers }
      );
      // Garmin sleep endpoint
      const sleepRes = await fetch(
        `${GARMIN_CONNECT_API}/wellness-api/rest/sleeps?uploadStartTimeInSeconds=${startSec}&uploadEndTimeInSeconds=${endSec}`,
        { headers }
      );
      const sleepData = await sleepRes.json();
      for (const s of Array.isArray(sleepData) ? sleepData : []) {
        const ts = new Date((s.startTimeInSeconds || 0) * 1000).toISOString();
        if (s.durationInSeconds != null) samples.push({ source: 'garmin', metric: 'sleep_total_min', value: s.durationInSeconds / 60, unit: 'min', recorded_at: ts });
        if (s.deepSleepDurationInSeconds != null) samples.push({ source: 'garmin', metric: 'sleep_deep_min', value: s.deepSleepDurationInSeconds / 60, unit: 'min', recorded_at: ts });
        if (s.lightSleepDurationInSeconds != null) samples.push({ source: 'garmin', metric: 'sleep_light_min', value: s.lightSleepDurationInSeconds / 60, unit: 'min', recorded_at: ts });
        if (s.remSleepInSeconds != null) samples.push({ source: 'garmin', metric: 'sleep_rem_min', value: s.remSleepInSeconds / 60, unit: 'min', recorded_at: ts });
        if (s.awakeDurationInSeconds != null) samples.push({ source: 'garmin', metric: 'sleep_awake_min', value: s.awakeDurationInSeconds / 60, unit: 'min', recorded_at: ts });
        if (s.averageSpO2Value != null) samples.push({ source: 'garmin', metric: 'spo2', value: s.averageSpO2Value, unit: '%', recorded_at: ts });
        if (s.averageRespirationValue != null) samples.push({ source: 'garmin', metric: 'respiratory_rate', value: s.averageRespirationValue, unit: 'brpm', recorded_at: ts });
      }
    } catch (e) { console.error('[Garmin] Sleep sync failed:', e); }

    // Activity Summaries (workouts)
    try {
      const res = await fetch(
        `${GARMIN_CONNECT_API}/wellness-api/rest/activities?uploadStartTimeInSeconds=${startSec}&uploadEndTimeInSeconds=${endSec}`,
        { headers }
      );
      const data = await res.json();
      for (const a of Array.isArray(data) ? data : []) {
        const ts = new Date((a.startTimeInSeconds || 0) * 1000).toISOString();
        if (a.activeKilocalories != null) samples.push({ source: 'garmin', metric: 'workout_calories', value: a.activeKilocalories, unit: 'kcal', recorded_at: ts, metadata: { type: a.activityType } });
        if (a.averageHeartRateInBeatsPerMinute != null) samples.push({ source: 'garmin', metric: 'workout_avg_hr', value: a.averageHeartRateInBeatsPerMinute, unit: 'bpm', recorded_at: ts });
        if (a.durationInSeconds != null) samples.push({ source: 'garmin', metric: 'workout_duration', value: a.durationInSeconds / 60, unit: 'min', recorded_at: ts });
      }
    } catch (e) { console.error('[Garmin] Activity sync failed:', e); }

    // HRV (if available)
    try {
      const res = await fetch(
        `${GARMIN_CONNECT_API}/wellness-api/rest/hrv?uploadStartTimeInSeconds=${startSec}&uploadEndTimeInSeconds=${endSec}`,
        { headers }
      );
      const data = await res.json();
      for (const h of Array.isArray(data) ? data : []) {
        const ts = new Date((h.startTimeInSeconds || 0) * 1000).toISOString();
        if (h.hrvValue != null) samples.push({ source: 'garmin', metric: 'hrv_rmssd', value: h.hrvValue, unit: 'ms', recorded_at: ts });
      }
    } catch (e) { console.error('[Garmin] HRV sync failed:', e); }

    return samples;
  }

  getCapabilities(): string[] {
    return ['steps', 'active_calories', 'resting_hr', 'heart_rate', 'stress_level', 'body_battery', 'sleep_total_min', 'sleep_deep_min', 'sleep_light_min', 'sleep_rem_min', 'spo2', 'respiratory_rate', 'hrv_rmssd', 'workout_calories', 'workout_avg_hr', 'workout_duration'];
  }
}
