export interface OAuthTokens {
  accessToken: string;
  refreshToken?: string;
  expiresAt?: Date;
  scopes?: string[];
}

export interface HealthSample {
  source: string;
  metric: string;
  value: number;
  unit: string;
  recorded_at: string;
  metadata?: Record<string, unknown>;
}

export interface WearableAdapter {
  name: string;
  getAuthUrl(redirectUri: string, state: string): string;
  exchangeCode(code: string, redirectUri: string): Promise<OAuthTokens>;
  refreshToken(refreshToken: string): Promise<OAuthTokens>;
  sync(accessToken: string, since: Date): Promise<HealthSample[]>;
  getCapabilities(): string[];
}
