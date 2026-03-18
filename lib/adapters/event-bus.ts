export interface EventBus {
  publish(channel: string, data: unknown): Promise<void>
  subscribe(channel: string, handler: (data: unknown) => void): void
}

export class PgNotifyEventBus implements EventBus {
  constructor(private supabaseUrl: string, private serviceKey: string) {}

  async publish(channel: string, data: unknown): Promise<void> {
    // Uses Supabase REST to execute pg_notify
    await fetch(`${this.supabaseUrl}/rest/v1/rpc/pg_notify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.serviceKey}`,
        'apikey': this.serviceKey,
      },
      body: JSON.stringify({ channel, payload: JSON.stringify(data) }),
    })
  }

  subscribe(_channel: string, _handler: (data: unknown) => void): void {
    // In PG implementation, subscription happens via Supabase Realtime
    // or Database Webhooks — Edge Functions are stateless
  }
}
