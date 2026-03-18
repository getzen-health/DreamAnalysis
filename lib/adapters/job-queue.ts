export interface JobOpts {
  delay?: number
  priority?: number
  retries?: number
}

export interface JobQueue {
  enqueue(job: string, payload: unknown, opts?: JobOpts): Promise<void>
  process(job: string, handler: (payload: unknown) => Promise<void>): void
}

export class DirectCallQueue implements JobQueue {
  constructor(private supabaseUrl: string, private serviceKey: string) {}

  async enqueue(job: string, payload: unknown, _opts?: JobOpts): Promise<void> {
    const response = await fetch(
      `${this.supabaseUrl}/functions/v1/${job}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.serviceKey}`,
        },
        body: JSON.stringify(payload),
      }
    )
    if (!response.ok) {
      console.error(`[DirectCallQueue] ${job} failed:`, await response.text())
    }
  }

  process(_job: string, _handler: (payload: unknown) => Promise<void>): void {
    // No-op for direct call — Edge Functions handle their own processing
  }
}
