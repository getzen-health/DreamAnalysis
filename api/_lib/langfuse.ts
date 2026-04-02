/**
 * Langfuse observability client — LLM tracing for all AI calls.
 *
 * Activated only when LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY are set.
 * Safe no-op otherwise — zero overhead in development.
 *
 * Usage:
 *   import { getLangfuse } from './_lib/langfuse.js';
 *   const lf = getLangfuse();
 *   const trace = lf?.trace({ name: 'dream-analysis', userId });
 *   const gen = trace?.generation({ name: 'llm-call', model, input: messages });
 *   // ... await llm call ...
 *   gen?.end({ output: response.choices[0].message.content });
 *   await lf?.flushAsync();
 */

import { Langfuse } from 'langfuse';

let _client: Langfuse | null = null;
let _initialized = false;

export function getLangfuse(): Langfuse | null {
  if (_initialized) return _client;
  _initialized = true;

  const secretKey  = process.env.LANGFUSE_SECRET_KEY;
  const publicKey  = process.env.LANGFUSE_PUBLIC_KEY;
  const baseUrl    = process.env.LANGFUSE_BASEURL ?? 'https://cloud.langfuse.com';

  if (!secretKey || !publicKey) return null;

  _client = new Langfuse({
    secretKey,
    publicKey,
    baseUrl,
    release: process.env.npm_package_version ?? '1.0.0',
    environment: process.env.NODE_ENV ?? 'production',
    flushInterval: 2000,
  });

  return _client;
}

/** Flush pending traces — call at the end of each serverless invocation. */
export async function flushLangfuse(): Promise<void> {
  if (_client) await _client.flushAsync().catch(() => {});
}
