import OpenAI from 'openai';
import { observeOpenAI } from 'langfuse';

let cachedClient: OpenAI | null = null;

// Uses Groq's OpenAI-compatible API — free Llama 3.3 70B inference.
// 30 requests/minute, no credit card required.
// Get a free API key at https://console.groq.com/keys
export function getOpenAIClient(): OpenAI {
  if (cachedClient) {
    return cachedClient;
  }

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    throw new Error('GROQ_API_KEY environment variable is not set');
  }

  const raw = new OpenAI({
    apiKey,
    baseURL: 'https://api.groq.com/openai/v1',
  });

  // Wrap with Langfuse observability when keys are present (no-op otherwise)
  const secretKey = process.env.LANGFUSE_SECRET_KEY;
  const publicKey = process.env.LANGFUSE_PUBLIC_KEY;
  if (secretKey && publicKey) {
    cachedClient = observeOpenAI(raw, {
      clientInitParams: {
        secretKey,
        publicKey,
        baseUrl: process.env.LANGFUSE_BASEURL ?? 'https://cloud.langfuse.com',
        release: process.env.npm_package_version ?? '1.0.0',
        environment: process.env.NODE_ENV ?? 'production',
        flushInterval: 2000,
      },
      generationName: 'groq-llm',
    }) as unknown as OpenAI;
  } else {
    cachedClient = raw;
  }

  return cachedClient;
}
