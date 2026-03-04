import OpenAI from 'openai';

let cachedClient: OpenAI | null = null;

// Uses Cerebras's OpenAI-compatible API — open-source Llama 3.1 70B, 1M tokens/day free,
// no credit card required, no IP blocking from Vercel/AWS.
// Get a free API key at https://cloud.cerebras.ai
export function getOpenAIClient(): OpenAI {
  if (cachedClient) {
    return cachedClient;
  }

  const apiKey = process.env.CEREBRAS_API_KEY;
  if (!apiKey) {
    throw new Error('CEREBRAS_API_KEY environment variable is not set');
  }

  cachedClient = new OpenAI({
    apiKey,
    baseURL: 'https://api.cerebras.ai/v1',
  });

  return cachedClient;
}
