import OpenAI from 'openai';

let cachedClient: OpenAI | null = null;

// Uses Together AI's OpenAI-compatible API to serve open-source models (Llama 3.3 70B).
// Get a free API key at https://api.together.xyz (free $1 credit, no card required)
export function getOpenAIClient(): OpenAI {
  if (cachedClient) {
    return cachedClient;
  }

  const apiKey = process.env.TOGETHER_API_KEY;
  if (!apiKey) {
    throw new Error('TOGETHER_API_KEY environment variable is not set');
  }

  cachedClient = new OpenAI({
    apiKey,
    baseURL: 'https://api.together.xyz/v1',
  });

  return cachedClient;
}
