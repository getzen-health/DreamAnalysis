import OpenAI from 'openai';

let cachedClient: OpenAI | null = null;

// Uses Groq's OpenAI-compatible API to serve open-source models (Llama 3.3 70B).
// Get a free API key at https://console.groq.com
export function getOpenAIClient(): OpenAI {
  if (cachedClient) {
    return cachedClient;
  }

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    throw new Error('GROQ_API_KEY environment variable is not set');
  }

  cachedClient = new OpenAI({
    apiKey,
    baseURL: 'https://api.groq.com/openai/v1',
  });

  return cachedClient;
}
