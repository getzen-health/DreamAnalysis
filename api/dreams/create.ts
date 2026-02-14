import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getDb } from '../_lib/db';
import { getOpenAIClient } from '../_lib/openai';
import { success, error, badRequest, methodNotAllowed } from '../_lib/response';
import * as schema from '../../shared/schema';
import { eq, desc } from 'drizzle-orm';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res);

  try {
    const { dreamText, userId, tags, sleepQuality, sleepDuration } = req.body;

    if (!dreamText || !userId) {
      return badRequest(res, 'dreamText and userId are required');
    }

    const openai = getOpenAIClient();
    const db = getDb();

    // Fetch recent dreams for context
    const recentDreams = await db
      .select({ dreamText: schema.dreamAnalysis.dreamText, symbols: schema.dreamAnalysis.symbols })
      .from(schema.dreamAnalysis)
      .where(eq(schema.dreamAnalysis.userId, userId))
      .orderBy(desc(schema.dreamAnalysis.timestamp))
      .limit(5);

    const historyContext = recentDreams.length > 0
      ? `\n\nRecent dream themes: ${recentDreams.map(d => {
          const syms = d.symbols as string[] | null;
          return syms?.join(', ') || 'unknown';
        }).join('; ')}`
      : '';

    // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
    const response = await openai.chat.completions.create({
      model: "gpt-5",
      messages: [
        {
          role: "system",
          content: `You are an expert dream analyst combining Jungian archetypal analysis, Freudian symbolism, and modern neuroscience perspectives. Analyze the dream and respond with JSON:
{
  "symbols": ["symbol1", "symbol2", ...],
  "emotions": [{"emotion": "name", "intensity": 0-10}],
  "analysis": "Detailed interpretation paragraph",
  "lucidityScore": 1-10,
  "themes": ["theme1", "theme2"],
  "wakingLifeConnections": "How this might relate to waking experiences",
  "recurringPatterns": "Any patterns noticed with past dreams"
}${historyContext}`
        },
        {
          role: "user",
          content: `Analyze this dream: ${dreamText}`
        }
      ],
      response_format: { type: "json_object" }
    });

    const analysis = JSON.parse(response.choices[0].message.content || "{}");

    const [dreamEntry] = await db.insert(schema.dreamAnalysis).values({
      userId,
      dreamText,
      symbols: analysis.symbols || [],
      emotions: analysis.emotions || [],
      aiAnalysis: analysis.analysis || "",
      lucidityScore: analysis.lucidityScore || null,
      sleepQuality: sleepQuality || null,
      sleepDuration: sleepDuration || null,
      tags: tags || [],
    }).returning();

    // Upsert dream symbols
    if (analysis.symbols) {
      for (const symbol of analysis.symbols) {
        const meaning = analysis.themes?.find((t: string) =>
          t.toLowerCase().includes(symbol.toLowerCase())
        );
        await db.insert(schema.dreamSymbols).values({
          userId,
          symbol,
          meaning: meaning || null,
          frequency: 1,
        }).onConflictDoNothing();
      }
    }

    return success(res, { ...dreamEntry, themes: analysis.themes, wakingLifeConnections: analysis.wakingLifeConnections }, 201);
  } catch (err) {
    return error(res, 'Failed to analyze dream');
  }
}
