import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getDb } from '../_lib/db';
import { getOpenAIClient } from '../_lib/openai';
import { success, error, methodNotAllowed } from '../_lib/response';
import * as schema from '../../shared/schema';
import { eq, desc, gte } from 'drizzle-orm';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res);

  try {
    const userId = req.query.userId as string;
    if (!userId) return error(res, 'userId required', 400);

    const db = getDb();
    const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);

    const [dreams, emotions, metrics] = await Promise.all([
      db.select().from(schema.dreamAnalysis)
        .where(eq(schema.dreamAnalysis.userId, userId))
        .orderBy(desc(schema.dreamAnalysis.timestamp))
        .limit(10),
      db.select().from(schema.emotionReadings)
        .where(eq(schema.emotionReadings.userId, userId))
        .orderBy(desc(schema.emotionReadings.timestamp))
        .limit(50),
      db.select().from(schema.healthMetrics)
        .where(eq(schema.healthMetrics.userId, userId))
        .orderBy(desc(schema.healthMetrics.timestamp))
        .limit(50),
    ]);

    const openai = getOpenAIClient();

    const dataContext = {
      dreamCount: dreams.length,
      dreamSymbols: dreams.flatMap(d => (d.symbols as string[]) || []),
      avgStress: emotions.length ? emotions.reduce((s, e) => s + e.stress, 0) / emotions.length : null,
      avgFocus: emotions.length ? emotions.reduce((s, e) => s + e.focus, 0) / emotions.length : null,
      avgSleepQuality: metrics.length ? metrics.reduce((s, m) => s + m.sleepQuality, 0) / metrics.length : null,
      dominantEmotions: emotions.slice(0, 10).map(e => e.dominantEmotion),
    };

    // the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
    const response = await openai.chat.completions.create({
      model: "gpt-5",
      messages: [
        {
          role: "system",
          content: `You are an AI neuroscience wellness advisor. Generate 4 personalized weekly insights combining dream analysis, emotion data, and brain health metrics. Respond with JSON:
{
  "insights": [
    { "title": "...", "description": "...", "type": "success|warning|info|secondary", "icon": "brain|heart|moon|lightbulb" }
  ],
  "weeklyScore": 0-100,
  "recommendation": "One key recommendation for next week"
}`
        },
        {
          role: "user",
          content: `Generate weekly insights for this data: ${JSON.stringify(dataContext)}`
        }
      ],
      response_format: { type: "json_object" }
    });

    const insights = JSON.parse(response.choices[0].message.content || "{}");
    return success(res, insights);
  } catch (err) {
    return error(res, 'Failed to generate weekly insights');
  }
}
