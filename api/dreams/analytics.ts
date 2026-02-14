import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getDb } from '../_lib/db';
import { success, error, methodNotAllowed } from '../_lib/response';
import * as schema from '../../shared/schema';
import { eq, desc } from 'drizzle-orm';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') return methodNotAllowed(res);

  try {
    const userId = req.query.userId as string;
    if (!userId) return error(res, 'userId required', 400);

    const db = getDb();

    const dreams = await db
      .select()
      .from(schema.dreamAnalysis)
      .where(eq(schema.dreamAnalysis.userId, userId))
      .orderBy(desc(schema.dreamAnalysis.timestamp));

    const symbols = await db
      .select()
      .from(schema.dreamSymbols)
      .where(eq(schema.dreamSymbols.userId, userId));

    // Compute analytics
    const totalDreams = dreams.length;
    const avgSleepQuality = dreams.reduce((sum, d) => sum + (d.sleepQuality || 0), 0) / Math.max(totalDreams, 1);
    const avgLucidity = dreams.reduce((sum, d) => sum + (d.lucidityScore || 0), 0) / Math.max(totalDreams, 1);

    const tagCounts: Record<string, number> = {};
    dreams.forEach(d => {
      const tags = d.tags as string[] | null;
      if (tags) tags.forEach(t => { tagCounts[t] = (tagCounts[t] || 0) + 1; });
    });

    const emotionCounts: Record<string, number> = {};
    dreams.forEach(d => {
      const emotions = d.emotions as Array<{ emotion: string }> | null;
      if (emotions) emotions.forEach(e => { emotionCounts[e.emotion] = (emotionCounts[e.emotion] || 0) + 1; });
    });

    return success(res, {
      totalDreams,
      avgSleepQuality: Math.round(avgSleepQuality * 10) / 10,
      avgLucidity: Math.round(avgLucidity * 10) / 10,
      tagDistribution: tagCounts,
      emotionDistribution: emotionCounts,
      topSymbols: symbols.sort((a, b) => (b.frequency || 0) - (a.frequency || 0)).slice(0, 10),
      dreamsPerWeek: Math.round(totalDreams / Math.max(1, Math.ceil((Date.now() - (dreams[dreams.length - 1]?.timestamp?.getTime?.() || Date.now())) / (7 * 24 * 60 * 60 * 1000))) * 10) / 10,
    });
  } catch (err) {
    return error(res, 'Failed to compute dream analytics');
  }
}
