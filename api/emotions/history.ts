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

    const limit = parseInt(req.query.limit as string) || 50;
    const db = getDb();

    const readings = await db
      .select()
      .from(schema.emotionReadings)
      .where(eq(schema.emotionReadings.userId, userId))
      .orderBy(desc(schema.emotionReadings.timestamp))
      .limit(limit);

    return success(res, readings);
  } catch (err) {
    return error(res, 'Failed to fetch emotion history');
  }
}
