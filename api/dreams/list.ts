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

    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 20;
    const offset = (page - 1) * limit;

    const db = getDb();
    const dreams = await db
      .select()
      .from(schema.dreamAnalysis)
      .where(eq(schema.dreamAnalysis.userId, userId))
      .orderBy(desc(schema.dreamAnalysis.timestamp))
      .limit(limit)
      .offset(offset);

    return success(res, { dreams, page, limit });
  } catch (err) {
    return error(res, 'Failed to fetch dreams');
  }
}
