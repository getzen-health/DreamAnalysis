import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getDb } from '../_lib/db';
import { success, error, methodNotAllowed } from '../_lib/response';
import { aiChats } from '../../shared/schema';
import { eq, asc } from 'drizzle-orm';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const { userId } = req.query;

  if (typeof userId !== 'string') {
    return error(res, 'Invalid user ID', 400);
  }

  if (req.method === 'GET') {
    try {
      const db = getDb();
      // asc order so the client renders oldest message at top, newest at bottom
      const chats = await db
        .select()
        .from(aiChats)
        .where(eq(aiChats.userId, userId))
        .orderBy(asc(aiChats.timestamp))
        .limit(50);

      return success(res, chats);
    } catch (err) {
      console.error('Error fetching chat history:', err);
      return error(res, 'Failed to fetch chat history');
    }
  }

  return methodNotAllowed(res, ['GET']);
}
