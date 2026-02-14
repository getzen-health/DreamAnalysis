import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getDb } from '../_lib/db';
import { success, error, badRequest, methodNotAllowed } from '../_lib/response';
import * as schema from '../../shared/schema';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res);

  try {
    const { userId, endpoint, keys } = req.body;

    if (!userId || !endpoint || !keys) {
      return badRequest(res, 'userId, endpoint, and keys are required');
    }

    const db = getDb();
    const [subscription] = await db.insert(schema.pushSubscriptions).values({
      userId,
      endpoint,
      keys,
    }).returning();

    return success(res, subscription, 201);
  } catch (err) {
    return error(res, 'Failed to save push subscription');
  }
}
