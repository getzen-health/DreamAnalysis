import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getDb } from '../_lib/db';
import { success, error, badRequest, methodNotAllowed } from '../_lib/response';
import * as schema from '../../shared/schema';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res);

  try {
    const { userId, stress, happiness, focus, energy, dominantEmotion, valence, arousal, eegSnapshot } = req.body;

    if (!userId || stress === undefined || happiness === undefined || focus === undefined || energy === undefined || !dominantEmotion) {
      return badRequest(res, 'Missing required fields');
    }

    const db = getDb();
    const [reading] = await db.insert(schema.emotionReadings).values({
      userId,
      stress,
      happiness,
      focus,
      energy,
      dominantEmotion,
      valence: valence ?? null,
      arousal: arousal ?? null,
      eegSnapshot: eegSnapshot ?? null,
    }).returning();

    return success(res, reading, 201);
  } catch (err) {
    return error(res, 'Failed to record emotion reading');
  }
}
