import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getDb } from '../_lib/db';
import { getOpenAIClient } from '../_lib/openai';
import { success, error, badRequest, methodNotAllowed } from '../_lib/response';
import * as schema from '../../shared/schema';
import { eq } from 'drizzle-orm';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return methodNotAllowed(res);

  try {
    const { dreamId } = req.body;
    if (!dreamId) return badRequest(res, 'dreamId is required');

    const db = getDb();
    const [dream] = await db
      .select()
      .from(schema.dreamAnalysis)
      .where(eq(schema.dreamAnalysis.id, dreamId));

    if (!dream) return error(res, 'Dream not found', 404);

    const openai = getOpenAIClient();

    // Generate image from dream description
    const imageResponse = await openai.images.generate({
      model: "dall-e-3",
      prompt: `Surreal, dreamlike digital art of: ${dream.dreamText.substring(0, 500)}. Style: ethereal, mystical, glowing colors, floating elements, cosmic atmosphere. No text.`,
      n: 1,
      size: "1024x1024",
      quality: "standard",
    });

    const imageUrl = imageResponse.data?.[0]?.url;

    if (imageUrl) {
      await db
        .update(schema.dreamAnalysis)
        .set({ imageUrl })
        .where(eq(schema.dreamAnalysis.id, dreamId));
    }

    return success(res, { imageUrl });
  } catch (err) {
    return error(res, 'Failed to generate dream image');
  }
}
