import type { VercelRequest, VercelResponse } from '@vercel/node';
import { eq } from 'drizzle-orm';
import { getDb } from '../_lib/db';
import { success, error, methodNotAllowed } from '../_lib/response';
import { requireAuth } from '../_lib/auth';
import * as schema from '../../shared/schema';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return methodNotAllowed(res, ['GET']);
  }

  // Verify authentication - returns null and sends 401 if invalid
  const authPayload = requireAuth(req, res);
  if (!authPayload) {
    return;
  }

  try {
    const db = getDb();

    // Fetch user from database
    const [user] = await db
      .select()
      .from(schema.users)
      .where(eq(schema.users.id, authPayload.userId));

    if (!user) {
      return error(res, 'User not found', 404);
    }

    // Return user without password
    const { password: _, ...userWithoutPassword } = user;
    return success(res, userWithoutPassword);
  } catch (err: any) {
    console.error('Error fetching current user:', err);
    return error(res, 'Failed to fetch user data');
  }
}
