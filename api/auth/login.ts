import type { VercelRequest, VercelResponse } from '@vercel/node';
import { scrypt, timingSafeEqual } from 'crypto';
import { promisify } from 'util';
import { eq } from 'drizzle-orm';
import { getDb } from '../_lib/db';
import { success, error, badRequest, methodNotAllowed, unauthorized } from '../_lib/response';
import { generateToken, setAuthCookie } from '../_lib/auth';
import * as schema from '../../shared/schema';

const scryptAsync = promisify(scrypt);

async function verifyPassword(storedHash: string, suppliedPassword: string): Promise<boolean> {
  const [salt, hash] = storedHash.split(':');
  if (!salt || !hash) {
    return false;
  }
  const derivedKey = (await scryptAsync(suppliedPassword, salt, 64)) as Buffer;
  const storedBuffer = Buffer.from(hash, 'hex');
  return timingSafeEqual(derivedKey, storedBuffer);
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return methodNotAllowed(res, ['POST']);
  }

  try {
    const { username, password } = req.body;

    // Validate inputs
    if (!username || typeof username !== 'string') {
      return badRequest(res, 'Username is required');
    }

    if (!password || typeof password !== 'string') {
      return badRequest(res, 'Password is required');
    }

    const db = getDb();

    // Find user by username
    const [user] = await db
      .select()
      .from(schema.users)
      .where(eq(schema.users.username, username.trim()));

    if (!user) {
      return unauthorized(res, 'Invalid username or password');
    }

    // Verify password hash using timing-safe comparison
    const isValid = await verifyPassword(user.password, password);

    if (!isValid) {
      return unauthorized(res, 'Invalid username or password');
    }

    // Generate JWT token
    const token = generateToken({
      userId: user.id,
      username: user.username,
    });

    // Set auth cookie
    setAuthCookie(res, token);

    // Return user without password
    const { password: _, ...userWithoutPassword } = user;
    return success(res, { user: userWithoutPassword, token });
  } catch (err: any) {
    console.error('Login error:', err);
    return error(res, 'Failed to log in');
  }
}
