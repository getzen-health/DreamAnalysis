import type { VercelRequest, VercelResponse } from '@vercel/node';
import { scrypt, randomBytes } from 'crypto';
import { promisify } from 'util';
import { eq } from 'drizzle-orm';
import { getDb } from '../_lib/db';
import { success, error, badRequest, methodNotAllowed } from '../_lib/response';
import { generateToken, setAuthCookie } from '../_lib/auth';
import * as schema from '../../shared/schema';

const scryptAsync = promisify(scrypt);

async function hashPassword(password: string): Promise<string> {
  const salt = randomBytes(16).toString('hex');
  const derivedKey = (await scryptAsync(password, salt, 64)) as Buffer;
  return `${salt}:${derivedKey.toString('hex')}`;
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return methodNotAllowed(res, ['POST']);
  }

  try {
    const { username, password, email } = req.body;

    // Validate username
    if (!username || typeof username !== 'string' || username.trim().length < 3) {
      return badRequest(res, 'Username must be at least 3 characters long');
    }

    // Validate password
    if (!password || typeof password !== 'string' || password.length < 6) {
      return badRequest(res, 'Password must be at least 6 characters long');
    }

    const db = getDb();

    // Check if username already exists
    const [existingUser] = await db
      .select()
      .from(schema.users)
      .where(eq(schema.users.username, username.trim()));

    if (existingUser) {
      return badRequest(res, 'Username already exists');
    }

    // Hash password
    const hashedPassword = await hashPassword(password);

    // Insert user into database
    const [newUser] = await db
      .insert(schema.users)
      .values({
        username: username.trim(),
        password: hashedPassword,
        email: email || null,
      })
      .returning();

    // Generate JWT token
    const token = generateToken({
      userId: newUser.id,
      username: newUser.username,
    });

    // Set auth cookie
    setAuthCookie(res, token);

    // Return user without password
    const { password: _, ...userWithoutPassword } = newUser;
    return success(res, { user: userWithoutPassword, token }, 201);
  } catch (err: any) {
    console.error('Registration error:', err);
    return error(res, 'Failed to register user');
  }
}
