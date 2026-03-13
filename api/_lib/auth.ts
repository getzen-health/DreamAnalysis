import type { VercelRequest, VercelResponse } from '@vercel/node';
import jwt from 'jsonwebtoken';
import { unauthorized } from './response.js';

export interface JWTPayload {
  userId: string;
  username: string;
}

function getJWTSecret(): string {
  const secret = process.env.JWT_SECRET;
  if (!secret) {
    throw new Error('JWT_SECRET environment variable is not set. This is required for secure authentication.');
  }
  return secret;
}

const JWT_EXPIRES_IN = '7d';

export function generateToken(payload: JWTPayload): string {
  return jwt.sign(payload, getJWTSecret(), { expiresIn: JWT_EXPIRES_IN });
}

export function verifyToken(token: string): JWTPayload | null {
  try {
    return jwt.verify(token, getJWTSecret()) as JWTPayload;
  } catch {
    return null;
  }
}

export function getAuthToken(req: VercelRequest): string | null {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    return authHeader.substring(7);
  }
  
  const cookieHeader = req.headers.cookie;
  if (cookieHeader) {
    const cookies = cookieHeader.split(';').reduce((acc, cookie) => {
      const [key, value] = cookie.trim().split('=');
      acc[key] = value;
      return acc;
    }, {} as Record<string, string>);
    
    return cookies['auth_token'] || null;
  }
  
  return null;
}

export function requireAuth(req: VercelRequest, res: VercelResponse): JWTPayload | null {
  const token = getAuthToken(req);
  
  if (!token) {
    unauthorized(res, 'No authentication token provided');
    return null;
  }
  
  const payload = verifyToken(token);
  if (!payload) {
    unauthorized(res, 'Invalid or expired token');
    return null;
  }
  
  return payload;
}

export function setAuthCookie(res: VercelResponse, token: string) {
  res.setHeader('Set-Cookie', `auth_token=${token}; HttpOnly; Secure; SameSite=Strict; Max-Age=${7 * 24 * 60 * 60}; Path=/`);
}

export function clearAuthCookie(res: VercelResponse) {
  res.setHeader('Set-Cookie', 'auth_token=; HttpOnly; Secure; SameSite=Strict; Max-Age=0; Path=/');
}

/**
 * Require auth AND verify the authenticated user matches the requested userId.
 * Returns the JWT payload if authorized, null if not (response already sent).
 */
export function requireOwner(req: VercelRequest, res: VercelResponse, requestedUserId: string): JWTPayload | null {
  const auth = requireAuth(req, res);
  if (!auth) return null; // 401 already sent
  if (auth.userId !== requestedUserId) {
    res.status(403).json({ error: 'Forbidden — you can only access your own data' });
    return null;
  }
  return auth;
}

/** List of usernames that have admin access to study endpoints. */
const ADMIN_USERNAMES = new Set(['sravya', 'admin']);

/**
 * Require auth AND verify the user has admin role.
 * Returns the JWT payload if authorized, null if not (response already sent).
 */
export function requireAdmin(req: VercelRequest, res: VercelResponse): JWTPayload | null {
  const auth = requireAuth(req, res);
  if (!auth) return null; // 401 already sent
  if (!ADMIN_USERNAMES.has(auth.username?.toLowerCase())) {
    res.status(403).json({ error: 'Forbidden — admin access required' });
    return null;
  }
  return auth;
}
