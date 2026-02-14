import type { VercelRequest, VercelResponse } from '@vercel/node';
import { success, methodNotAllowed } from '../_lib/response';
import { clearAuthCookie } from '../_lib/auth';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return methodNotAllowed(res, ['POST']);
  }

  // Clear the auth cookie
  clearAuthCookie(res);

  return success(res, { message: 'Logged out successfully' });
}
