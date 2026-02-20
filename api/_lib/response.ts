import type { VercelRequest, VercelResponse } from '@vercel/node';

export function success(res: VercelResponse, data: any, status: number = 200) {
  return res.status(status).json(data);
}

export function error(res: VercelResponse, message: string, status: number = 500) {
  return res.status(status).json({ error: message });
}

export function methodNotAllowed(res: VercelResponse, allowedMethods: string[] = []) {
  res.setHeader('Allow', allowedMethods.join(', '));
  return res.status(405).json({ error: 'Method not allowed' });
}

export function unauthorized(res: VercelResponse, message: string = 'Unauthorized') {
  return res.status(401).json({ error: message });
}

export function badRequest(res: VercelResponse, message: string = 'Bad request') {
  return res.status(400).json({ error: message });
}
