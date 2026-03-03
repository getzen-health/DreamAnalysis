import type { VercelRequest, VercelResponse } from '@vercel/node';
import { drizzle } from 'drizzle-orm/neon-http';

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.status(200).json({ ok: true, drizzle: typeof drizzle });
}
