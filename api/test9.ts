import type { VercelRequest, VercelResponse } from '@vercel/node';
import { neon } from '@neondatabase/serverless';

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.status(200).json({ ok: true, neon: typeof neon });
}
