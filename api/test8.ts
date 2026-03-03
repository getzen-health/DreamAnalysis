import type { VercelRequest, VercelResponse } from '@vercel/node';
import { getDb } from './_lib/db';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    res.status(200).json({ ok: true, hasDb: typeof getDb });
  } catch (e: any) {
    res.status(500).json({ error: String(e) });
  }
}
