import type { VercelRequest, VercelResponse } from '@vercel/node';
import * as schema from '../shared/schema';
export default function handler(req: VercelRequest, res: VercelResponse) {
  res.status(200).json({ ok: true, keys: Object.keys(schema).slice(0,5) });
}
