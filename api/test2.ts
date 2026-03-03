import type { VercelRequest, VercelResponse } from '@vercel/node';
import { neon } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-http';
import * as schema from '../shared/schema';
export default function handler(req: VercelRequest, res: VercelResponse) {
  res.status(200).json({ ok: true, tables: Object.keys(schema).length });
}
