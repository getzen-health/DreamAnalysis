import type { VercelRequest, VercelResponse } from '@vercel/node';
import { eq, desc, asc, and, gte, lt, sql } from 'drizzle-orm';
import { getDb } from './_lib/db';
import * as schema from '../shared/schema';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    res.status(200).json({ ok: true, tables: Object.keys(schema).filter(k => !k.startsWith('insert') && !k.startsWith('type')).length });
  } catch (e: any) {
    res.status(500).json({ error: String(e) });
  }
}
