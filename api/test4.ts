import type { VercelRequest, VercelResponse } from '@vercel/node';
import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, serial, jsonb, timestamp, real, boolean, index } from "drizzle-orm/pg-core";
export default function handler(req: VercelRequest, res: VercelResponse) {
  res.status(200).json({ ok: true });
}
