import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";
export default function handler(req: VercelRequest, res: VercelResponse) {
  res.status(200).json({ ok: true });
}
