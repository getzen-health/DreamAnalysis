import type { VercelRequest, VercelResponse } from '@vercel/node';
import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, serial, jsonb, timestamp, real, boolean, index, uniqueIndex } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
});
const insertUserSchema = createInsertSchema(users);

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.status(200).json({ ok: true, type: typeof insertUserSchema });
}
