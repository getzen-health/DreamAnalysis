import { neon } from '@neondatabase/serverless';
import { drizzle } from 'drizzle-orm/neon-http';

let cachedDb: ReturnType<typeof drizzle> | null = null;

export function getDb() {
  if (cachedDb) {
    return cachedDb;
  }

  const databaseUrl = process.env.DATABASE_URL;
  if (!databaseUrl) {
    throw new Error('DATABASE_URL environment variable is not set');
  }

  // @neondatabase/serverless's HTTP driver works with any PostgreSQL
  // (including Supabase) — ideal for Vercel serverless functions
  const sql = neon(databaseUrl);
  cachedDb = drizzle(sql);

  return cachedDb;
}
