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

  const sql = neon(databaseUrl);
  // No schema passed — handlers use .from(schema.table) style, not db.query.* style
  cachedDb = drizzle(sql);

  return cachedDb;
}
