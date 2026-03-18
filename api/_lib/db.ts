import postgres from 'postgres';
import { drizzle } from 'drizzle-orm/postgres-js';

let cachedDb: ReturnType<typeof drizzle> | null = null;

export function getDb() {
  if (cachedDb) {
    return cachedDb;
  }

  const databaseUrl = process.env.DATABASE_URL;
  if (!databaseUrl) {
    throw new Error('DATABASE_URL environment variable is not set');
  }

  const client = postgres(databaseUrl, { prepare: false });
  cachedDb = drizzle(client);

  return cachedDb;
}
