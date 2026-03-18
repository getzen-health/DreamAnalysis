import pg from 'pg';
import { drizzle } from 'drizzle-orm/node-postgres';

const { Pool } = pg;

let cachedDb: ReturnType<typeof drizzle> | null = null;
let cachedPool: InstanceType<typeof Pool> | null = null;

export function getDb() {
  if (cachedDb) {
    return cachedDb;
  }

  const databaseUrl = process.env.DATABASE_URL;
  if (!databaseUrl) {
    throw new Error('DATABASE_URL environment variable is not set');
  }

  cachedPool = new Pool({
    connectionString: databaseUrl,
    ssl: { rejectUnauthorized: false },
    max: 1, // serverless — one connection per invocation
  });

  cachedDb = drizzle(cachedPool);

  return cachedDb;
}
