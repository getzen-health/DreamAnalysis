-- Add role column to users table for database-backed RBAC
ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(10) NOT NULL DEFAULT 'user';

-- Grant admin role to the primary admin user
UPDATE users SET role = 'admin' WHERE username = 'sravya';

-- Ensure RLS: users can only read their own role; role updates are service_role only
-- (users table RLS already enabled from 007_drizzle_tables_rls.sql)
