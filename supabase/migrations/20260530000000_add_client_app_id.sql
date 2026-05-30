-- supabase/migrations/20260530000000_add_client_app_id.sql

-- Add client_app_id column to client_queries table
ALTER TABLE client_queries ADD COLUMN IF NOT EXISTS client_app_id VARCHAR(100) DEFAULT 'webapp' NOT NULL;

-- Add client_app_id column to classified_queries table
ALTER TABLE classified_queries ADD COLUMN IF NOT EXISTS client_app_id VARCHAR(100) DEFAULT 'webapp' NOT NULL;
