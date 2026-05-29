-- supabase/migrations/20260529000000_queries_schema.sql

-- 1. Drop existing tables if they exist to clean up schema
DROP TABLE IF EXISTS flagged_predictions CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS feedback CASCADE;
DROP TABLE IF EXISTS queries CASCADE;
DROP TABLE IF EXISTS client_queries CASCADE;
DROP TABLE IF EXISTS classified_queries CASCADE;

-- 2. Create Ingestion Queue Table: client_queries
CREATE TABLE client_queries (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    message TEXT NOT NULL,
    processed BOOLEAN DEFAULT false NOT NULL
);

-- 3. Create Classification Results Table: classified_queries
CREATE TABLE classified_queries (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    query_id UUID REFERENCES client_queries(id) ON DELETE CASCADE,
    message TEXT NOT NULL,
    intent VARCHAR(50),
    priority VARCHAR(50),
    intent_confidence FLOAT,
    priority_confidence FLOAT,
    flagged BOOLEAN DEFAULT false,
    error TEXT
);

-- 4. Create Feedback Table: feedback
CREATE TABLE feedback (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    query TEXT NOT NULL,
    expected_intent VARCHAR(50),
    expected_priority VARCHAR(50),
    source VARCHAR(50) DEFAULT 'user_feedback' NOT NULL
);

-- 5. Disable Row Level Security on all tables to allow publishable key access
ALTER TABLE client_queries DISABLE ROW LEVEL SECURITY;
ALTER TABLE classified_queries DISABLE ROW LEVEL SECURITY;
ALTER TABLE feedback DISABLE ROW LEVEL SECURITY;

-- 6. Grant permissions to public/anon role
GRANT USAGE ON SCHEMA public TO anon;
GRANT ALL ON TABLE client_queries TO anon;
GRANT ALL ON TABLE classified_queries TO anon;
GRANT ALL ON TABLE feedback TO anon;
