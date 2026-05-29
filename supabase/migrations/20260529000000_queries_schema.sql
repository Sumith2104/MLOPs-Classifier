-- supabase/migrations/20260529000000_queries_schema.sql

-- 4. MLOps Raw Queries Table
CREATE TABLE IF NOT EXISTS mlops_queries (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    query TEXT NOT NULL,
    processed BOOLEAN DEFAULT false NOT NULL
);

-- 5. MLOps Classified Queries Table
CREATE TABLE IF NOT EXISTS mlops_classified_queries (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    query_id UUID REFERENCES mlops_queries(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    intent VARCHAR(50),
    priority VARCHAR(50),
    intent_confidence FLOAT,
    priority_confidence FLOAT,
    flagged BOOLEAN DEFAULT false,
    error TEXT
);

-- Disable Row Level Security for dev publishable key access
ALTER TABLE mlops_queries DISABLE ROW LEVEL SECURITY;
ALTER TABLE mlops_classified_queries DISABLE ROW LEVEL SECURITY;
