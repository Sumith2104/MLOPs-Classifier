-- supabase/migrations/20260529000000_queries_schema.sql

-- 1. Alter existing queries table to add processed flag if not present
ALTER TABLE queries ADD COLUMN IF NOT EXISTS processed BOOLEAN DEFAULT false NOT NULL;

-- 2. Create Classified Queries Table referencing queries.id
CREATE TABLE IF NOT EXISTS classified_queries (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    query_id UUID REFERENCES queries(id) ON DELETE CASCADE,
    message TEXT NOT NULL,
    intent VARCHAR(50),
    priority VARCHAR(50),
    intent_confidence FLOAT,
    priority_confidence FLOAT,
    flagged BOOLEAN DEFAULT false,
    error TEXT
);

-- 3. Disable Row Level Security to allow direct access
ALTER TABLE queries DISABLE ROW LEVEL SECURITY;
ALTER TABLE classified_queries DISABLE ROW LEVEL SECURITY;

-- 4. Fix permissions for the anon role
GRANT ALL ON TABLE queries TO anon;
GRANT ALL ON TABLE classified_queries TO anon;
