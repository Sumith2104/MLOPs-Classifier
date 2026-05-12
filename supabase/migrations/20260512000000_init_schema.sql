-- supabase_schema.sql
-- Run this in your Supabase SQL Editor

-- 1. Predictions Table (Replaces app.log for analytics)
CREATE TABLE IF NOT EXISTS predictions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    query TEXT NOT NULL,
    intent VARCHAR(50),
    priority VARCHAR(50),
    intent_confidence FLOAT,
    priority_confidence FLOAT
);

-- 2. Feedback Table (Replaces data/raw/feedback.csv)
CREATE TABLE IF NOT EXISTS feedback (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    query TEXT NOT NULL,
    expected_intent VARCHAR(50),
    expected_priority VARCHAR(50),
    source VARCHAR(50) DEFAULT 'user_feedback'
);

-- 3. Flagged Predictions Table (Replaces logs/flagged_predictions.jsonl)
CREATE TABLE IF NOT EXISTS flagged_predictions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    query TEXT NOT NULL,
    intent VARCHAR(50),
    priority VARCHAR(50),
    intent_confidence FLOAT,
    priority_confidence FLOAT,
    threshold FLOAT
);
