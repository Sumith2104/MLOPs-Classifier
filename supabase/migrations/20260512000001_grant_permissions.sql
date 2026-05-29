-- Fix permissions for the anon role
GRANT USAGE ON SCHEMA public TO anon;
GRANT ALL ON TABLE predictions TO anon;
GRANT ALL ON TABLE feedback TO anon;
GRANT ALL ON TABLE flagged_predictions TO anon;