-- FairLens AI - Row Level Security & Storage Policies
-- Apply after running schema.sql in Supabase SQL Editor.

-- Enable RLS on all tables
ALTER TABLE datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE models ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE incoming_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE fairness_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE mitigation_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE monitoring_batches ENABLE ROW LEVEL SECURITY;
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

-- Note: For academic / single-tenant demo deployments using the Supabase anon key,
-- restrict operations to authenticated or anon academic clients as appropriate.
-- Never grant service-role credentials to client code.

DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'datasets', 'analyses', 'models', 'predictions',
        'incoming_records', 'fairness_results', 'mitigation_results',
        'monitoring_batches', 'reports'
    ]
    LOOP
        EXECUTE format('DROP POLICY IF EXISTS "allow_read_%s" ON %I;', tbl, tbl);
        EXECUTE format('CREATE POLICY "allow_read_%s" ON %I FOR SELECT TO anon, authenticated USING (true);', tbl, tbl);

        EXECUTE format('DROP POLICY IF EXISTS "allow_insert_%s" ON %I;', tbl, tbl);
        EXECUTE format('CREATE POLICY "allow_insert_%s" ON %I FOR INSERT TO anon, authenticated WITH CHECK (true);', tbl, tbl);

        EXECUTE format('DROP POLICY IF EXISTS "allow_update_%s" ON %I;', tbl, tbl);
        EXECUTE format('CREATE POLICY "allow_update_%s" ON %I FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);', tbl, tbl);
    END LOOP;
END $$;

-- Storage buckets for datasets, trained models, and PDF reports
INSERT INTO storage.buckets (id, name, public)
VALUES
    ('fairlens-datasets', 'fairlens-datasets', false),
    ('fairlens-models', 'fairlens-models', false),
    ('fairlens-reports', 'fairlens-reports', false)
ON CONFLICT (id) DO NOTHING;

-- Storage access policies
DROP POLICY IF EXISTS "fairlens_storage_select" ON storage.objects;
CREATE POLICY "fairlens_storage_select"
ON storage.objects FOR SELECT
TO anon, authenticated
USING (bucket_id IN ('fairlens-datasets', 'fairlens-models', 'fairlens-reports'));

DROP POLICY IF EXISTS "fairlens_storage_insert" ON storage.objects;
CREATE POLICY "fairlens_storage_insert"
ON storage.objects FOR INSERT
TO anon, authenticated
WITH CHECK (bucket_id IN ('fairlens-datasets', 'fairlens-models', 'fairlens-reports'));
