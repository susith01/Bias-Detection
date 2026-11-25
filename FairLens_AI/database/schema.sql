-- FairLens AI - Supabase PostgreSQL Schema
-- Run this script in the Supabase SQL Editor.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 1. Datasets metadata
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_name TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('synthetic', 'historical', 'uploaded')),
    domain TEXT NOT NULL,
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    column_count INTEGER NOT NULL CHECK (column_count >= 0),
    target_column TEXT NOT NULL,
    sensitive_columns JSONB NOT NULL DEFAULT '[]'::jsonb,
    missing_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    duplicate_count INTEGER NOT NULL DEFAULT 0,
    storage_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Analyses sessions
CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
    analysis_name TEXT NOT NULL,
    protected_attributes JSONB NOT NULL DEFAULT '[]'::jsonb,
    bias_risk_score NUMERIC(5,2),
    risk_level TEXT CHECK (risk_level IN ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')),
    status TEXT NOT NULL DEFAULT 'completed',
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Trained ML models
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES analyses(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES datasets(id) ON DELETE SET NULL,
    model_name TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    accuracy NUMERIC(6,4),
    precision_score NUMERIC(6,4),
    recall_score NUMERIC(6,4),
    f1_score NUMERIC(6,4),
    roc_auc NUMERIC(6,4),
    train_size INTEGER,
    test_size INTEGER,
    storage_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Individual predictions
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    analysis_id UUID REFERENCES analyses(id) ON DELETE SET NULL,
    data_origin TEXT NOT NULL CHECK (data_origin IN ('simulated', 'uploaded_batch', 'evaluation')),
    input_features JSONB NOT NULL,
    protected_attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    predicted_label INTEGER NOT NULL,
    predicted_probability NUMERIC(6,4) NOT NULL,
    actual_label INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Incoming synthetic/monitoring records
CREATE TABLE IF NOT EXISTS incoming_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID,
    domain TEXT NOT NULL,
    data_origin TEXT NOT NULL DEFAULT 'simulated',
    record_payload JSONB NOT NULL,
    predicted_label INTEGER,
    predicted_probability NUMERIC(6,4),
    simulated_ground_truth INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. Fairness evaluation results
CREATE TABLE IF NOT EXISTS fairness_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES analyses(id) ON DELETE CASCADE,
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    sensitive_attribute TEXT NOT NULL,
    demographic_parity_diff NUMERIC(6,4),
    demographic_parity_ratio NUMERIC(6,4),
    equal_opportunity_diff NUMERIC(6,4),
    equalized_odds_diff NUMERIC(6,4),
    overall_status TEXT CHECK (overall_status IN ('PASS', 'WARN', 'FAIL')),
    group_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence_intervals JSONB NOT NULL DEFAULT '{}'::jsonb,
    small_group_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 7. Bias mitigation comparisons
CREATE TABLE IF NOT EXISTS mitigation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES analyses(id) ON DELETE CASCADE,
    baseline_model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    mitigation_strategy TEXT NOT NULL,
    sensitive_attribute TEXT NOT NULL,
    baseline_accuracy NUMERIC(6,4),
    mitigated_accuracy NUMERIC(6,4),
    baseline_dp_diff NUMERIC(6,4),
    mitigated_dp_diff NUMERIC(6,4),
    baseline_eodds_diff NUMERIC(6,4),
    mitigated_eodds_diff NUMERIC(6,4),
    tradeoff_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 8. Drift monitoring batches
CREATE TABLE IF NOT EXISTS monitoring_batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES analyses(id) ON DELETE CASCADE,
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    batch_label TEXT NOT NULL,
    data_origin TEXT NOT NULL DEFAULT 'simulated',
    sample_size INTEGER NOT NULL,
    dp_diff NUMERIC(6,4),
    selection_rate NUMERIC(6,4),
    feature_drift_psi NUMERIC(6,4),
    alert_triggered BOOLEAN NOT NULL DEFAULT FALSE,
    alert_message TEXT,
    metrics_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 9. Generated PDF reports metadata
CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES analyses(id) ON DELETE CASCADE,
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    report_title TEXT NOT NULL,
    file_name TEXT NOT NULL,
    storage_path TEXT,
    gemini_summary_included BOOLEAN NOT NULL DEFAULT FALSE,
    bias_risk_score NUMERIC(5,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Helpful indexes for history queries
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fairness_analysis ON fairness_results (analysis_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_monitoring_created_at ON monitoring_batches (created_at DESC);
