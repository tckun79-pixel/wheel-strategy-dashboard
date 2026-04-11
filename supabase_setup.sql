-- ========================================
-- Supabase Setup for Wheel Strategy Dashboard
-- Run these in Supabase SQL Editor
-- ========================================

-- 1. Positions table (active option trades)
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    "Ticker" TEXT NOT NULL,
    "Type" TEXT NOT NULL,           -- 'Put' or 'Call'
    "Strike" NUMERIC NOT NULL,
    "Premium" NUMERIC NOT NULL,
    "Contracts" INTEGER NOT NULL,
    "Expiry" TEXT NOT NULL,
    "OpenDate" TEXT NOT NULL,
    "owner" TEXT NOT NULL DEFAULT 'admin'
);

-- 2. History table (completed/closed trades)
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,
    "Ticker" TEXT NOT NULL,
    "Type" TEXT,
    "Strike" NUMERIC,
    "Premium" NUMERIC,
    "Contracts" INTEGER,
    "Expiry" TEXT,
    "OpenDate" TEXT,
    "CloseDate" TEXT,
    "Result" TEXT,                  -- 'Expired', 'Assigned', 'Rolled', 'Closed Early'
    "Profit" NUMERIC,
    "owner" TEXT NOT NULL DEFAULT 'admin',
    "CostPrice" NUMERIC,
    "Shares" INTEGER
);

-- 3. Holdings table (stock inventory from assignments)
CREATE TABLE IF NOT EXISTS holdings (
    id TEXT PRIMARY KEY,
    "Ticker" TEXT NOT NULL,
    "Shares" INTEGER NOT NULL,
    "CostPrice" NUMERIC NOT NULL,
    "Date" TEXT NOT NULL,
    "owner" TEXT NOT NULL DEFAULT 'admin'
);

-- 3b. Screener Presets table (saved screener configurations)
CREATE TABLE IF NOT EXISTS screener_presets (
    id TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "criteria_json" JSONB NOT NULL,
    "created_at" TIMESTAMPTZ DEFAULT NOW(),
    "owner" TEXT NOT NULL DEFAULT 'admin'
);

CREATE INDEX IF NOT EXISTS idx_screener_presets_owner ON screener_presets(owner);

-- 4. Row Level Security (RLS) Policies
-- Enable RLS on all tables
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE history ENABLE ROW LEVEL SECURITY;
ALTER TABLE holdings ENABLE ROW LEVEL SECURITY;
ALTER TABLE screener_presets ENABLE ROW LEVEL SECURITY;

-- RLS Policies: Users can only access their own records based on owner field
-- For single-user mode, owner='admin'. When adding multi-user auth, set owner appropriately.
CREATE POLICY "Users can access own positions" ON positions FOR ALL USING (owner = current_setting('app.current_user', true)::text);
CREATE POLICY "Users can access own history" ON history FOR ALL USING (owner = current_setting('app.current_user', true)::text);
CREATE POLICY "Users can access own holdings" ON holdings FOR ALL USING (owner = current_setting('app.current_user', true)::text);
CREATE POLICY "Users can access own presets" ON screener_presets FOR ALL USING (owner = current_setting('app.current_user', true)::text);

-- For simplicity in single-user mode without proper auth, the app bypasses RLS by not setting app.current_user.
-- In that case, all records with owner='admin' are accessible to anyone with DB access.
-- When implementing proper auth, set app.current_user per session.
