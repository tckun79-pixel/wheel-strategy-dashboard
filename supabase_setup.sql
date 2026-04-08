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
    "OpenDate" TEXT NOT NULL
);

-- 2. History table (completed/closed trades)
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,
    "Ticker" TEXT NOT NULL,
    "Type" TEXT TEXT,
    "Strike" NUMERIC,
    "Premium" NUMERIC,
    "Contracts" INTEGER,
    "Expiry" TEXT,
    "OpenDate" TEXT,
    "CloseDate" TEXT,
    "Result" TEXT,                  -- 'Expired', 'Assigned', 'Rolled', 'Closed Early'
    "Profit" NUMERIC
);

-- 3. Holdings table (stock inventory from assignments)
CREATE TABLE IF NOT EXISTS holdings (
    id TEXT PRIMARY KEY,
    "Ticker" TEXT NOT NULL,
    "Shares" INTEGER NOT NULL,
    "CostPrice" NUMERIC NOT NULL,
    "Date" TEXT NOT NULL
);

-- 4. Row Level Security (RLS) Policies
-- Enable RLS on all tables
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE history ENABLE ROW LEVEL SECURITY;
ALTER TABLE holdings ENABLE ROW LEVEL SECURITY;

-- Allow all operations with anon key (since the app uses the anon/service key for CRUD)
-- If you want stricter security, replace these with authenticated-only policies
CREATE POLICY "Enable all access on positions" ON positions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Enable all access on history" ON history FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Enable all access on holdings" ON holdings FOR ALL USING (true) WITH CHECK (true);
