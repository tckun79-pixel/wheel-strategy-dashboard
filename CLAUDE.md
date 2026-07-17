# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A single-user Streamlit dashboard for running the options **Wheel Strategy** (cash-secured puts → assignment → covered calls) plus bull put spreads. It tracks open positions, assigned stock, closed trades, and campaign performance, and includes a quantitative screener and a Black-Scholes calculator. Live market data comes from `yfinance`; persistence is a Supabase (Postgres) backend.

## Commands

```bash
# Install deps (no lockfile; requirements.txt only)
pip install -r requirements.txt

# Run the app locally (default port 8501)
streamlit run wheel_dashboard.py

# Run all tests
python3 -m pytest tests/ -v

# Run a single test file / class / case
python3 -m pytest tests/test_units.py -v
python3 -m pytest tests/test_units.py::TestValidateBullPutSpread -v
python3 -m pytest tests/test_units.py::TestKPIPremiumFormula::test_spreads_added -v
```

There is no linter, formatter, or CI configured. The `.devcontainer/` runs `pip install -r requirements.txt` on create and auto-launches `streamlit run wheel_dashboard.py` on attach (Codespaces).

## Architecture

Only two Python modules carry the app; everything else is config/SQL/tests.

- **`wheel_dashboard.py`** (~1700 lines) is the entire application — there is no package structure. It runs top-to-bottom as a Streamlit script on every rerun: config load → validation helpers → auth → DB connection → global data load → sidebar entry forms → five `st.tabs`. When editing, respect that a change anywhere re-executes the whole file on each interaction.
  - **Tabs**: `tab1` Active Positions (open options, assigned stock inventory, open spreads, trade actions: close/assign/rollover), `tab2` Campaign Analysis (per-ticker aggregates from closed history), `tab3` Trade History (closed trades + closed spreads), `tab4` Screener, `tab5` Black-Scholes Option Calculator.
- **`wheel_screener.py`** is the pure quant layer, imported by the dashboard. `get_real_market_data()` pulls a `yfinance` option chain, selecting CSP/CC strikes by *target delta* (falling back to %-OTM when the chain has no delta column), and computes IV/IVR vs 30-day historical volatility. `analyze_strategy_optimized()` applies the Wheel formulas (CSP ROC, full-loop blended ROC, defensive-roll and stop-loss metrics) and returns ranked results. Keep this module free of Streamlit UI except the caching decorators it already uses.

### Persistence (Supabase)

- Schema lives in `supabase_setup.sql` (run manually in the Supabase SQL editor). Five tables: `positions`, `history`, `holdings`, `screener_presets`, `spreads`. Column names are **quoted PascalCase** (e.g. `"Ticker"`, `"NetCredit"`) — match that casing exactly in inserts/queries.
- All DB access goes through the helpers `load_collection` / `add_document` / `delete_document`. Every row carries an `owner` field and these helpers filter on it. `add_document` auto-assigns a `uuid4` `id` and the owner.
- The app is **single-user**: `DEFAULT_OWNER = "admin"` and `get_current_owner()` always returns `"admin"`. RLS policies exist but are bypassed in single-user mode (no `app.current_user` is set). The `owner` plumbing is intentional scaffolding for future multi-user auth — preserve it rather than removing it.

### Auth & secrets

- Login is a single admin password compared against `st.secrets["admin_password"]`; auth state is `st.session_state.authenticated`. Guard any write/mutation UI behind `check_auth()`.
- Supabase credentials are read from `st.secrets` (`supabase_url`, then `SUPABASE_KEY` / `supabase_key`) with an `os.environ["SUPABASE_KEY"]` fallback. Secrets go in `.streamlit/secrets.toml` (gitignored) — never commit them. The Supabase project URL is hardcoded as a default in both `config.yaml` and `wheel_dashboard.py`.

### Config & risk profiles

`config.yaml` holds `screener_defaults` and three named risk profiles (`Conservative` / `Moderate` / `Aggressive`) that preset every screener knob (DTE, premium, delta ranges, IV/IVR, target deltas). The dashboard loads it at startup and falls back to an inline copy of the same defaults if the file is missing — if you change one, change both.

## Conventions & gotchas

- **Caching**: `get_db()` uses `@st.cache_resource`; price/data fetches use `@st.cache_data` with TTLs. `tests/test_app.py` is deliberately a *separate file* from `tests/test_units.py` because `@st.cache_resource` on `get_db()` persists across `AppTest` runs in one process — the missing-credentials test must run before the dummy-secrets test. Keep AppTest cases that depend on cache state isolated this way.
- **Testing strategy**: `test_units.py` re-implements pure validators (`validate_option_trade`, `validate_bull_put_spread`) and DB-helper contracts *locally* rather than importing them, to avoid triggering the app module's import-time side effects (Supabase client, secrets). If you change a validator's rules or a helper's return contract, update the replica in the test too.
- **DB helper return contract**: `add_document` / `delete_document` return `True`/`False` (False when `db` is `None` or on exception) — callers branch on this; don't change them to raise.
- **`requirements.txt` lists `mibian`, but the Option Calculator (tab5) uses a manual Black-Scholes implementation via `scipy.stats.norm`** — `mibian` is not actually used in the current code.
