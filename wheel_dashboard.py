import streamlit as st
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, date, timedelta
from wheel_screener import analyze_strategy_optimized
import uuid
import json
import re
import time
import logging
import yaml
from typing import Optional
from supabase import create_client, Client

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
CONFIG_PATH = "config.yaml"
CONFIG = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", CONFIG_PATH)
except Exception as e:
    logger.warning("Failed to load config.yaml: %s. Using defaults.", e)
    CONFIG = {
        "screener_defaults": {
            "max_capital": 30000, "min_dte": 7, "max_dte": 45,
            "min_premium": 0.30, "min_csp_roc": 0.0, "min_blended_roc": 0.0,
            "max_cap_per_trade": 30000, "min_iv": 0.0, "min_ivr": 0.0,
            "csp_target_delta": 0.18, "cc_target_delta": 0.25
        },
        "profiles": {
            "Conservative": {
                "max_capital": 20000, "min_dte": 14, "max_dte": 30,
                "min_premium": 0.50, "max_cap_per_trade": 5000, "min_iv": 30,
                "min_csp_delta_abs": 0.05, "max_csp_delta_abs": 0.20,
                "min_cc_delta": 0.05, "max_cc_delta": 0.20, "min_ivr": 80,
                "csp_target_delta": 0.10, "cc_target_delta": 0.15
            },
            "Moderate": {
                "max_capital": 30000, "min_dte": 7, "max_dte": 45,
                "min_premium": 0.30, "max_cap_per_trade": 10000, "min_iv": 0.0,
                "min_csp_delta_abs": 0.0, "max_csp_delta_abs": 1.0,
                "min_cc_delta": 0.0, "max_cc_delta": 1.0, "min_ivr": 0.0,
                "csp_target_delta": 0.18, "cc_target_delta": 0.25
            },
            "Aggressive": {
                "max_capital": 50000, "min_dte": 5, "max_dte": 21,
                "min_premium": 0.10, "max_cap_per_trade": 20000, "min_iv": 0.0,
                "min_csp_delta_abs": 0.20, "max_csp_delta_abs": 0.50,
                "min_cc_delta": 0.20, "max_cc_delta": 0.50, "min_ivr": 0.0,
                "csp_target_delta": 0.30, "cc_target_delta": 0.35
            }
        }
    }

# --- Input Validation Functions ---
def validate_option_trade(ticker: str, strike: float, contracts: int, premium: float, expiry: date) -> tuple[bool, str]:
    """Validate option trade inputs. Returns (valid, error_message)."""
    if not ticker or not ticker.strip():
        return False, "Ticker cannot be empty"
    if strike <= 0:
        return False, "Strike must be greater than 0"
    if contracts < 1:
        return False, "Contracts must be at least 1"
    if expiry <= date.today():
        return False, "Expiry must be after today"
    if premium < 0:
        return False, "Premium cannot be negative"
    return True, ""

def validate_stock_holding(shares: int, cost_price: float, acquisition_date: date) -> tuple[bool, str]:
    """Validate stock holding inputs. Returns (valid, error_message)."""
    if shares < 1:
        return False, "Shares must be at least 1"
    if cost_price < 0:
        return False, "Cost price cannot be negative"
    if acquisition_date > date.today():
        return False, "Acquisition date cannot be in the future"
    return True, ""

def validate_screener_inputs(min_dte: int, max_dte: int, tickers: list) -> tuple[bool, str]:
    """Validate screener inputs. Returns (valid, error_message)."""
    if min_dte > max_dte:
        return False, "Min DTE cannot be greater than Max DTE"
    if not tickers:
        return False, "Ticker list cannot be empty"
    return True, ""

# --- Page Configuration ---
st.set_page_config(page_title="Wheel Strategy Pro", layout="wide", page_icon="☸️")

# --- Styling ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 24px; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    td:nth-child(6) { font-weight: bold; color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- Authentication Logic ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_auth():
    """Checks if the user is logged in."""
    if st.session_state.authenticated:
        return True
    return False

# --- Supabase Connection ---
SUPABASE_URL = "https://xqatmutydxsxvrgnuwgm.supabase.co"

@st.cache_resource
def get_db():
    try:
        # Try secrets first, fall back to env vars
        supabase_url = st.secrets.get("supabase_url", SUPABASE_URL)
        supabase_key = st.secrets.get("supabase_key", os.environ.get("SUPABASE_KEY", ""))

        if not supabase_key:
            logger.error("'supabase_key' not found in secrets or SUPABASE_KEY env var")
            st.error("Internal Error: Supabase key not configured. Please check your .streamlit/secrets.toml file.")
            return None

        db = create_client(supabase_url, supabase_key)

        # Quick connectivity check
        try:
            db.table("positions").select("id").limit(1).execute()
        except Exception as conn_err:
            err_str = str(conn_err).lower()
            if "jwt" in err_str or "unauthorized" in err_str or "apikey" in err_str:
                logger.error("Supabase auth error: %s", conn_err)
                st.error("Supabase authentication failed. Please check your API key.")
                return None
            logger.warning("Supabase connectivity check skipped: %s", conn_err)

        return db

    except Exception as e:
        logger.error("Connection Error: %s", e)
        st.error("Connection Error: Unable to connect to database. Please check your configuration.")
        return None

db = get_db()

# --- Database Helpers ---
def load_collection(collection_name):
    """Load all rows from a Supabase table, returns list of dicts."""
    if not db: return []
    result = db.table(collection_name).select("*").execute()
    if not result.data:
        return []
    return result.data

def add_document(collection_name, data):
    """Insert a document into a Supabase table."""
    if not db: return
    doc_id = data.get('id', str(uuid.uuid4()))
    data['id'] = doc_id
    db.table(collection_name).insert(data).execute()

def delete_document(collection_name, doc_id):
    """Delete a document by id."""
    if not db: return
    db.table(collection_name).delete().eq("id", doc_id).execute()

# --- Market Data (with retry & rate limiting) ---
def get_current_price_with_retry(ticker: str) -> Optional[float]:
    """Fetch current price with retry logic and fallback."""
    ticker = ticker.strip().upper()
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            tk = yf.Ticker(ticker)
            # Try fast_info first
            price = tk.fast_info.last_price
            if price is not None and price > 0:
                logger.info("Price fetched for %s: $%.2f", ticker, price)
                return float(price)
            # Fallback to history
            hist = tk.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                if price > 0:
                    logger.info("Price fetched for %s via fallback: $%.2f", ticker, price)
                    return float(price)
            raise ValueError(f"Invalid price returned: {price}")
        except Exception as e:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt + 1, max_attempts, ticker, e)
            if attempt < max_attempts - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
    logger.error("All attempts failed for %s", ticker)
    return None

@st.cache_data(ttl=60)
def get_current_price(ticker):
    return get_current_price_with_retry(ticker)

# --- Main App ---
st.title("☸️ Wheel Strategy Manager Pro")

if not db:
    st.stop()

# --- Global Data Load ---
positions_data = load_collection('positions')
history_data = load_collection('history')
holdings_data = load_collection('holdings')

# --- Sidebar: Authentication & Actions ---
st.sidebar.header("🔐 Security")

if not st.session_state.authenticated:
    with st.sidebar.expander("Admin Login", expanded=True):
        admin_pass = st.text_input("Password", type="password")
        if st.button("Login"):
            if "admin_password" in st.secrets and admin_pass == st.secrets["admin_password"]:
                st.session_state.authenticated = True
                st.rerun()
            else:
                logger.warning("Failed login attempt with password: %s", "*" * len(admin_pass))
                st.error("Incorrect Password")
else:
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    st.sidebar.success("Logged in as Admin")

st.sidebar.divider()
st.sidebar.header("➕ Portfolio Actions")

if check_auth():
    add_mode = st.sidebar.radio("Entry Type", ["Option Trade", "Stock Holding (Assignment)"], horizontal=True)

    with st.sidebar.form("add_entry_form", clear_on_submit=True):
        if add_mode == "Option Trade":
            st.subheader("Add Option")
            col1, col2 = st.columns(2)
            ticker_in = col1.text_input("Ticker").upper()
            strike_in = col1.number_input("Strike", min_value=0.0, step=0.5)
            type_in = col2.selectbox("Type", ["Put", "Call"])
            contracts_in = col2.number_input("Contracts", min_value=1, step=1)
            premium_in = st.number_input("Premium Received (Per Share)", min_value=0.0, step=0.01)
            expiry_in = st.date_input("Expiry Date")
            
            submitted = st.form_submit_button("Submit Option")
            if submitted and ticker_in:
                valid, err_msg = validate_option_trade(ticker_in, strike_in, contracts_in, premium_in, expiry_in)
                if not valid:
                    st.error(f"Validation: {err_msg}")
                else:
                    new_trade = {
                        'id': str(uuid.uuid4()),
                        'Ticker': ticker_in.strip().upper(),
                        'Type': type_in,
                        'Strike': strike_in,
                        'Premium': premium_in,
                        'Contracts': contracts_in,
                        'Expiry': str(expiry_in), 
                        'OpenDate': str(date.today())
                    }
                    add_document('positions', new_trade)
                    logger.info("Option trade added: %s %s $%s", ticker_in, type_in, strike_in)
                    st.toast("Option Saved! ☁️")
                    st.rerun()

        else:
            st.subheader("Add Stock Inventory")
            st.caption("Use this when you get assigned shares.")
            col1, col2 = st.columns(2)
            s_ticker = col1.text_input("Ticker Symbol").upper()
            s_shares = col2.number_input("Shares Owned", min_value=1, step=100)
            s_cost = st.number_input("Cost Basis Per Share", min_value=0.0, step=0.01)
            s_date = st.date_input("Acquisition Date")
            
            submitted = st.form_submit_button("Add Stock")
            if submitted and s_ticker:
                valid, err_msg = validate_stock_holding(s_shares, s_cost, s_date)
                if not valid:
                    st.error(f"Validation: {err_msg}")
                else:
                    new_holding = {
                        'id': str(uuid.uuid4()),
                        'Ticker': s_ticker.strip().upper(),
                        'Shares': s_shares,
                        'CostPrice': s_cost,
                        'Date': str(s_date)
                    }
                    add_document('holdings', new_holding)
                    logger.info("Stock holding added: %s %d shares @ $%s", s_ticker, s_shares, s_cost)
                    st.toast("Stock Inventory Added! ☁️")
                    st.rerun()
else:
    st.sidebar.info("Login to add new trades or stocks.")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Active Positions", "📉 Campaign Analysis", "📜 Trade History", "🔍 Screener"])

# ==========================
# TAB 1: ACTIVE POSITIONS
# ==========================
with tab1:
    
    total_stock_unrealized = 0
    total_option_premium = 0
    option_risk_count = 0
    
    h_df = pd.DataFrame()
    if holdings_data:
        h_df = pd.DataFrame(holdings_data)
        with st.spinner('Syncing market data...'):
            unique_tickers = list(set(h_df['Ticker'].unique().tolist() + [p['Ticker'] for p in positions_data]))
            prices = {}
            for t in unique_tickers:
                prices[t] = get_current_price(t)
                time.sleep(0.5)  # Rate limiting: 500ms between ticker fetches
            
        def calc_holding(row):
            curr = prices.get(row['Ticker'], 0) or 0
            mkt_val = curr * row['Shares']
            cost_val = row['CostPrice'] * row['Shares']
            unreal_pl = mkt_val - cost_val
            pl_pct = ((curr - row['CostPrice']) / row['CostPrice']) * 100 if row['CostPrice'] > 0 else 0
            return pd.Series([curr, mkt_val, unreal_pl, pl_pct])

        if not h_df.empty:
            h_stats = h_df.apply(calc_holding, axis=1)
            h_stats.columns = ['Current Price', 'Market Value', 'Unrealized P&L', 'Return %']
            h_df = pd.concat([h_df, h_stats], axis=1)
            total_stock_unrealized = h_df['Unrealized P&L'].sum()
    else:
        with st.spinner('Syncing market data...'):
            unique_tickers = list(set([p['Ticker'] for p in positions_data]))
            prices = {}
            for t in unique_tickers:
                prices[t] = get_current_price(t)
                time.sleep(0.5)  # Rate limiting: 500ms between ticker fetches

    df = pd.DataFrame()
    if positions_data:
        df = pd.DataFrame(positions_data)
        total_option_premium = (df['Premium'] * df['Contracts'] * 100).sum()
        
        for _, row in df.iterrows():
            curr = prices.get(row['Ticker'], 0) or 0
            is_itm = (row['Type'] == 'Put' and curr < row['Strike']) or \
                     (row['Type'] == 'Call' and curr > row['Strike'])
            if is_itm: option_risk_count += 1

    m1, m2, m3 = st.columns(3)
    
    m1.metric(
        "Inventory Unrealized P&L", 
        f"${total_stock_unrealized:,.2f}", 
        delta_color="normal"
    )
    
    m2.metric(
        "Unrealized Premium Held", 
        f"${total_option_premium:,.0f}",
        help="This is the max profit if all options expire worthless."
    )
    
    m3.metric(
        "Options At Risk (ITM)", 
        f"{option_risk_count}", 
        delta_color="inverse", 
        delta="Alert" if option_risk_count > 0 else "Safe"
    )
    
    st.divider()

    if not h_df.empty:
        st.subheader("📦 Stock Inventory (Assigned)")
        
        def color_pnl(val):
            try:
                # Safely convert to float, return empty string for non-numeric
                num = float(val)
                return 'color: green' if num > 0 else 'color: red' if num < 0 else ''
            except (ValueError, TypeError):
                return ''

        st.dataframe(
            h_df[['Ticker', 'Shares', 'CostPrice', 'Current Price', 'Market Value', 'Unrealized P&L', 'Return %']]
            .style.format({'CostPrice': '${:.2f}', 'Current Price': '${:.2f}', 'Market Value': '${:,.0f}', 'Unrealized P&L': '${:,.0f}', 'Return %': '{:.2f}%'})
            .map(color_pnl, subset=['Unrealized P&L']),
            use_container_width=True,
            height=150
        )
        st.download_button(
            "📥 Export Stock Inventory CSV",
            h_df[['Ticker', 'Shares', 'CostPrice', 'Current Price', 'Market Value', 'Unrealized P&L', 'Return %']].to_csv(index=False),
            "stock_inventory.csv",
            "text/csv",
            key="dl_stock_inventory"
        )
        
        # --- Stock Actions ---
        if check_auth():
            st.markdown("### 🛠️ Stock Actions")
            stock_list = {}
            for h in holdings_data:
                label = f"{h['Ticker']} — {h['Shares']} shares @ ${h['CostPrice']} (Cost)"
                stock_list[label] = h
            
            if stock_list:
                selected_stock_label = st.selectbox("Select Stock to Manage", list(stock_list.keys()))
                sel_stock = stock_list[selected_stock_label]
                
                c1, c2 = st.columns(2)
                
                # Close (Sell) action
                if c1.button("Sell All Shares (Close)"):
                    current_price = get_current_price(sel_stock['Ticker']) or 0
                    cost_total = float(sel_stock['CostPrice']) * int(sel_stock['Shares'])
                    proceeds = current_price * int(sel_stock['Shares'])
                    realized_pl = proceeds - cost_total
                    
                    hist_entry = {
                        'id': str(uuid.uuid4()),
                        'Ticker': sel_stock['Ticker'],
                        'Type': 'Stock',
                        'Strike': 0,
                        'Premium': current_price,
                        'Contracts': sel_stock['Shares'],
                        'Expiry': '',
                        'OpenDate': sel_stock.get('Date', str(date.today())),
                        'CloseDate': str(date.today()),
                        'Result': 'Sold',
                        'Profit': realized_pl,
                        'CostPrice': sel_stock['CostPrice'],
                        'Shares': sel_stock['Shares']
                    }
                    add_document('history', hist_entry)
                    delete_document('holdings', sel_stock['id'])
                    st.success(f"Sold {sel_stock['Shares']} {sel_stock['Ticker']} @ ${current_price:.2f}. Realized P&L: ${realized_pl:.2f}")
                    st.rerun()
                
                # Delete entry (for errors)
                if c2.button("Delete Entry (Erroneous)"):
                    delete_document('holdings', sel_stock['id'])
                    st.warning("Entry deleted.")
                    st.rerun()
        
        st.divider()

    st.subheader("⚡ Active Options")
    if positions_data and not df.empty:
        def calc_row(row):
            curr = prices.get(row['Ticker'], 0) or 0
            dist = 0
            is_itm = False
            if curr > 0:
                dist = ((curr - row['Strike']) / curr) * 100
                is_itm = (row['Type'] == 'Put' and curr < row['Strike']) or \
                         (row['Type'] == 'Call' and curr > row['Strike'])
            
            try:
                exp_date = datetime.strptime(row['Expiry'], '%Y-%m-%d').date()
                dte = max(1, (exp_date - date.today()).days)
            except: dte = 30
            
            collateral = row['Strike'] * 100 * row['Contracts']
            total_prem = row['Premium'] * 100 * row['Contracts']
            annualized = (total_prem / collateral) * (365 / dte) * 100 if collateral > 0 else 0
            
            covered_info = "Naked"
            if row['Type'] == 'Call' and holdings_data:
                matching_stock = next((h for h in holdings_data if h['Ticker'] == row['Ticker']), None)
                if matching_stock:
                    stock_cost = matching_stock['CostPrice']
                    shares_owned = matching_stock['Shares']
                    req_shares = row['Contracts'] * 100
                    
                    if shares_owned >= req_shares:
                        is_profitable_assign = row['Strike'] >= stock_cost
                        icon = "✅" if is_profitable_assign else "⚠️"
                        covered_info = f"{icon} Covered (Basis ${stock_cost})"
                    else:
                        covered_info = f"⚠️ Partial ({shares_owned}/{req_shares})"
            elif row['Type'] == 'Put':
                covered_info = "Cash Sec."

            return pd.Series([curr, dist, is_itm, annualized, dte, covered_info])

        stats = df.apply(calc_row, axis=1)
        stats.columns = ['Current Price', 'DistPct', 'Is ITM', 'AROC %', 'DTE', 'Status']
        
        df = df.drop(columns=[c for c in stats.columns if c in df.columns])
        df = pd.concat([df, stats], axis=1)

        def color_risk(row):
            try:
                if row['Is ITM']: return ['background-color: #ffcccc; color: #8B0000'] * len(row)
                if row['Type'] == 'Put' and row['DistPct'] > 10: return ['background-color: #e6fffa; color: #004d40'] * len(row)
            except: pass
            return [''] * len(row)

        display_df = df.copy()
        
        for col in ['DistPct', 'AROC %']:
            col_data = display_df[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            display_df[col] = pd.to_numeric(col_data, errors='coerce').fillna(0)
        
        display_df['Dist to Strike'] = display_df.apply(lambda x: f"{x['DistPct']:.1f}%" if x['Type']=='Put' else f"{abs(x['DistPct']):.1f}%", axis=1)
        display_df['Annualized'] = display_df['AROC %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_df[['Ticker', 'Type', 'Strike', 'Current Price', 'Dist to Strike', 'Annualized', 'DTE', 'Is ITM', 'Status']]
            .style.apply(color_risk, axis=1)
            .format({'Strike': '${:.2f}', 'Current Price': '${:.2f}'}),
            use_container_width=True, 
            height=300,
            column_config={
                "Is ITM": st.column_config.CheckboxColumn("ITM", disabled=True),
                "Status": st.column_config.TextColumn("Collateral Status", width="medium")
            }
        )
        st.download_button(
            "📥 Export Active Positions CSV",
            display_df[['Ticker', 'Type', 'Strike', 'Current Price', 'Dist to Strike', 'Annualized', 'DTE', 'Is ITM', 'Status']].to_csv(index=False),
            "active_positions.csv",
            "text/csv",
            key="dl_active_positions"
        )
    else:
        st.info("No active options.")

    st.divider()
    st.subheader("🛠️ Trade Actions")
    
    if check_auth():
        pos_list = {}
        if positions_data:
            for p in positions_data:
                label = f"{p['Ticker']} {p['Type']} ${p['Strike']} Exp: {p['Expiry']}"
                pos_list[label] = p

            selected_label = st.selectbox("Select Position to Close/Expire", list(pos_list.keys()))
            sel = pos_list[selected_label]
            
            c1, c2, c3, c4 = st.columns(4)
            if c1.button("Mark as Expired (Full Profit)"):
                hist_entry = sel.copy()
                hist_entry['CloseDate'] = str(date.today())
                hist_entry['Result'] = "Expired"
                hist_entry['Profit'] = sel['Premium'] * 100 * sel['Contracts']
                add_document('history', hist_entry)
                delete_document('positions', sel['id'])
                st.success("Moved to History!")
                st.rerun()
                
            if c2.button("Mark as Assigned"):
                hist_entry = sel.copy()
                hist_entry['CloseDate'] = str(date.today())
                hist_entry['Result'] = "Assigned"
                hist_entry['Profit'] = 0
                add_document('history', hist_entry)
                
                if sel['Type'] == 'Put':
                    new_holding = {
                        'id': str(uuid.uuid4()),
                        'Ticker': sel['Ticker'],
                        'Shares': sel['Contracts'] * 100,
                        'CostPrice': sel['Strike'],
                        'Date': str(date.today())
                    }
                    add_document('holdings', new_holding)
                    st.toast("Stock added to Inventory!")
                
                delete_document('positions', sel['id'])
                st.success("Assignment Processed!")
                st.rerun()

            if c3.button("Delete (Error Entry)"):
                delete_document('positions', sel['id'])
                st.warning("Deleted.")
                st.rerun()

            with c4.expander("💸 Close Early"):
                close_price = st.number_input("Price Paid/Received (Per Share)", min_value=0.0, step=0.01, help="The price you paid to buy back (BTC) or received to sell (STC) the option.")
                if st.button("Execute Early Close"):
                    hist_entry = sel.copy()
                    hist_entry['CloseDate'] = str(date.today())
                    hist_entry['Result'] = "Closed Early"
                    hist_entry['Profit'] = (sel['Premium'] - close_price) * 100 * sel['Contracts']
                    add_document('history', hist_entry)
                    delete_document('positions', sel['id'])
                    st.success(f"Trade closed! Profit/Loss: ${hist_entry['Profit']:,.2f}")
                    st.rerun()

            st.markdown("---")
            with st.expander(f"🔄 Rollover: {selected_label}"):
                st.write("Close this position and open a new one.")
                r_col1, r_col2 = st.columns(2)
                
                new_strike = r_col1.number_input("New Strike", value=float(sel['Strike']), step=0.5, key="roll_strike")
                new_expiry = r_col2.date_input("New Expiry", value=date.today() + timedelta(days=7), key="roll_expiry")
                
                close_cost = r_col1.number_input("Cost to Close (Total)", min_value=0.0, step=1.0, help="Total amount paid to buy back/close the current position.")
                new_credit = r_col2.number_input("New Premium (Per Share)", min_value=0.0, step=0.01, value=float(sel['Premium']), help="Premium received for the new position.")
                
                net_credit = (new_credit * 100 * sel['Contracts']) - close_cost
                st.info(f"Net Rollover Credit: ${net_credit:,.2f}")
                
                if st.button("Execute Rollover"):
                    hist_entry = sel.copy()
                    hist_entry['CloseDate'] = str(date.today())
                    hist_entry['Result'] = "Rolled"
                    hist_entry['Profit'] = (sel['Premium'] * 100 * sel['Contracts']) - close_cost
                    add_document('history', hist_entry)
                    delete_document('positions', sel['id'])
                    
                    new_trade = {
                        'id': str(uuid.uuid4()),
                        'Ticker': sel['Ticker'],
                        'Type': sel['Type'],
                        'Strike': new_strike,
                        'Premium': new_credit,
                        'Contracts': sel['Contracts'],
                        'Expiry': str(new_expiry), 
                        'OpenDate': str(date.today())
                    }
                    add_document('positions', new_trade)
                    
                    st.success(f"Rollover complete! Net Credit: ${net_credit:,.2f}")
                    st.rerun()
    else:
        st.info("Login to manage trades.")

# ==========================
# TAB 2: CAMPAIGN ANALYSIS
# ==========================
with tab2:
    st.subheader("📊 Campaign Tracking")
    st.write("Grouped view of your wheel performance per ticker.")
    
    if history_data:
        hist_df = pd.DataFrame(history_data)
        
        hist_df['Profit'] = pd.to_numeric(hist_df['Profit'], errors='coerce').fillna(0)
        
        summary = hist_df.groupby('Ticker').agg({
            'Profit': ['sum', 'count'],
            'Result': [
                lambda x: (x == 'Expired').sum(),
                lambda x: (x == 'Rolled').sum(),
                lambda x: (x == 'Assigned').sum()
            ]
        }).reset_index()
        
        summary.columns = ['Ticker', 'Total Realized P&L', 'Trade Count', 'Expired', 'Rolled', 'Assigned']
        summary = summary.sort_values('Total Realized P&L', ascending=False)
        
        total_p_l = summary['Total Realized P&L'].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Realized P&L", f"${total_p_l:,.2f}")
        c2.metric("Winning Trades", f"{summary['Expired'].sum()}")
        c3.metric("Total Trades", f"{summary['Trade Count'].sum()}")
        
        st.markdown("### 🏷️ Performance by Ticker")
        def color_pl(val):
            try:
                num = float(val)
                return 'color: green' if num > 0 else 'color: red' if num < 0 else ''
            except (ValueError, TypeError):
                return ''

        st.dataframe(
            summary.style.format({'Total Realized P&L': '${:,.2f}'})
            .map(color_pl, subset=['Total Realized P&L']),
            use_container_width=True
        )
    else:
        st.info("No trade history available for analysis.")

# ==========================
# TAB 3: TRADE HISTORY
# ==========================
with tab3:
    st.subheader("📜 Completed Trades")
    if history_data:
        h_df_view = pd.DataFrame(history_data)
        st.dataframe(h_df_view.sort_values('CloseDate', ascending=False), use_container_width=True)
        
        st.download_button(
            "📥 Export Trade History CSV",
            h_df_view.sort_values('CloseDate', ascending=False).to_csv(index=False),
            "trade_history.csv",
            "text/csv",
            key="dl_trade_history"
        )
        
        if check_auth():
            if st.button("Clear History"):
                for item in history_data:
                    delete_document('history', item['id'])
                st.rerun()
    else:
        st.info("No completed trades.")

# ==========================
# TAB 4: SCREENER
# ==========================
with tab4:
    st.subheader("🔍 Quantitative Wheel Strategy Screener")
    st.write("Scan the market for high-probability Cash-Secured Put and Covered Call opportunities.")
    
    col1, col2 = st.columns([3, 1])
    ticker_list_input = col1.text_input("Enter Tickers (comma-separated)", value="NVDA, TSLA, AAPL, IREN, NBIS")
    max_cap_input = col2.number_input("Max Total Capital ($)", value=30000, step=5000)
    
    # Per‑ticker profile overrides (format: "TICKER:Profile, TICKER:Profile")
    ticker_profiles_input = st.text_input(
        "Per‑Ticker Profile Overrides",
        value="",
        placeholder="e.g. NVDA:Conservative, TSLA:Aggressive",
        help="Override the default profile for specific tickers. Format: TICKER:Profile (comma‑separated). Unlisted tickers use the Default Profile below."
    )
    default_profile = st.selectbox(
        "Default Profile (unlisted tickers)",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
        help="Applied to all tickers not listed in the Per‑Ticker Overrides above."
    )
    
    # --- Screening Criteria ---
    with st.expander("⚙️ Screening Criteria", expanded=False):
        # ── Profile presets ──────────────────────────────────────────────────────────
        st.markdown("**Strategy Profile**")
        profile_options = ["Custom", "Conservative", "Moderate", "Aggressive"]
        # Initialize defaults on first run
        if "screener_profile" not in st.session_state:
            st.session_state["screener_profile"] = "Custom"
            st.session_state["screener_max_cap"] = 30000.0
            st.session_state["screener_min_dte"] = 7
            st.session_state["screener_max_dte"] = 45
            st.session_state["screener_min_premium"] = 0.30
            st.session_state["screener_min_csp_roc"] = 0.0
            st.session_state["screener_min_blended_roc"] = 0.0
            st.session_state["screener_max_cap_per_trade"] = 30000.0
            st.session_state["screener_min_iv"] = 0.0
            st.session_state["screener_min_csp_delta_abs"] = 0.0
            st.session_state["screener_max_csp_delta_abs"] = 1.0
            st.session_state["screener_min_cc_delta"] = 0.0
            st.session_state["screener_max_cc_delta"] = 1.0
            st.session_state["screener_min_ivr"] = 0.0
            st.session_state["screener_csp_target_delta"] = 0.18
            st.session_state["screener_cc_target_delta"] = 0.25
        
        profiles = {
            "Conservative": dict(
                screener_max_cap=20000.0, screener_min_dte=14, screener_max_dte=30,
                screener_min_premium=0.50, screener_min_csp_roc=0.0, screener_min_blended_roc=0.0,
                screener_max_cap_per_trade=5000.0, screener_min_iv=30.0,
                screener_min_csp_delta_abs=0.05, screener_max_csp_delta_abs=0.20,
                screener_min_cc_delta=0.05, screener_max_cc_delta=0.20,
                screener_min_ivr=80.0, screener_csp_target_delta=0.10, screener_cc_target_delta=0.15,
            ),
            "Moderate": dict(
                screener_max_cap=30000.0, screener_min_dte=7, screener_max_dte=45,
                screener_min_premium=0.30, screener_min_csp_roc=0.0, screener_min_blended_roc=0.0,
                screener_max_cap_per_trade=10000.0, screener_min_iv=0.0,
                screener_min_csp_delta_abs=0.0, screener_max_csp_delta_abs=1.0,
                screener_min_cc_delta=0.0, screener_max_cc_delta=1.0,
                screener_min_ivr=0.0, screener_csp_target_delta=0.18, screener_cc_target_delta=0.25,
            ),
            "Aggressive": dict(
                screener_max_cap=50000.0, screener_min_dte=5, screener_max_dte=21,
                screener_min_premium=0.10, screener_min_csp_roc=0.0, screener_min_blended_roc=0.0,
                screener_max_cap_per_trade=20000.0, screener_min_iv=0.0,
                screener_min_csp_delta_abs=0.20, screener_max_csp_delta_abs=0.50,
                screener_min_cc_delta=0.20, screener_max_cc_delta=0.50,
                screener_min_ivr=0.0, screener_csp_target_delta=0.30, screener_cc_target_delta=0.35,
            ),
        }
        
        def apply_profile():
            p = st.session_state.screener_profile
            if p == "Custom":
                return  # keep current values
            vals = profiles[p]
            for k, v in vals.items():
                st.session_state[k] = v
        
        st.selectbox(
            "Profile", profile_options, index=profile_options.index(st.session_state["screener_profile"]),
            key="screener_profile", on_change=apply_profile,
            help="Select a profile to auto-fill criteria. Switch to Custom to override."
        )
        st.divider()
        
        # ── Criteria inputs (reference session_state) ─────────────────────────────────
        cr1, cr2, cr3 = st.columns(3)
        min_dte = cr1.number_input("Min DTE", min_value=1, value=st.session_state["screener_min_dte"], step=1, help="Minimum days to expiration", key="screener_min_dte")
        max_dte = cr2.number_input("Max DTE", min_value=min_dte, value=st.session_state["screener_max_dte"], step=1, help="Maximum days to expiration", key="screener_max_dte")
        min_premium = cr3.number_input("Min Premium ($)", min_value=0.0, value=st.session_state["screener_min_premium"], step=0.05, help="Minimum option premium per share", key="screener_min_premium")
        
        cr4, cr5, cr6 = st.columns(3)
        min_csp_roc = cr4.number_input("Min CSP ROC %", min_value=0.0, value=st.session_state["screener_min_csp_roc"], step=1.0, help="Minimum annualized return for cash-secured put", key="screener_min_csp_roc")
        min_blended_roc = cr5.number_input("Min Blended ROC %", min_value=0.0, value=st.session_state["screener_min_blended_roc"], step=1.0, help="Minimum annualized blended return (full wheel loop)", key="screener_min_blended_roc")
        max_cap_per_trade = cr6.number_input("Max Capital per Trade ($)", min_value=0.0, value=st.session_state["screener_max_cap_per_trade"], step=1000.0, help="Max capital requirement per single trade", key="screener_max_cap_per_trade")
        
        cr7, cr8 = st.columns(2)
        min_iv = cr7.number_input("Min Implied Volatility (%)", min_value=0.0, value=st.session_state["screener_min_iv"], step=5.0, help="Minimum IV% of the option", key="screener_min_iv")
        sort_by = cr8.selectbox("Sort By", ["CSP ROC %", "Blended ROC %", "DTE", "Capital Req", "Premium", "IV %", "IVR %", "CSP Δ", "CC Δ"], index=0)
        
        # Delta range filters
        cr9, cr10 = st.columns(2)
        with cr9:
            st.markdown("**CSP Delta Range (abs)**")
            min_csp_delta_abs = st.number_input("Min CSP Δ", min_value=0.0, value=st.session_state["screener_min_csp_delta_abs"], step=0.05, help="Minimum absolute delta for CSP", key="screener_min_csp_delta_abs")
            max_csp_delta_abs = st.number_input("Max CSP Δ", min_value=0.0, value=st.session_state["screener_max_csp_delta_abs"], step=0.05, help="Maximum absolute delta for CSP", key="screener_max_csp_delta_abs")
        with cr10:
            st.markdown("**CC Delta Range**")
            min_cc_delta = st.number_input("Min CC Δ", min_value=0.0, value=st.session_state["screener_min_cc_delta"], step=0.05, help="Minimum delta for CC", key="screener_min_cc_delta")
            max_cc_delta = st.number_input("Max CC Δ", min_value=0.0, value=st.session_state["screener_max_cc_delta"], step=0.05, help="Maximum delta for CC", key="screener_max_cc_delta")
    
        # Min IVR filter
        min_ivr = st.number_input("Min IVR %", min_value=0.0, value=st.session_state["screener_min_ivr"], step=1.0, help="Minimum implied volatility rank relative to 30‑day HV", key="screener_min_ivr")
        
        # Target deltas for strike selection
        cr12, cr13 = st.columns(2)
        with cr12:
            csp_target_delta = st.number_input("CSP Target Δ", min_value=0.01, value=st.session_state["screener_csp_target_delta"], step=0.01, help="Desired absolute delta for CSP strike selection (e.g. 0.18 = deeper OTM)", key="screener_csp_target_delta")
        with cr13:
            cc_target_delta = st.number_input("CC Target Δ", min_value=0.01, value=st.session_state["screener_cc_target_delta"], step=0.01, help="Desired delta for CC strike selection (e.g. 0.25 = 25 delta)", key="screener_cc_target_delta")
    
    # Per‑profile delta targets (must match the profiles dict values above)
    PROFILE_DELTA_TARGETS = {
        "Conservative": {"csp_target_delta": 0.10, "cc_target_delta": 0.15},
        "Moderate":     {"csp_target_delta": 0.18, "cc_target_delta": 0.25},
        "Aggressive":   {"csp_target_delta": 0.30, "cc_target_delta": 0.35},
    }
    
    def parse_ticker_profiles(raw: str, default: str) -> dict:
        """Parse 'TICKER:Profile,TICKER:Profile' into {ticker: profile}."""
        mapping = {}
        if not raw:
            return mapping
        for entry in raw.split(","):
            if ":" not in entry:
                continue
            key, val = entry.split(":", 1)
            ticker = key.strip().upper()
            profile = val.strip()
            if profile in PROFILE_DELTA_TARGETS:
                mapping[ticker] = profile
            else:
                st.warning(f"Unknown profile '{profile}' for {ticker} — using {default}.")
                mapping[ticker] = default
        return mapping
    
    if st.button("🚀 Run Analysis"):
        all_tickers = [t.strip().upper() for t in ticker_list_input.split(",") if t.strip()]
        valid, err_msg = validate_screener_inputs(min_dte, max_dte, all_tickers)
        if not valid:
            st.error(f"Validation: {err_msg}")
        else:
            with st.spinner("Analyzing real-time options data..."):
                ticker_overrides = parse_ticker_profiles(ticker_profiles_input, default_profile)
                
                # Group tickers by profile
                profile_groups = {}
                for t in all_tickers:
                    p = ticker_overrides.get(t, default_profile)
                    profile_groups.setdefault(p, []).append(t)
                
                # Run analysis per profile group and merge results
                all_results = []
                for profile_name, tickers in profile_groups.items():
                    deltas = PROFILE_DELTA_TARGETS.get(profile_name, {"csp_target_delta": csp_target_delta, "cc_target_delta": cc_target_delta})
                    result = analyze_strategy_optimized(
                        tickers, float(max_cap_input),
                        csp_target_delta=deltas["csp_target_delta"],
                        cc_target_delta=deltas["cc_target_delta"],
                    )
                    for r in result.get("results", []):
                        r["profile"] = profile_name   # tag result with profile used
                    all_results.extend(result.get("results", []))
                
                total_capital = sum(r.get("capital_req", 0) for r in all_results)
            max_exceeded = total_capital > float(max_cap_input)
            analysis = {"results": all_results, "total_capital": total_capital, "max_exceeded": max_exceeded}
            
            if analysis["results"]:
                if analysis["max_exceeded"]:
                    st.warning(f"⚠️ Aggregate Capital Required (${analysis['total_capital']:,.2f}) exceeds Constraint (${max_cap_input:,.2f})")
                
                res_df = pd.DataFrame(analysis["results"])
                
                # --- Apply Screening Criteria ---
                mask = pd.Series([True] * len(res_df), index=res_df.index)
                if 'dte' in res_df.columns:
                    mask &= (res_df['dte'] >= min_dte) & (res_df['dte'] <= max_dte)
                if 'csp_premium' in res_df.columns:
                    mask &= (res_df['csp_premium'] >= min_premium)
                if 'csp_roc' in res_df.columns:
                    mask &= (res_df['csp_roc'] >= min_csp_roc)
                if 'blended_roc' in res_df.columns:
                    mask &= (res_df['blended_roc'] >= min_blended_roc)
                if 'capital_req' in res_df.columns:
                    mask &= (res_df['capital_req'] <= max_cap_per_trade)
                if 'iv' in res_df.columns:
                    mask &= (res_df['iv'] >= min_iv)
                if 'ivr' in res_df.columns:
                    mask &= (res_df['ivr'] >= min_ivr)
                if 'csp_delta_abs' in res_df.columns:
                    mask &= (res_df['csp_delta_abs'] >= min_csp_delta_abs) & (res_df['csp_delta_abs'] <= max_csp_delta_abs)
                if 'cc_delta' in res_df.columns:
                    mask &= (res_df['cc_delta'] >= min_cc_delta) & (res_df['cc_delta'] <= max_cc_delta)
                
                filtered_df = res_df[mask].copy()
                
                # --- Sort ---
                sort_map = {
                    "CSP ROC %": ("csp_roc", True),
                    "Blended ROC %": ("blended_roc", True),
                    "DTE": ("dte", True),
                    "Capital Req": ("capital_req", False),
                    "Premium": ("csp_premium", True),
                    "IV %": ("iv", True),
                    "IVR %": ("ivr", True),
                    "CSP Δ": ("csp_delta_abs", True),
                    "CC Δ": ("cc_delta", True),
                }
                sort_col, ascending = sort_map.get(sort_by, ("csp_roc", True))
                if sort_col in filtered_df.columns:
                    filtered_df = filtered_df.sort_values(sort_col, ascending=ascending)
                
                filtered_count = len(filtered_df)
                total_count = len(res_df)
                
                if filtered_count == 0:
                    st.warning(f"No results match your criteria ({total_count} tickers scanned).")
                else:
                    st.markdown(f"### 📊 Screener Results — {filtered_count} / {total_count} matched")
                    
                    # Dynamically select columns that exist to avoid KeyError
                    primary_cols = ['ticker', 'price', 'csp_strike', 'csp_premium', 'csp_roc', 'capital_req', 'dte', 'iv', 'ivr', 'csp_delta_abs', 'cc_delta', 'profile']
                    available_primary = [c for c in primary_cols if c in filtered_df.columns]
                    
                    rename_map = {
                        'ticker': 'Ticker',
                        'price': 'Price',
                        'csp_strike': 'CSP Strike',
                        'csp_premium': 'Premium',
                        'csp_roc': 'CSP ROC %',
                        'capital_req': 'Cap Req',
                        'dte': 'DTE',
                        'iv': 'IV %',
                        'ivr': 'IVR %',
                        'csp_delta_abs': 'CSP Δ',
                        'cc_delta': 'CC Δ',
                        'profile': 'Profile'
                    }
                    format_dict = {
                        'Price': '${:.2f}', 
                        'CSP Strike': '${:.2f}', 
                        'Premium': '${:.2f}', 
                        'CSP ROC %': '{:.2f}%', 
                        'Cap Req': '${:,.0f}',
                        'DTE': '{:.0f}',
                        'IV %': '{:.1f}%',
                        'IVR %': '{:.1f}%',
                        'CSP Δ': '{:.2f}',
                        'CC Δ': '{:.2f}'
                    }
                    
                    display_df = filtered_df[available_primary].rename(columns={k: v for k, v in rename_map.items() if k in available_primary})
                    styler = display_df.style.format(format_dict)
                    if 'CSP ROC %' in display_df.columns:
                        styler = styler.background_gradient(subset=['CSP ROC %'], cmap='YlGn')
                    if 'Profile' in display_df.columns:
                        profile_colors = {
                            'Conservative': 'background-color: #2e7d32; color: white; font-weight: bold',
                            'Moderate':     'background-color: #e65100; color: white; font-weight: bold',
                            'Aggressive':   'background-color: #b71c1c; color: white; font-weight: bold',
                        }
                        styler = styler.map(profile_colors, subset=['Profile'])
                    
                    st.dataframe(
                        styler,
                        use_container_width=True,
                        height=400
                    )
                    
                    st.download_button(
                        "📥 Export Screener Results CSV",
                        filtered_df.to_csv(index=False),
                        "screener_results.csv",
                        "text/csv",
                        key="dl_screener_results"
                    )
                    
                    st.markdown("### 🛡️ Risk & Full-Loop Metrics")
                    # Dynamically select available columns for risk view
                    risk_cols = ['ticker', 'cc_strike', 'cc_premium', 'blended_roc', 'defense_roll_strike', 'stop_loss_loss', 'capital_protected']
                    available_risk = [c for c in risk_cols if c in filtered_df.columns]
                    
                    risk_rename = {
                        'ticker': 'Ticker',
                        'cc_strike': 'CC Strike',
                        'cc_premium': 'CC Premium',
                        'blended_roc': 'Blended ROC %',
                        'defense_roll_strike': 'Roll Strike',
                        'stop_loss_loss': 'Max Loss (Stop)',
                        'capital_protected': 'Capital Protected %'
                    }
                    risk_format = {
                        'CC Strike': '${:.2f}', 
                        'CC Premium': '${:.2f}',
                        'Blended ROC %': '{:.2f}%', 
                        'Roll Strike': '${:.2f}', 
                        'Max Loss (Stop)': '${:,.0f}',
                        'Capital Protected %': '{:.1f}%'
                    }
                    
                    if available_risk:
                        st.dataframe(
                            filtered_df[available_risk]
                            .rename(columns={k: v for k, v in risk_rename.items() if k in available_risk})
                            .style.format(risk_format),
                            use_container_width=True,
                            height=400
                        )
                    else:
                        st.warning("No risk metrics available.")
            else:
                st.error("No data found for the provided tickers. Please check the symbols and try again.")