import streamlit as st
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, date, timedelta
from wheel_screener import analyze_strategy_optimized
import uuid
import json
import re
from supabase import create_client, Client

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
            st.error("Internal Error: 'supabase_key' not found in secrets or SUPABASE_KEY env var.")
            return None

        db = create_client(supabase_url, supabase_key)

        # Quick connectivity check
        try:
            db.table("positions").select("id").limit(1).execute()
        except Exception as conn_err:
            err_str = str(conn_err).lower()
            if "jwt" in err_str or "unauthorized" in err_str or "apikey" in err_str:
                st.error(f"Supabase auth error: {conn_err}")
                return None
            # Table might not exist yet - continue anyway

        return db

    except Exception as e:
        st.error(f"Connection Error: {e}")
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

# --- Market Data ---
@st.cache_data(ttl=60)
def get_current_price(ticker):
    try:
        return yf.Ticker(ticker).fast_info.last_price
    except:
        return None

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
                new_trade = {
                    'id': str(uuid.uuid4()),
                    'Ticker': ticker_in,
                    'Type': type_in,
                    'Strike': strike_in,
                    'Premium': premium_in,
                    'Contracts': contracts_in,
                    'Expiry': str(expiry_in), 
                    'OpenDate': str(date.today())
                }
                add_document('positions', new_trade)
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
                new_holding = {
                    'id': str(uuid.uuid4()),
                    'Ticker': s_ticker,
                    'Shares': s_shares,
                    'CostPrice': s_cost,
                    'Date': str(s_date)
                }
                add_document('holdings', new_holding)
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
            prices = {t: get_current_price(t) for t in unique_tickers}
            
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
            prices = {t: get_current_price(t) for t in unique_tickers}

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
    
    # --- Screening Criteria ---
    with st.expander("⚙️ Screening Criteria", expanded=False):
        cr1, cr2, cr3 = st.columns(3)
        min_dte = cr1.number_input("Min DTE", min_value=1, value=7, step=1, help="Minimum days to expiration")
        max_dte = cr2.number_input("Max DTE", min_value=min_dte, value=45, step=1, help="Maximum days to expiration")
        min_premium = cr3.number_input("Min Premium ($)", min_value=0.0, value=0.3, step=0.05, help="Minimum option premium per share")
        
        cr4, cr5, cr6 = st.columns(3)
        min_csp_roc = cr4.number_input("Min CSP ROC %", min_value=0.0, value=0.0, step=1.0, help="Minimum annualized return for cash-secured put")
        min_blended_roc = cr5.number_input("Min Blended ROC %", min_value=0.0, value=0.0, step=1.0, help="Minimum annualized blended return (full wheel loop)")
        max_cap_per_trade = cr6.number_input("Max Capital per Trade ($)", min_value=0.0, value=30000.0, step=1000.0, help="Max capital requirement per single trade")
        
        cr7, cr8 = st.columns(2)
        min_iv = cr7.number_input("Min Implied Volatility (%)", min_value=0.0, value=0.0, step=5.0, help="Minimum IV% of the option")
        sort_by = cr8.selectbox("Sort By", ["CSP ROC %", "Blended ROC %", "DTE", "Capital Req", "Premium", "IV %"], index=0)
    
    if st.button("🚀 Run Analysis"):
        with st.spinner("Analyzing real-time options data..."):
            tickers = [t.strip().upper() for t in ticker_list_input.split(",") if t.strip()]
            analysis = analyze_strategy_optimized(tickers, float(max_cap_input))
            
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
                
                filtered_df = res_df[mask].copy()
                
                # --- Sort ---
                sort_map = {
                    "CSP ROC %": ("csp_roc", True),
                    "Blended ROC %": ("blended_roc", True),
                    "DTE": ("dte", True),
                    "Capital Req": ("capital_req", False),
                    "Premium": ("csp_premium", True),
                    "IV %": ("iv", True),
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
                    primary_cols = ['ticker', 'price', 'csp_strike', 'csp_premium', 'csp_roc', 'capital_req', 'dte', 'iv']
                    available_primary = [c for c in primary_cols if c in filtered_df.columns]
                    
                    rename_map = {
                        'ticker': 'Ticker',
                        'price': 'Price',
                        'csp_strike': 'CSP Strike',
                        'csp_premium': 'Premium',
                        'csp_roc': 'CSP ROC %',
                        'capital_req': 'Cap Req',
                        'dte': 'DTE',
                        'iv': 'IV %'
                    }
                    format_dict = {
                        'Price': '${:.2f}', 
                        'CSP Strike': '${:.2f}', 
                        'Premium': '${:.2f}', 
                        'CSP ROC %': '{:.2f}%', 
                        'Cap Req': '${:,.0f}',
                        'DTE': '{:.0f}',
                        'IV %': '{:.1f}%'
                    }
                    
                        display_df = filtered_df[available_primary].rename(columns={k: v for k, v in rename_map.items() if k in available_primary})
                        styler = display_df.style.format(format_dict)
                        if 'CSP ROC %' in display_df.columns:
                            styler = styler.background_gradient(subset=['CSP ROC %'], cmap='YlGn')
                        
                        st.dataframe(
                            styler,
                            use_container_width=True,
                            height=400
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