import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
import uuid
import json
import re
from google.cloud import firestore
from google.oauth2 import service_account

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
    """Checks if the user is logged in. Returns True if admin."""
    if st.session_state.authenticated:
        return True
    return False

# --- Firestore Connection (ROBUST VERSION) ---
@st.cache_resource
def get_db():
    try:
        if "textkey" not in st.secrets:
            st.error("Internal Error: 'textkey' not found in secrets.")
            return None

        secret_val = st.secrets["textkey"]
        
        if isinstance(secret_val, dict):
            key_dict = secret_val
        else:
            try:
                key_dict = json.loads(secret_val, strict=False)
            except json.JSONDecodeError:
                clean_val = re.sub(r'[\r\n]+', '', secret_val) 
                try:
                     key_dict = json.loads(clean_val, strict=False)
                except:
                     st.error("JSON Error in Secrets. Please check format.")
                     return None

        creds = service_account.Credentials.from_service_account_info(key_dict)
        db = firestore.Client(credentials=creds, project=key_dict["project_id"])
        return db
        
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

db = get_db()

# --- Database Helpers ---
def load_collection(collection_name):
    if not db: return []
    docs = db.collection(collection_name).stream()
    items = []
    for doc in docs:
        item = doc.to_dict()
        item['id'] = doc.id
        items.append(item)
    return items

def add_document(collection_name, data):
    if not db: return
    doc_id = data.get('id', str(uuid.uuid4()))
    db.collection(collection_name).document(doc_id).set(data)

def delete_document(collection_name, doc_id):
    if not db: return
    db.collection(collection_name).document(doc_id).delete()

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

# Only show Write Forms if Authenticated
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
tab1, tab2, tab3 = st.tabs(["🚀 Active Positions", "📉 Campaign Analysis", "📜 Trade History"])

# ==========================
# TAB 1: ACTIVE POSITIONS
# ==========================
with tab1:
    # --- SECTION A: STOCK HOLDINGS ---
    if holdings_data:
        st.subheader("📦 Stock Inventory (Assigned)")
        h_df = pd.DataFrame(holdings_data)
        
        # Get Prices
        with st.spinner('Syncing prices...'):
             h_prices = {t: get_current_price(t) for t in h_df['Ticker'].unique()}
        
        def calc_holding(row):
            curr = h_prices.get(row['Ticker'], 0) or 0
            mkt_val = curr * row['Shares']
            cost_val = row['CostPrice'] * row['Shares']
            unreal_pl = mkt_val - cost_val
            pl_pct = ((curr - row['CostPrice']) / row['CostPrice']) * 100 if row['CostPrice'] > 0 else 0
            return pd.Series([curr, mkt_val, unreal_pl, pl_pct])

        h_stats = h_df.apply(calc_holding, axis=1)
        h_stats.columns = ['Current Price', 'Market Value', 'Unrealized P&L', 'Return %']
        h_df = pd.concat([h_df, h_stats], axis=1)
        
        st.dataframe(
            h_df[['Ticker', 'Shares', 'CostPrice', 'Current Price', 'Market Value', 'Unrealized P&L', 'Return %']]
            .style.format({'CostPrice': '${:.2f}', 'Current Price': '${:.2f}', 'Market Value': '${:,.0f}', 'Unrealized P&L': '${:,.0f}', 'Return %': '{:.2f}%'})
            .applymap(lambda v: 'color: green' if v > 0 else 'color: red', subset=['Unrealized P&L']),
            use_container_width=True,
            height=150
        )
        st.divider()

    # --- SECTION B: OPTIONS ---
    st.subheader("⚡ Active Options")
    if not positions_data:
        st.info("No active options.")
    else:
        df = pd.DataFrame(positions_data)
        prices = {t: get_current_price(t) for t in df['Ticker'].unique()}
        
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
            
            # Covered Logic
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
        df = pd.concat([df, stats], axis=1)

        def color_risk(row):
            try:
                if row['Is ITM']: return ['background-color: #ffcccc; color: #8B0000'] * len(row)
                if row['Type'] == 'Put' and row['DistPct'] > 10: return ['background-color: #e6fffa; color: #004d40'] * len(row)
            except: pass
            return [''] * len(row)

        display_df = df.copy()
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

        # 5. ACTION CENTER (ROLL & EDIT) - PROTECTED
        st.divider()
        st.subheader("🛠️ Trade Actions")
        
        if check_auth():
            pos_list = {f"OPTION: {r['Ticker']} ${r['Strike']} {r['Type']}": ('pos', r['id']) for _, r in df.iterrows()}
            if holdings_data:
                h_df_temp = pd.DataFrame(holdings_data)
                pos_list.update({f"STOCK: {r['Ticker']} ({r['Shares']} shares)": ('hold', r['id']) for _, r in h_df_temp.iterrows()})

            if pos_list:
                selected_label = st.selectbox("Select Asset to Manage:", list(pos_list.keys()))
                sel_type, sel_id = pos_list[selected_label]

                if sel_type == 'hold':
                    if st.button("Delete Stock Holding"):
                        delete_document('holdings', sel_id)
                        st.success("Stock deleted."); st.rerun()
                else:
                    selected_row = df[df['id'] == sel_id].iloc[0]
                    action_type = st.radio("Action Type", ["Close / Exit", "Roll Position"], horizontal=True)

                    if action_type == "Close / Exit":
                        c1, c2 = st.columns(2)
                        with c1:
                            close_reason = st.selectbox("Exit Reason", ["Buy to Close (BTC)", "Expired Worthless", "Assignment", "Delete (Mistake)"])
                        with c2:
                            if close_reason == "Buy to Close (BTC)":
                                close_price = st.number_input("Price Paid to Close", min_value=0.0, value=0.05, step=0.01)
                            else:
                                close_price = 0.0
                        
                        if st.button("Confirm Exit"):
                            if close_reason == "Delete (Mistake)":
                                delete_document('positions', sel_id)
                            else:
                                profit = (selected_row['Premium'] - close_price) * 100 * selected_row['Contracts']
                                history_record = selected_row.to_dict()
                                history_record.update({'CloseDate': str(date.today()), 'ClosePrice': close_price, 'Reason': close_reason, 'Profit': profit})
                                add_document('history', history_record)
                                delete_document('positions', sel_id)
                            st.toast("Updated!"); st.rerun()

                    elif action_type == "Roll Position":
                        st.info("🔄 Roll Logic Active")
                        col_old, col_new = st.columns(2)
                        with col_old:
                            btc_price = st.number_input("Buy-to-Close (Old)", value=selected_row['Premium']*0.5)
                        with col_new:
                            new_expiry = st.date_input("New Expiry", value=date.today() + timedelta(days=30))
                            new_strike = st.number_input("New Strike", value=selected_row['Strike'])
                            new_premium = st.number_input("New Premium", value=selected_row['Premium'])
                        
                        if st.button("Execute Roll"):
                            old_pnl = (selected_row['Premium'] - btc_price) * 100 * selected_row['Contracts']
                            hist_rec = selected_row.to_dict()
                            hist_rec.update({'CloseDate': str(date.today()), 'ClosePrice': btc_price, 'Reason': "Rolled", 'Profit': old_pnl})
                            add_document('history', hist_rec)
                            delete_document('positions', sel_id)
                            new_trade = selected_row.to_dict()
                            new_trade.update({'id': str(uuid.uuid4()), 'Strike': new_strike, 'Premium': new_premium, 'Expiry': str(new_expiry), 'OpenDate': str(date.today())})
                            add_document('positions', new_trade)
                            st.balloons(); st.rerun()
        else:
            st.warning("🔒 Login in the sidebar to Manage/Edit trades.")

# ==========================
# TAB 2: CAMPAIGN ANALYSIS
# ==========================
with tab2:
    st.header("📉 Total Campaign P&L")
    
    all_tickers = set()
    if positions_data: all_tickers.update([x['Ticker'] for x in positions_data])
    if history_data: all_tickers.update([x['Ticker'] for x in history_data])
    if holdings_data: all_tickers.update([x['Ticker'] for x in holdings_data])
    
    if not all_tickers:
        st.info("No data.")
    else:
        campaigns = []
        for ticker in all_tickers:
            hist_trades = [h for h in history_data if h['Ticker'] == ticker]
            opt_realized = sum([h['Profit'] for h in hist_trades])
            
            active_opts = [p for p in positions_data if p['Ticker'] == ticker]
            opt_premium_held = sum([p['Premium'] * p['Contracts'] * 100 for p in active_opts])
            
            stock_holdings = [h for h in holdings_data if h['Ticker'] == ticker]
            stock_unrealized = 0
            stock_cost_basis = 0
            shares_count = 0
            
            curr_price = get_current_price(ticker) or 0
            
            if stock_holdings and curr_price > 0:
                for h in stock_holdings:
                    shares_count += h['Shares']
                    stock_cost_basis += (h['CostPrice'] * h['Shares'])
                    stock_unrealized += (curr_price - h['CostPrice']) * h['Shares']
            
            total_campaign_pl = opt_realized + stock_unrealized + opt_premium_held
            
            if shares_count > 0:
                net_cost_total = stock_cost_basis - opt_realized
                adj_cost_per_share = net_cost_total / shares_count
            else:
                adj_cost_per_share = 0
            
            campaigns.append({
                "Ticker": ticker,
                "Shares": shares_count,
                "Option Profit (Realized)": opt_realized,
                "Stock Profit (Unrealized)": stock_unrealized,
                "Total Campaign P&L": total_campaign_pl,
                "Adj. Cost Basis": adj_cost_per_share
            })
            
        camp_df = pd.DataFrame(campaigns)
        
        cols = st.columns(3)
        for idx, row in camp_df.iterrows():
            with cols[idx % 3]:
                with st.container(border=True):
                    st.subheader(f"{row['Ticker']}")
                    
                    color = "green" if row['Total Campaign P&L'] > 0 else "red"
                    st.markdown(f"<h3 style='color:{color}'>${row['Total Campaign P&L']:,.0f}</h3>", unsafe_allow_html=True)
                    st.caption("Total P&L (Options + Stock Growth)")
                    
                    st.divider()
                    
                    if row['Shares'] > 0:
                        st.write(f"**Shares Held:** {row['Shares']}")
                        st.write(f"**Adj. Cost Basis:** :blue[${row['Adj. Cost Basis']:.2f}]")
                    else:
                        st.write("**Strategy:** Wheel (Currently Cash Secured)")
                        st.write(f"**Realized Gains:** ${row['Option Profit (Realized)']:,.2f}")

# ==========================
# TAB 3: HISTORY
# ==========================
with tab3:
    if not history_data:
        st.write("No closed trades.")
    else:
        hist_df = pd.DataFrame(history_data)
        hist_df['CloseDateDT'] = pd.to_datetime(hist_df['CloseDate'])
        hist_df['Month'] = hist_df['CloseDateDT'].dt.strftime('%Y-%m')
        monthly_pnl = hist_df.groupby('Month')['Profit'].sum()
        
        st.subheader("💰 Monthly Income")
        st.bar_chart(monthly_pnl, color="#4CAF50")
        
        st.dataframe(
            hist_df[['CloseDate', 'Ticker', 'Type', 'Strike', 'Reason', 'Profit']]
            .sort_values(by='CloseDate', ascending=False)
            .style.format({'Profit': '${:,.2f}', 'Strike': '${:.1f}'})
            .applymap(lambda v: 'color: green' if v > 0 else 'color: red', subset=['Profit']),
            use_container_width=True
        )