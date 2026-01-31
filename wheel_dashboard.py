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
    /* Success color for profitable rolls */
    .highlight-green { color: #4CAF50; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Firestore Connection (ROBUST VERSION) ---
@st.cache_resource
def get_db():
    try:
        if "textkey" not in st.secrets:
            st.error("Internal Error: 'textkey' not found in secrets.")
            return None

        secret_val = st.secrets["textkey"]

        # 1. AUTO-FIX: Check if Streamlit already parsed it as a Dict
        if isinstance(secret_val, dict):
            key_dict = secret_val
        else:
            # 2. AUTO-FIX: It's a string. Try to parse.
            try:
                key_dict = json.loads(secret_val, strict=False)
            except json.JSONDecodeError:
                # Remove non-printable control characters
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
# We load data once here to use across tabs
positions_data = load_collection('positions')
history_data = load_collection('history')

# --- Sidebar: Add New Position ---
st.sidebar.header("➕ Add New Trade")
with st.sidebar.form("add_position_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    ticker_in = col1.text_input("Ticker").upper()
    strike_in = col1.number_input("Strike", min_value=0.0, step=0.5)
    type_in = col2.selectbox("Type", ["Put", "Call"])
    contracts_in = col2.number_input("Contracts", min_value=1, step=1)
    premium_in = st.number_input("Premium Received (Per Share)", min_value=0.0, step=0.01)
    expiry_in = st.date_input("Expiry Date")
    
    if st.form_submit_button("Submit Trade"):
        if ticker_in:
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
            st.toast("Trade Saved! ☁️")
            st.rerun()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🚀 Active Positions", "📉 Campaign Analysis", "📜 Trade History"])

# ==========================
# TAB 1: ACTIVE POSITIONS
# ==========================
with tab1:
    if not positions_data:
        st.info("No active trades found.")
    else:
        df = pd.DataFrame(positions_data)
        
        # 1. Market Data Sync
        with st.spinner('Syncing market data...'):
            prices = {t: get_current_price(t) for t in df['Ticker'].unique()}
        
        # 2. Calculations
        def calc_row(row):
            curr = prices.get(row['Ticker'], 0) or 0
            
            # Distance / ITM
            dist = 0
            is_itm = False
            if curr > 0:
                dist = ((curr - row['Strike']) / curr) * 100
                is_itm = (row['Type'] == 'Put' and curr < row['Strike']) or \
                         (row['Type'] == 'Call' and curr > row['Strike'])
            
            # AROC & DTE
            try:
                exp_date = datetime.strptime(row['Expiry'], '%Y-%m-%d').date()
                dte = (exp_date - date.today()).days
                dte = max(1, dte)
            except:
                dte = 30
            
            collateral = row['Strike'] * 100 * row['Contracts']
            total_prem = row['Premium'] * 100 * row['Contracts']
            
            # Annualized Return
            if collateral > 0:
                raw_return = total_prem / collateral
                annualized = raw_return * (365 / dte) * 100
            else:
                annualized = 0
            
            return pd.Series([curr, dist, is_itm, annualized, dte])

        stats = df.apply(calc_row, axis=1)
        stats.columns = ['Current Price', 'DistPct', 'Is ITM', 'AROC %', 'DTE']
        df = pd.concat([df, stats], axis=1)

        # 3. Metrics
        total_open_premium = (df['Premium'] * df['Contracts'] * 100).sum()
        risk_count = df['Is ITM'].sum()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Unrealized Premium", f"${total_open_premium:,.0f}")
        m2.metric("Portfolio Efficiency (Avg AROC)", f"{df['AROC %'].mean():.1f}%")
        m3.metric("At Risk (ITM)", f"{risk_count}", delta_color="inverse", delta="Alert" if risk_count > 0 else "Safe")
        
        # 4. Data Table
        st.subheader("Portfolio Monitor")
        
        def color_risk(row):
            try:
                if row['Is ITM']: return ['background-color: #ffcccc; color: #8B0000'] * len(row)
                if row['Type'] == 'Put' and row['DistPct'] > 10: return ['background-color: #e6fffa; color: #004d40'] * len(row)
            except: pass
            return [''] * len(row)

        display_df = df.copy()
        display_df['Dist to Strike'] = display_df.apply(
            lambda x: f"{x['DistPct']:.1f}%" if x['Type']=='Put' else f"{abs(x['DistPct']):.1f}%", axis=1
        )
        display_df['Annualized'] = display_df['AROC %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_df[['Ticker', 'Type', 'Strike', 'Current Price', 'Dist to Strike', 'Annualized', 'DTE', 'Is ITM', 'DistPct']]
            .style.apply(color_risk, axis=1)
            .format({'Strike': '${:.2f}', 'Current Price': '${:.2f}'}),
            use_container_width=True, 
            height=300,
            column_config={
                "DistPct": None,
                "Is ITM": st.column_config.CheckboxColumn("ITM Alert", disabled=True),
                "DTE": st.column_config.NumberColumn("DTE", help="Days to Expiration")
            }
        )

        # 5. ACTION CENTER (ROLL & EDIT)
        st.divider()
        st.subheader("🛠️ Trade Actions")
        
        position_options = {f"{row['Ticker']} ${row['Strike']} {row['Type']} (Exp: {row['Expiry']})": row['id'] for _, row in df.iterrows()}
        selected_label = st.selectbox("Select Position:", list(position_options.keys()))
        selected_id = position_options[selected_label]
        selected_row = df[df['id'] == selected_id].iloc[0]

        # Toggle between Close and Roll
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
                    delete_document('positions', selected_id)
                else:
                    profit = (selected_row['Premium'] - close_price) * 100 * selected_row['Contracts']
                    history_record = selected_row.to_dict()
                    history_record.update({
                        'CloseDate': str(date.today()), 'ClosePrice': close_price, 
                        'Reason': close_reason, 'Profit': profit
                    })
                    add_document('history', history_record)
                    delete_document('positions', selected_id)
                st.toast("Updated!"); st.rerun()

        elif action_type == "Roll Position":
            st.info("🔄 Rolling: Close existing trade and open a new one immediately.")
            
            col_old, col_new = st.columns(2)
            
            with col_old:
                st.markdown("##### Step 1: Close Old Trade")
                btc_price = st.number_input("Buy-to-Close Price (Old)", min_value=0.0, value=selected_row['Premium']*0.5, step=0.01)
                old_pnl = (selected_row['Premium'] - btc_price) * 100 * selected_row['Contracts']
                st.caption(f"Realized P&L on Old Leg: ${old_pnl:,.2f}")

            with col_new:
                st.markdown("##### Step 2: Open New Trade")
                new_expiry = st.date_input("New Expiry", value=date.today() + timedelta(days=30))
                new_strike = st.number_input("New Strike", value=selected_row['Strike'], step=0.5)
                new_premium = st.number_input("New Premium Received", min_value=0.0, value=selected_row['Premium'], step=0.01)
                
                net_credit = (new_premium - btc_price) * 100 * selected_row['Contracts']
                st.markdown(f"**Net Credit for Roll:** :green[${net_credit:,.2f}]")

            if st.button("Execute Roll"):
                # 1. Save Old to History
                history_record = selected_row.to_dict()
                history_record.update({
                    'CloseDate': str(date.today()), 'ClosePrice': btc_price, 
                    'Reason': "Rolled", 'Profit': old_pnl
                })
                add_document('history', history_record)
                
                # 2. Delete Old Active
                delete_document('positions', selected_id)
                
                # 3. Create New Active
                new_trade = {
                    'id': str(uuid.uuid4()),
                    'Ticker': selected_row['Ticker'],
                    'Type': selected_row['Type'],
                    'Strike': new_strike,
                    'Premium': new_premium,
                    'Contracts': selected_row['Contracts'],
                    'Expiry': str(new_expiry),
                    'OpenDate': str(date.today())
                }
                add_document('positions', new_trade)
                st.balloons()
                st.rerun()

# ==========================
# TAB 2: CAMPAIGN ANALYSIS (NEW)
# ==========================
with tab2:
    st.header("📉 Net Stock Cost Basis (Campaign View)")
    st.caption("Aggregates all historical trades per Ticker to show your effective cost reduction.")
    
    if not history_data and not positions_data:
        st.info("No data available.")
    else:
        # Combine historical P&L with active tickers
        all_tickers = set([p['Ticker'] for p in positions_data] + [h['Ticker'] for h in history_data])
        
        campaign_stats = []
        
        for ticker in all_tickers:
            # Get history for this ticker
            hist_trades = [h for h in history_data if h['Ticker'] == ticker]
            total_realized_pnl = sum([h['Profit'] for h in hist_trades])
            total_trades_count = len(hist_trades)
            
            # Get active contracts count (to estimate cost basis reduction per share)
            # Assumption: We normalize based on currently held contracts or default to 100 shares if 0
            active_contracts = sum([p['Contracts'] for p in positions_data if p['Ticker'] == ticker])
            share_divisor = max(active_contracts * 100, 100) # Avoid divide by zero, assume at least 1 lot
            
            cost_reduction_per_share = total_realized_pnl / share_divisor
            
            campaign_stats.append({
                "Ticker": ticker,
                "Realized P&L": total_realized_pnl,
                "Trades Closed": total_trades_count,
                "Current Active Contracts": active_contracts,
                "Cost Basis Reducer": cost_reduction_per_share
            })
            
        camp_df = pd.DataFrame(campaign_stats)
        
        # Display Cards
        cols = st.columns(3)
        for idx, row in camp_df.iterrows():
            with cols[idx % 3]:
                with st.container(border=True):
                    st.subheader(row['Ticker'])
                    st.metric("Total Realized P&L", f"${row['Realized P&L']:,.2f}")
                    
                    st.markdown(f"**Cost Basis Impact:**")
                    st.markdown(f"You have reduced your breakeven by **${row['Cost Basis Reducer']:.2f} per share**.")
                    
                    if row['Current Active Contracts'] > 0:
                        st.caption(f"Based on {row['Current Active Contracts']} active contracts.")
                    else:
                        st.caption("No active contracts. (Calculated on 100 shares)")

# ==========================
# TAB 3: HISTORY
# ==========================
with tab3:
    if not history_data:
        st.write("No closed trades yet.")
    else:
        hist_df = pd.DataFrame(history_data)
        
        # Monthly Chart
        st.subheader("💰 Monthly Income")
        hist_df['CloseDateDT'] = pd.to_datetime(hist_df['CloseDate'])
        hist_df['Month'] = hist_df['CloseDateDT'].dt.strftime('%Y-%m')
        monthly_pnl = hist_df.groupby('Month')['Profit'].sum()
        st.bar_chart(monthly_pnl, color="#4CAF50")

        # Table
        st.divider()
        st.dataframe(
            hist_df[['CloseDate', 'Ticker', 'Type', 'Strike', 'Reason', 'Profit']]
            .sort_values(by='CloseDate', ascending=False)
            .style.format({'Profit': '${:,.2f}', 'Strike': '${:.1f}'})
            .applymap(lambda v: 'color: green' if v > 0 else 'color: red', subset=['Profit']),
            use_container_width=True
        )