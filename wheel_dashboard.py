import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date
import uuid
import json
import re # Added for regex
from google.cloud import firestore
from google.oauth2 import service_account

# --- Page Configuration ---
st.set_page_config(page_title="Wheel Strategy Dashboard", layout="wide", page_icon="☸️")

# --- Styling ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 24px; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    /* Highlight the Annualized Return column */
    td:nth-child(6) { font-weight: bold; color: #4CAF50; }
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

        # 1. AUTO-FIX: Check if Streamlit already parsed it as a Dict (Native TOML)
        if isinstance(secret_val, dict):
            key_dict = secret_val
        else:
            # 2. AUTO-FIX: It's a string. Try to parse, handling common copy-paste corruptions.
            try:
                # Strict=False allows some control characters
                key_dict = json.loads(secret_val, strict=False)
            except json.JSONDecodeError:
                st.warning("JSON formatting issue detected. Attempting auto-repair...")
                # Remove non-printable control characters
                clean_val = re.sub(r'[\r\n]+', '', secret_val) 
                try:
                     key_dict = json.loads(clean_val, strict=False)
                except:
                     st.error("Could not auto-repair JSON. Please ensure 'private_key' is on one single line in your secrets.")
                     return None

        creds = service_account.Credentials.from_service_account_info(key_dict)
        db = firestore.Client(credentials=creds, project=key_dict["project_id"])
        return db
        
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

db = get_db()

# --- Database Helper Functions ---
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

# --- Market Data Helper ---
@st.cache_data(ttl=60)
def get_current_price(ticker):
    try:
        return yf.Ticker(ticker).fast_info.last_price
    except:
        return None

# --- Main App Logic ---

st.title("☸️ Wheel Strategy Manager Pro")

if not db:
    st.stop()

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
            st.toast("Trade Saved to Cloud! ☁️")
            st.rerun()

# --- Tabs ---
tab1, tab2 = st.tabs(["🚀 Active Positions", "📊 Analytics & History"])

# ==========================
# TAB 1: ACTIVE POSITIONS
# ==========================
with tab1:
    positions_data = load_collection('positions')
    
    if not positions_data:
        st.info("No active trades found in database.")
    else:
        df = pd.DataFrame(positions_data)
        
        # 1. Fetch Prices & Calc Status
        with st.spinner('Syncing with market...'):
            prices = {t: get_current_price(t) for t in df['Ticker'].unique()}
        
        def calc_row(row):
            curr = prices.get(row['Ticker'], 0) or 0
            
            # Basic Math
            dist = 0
            is_itm = False
            if curr > 0:
                dist = ((curr - row['Strike']) / curr) * 100
                is_itm = (row['Type'] == 'Put' and curr < row['Strike']) or \
                         (row['Type'] == 'Call' and curr > row['Strike'])
            
            # NEW: Efficiency Metrics (AROC)
            # Days to Expiration (DTE)
            try:
                exp_date = datetime.strptime(row['Expiry'], '%Y-%m-%d').date()
                dte = (exp_date - date.today()).days
                dte = max(1, dte) # Avoid divide by zero
            except:
                dte = 30
            
            collateral = row['Strike'] * 100 * row['Contracts']
            total_prem = row['Premium'] * 100 * row['Contracts']
            
            # (Premium / Collateral) * (365 / DTE)
            if collateral > 0:
                raw_return = total_prem / collateral
                annualized = raw_return * (365 / dte) * 100
            else:
                annualized = 0
            
            return pd.Series([curr, dist, is_itm, annualized, dte])

        stats = df.apply(calc_row, axis=1)
        stats.columns = ['Current Price', 'DistPct', 'Is ITM', 'AROC %', 'DTE']
        df = pd.concat([df, stats], axis=1)

        # 2. Metrics Header
        total_open_premium = (df['Premium'] * df['Contracts'] * 100).sum()
        risk_count = df['Is ITM'].sum()
        avg_aroc = df['AROC %'].mean()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Unrealized Premium", f"${total_open_premium:,.0f}")
        m2.metric("Portfolio efficiency (Avg AROC)", f"{avg_aroc:.1f}%")
        m3.metric("At Risk (ITM)", f"{risk_count}", delta_color="inverse", delta="Alert" if risk_count > 0 else "Safe")
        
        # 3. Active Table
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
        
        cols_to_show = ['Ticker', 'Type', 'Strike', 'Current Price', 'Dist to Strike', 'Annualized', 'DTE', 'Is ITM', 'DistPct']
        
        st.dataframe(
            display_df[cols_to_show]
            .style.apply(color_risk, axis=1)
            .format({'Strike': '${:.2f}', 'Current Price': '${:.2f}'}),
            use_container_width=True, 
            height=300,
            column_config={
                "DistPct": None,
                "Is ITM": st.column_config.CheckboxColumn("ITM Alert", disabled=True),
                "DTE": st.column_config.NumberColumn("Days Left", help="Days to Expiration"),
                "Annualized": st.column_config.TextColumn("Yield (Ann.)", help="Annualized Return on Capital if held to expiry")
            }
        )

        # 4. Action Center
        st.divider()
        st.subheader("🛠️ Manage Trades")
        
        position_options = {f"{row['Ticker']} ${row['Strike']} {row['Type']} (Exp: {row['Expiry']})": row['id'] for _, row in df.iterrows()}
        selected_label = st.selectbox("Select Position:", list(position_options.keys()))
        selected_id = position_options[selected_label]
        selected_row = df[df['id'] == selected_id].iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1: st.info(f"**Closing:** {selected_label}")
        with c2: close_reason = st.radio("Exit Reason", ["Buy to Close (BTC)", "Expired Worthless", "Assignment", "Delete (Mistake)"])
            
        with c3:
            if close_reason == "Buy to Close (BTC)":
                close_price = st.number_input("Price Paid to Close", min_value=0.0, value=0.05, step=0.01)
            else:
                close_price = 0.0
                st.write("**Price:** $0.00")
        
        if st.button("Confirm Action"):
            if close_reason == "Delete (Mistake)":
                delete_document('positions', selected_id)
                st.success("Deleted from Cloud!")
                st.rerun()
            else:
                profit = (selected_row['Premium'] - close_price) * 100 * selected_row['Contracts']
                
                history_record = selected_row.to_dict()
                history_record['CloseDate'] = str(date.today())
                history_record['ClosePrice'] = close_price
                history_record['Reason'] = close_reason
                history_record['Profit'] = profit
                
                add_document('history', history_record)
                delete_document('positions', selected_id)
                
                st.balloons()
                st.rerun()

# ==========================
# TAB 2: ANALYTICS & HISTORY
# ==========================
with tab2:
    history_data = load_collection('history')
    
    if not history_data:
        st.write("No closed trades yet.")
    else:
        hist_df = pd.DataFrame(history_data)
        
        # --- NEW: Charting Section ---
        st.subheader("💰 Monthly Income")
        
        # Convert CloseDate to datetime objects
        hist_df['CloseDateDT'] = pd.to_datetime(hist_df['CloseDate'])
        # Create a 'Month' column (e.g., "2024-02")
        hist_df['Month'] = hist_df['CloseDateDT'].dt.strftime('%Y-%m')
        
        # Group by Month and Sum Profit
        monthly_pnl = hist_df.groupby('Month')['Profit'].sum()
        
        # Show Chart
        st.bar_chart(monthly_pnl, color="#4CAF50")

        # --- Summary Metrics ---
        total_pnl = hist_df['Profit'].sum()
        win_rate = (hist_df[hist_df['Profit'] > 0].shape[0] / hist_df.shape[0]) * 100
        
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Lifetime P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
        h2.metric("Total Trades", len(hist_df))
        h3.metric("Win Rate", f"{win_rate:.1f}%")
        
        st.divider()
        st.subheader("📜 Transaction Log")
        st.dataframe(
            hist_df[['CloseDate', 'Ticker', 'Type', 'Strike', 'Reason', 'Profit']]
            .sort_values(by='CloseDate', ascending=False)
            .style.format({'Profit': '${:,.2f}', 'Strike': '${:.1f}'})
            .applymap(lambda v: 'color: green' if v > 0 else 'color: red', subset=['Profit']),
            use_container_width=True
        )