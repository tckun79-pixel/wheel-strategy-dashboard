import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date
import uuid
import json
from google.cloud import firestore
from google.oauth2 import service_account

# --- Page Configuration ---
st.set_page_config(page_title="Wheel Strategy Dashboard", layout="wide", page_icon="☸️")

# --- Styling ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 24px; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- Firestore Connection (DEBUG VERSION) ---
@st.cache_resource
def get_db():
    """Initializes the Firestore client using Streamlit Secrets."""
    try:
        # 1. Debug: Check if the key exists at all
        if "textkey" not in st.secrets:
            st.error("DEBUG: Streamlit cannot find a secret named 'textkey'. Check your TOML header.")
            return None

        # 2. Debug: Try to parse the JSON
        try:
            key_dict = json.loads(st.secrets["textkey"])
        except json.JSONDecodeError as e:
            st.error(f"DEBUG: JSON Parsing Failed. Your JSON inside the quotes might be corrupted. Error: {e}")
            return None

        # 3. Debug: Try to connect to Google
        creds = service_account.Credentials.from_service_account_info(key_dict)
        db = firestore.Client(credentials=creds, project=key_dict["project_id"])
        return db
        
    except Exception as e:
        st.error(f"DEBUG: Unexpected Error: {e}")
        return None

db = get_db()

# --- Database Helper Functions ---
def load_collection(collection_name):
    """Fetches all documents from a Firestore collection."""
    if not db: return []
    docs = db.collection(collection_name).stream()
    items = []
    for doc in docs:
        item = doc.to_dict()
        item['id'] = doc.id # Ensure ID is attached
        items.append(item)
    return items

def add_document(collection_name, data):
    """Adds a new document to Firestore."""
    if not db: return
    # Use the 'id' field as the document key if it exists, otherwise auto-generate
    doc_id = data.get('id', str(uuid.uuid4()))
    db.collection(collection_name).document(doc_id).set(data)

def delete_document(collection_name, doc_id):
    """Deletes a document from Firestore."""
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

st.title("☸️ Wheel Strategy Manager (Cloud Synced)")

# CHECK: Database Connection
if not db:
    st.error("⚠️ Database Not Connected")
    st.warning("To use cloud storage, you must set up your Firebase credentials in Streamlit Secrets.")
    st.info("Check `SETUP_INSTRUCTIONS.md` for how to generate your `service-account.json`.")
    st.stop() # Stop execution if no DB

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
                # Convert dates to string for JSON/Firestore compatibility
                'Expiry': str(expiry_in), 
                'OpenDate': str(date.today())
            }
            add_document('positions', new_trade)
            st.toast("Trade Saved to Cloud! ☁️")
            st.rerun()

# --- Tabs ---
tab1, tab2 = st.tabs(["🚀 Active Positions", "📜 Trade History"])

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
            if curr == 0: return pd.Series([0, 0, False])
            
            # Distance & ITM
            dist = ((curr - row['Strike']) / curr) * 100
            is_itm = (row['Type'] == 'Put' and curr < row['Strike']) or \
                     (row['Type'] == 'Call' and curr > row['Strike'])
            
            return pd.Series([curr, dist, is_itm])

        stats = df.apply(calc_row, axis=1)
        stats.columns = ['Current Price', 'DistPct', 'Is ITM']
        df = pd.concat([df, stats], axis=1)

        # 2. Metrics Header
        total_open_premium = (df['Premium'] * df['Contracts'] * 100).sum()
        risk_count = df['Is ITM'].sum()
        
        m1, m2 = st.columns(2)
        m1.metric("Unrealized Premium Held", f"${total_open_premium:,.0f}")
        m2.metric("Positions at Risk (ITM)", f"{risk_count}", delta_color="inverse", delta="Alert" if risk_count > 0 else "Safe")
        
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
        
        cols_to_show = ['Ticker', 'Type', 'Strike', 'Current Price', 'Dist to Strike', 'Premium', 'Expiry', 'Contracts', 'Is ITM', 'DistPct']
        
        st.dataframe(
            display_df[cols_to_show]
            .style.apply(color_risk, axis=1)
            .format({'Strike': '${:.2f}', 'Current Price': '${:.2f}', 'Premium': '${:.2f}'}),
            use_container_width=True, 
            height=300,
            column_config={
                "DistPct": None,
                "Is ITM": st.column_config.CheckboxColumn("ITM Alert", disabled=True) 
            }
        )

        # 4. Action Center (Close Trades)
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
                
                # Transactional: Add to History -> Delete from Active
                add_document('history', history_record)
                delete_document('positions', selected_id)
                
                st.balloons()
                st.rerun()

# ==========================
# TAB 2: TRADE HISTORY
# ==========================
with tab2:
    history_data = load_collection('history')
    
    if not history_data:
        st.write("No closed trades yet.")
    else:
        hist_df = pd.DataFrame(history_data)
        
        total_pnl = hist_df['Profit'].sum()
        win_rate = (hist_df[hist_df['Profit'] > 0].shape[0] / hist_df.shape[0]) * 100
        
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Realized P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
        h2.metric("Trades Closed", len(hist_df))
        h3.metric("Win Rate", f"{win_rate:.1f}%")
        
        st.dataframe(
            hist_df[['CloseDate', 'Ticker', 'Type', 'Strike', 'Reason', 'Profit']]
            .sort_values(by='CloseDate', ascending=False)
            .style.format({'Profit': '${:,.2f}', 'Strike': '${:.1f}'})
            .applymap(lambda v: 'color: green' if v > 0 else 'color: red', subset=['Profit']),
            use_container_width=True
        )