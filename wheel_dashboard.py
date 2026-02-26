import streamlit as st
import pandas as pd
import yfinance as yf
from openai import OpenAI
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
    """Checks if the user is logged in."""
    if st.session_state.authenticated:
        return True
    return False

# --- Firestore Connection ---
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
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Active Positions", "📉 Campaign Analysis", "📜 Trade History", "🤖 AI Assistant"])

# ==========================
# TAB 1: ACTIVE POSITIONS
# ==========================
with tab1:
    
    # ---------------------------
    # 1. PRE-CALCULATE METRICS
    # ---------------------------
    total_stock_unrealized = 0
    total_option_premium = 0
    option_risk_count = 0
    
    # Calculate Stock P&L first (if data exists)
    h_df = pd.DataFrame()
    if holdings_data:
        h_df = pd.DataFrame(holdings_data)
        # Fetch prices for Stocks
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
        # If no stocks, still need prices for options
        with st.spinner('Syncing market data...'):
            unique_tickers = list(set([p['Ticker'] for p in positions_data]))
            prices = {t: get_current_price(t) for t in unique_tickers}

    # Calculate Option Metrics (if data exists)
    df = pd.DataFrame()
    if positions_data:
        df = pd.DataFrame(positions_data)
        total_option_premium = (df['Premium'] * df['Contracts'] * 100).sum()
        
        # Determine risk count
        for _, row in df.iterrows():
            curr = prices.get(row['Ticker'], 0) or 0
            is_itm = (row['Type'] == 'Put' and curr < row['Strike']) or \
                     (row['Type'] == 'Call' and curr > row['Strike'])
            if is_itm: option_risk_count += 1

    # ---------------------------
    # 2. DISPLAY TOP METRICS
    # ---------------------------
    m1, m2, m3 = st.columns(3)
    
    # Metric 1: Stock Inventory P&L (The new request)
    m1.metric(
        "Inventory Unrealized P&L", 
        f"${total_stock_unrealized:,.2f}", 
        delta_color="normal" # Green is good, Red is bad automatically
    )
    
    # Metric 2: Option Premium (Max Potential Profit)
    m2.metric(
        "Unrealized Premium Held", 
        f"${total_option_premium:,.0f}",
        help="This is the max profit if all options expire worthless."
    )
    
    # Metric 3: Risk Alert
    m3.metric(
        "Options At Risk (ITM)", 
        f"{option_risk_count}", 
        delta_color="inverse", 
        delta="Alert" if option_risk_count > 0 else "Safe"
    )
    
    st.divider()

    # ---------------------------
    # 3. STOCK INVENTORY TABLE
    # ---------------------------
    if not h_df.empty:
        st.subheader("📦 Stock Inventory (Assigned)")
        
        st.dataframe(
            h_df[['Ticker', 'Shares', 'CostPrice', 'Current Price', 'Market Value', 'Unrealized P&L', 'Return %']]
            .style.format({'CostPrice': '${:.2f}', 'Current Price': '${:.2f}', 'Market Value': '${:,.0f}', 'Unrealized P&L': '${:,.0f}', 'Return %': '{:.2f}%'})
            .applymap(lambda v: 'color: green' if v > 0 else 'color: red', subset=['Unrealized P&L']),
            use_container_width=True,
            height=150
        )
        st.divider()

    # ---------------------------
    # 4. ACTIVE OPTIONS TABLE
    # ---------------------------
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
        
        # Drop existing columns if they exist to avoid duplicates before concat
        df = df.drop(columns=[c for c in stats.columns if c in df.columns])
        df = pd.concat([df, stats], axis=1)

        def color_risk(row):
            try:
                if row['Is ITM']: return ['background-color: #ffcccc; color: #8B0000'] * len(row)
                if row['Type'] == 'Put' and row['DistPct'] > 10: return ['background-color: #e6fffa; color: #004d40'] * len(row)
            except: pass
            return [''] * len(row)

        display_df = df.copy()
        
        # Ensure DistPct and AROC % are numeric to avoid TypeError during formatting
        # Using .iloc[:, 0] in case of any remaining duplicates, though drop(columns) should handle it
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

    # ---------------------------
    # 5. ACTION CENTER
    # ---------------------------
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
                # Save to history
                hist_entry = sel.copy()
                hist_entry['CloseDate'] = str(date.today())
                hist_entry['Result'] = "Expired"
                hist_entry['Profit'] = sel['Premium'] * 100 * sel['Contracts']
                add_document('history', hist_entry)
                # Remove from active
                delete_document('positions', sel['id'])
                st.success("Moved to History!")
                st.rerun()
                
            if c2.button("Mark as Assigned"):
                # Save to history
                hist_entry = sel.copy()
                hist_entry['CloseDate'] = str(date.today())
                hist_entry['Result'] = "Assigned"
                hist_entry['Profit'] = 0 # Assignment usually neutral in premium terms
                add_document('history', hist_entry)
                
                # If Put, add to holdings
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
                
                # Remove from active
                delete_document('positions', sel['id'])
                st.success("Assignment Processed!")
                st.rerun()

            if c3.button("Delete (Error Entry)"):
                delete_document('positions', sel['id'])
                st.warning("Deleted.")
                st.rerun()

            # Early Close (BTC/STC) Section
            with c4.expander("💸 Close Early"):
                close_price = st.number_input("Price Paid/Received (Per Share)", min_value=0.0, step=0.01, help="The price you paid to buy back (BTC) or received to sell (STC) the option.")
                if st.button("Execute Early Close"):
                    hist_entry = sel.copy()
                    hist_entry['CloseDate'] = str(date.today())
                    hist_entry['Result'] = "Closed Early"
                    # Profit = (Original Premium - Close Price) * 100 * Contracts
                    hist_entry['Profit'] = (sel['Premium'] - close_price) * 100 * sel['Contracts']
                    add_document('history', hist_entry)
                    delete_document('positions', sel['id'])
                    st.success(f"Trade closed! Profit/Loss: ${hist_entry['Profit']:,.2f}")
                    st.rerun()

            # Rollover Section
            st.markdown("---")
            with st.expander(f"🔄 Rollover: {selected_label}"):
                st.write("Close this position and open a new one.")
                r_col1, r_col2 = st.columns(2)
                
                # New Position Details
                new_strike = r_col1.number_input("New Strike", value=float(sel['Strike']), step=0.5, key="roll_strike")
                new_expiry = r_col2.date_input("New Expiry", value=date.today() + timedelta(days=7), key="roll_expiry")
                
                # Financials
                close_cost = r_col1.number_input("Cost to Close (Total)", min_value=0.0, step=1.0, help="Total amount paid to buy back/close the current position.")
                new_credit = r_col2.number_input("New Premium (Per Share)", min_value=0.0, step=0.01, value=float(sel['Premium']), help="Premium received for the new position.")
                
                net_credit = (new_credit * 100 * sel['Contracts']) - close_cost
                st.info(f"Net Rollover Credit: ${net_credit:,.2f}")
                
                if st.button("Execute Rollover"):
                    # 1. Close current position
                    hist_entry = sel.copy()
                    hist_entry['CloseDate'] = str(date.today())
                    hist_entry['Result'] = "Rolled"
                    # Profit for the closed leg is (Original Premium * 100 * Contracts) - Close Cost
                    hist_entry['Profit'] = (sel['Premium'] * 100 * sel['Contracts']) - close_cost
                    add_document('history', hist_entry)
                    delete_document('positions', sel['id'])
                    
                    # 2. Open new position
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
        
        # Ensure Profit is numeric
        hist_df['Profit'] = pd.to_numeric(hist_df['Profit'], errors='coerce').fillna(0)
        
        # Group by ticker and aggregate metrics
        summary = hist_df.groupby('Ticker').agg({
            'Profit': ['sum', 'count'],
            'Result': [
                lambda x: (x == 'Expired').sum(),
                lambda x: (x == 'Rolled').sum(),
                lambda x: (x == 'Assigned').sum()
            ]
        }).reset_index()
        
        # Flatten multi-index columns
        summary.columns = ['Ticker', 'Total Realized P&L', 'Trade Count', 'Expired', 'Rolled', 'Assigned']
        
        # Sort by P&L descending
        summary = summary.sort_values('Total Realized P&L', ascending=False)
        
        # Overall Stats
        total_p_l = summary['Total Realized P&L'].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Realized P&L", f"${total_p_l:,.2f}")
        c2.metric("Winning Trades", f"{summary['Expired'].sum()}")
        c3.metric("Total Trades", f"{summary['Trade Count'].sum()}")
        
        st.markdown("### 🏷️ Performance by Ticker")
        st.dataframe(
            summary.style.format({'Total Realized P&L': '${:,.2f}'})
            .applymap(lambda v: 'color: green' if v > 0 else 'color: red' if v < 0 else '', subset=['Total Realized P&L']),
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
# TAB 4: AI ASSISTANT
# ==========================
with tab4:
    st.subheader("🤖 AI Portfolio Assistant")
    st.write("Ask questions about your holdings, active positions, or trade history.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know about your portfolio?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                # Check for API key in secrets
                if "OPENAI_API_KEY" not in st.secrets:
                    st.error("Missing OpenAI API Key. Please add 'OPENAI_API_KEY' to your Streamlit Secrets.")
                    st.stop()
                
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                
                # Prepare context
                context = f"""
                You are a professional options trading assistant for a user named CK. 
                CK is a test engineer with 10+ years of trading experience.
                
                Current Portfolio Data:
                - Active Option Positions: {json.dumps(positions_data, indent=2)}
                - Stock Holdings (Assigned): {json.dumps(holdings_data, indent=2)}
                - Trade History: {json.dumps(history_data[:20], indent=2)} (Last 20 trades)
                
                User Profile:
                - Based in Singapore.
                - Interested in IT, reading, history, and travel.
                - Vegetarian (no onion/garlic).
                
                Instructions:
                - Be objective and avoid hype.
                - Use the provided data to answer specific questions about positions.
                - If asked about performance, refer to the realized P&L and active positions.
                - Keep responses professional and concise.
                """
                
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": context},
                        *st.session_state.messages
                    ]
                )
                full_response = response.choices[0].message.content
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"AI Error: {e}")
