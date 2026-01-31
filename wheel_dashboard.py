import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date
import uuid

# --- Page Configuration ---
st.set_page_config(page_title="Wheel Strategy Dashboard", layout="wide", page_icon="☸️")

# --- Styling ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 24px; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data(ttl=60)
def get_current_price(ticker):
    try:
        return yf.Ticker(ticker).fast_info.last_price
    except:
        return None

# --- Session State Initialization ---
if 'positions' not in st.session_state:
    st.session_state.positions = [
        # Added IDs for unique identification
        {'id': str(uuid.uuid4()), 'Ticker': 'NVDA', 'Type': 'Put', 'Strike': 115.0, 'Premium': 2.50, 'Contracts': 1, 'Expiry': date(2026, 2, 20), 'OpenDate': date(2026, 1, 15)},
        {'id': str(uuid.uuid4()), 'Ticker': 'OKLO', 'Type': 'Put', 'Strike': 18.0, 'Premium': 0.80, 'Contracts': 5, 'Expiry': date(2026, 2, 20), 'OpenDate': date(2026, 1, 18)},
    ]

if 'history' not in st.session_state:
    st.session_state.history = []

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
            st.session_state.positions.append({
                'id': str(uuid.uuid4()),
                'Ticker': ticker_in,
                'Type': type_in,
                'Strike': strike_in,
                'Premium': premium_in,
                'Contracts': contracts_in,
                'Expiry': expiry_in,
                'OpenDate': date.today()
            })
            st.rerun()

# --- Main Dashboard ---
st.title("☸️ Wheel Strategy Manager")

# Tabs for Active vs History
tab1, tab2 = st.tabs(["🚀 Active Positions", "📜 Trade History"])

# ==========================
# TAB 1: ACTIVE POSITIONS
# ==========================
with tab1:
    if not st.session_state.positions:
        st.info("No active trades. Add one from the sidebar.")
    else:
        df = pd.DataFrame(st.session_state.positions)
        
        # 1. Fetch Prices & Calc Status
        with st.spinner('Updating market data...'):
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
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Unrealized Premium Held", f"${total_open_premium:,.0f}")
        m2.metric("Positions at Risk (ITM)", f"{risk_count}", delta_color="inverse", delta="Alert" if risk_count > 0 else "Safe")
        
        # 3. Active Table
        st.subheader("Portfolio Monitor")
        
        def color_risk(row):
            if row['Is ITM']: return ['background-color: #ffcccc; color: #8B0000'] * len(row)
            if row['Type'] == 'Put' and row['DistPct'] > 10: return ['background-color: #e6fffa; color: #004d40'] * len(row)
            return [''] * len(row)

        display_df = df.copy()
        display_df['Dist to Strike'] = display_df.apply(
            lambda x: f"{x['DistPct']:.1f}%" if x['Type']=='Put' else f"{abs(x['DistPct']):.1f}%", axis=1
        )
        
        st.dataframe(
            display_df[['Ticker', 'Type', 'Strike', 'Current Price', 'Dist to Strike', 'Premium', 'Expiry', 'Contracts']]
            .style.apply(color_risk, axis=1)
            .format({'Strike': '${:.2f}', 'Current Price': '${:.2f}', 'Premium': '${:.2f}'}),
            use_container_width=True, 
            height=300
        )

        # 4. Action Center (Close Trades)
        st.divider()
        st.subheader("🛠️ Manage Trades (Close / Edit)")
        
        # Create a selectbox with a readable label for each position
        position_options = {f"{row['Ticker']} ${row['Strike']} {row['Type']} (Exp: {row['Expiry']})": row['id'] for _, row in df.iterrows()}
        selected_label = st.selectbox("Select Position to Close/Manage:", list(position_options.keys()))
        selected_id = position_options[selected_label]
        
        # Get selected position data
        selected_row = df[df['id'] == selected_id].iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"**Closing:** {selected_label}")
        
        with c2:
            close_reason = st.radio("Exit Reason", ["Buy to Close (BTC)", "Expired Worthless", "Assignment", "Delete (Mistake)"])
            
        with c3:
            # Logic for closing price input
            if close_reason == "Buy to Close (BTC)":
                close_price = st.number_input("Price Paid to Close", min_value=0.0, value=0.05, step=0.01)
            elif close_reason == "Expired Worthless":
                close_price = 0.0
                st.write("**Price:** $0.00")
            elif close_reason == "Assignment":
                close_price = 0.0 # Technically you keep full premium, but take stock/cash risk
                st.write("**Price:** $0.00 (Premium kept)")
            else:
                close_price = 0.0
        
        if st.button("Confirm Exit / Update"):
            if close_reason == "Delete (Mistake)":
                # Remove from positions, don't add to history
                st.session_state.positions = [p for p in st.session_state.positions if p['id'] != selected_id]
                st.success("Deleted!")
                st.rerun()
            else:
                # Calculate P&L
                # Profit = (Entry Premium - Exit Price) * 100 * Contracts
                profit = (selected_row['Premium'] - close_price) * 100 * selected_row['Contracts']
                
                # Create History Record
                history_record = selected_row.to_dict()
                history_record['CloseDate'] = date.today()
                history_record['ClosePrice'] = close_price
                history_record['Reason'] = close_reason
                history_record['Profit'] = profit
                
                # Update State
                st.session_state.history.append(history_record)
                st.session_state.positions = [p for p in st.session_state.positions if p['id'] != selected_id]
                
                st.balloons()
                st.rerun()

# ==========================
# TAB 2: TRADE HISTORY
# ==========================
with tab2:
    if not st.session_state.history:
        st.write("No closed trades yet.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        
        # Summary Metrics
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