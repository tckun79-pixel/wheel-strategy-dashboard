import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date

# --- Page Configuration ---
st.set_page_config(
    page_title="Wheel Strategy Dashboard", 
    layout="wide", 
    page_icon="☸️"
)

# --- Styling (CSS) ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
    div.stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data(ttl=60)
def get_current_price(ticker):
    """Fetches the latest price using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info.last_price
        return price
    except Exception as e:
        return None

def calculate_status(row, current_price):
    """Calculates P&L, Distance to Strike, and Moneyness Status."""
    if current_price is None:
        return pd.Series([0, 0, False, 0])
    
    distance = ((current_price - row['Strike']) / current_price) * 100
    
    # ITM / OTM Logic
    if row['Type'] == 'Put':
        # Selling Puts: ITM if Price < Strike (Risk of assignment)
        is_itm = current_price < row['Strike']
        distance_str = f"{distance:.2f}%"
    else: 
        # Selling Calls: ITM if Price > Strike (Risk of shares called away)
        is_itm = current_price > row['Strike']
        distance_str = f"+{abs(distance):.2f}%" if distance > 0 else f"{distance:.2f}%"

    # Breakeven Calculation
    if row['Type'] == 'Put':
        breakeven = row['Strike'] - row['Premium']
    else:
        breakeven = row['Strike'] + row['Premium']

    return pd.Series([current_price, distance_str, is_itm, breakeven])

# --- Sidebar: Portfolio Input ---
st.sidebar.header("📝 Position Manager")

if 'positions' not in st.session_state:
    st.session_state.positions = [
        {'Ticker': 'NVDA', 'Type': 'Put', 'Strike': 115.0, 'Premium': 2.50, 'Contracts': 1, 'Expiry': date(2026, 2, 20)},
        {'Ticker': 'OKLO', 'Type': 'Put', 'Strike': 18.0, 'Premium': 0.80, 'Contracts': 5, 'Expiry': date(2026, 2, 20)},
        {'Ticker': 'CELH', 'Type': 'Put', 'Strike': 45.0, 'Premium': 1.20, 'Contracts': 2, 'Expiry': date(2026, 2, 20)},
        {'Ticker': 'HIMS', 'Type': 'Put', 'Strike': 16.0, 'Premium': 0.50, 'Contracts': 3, 'Expiry': date(2026, 2, 20)},
        {'Ticker': 'RKLB', 'Type': 'Call', 'Strike': 15.0, 'Premium': 0.40, 'Contracts': 10, 'Expiry': date(2026, 2, 13)}, 
    ]

with st.sidebar.form("add_position_form", clear_on_submit=True):
    st.subheader("Add New Trade")
    col1, col2 = st.columns(2)
    with col1:
        ticker_in = st.text_input("Ticker").upper()
        strike_in = st.number_input("Strike Price", min_value=0.0, step=0.5)
        contracts_in = st.number_input("Contracts", min_value=1, step=1)
    with col2:
        type_in = st.selectbox("Type", ["Put", "Call"])
        premium_in = st.number_input("Premium", min_value=0.0, step=0.01)
        expiry_in = st.date_input("Expiry")
    
    submitted = st.form_submit_button("Add Position")
    
    if submitted and ticker_in:
        st.session_state.positions.append({
            'Ticker': ticker_in,
            'Type': type_in,
            'Strike': strike_in,
            'Premium': premium_in,
            'Contracts': contracts_in,
            'Expiry': expiry_in
        })
        st.rerun()

# --- Main Dashboard ---
st.title("☸️ Wheel Strategy Monitor")
st.markdown(f"**Active Positions:** {len(st.session_state.positions)} | **Data Source:** Live Market Data (Yahoo Finance)")

if not st.session_state.positions:
    st.info("No positions added. Use the sidebar to add your Wheel trades.")
else:
    df = pd.DataFrame(st.session_state.positions)

    # 1. Fetch live data
    with st.spinner('Fetching live prices...'):
        unique_tickers = df['Ticker'].unique()
        current_prices = {t: get_current_price(t) for t in unique_tickers}

    # 2. Apply calculations
    results = df.apply(lambda x: calculate_status(x, current_prices.get(x['Ticker'])), axis=1)
    results.columns = ['Current Price', 'Dist to Strike', 'Is ITM', 'Breakeven']
    df = pd.concat([df, results], axis=1)

    # 3. Metrics
    total_premium = (df['Premium'] * df['Contracts'] * 100).sum()
    m1, m2 = st.columns(2)
    m1.metric("Total Premium Collected", f"${total_premium:,.2f}")
    
    itm_count = df['Is ITM'].sum()
    m2.metric("Positions At Risk (ITM)", f"{itm_count}", delta_color="inverse", delta=f"{itm_count} Alert" if itm_count > 0 else "Safe")
    
    # 4. Main Data Table
    st.subheader("Portfolio Status")
    
    def highlight_risk(row):
        # Red background if In The Money
        if row['Is ITM']:
            return ['background-color: #ffcccc; color: #8B0000'] * len(row)
        
        # Greenish for very safe OTM (>10% safety) for Puts
        try:
            dist = float(row['Dist to Strike'].strip('%').replace('+', ''))
            if row['Type'] == 'Put' and dist > 10:
                return ['background-color: #e6fffa; color: #004d40'] * len(row)
        except:
            pass
        return [''] * len(row)

    # FIXED: Added 'Is ITM' to this list so the highlighter can find it
    display_cols = ['Ticker', 'Type', 'Strike', 'Current Price', 'Dist to Strike', 'Is ITM', 'Breakeven', 'Expiry', 'Contracts']
    
    st.dataframe(
        df[display_cols].style.apply(highlight_risk, axis=1)
        .format({
            'Strike': '${:.2f}',
            'Current Price': '${:.2f}',
            'Breakeven': '${:.2f}',
            'Premium': '${:.2f}'
        }),
        use_container_width=True,
        height=400
    )

    # 5. Visual Stage Indicators
    st.subheader("Wheel Cycle Visualization")
    
    grid_cols = st.columns(3)
    
    for idx, ticker in enumerate(unique_tickers):
        ticker_rows = df[df['Ticker'] == ticker]
        
        for _, row in ticker_rows.iterrows():
            with grid_cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"**{row['Ticker']}** ({row['Type']} @ ${row['Strike']})")
                    curr = row['Current Price']
                    strike = row['Strike']
                    
                    if row['Type'] == 'Put':
                        if curr > strike:
                            buffer_pct = (curr - strike) / curr
                            st.progress(min(buffer_pct * 5, 1.0), text=f"Safety Buffer: {buffer_pct:.1%}")
                            st.caption("🟢 Selling Puts (OTM)")
                        else:
                            st.progress(1.0, text="ITM - ASSIGNMENT RISK")
                            st.error("🚨 Price below Strike")
                            
                    elif row['Type'] == 'Call':
                        if curr < strike:
                            upside_pct = (strike - curr) / curr
                            st.progress(min(upside_pct * 5, 1.0), text=f"Upside to Strike: {upside_pct:.1%}")
                            st.caption("🔵 Covered Call (OTM)")
                        else:
                            st.progress(1.0, text="ITM - MAX PROFIT REACHED")
                            st.warning("⚠️ Shares likely called away")

# --- Footer ---
st.divider()
if st.button("⚠️ Reset Portfolio Data"):
    st.session_state.positions = []
    st.rerun()