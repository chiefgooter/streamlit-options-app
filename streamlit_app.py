import streamlit as st
import pandas as pd
import numpy as np
import time

# --- MOCK DATA SIMULATION (Translated from JavaScript) ---

def generate_mock_data(ticker):
    """
    Generates mock options chain data for a given ticker, simulating a real API response.
    """
    try:
        # Simulate an error condition
        if ticker.upper() == 'ERROR':
            raise ValueError("API Limit Exceeded or Connection Failure.")

        base_price = 175.50
        expiration = pd.to_datetime(pd.Timestamp.now() + pd.Timedelta(days=30)).strftime('%Y-%m-%d')

        # Stock Summary Data
        stock_summary = {
            'ticker': ticker.upper(),
            'lastPrice': base_price,
            # Generate random values for simulation
            'changePercent': round(np.random.uniform(-2.5, 2.5), 2),
            'volume': round(np.random.uniform(10, 60), 1),
            'impliedVolatility': round(np.random.uniform(0.2, 0.4) * 100, 2), # Display as percentage
            'expiration': expiration
        }

        options_data = []
        strikes = [base_price + i * 5 for i in range(-5, 6)]

        for strike in strikes:
            strike_price = round(strike, 2)

            # Generate mock contract data
            def create_contract(is_call):
                bid = strike * np.random.uniform(0.01, 0.05)
                ask = bid + np.random.uniform(0.1, 0.5)
                return {
                    'Bid Size': int(np.random.randint(50, 550)),
                    'Bid': round(bid, 2),
                    'Ask': round(ask, 2),
                    'Ask Size': int(np.random.randint(50, 550)),
                    'Open Interest': int(np.random.randint(10000, 60000)),
                    'Volume': int(np.random.randint(0, 10000)),
                }

            call = create_contract(True)
            put = create_contract(False)

            options_data.append({
                'Strike': strike_price,
                'CALL_Volume': call['Volume'],
                'CALL_Open Interest': call['Open Interest'],
                'CALL_Bid Size': call['Bid Size'],
                'CALL_Bid': call['Bid'],
                'CALL_Ask': call['Ask'],
                'CALL_Ask Size': call['Ask Size'],
                'PUT_Bid Size': put['Bid Size'],
                'PUT_Bid': put['Bid'],
                'PUT_Ask': put['Ask'],
                'PUT_Ask Size': put['Ask Size'],
                'PUT_Open Interest': put['Open Interest'],
                'PUT_Volume': put['Volume'],
                # Flags for conditional styling in the Streamlit dataframe
                'Is_ITM_Call': strike < stock_summary['lastPrice'],
                'Is_ITM_Put': strike > stock_summary['lastPrice'],
            })
        
        df = pd.DataFrame(options_data)
        return stock_summary, df

    except ValueError as e:
        st.error(f"Data Error: {e}")
        return None, None

# --- STREAMLIT UI LAYOUT AND LOGIC ---

# Set wide layout and title
st.set_page_config(layout="wide", page_title="Options Flow & Chain Analyzer")

st.title("Options Flow & Chain Analyzer")
st.markdown("Real-time mock data display for bids, asks, volume, and open interest.")

# Initialize session state for the ticker and data, mimicking the JavaScript variables
if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None
if 'data_chain' not in st.session_state:
    st.session_state.data_chain = None

# Input and Button in a container
with st.container():
    col1, col2 = st.columns([3, 1])

    with col1:
        st.session_state.ticker = st.text_input(
            "Enter Stock Ticker",
            value=st.session_state.ticker,
            placeholder="e.g., AAPL",
            key="ticker_input_box"
        )
    with col2:
        # Add some vertical space to align the button
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Get Options Data", type="primary", use_container_width=True):
            # Fetch data when button is clicked
            st.session_state.data_summary, st.session_state.data_chain = generate_mock_data(st.session_state.ticker)


# --- DISPLAY RESULTS ---

# This handles the initial load and updates when the button is clicked
if st.session_state.data_summary is not None:
    summary = st.session_state.data_summary
    df = st.session_state.data_chain

    # 1. Stock Summary/Quote
    st.subheader(f"{summary['ticker']} Quote")
    
    # Use columns for the summary metrics, similar to the grid layout in HTML
    col_price, col_change, col_volume, col_iv = st.columns(4)

    # Determine color for change percent
    change_color = "inverse" if summary['changePercent'] >= 0 else "off"

    col_price.metric("Last Price", f"${summary['lastPrice']:.2f}")
    col_change.metric("Change (%)", f"{summary['changePercent']:.2f}%", delta_color=change_color)
    col_volume.metric("Volume", f"{summary['volume']:.1f}M")
    col_iv.metric("Implied Volatility (IV)", f"{summary['impliedVolatility']:.2f}%")

    st.markdown("---")

    # 2. Options Chain Table
    st.subheader(f"Options Chain (Exp: {summary['expiration']})")
    st.write("Note: Bid/Ask Size represents the number of contracts available.")


    # Custom function to apply conditional formatting based on ITM/OTM
    def highlight_itm(s):
        is_call_itm = s['Is_ITM_Call']
        is_put_itm = s['Is_ITM_Put']
        
        # Apply a light green background for ITM calls
        if is_call_itm:
            # Highlight the call columns (Volume, OI, Sizes, Bid, Ask)
            call_cols = [f'CALL_{col}' for col in ['Volume', 'Open Interest', 'Bid Size', 'Bid', 'Ask', 'Ask Size']]
            styles = [f'background-color: #ecfdf5' for _ in call_cols] # Tailwind green-50
            
            # Apply light gray to strike
            styles.append(f'background-color: #e5e7eb') # Tailwind gray-200
            
            # Put columns are OTM here
            put_cols = [f'PUT_{col}' for col in ['Bid Size', 'Bid', 'Ask', 'Ask Size', 'Open Interest', 'Volume']]
            styles.extend([''] * len(put_cols))
            
        # Apply a light red background for ITM puts
        elif is_put_itm:
            # Call columns are OTM here
            call_cols = [f'CALL_{col}' for col in ['Volume', 'Open Interest', 'Bid Size', 'Bid', 'Ask', 'Ask Size']]
            styles = [''] * len(call_cols)
            
            # Apply light gray to strike
            styles.append(f'background-color: #e5e7eb') # Tailwind gray-200
            
            # Highlight the put columns (Sizes, Bid, Ask, OI, Volume)
            put_cols = [f'PUT_{col}' for col in ['Bid Size', 'Bid', 'Ask', 'Ask Size', 'Open Interest', 'Volume']]
            styles.extend([f'background-color: #fef2f2' for _ in put_cols]) # Tailwind red-50
            
        else:
            # At-The-Money or OTM (no specific highlight)
            styles = [''] * len(s)
            
        return styles

    # Drop the temporary ITM columns before styling/display
    display_df = df.drop(columns=['Is_ITM_Call', 'Is_ITM_Put'])

    # Apply the styling (Note: Streamlit styling is limited compared to custom CSS)
    styled_df = display_df.style.apply(highlight_itm, axis=1)

    # Use markdown to create the grouped headers (Calls/Puts)
    st.markdown("""
        <style>
        .stDataFrame table {
            border-collapse: separate;
        }
        .stDataFrame th {
            text-align: center !important;
        }
        .stDataFrame table th:nth-child(2), 
        .stDataFrame table th:nth-child(8) {
            border-left: 2px solid #ccc; /* Separator for Strike column */
        }
        .stDataFrame .col_heading.level0 {
            background-color: #f0f0f0;
            font-weight: bold;
        }
        </style>
        <div style='text-align: center; display: flex; width: 100%; border-radius: 0.5rem; overflow: hidden;'>
            <div style='flex: 6; background-color: #d1d5db; color: #1f2937; padding: 5px; border-right: 1px solid #9ca3af;'>CALLS</div>
            <div style='flex: 1; background-color: #9ca3af; color: #1f2937; padding: 5px; font-weight: bold;'>STRIKE</div>
            <div style='flex: 6; background-color: #d1d5db; color: #1f2937; padding: 5px; border-left: 1px solid #9ca3af;'>PUTS</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Display the styled dataframe
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    # --- Real-Time Polling Simulation ---
    # Streamlit doesn't automatically poll like JS, but we can simulate it with a loop or st.rerun
    st.warning("Note: In a real Streamlit app, you would need to use `st.rerun()` or external tools for true real-time streaming updates.")


# Optional: Automatically run the initial fetch when the app starts
if st.session_state.data_summary is None:
     st.session_state.data_summary, st.session_state.data_chain = generate_mock_data(st.session_state.ticker)
     st.rerun() # Rerun to display the initial data
