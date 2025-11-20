import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time

# --- DATA FETCHING WITH YFINANCE ---

def fetch_options_data(ticker_symbol):
    """
    Fetches real stock data and the options chain using the yfinance library.
    
    Args:
        ticker_symbol (str): The stock ticker (e.g., 'AAPL').
        
    Returns:
        tuple: (stock_summary_dict, options_chain_df) or (None, None) on error.
    """
    try:
        if not ticker_symbol:
            st.error("Please enter a valid stock ticker.")
            return None, None

        # 1. Initialize Ticker Object
        ticker = yf.Ticker(ticker_symbol)

        # 2. Get Stock Summary
        info = ticker.info
        last_price = info.get('currentPrice') or info.get('previousClose')
        
        if last_price is None:
            st.error(f"Could not find current price data for ticker: {ticker_symbol.upper()}. Check the ticker symbol.")
            return None, None
            
        # Calculate change percent
        prev_close = info.get('previousClose', last_price)
        change_percent = ((last_price - prev_close) / prev_close) * 100 if prev_close and prev_close != 0 else 0

        stock_summary = {
            'ticker': ticker_symbol.upper(),
            'lastPrice': last_price,
            'changePercent': round(change_percent, 2),
            'volume': round(info.get('volume', 0) / 1000000, 1), # Convert to Millions
            'impliedVolatility': info.get('impliedVolatility', 0) * 100, # Display as percentage
            'expiration': None # To be updated after fetching chain
        }

        # 3. Get Options Chain
        expirations = ticker.options
        if not expirations:
            st.warning(f"No options chain found for {ticker_symbol.upper()}.")
            return stock_summary, None

        # Use the first available expiration date
        selected_expiration = expirations[0]
        stock_summary['expiration'] = selected_expiration
        
        # Fetch the options chain data for that date
        option_chain = ticker.option_chain(selected_expiration)
        
        # 4. Process Calls and Puts
        calls = option_chain.calls
        puts = option_chain.puts
        
        # 5. Merge DataFrames on 'strike'
        # Select and rename columns for clarity and consistency
        call_cols = calls[['strike', 'volume', 'openInterest', 'bid', 'ask']].copy()
        call_cols.rename(columns={
            'strike': 'Strike',
            'volume': 'CALL_Volume',
            'openInterest': 'CALL_Open Interest',
            'bid': 'CALL_Bid',
            'ask': 'CALL_Ask'
        }, inplace=True)
        
        put_cols = puts[['strike', 'volume', 'openInterest', 'bid', 'ask']].copy()
        put_cols.rename(columns={
            'strike': 'Strike',
            'volume': 'PUT_Volume',
            'openInterest': 'PUT_Open Interest',
            'bid': 'PUT_Bid',
            'ask': 'PUT_Ask'
        }, inplace=True)

        # Merge Calls and Puts on the Strike price
        # Using an outer merge to ensure all strikes are present
        merged_df = pd.merge(call_cols, put_cols, on='Strike', how='outer')
        
        # Add ITM/OTM Flags for conditional styling
        merged_df['Is_ITM_Call'] = merged_df['Strike'] < last_price
        merged_df['Is_ITM_Put'] = merged_df['Strike'] > last_price
        
        # Sort by Strike price
        merged_df.sort_values(by='Strike', inplace=True)
        merged_df.fillna(0, inplace=True) # Fill NaNs (if any) with 0 for cleaner display
        
        return stock_summary, merged_df

    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        st.warning("Please ensure the ticker symbol is correct and you are connected to the internet.")
        return None, None

# --- STREAMLIT UI LAYOUT AND LOGIC ---

# Set wide layout and title
st.set_page_config(layout="wide", page_title="Options Flow & Chain Analyzer")

st.title("Options Flow & Chain Analyzer (YFinance Data)")
st.markdown("Delayed or end-of-day options data provided by `yfinance` for testing.")

# Initialize session state for the ticker and data
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
        ).upper().strip() # Ensure ticker is clean and uppercase
    with col2:
        # Add some vertical space to align the button
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Fetch Options Data", type="primary", use_container_width=True):
            # Show a spinner while fetching data
            with st.spinner(f"Fetching data for {st.session_state.ticker}..."):
                st.session_state.data_summary, st.session_state.data_chain = fetch_options_data(st.session_state.ticker)
            st.rerun() # Rerun to display the new data


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
    # NOTE: delta_color="inverse" means green for positive, red for negative
    change_color = "inverse" 

    col_price.metric("Last Price", f"${summary['lastPrice']:.2f}")
    col_change.metric("Change (%)", f"{summary['changePercent']:.2f}%", delta=f"{summary['changePercent']:.2f}%", delta_color=change_color)
    col_volume.metric("Volume (M)", f"{summary['volume']:.1f}")
    
    # IV from yfinance is often unavailable or returns 0. If it's zero, use 'N/A'
    iv_display = f"{summary['impliedVolatility']:.2f}%" if summary['impliedVolatility'] > 0 else "N/A"
    col_iv.metric("Implied Volatility (IV)", iv_display)

    st.markdown("---")
    
    # 2. Options Chain Table
    if df is not None and not df.empty:
        st.subheader(f"Options Chain (Exp: {summary['expiration']})")
        st.write("Data is often delayed (end-of-day). Open Interest (OI) represents the total number of outstanding contracts.")

        # Custom function to apply conditional formatting based on ITM/OTM
        def highlight_itm(s):
            is_call_itm = s['Is_ITM_Call']
            is_put_itm = s['Is_ITM_Put']
            
            # Styles for: CALL_Volume, CALL_Open Interest, CALL_Bid, CALL_Ask, STRIKE, PUT_Volume, PUT_Open Interest, PUT_Bid, PUT_Ask, Is_ITM_Call, Is_ITM_Put
            # yfinance columns: CALL_Volume, CALL_Open Interest, CALL_Bid, CALL_Ask, Strike, PUT_Volume, PUT_Open Interest, PUT_Bid, PUT_Ask
            
            # Initialize styles array
            styles = [''] * len(s) 
            
            # Index positions (adjusting for the columns present in the dataframe before styling)
            STRIKE_IDX = 0
            CALL_START_IDX = 1
            CALL_END_IDX = 4
            PUT_START_IDX = 5
            PUT_END_IDX = 8

            # Call ITM - Highlight Call Columns
            if is_call_itm:
                # Highlight call columns
                for i in range(CALL_START_IDX, CALL_END_IDX + 1):
                    styles[i] = 'background-color: #ecfdf5' # Tailwind green-50
                # Gray for strike
                styles[STRIKE_IDX] = 'background-color: #e5e7eb' # Tailwind gray-200
                
            # Put ITM - Highlight Put Columns
            elif is_put_itm:
                # Highlight put columns
                for i in range(PUT_START_IDX, PUT_END_IDX + 1):
                    styles[i] = 'background-color: #fef2f2' # Tailwind red-50
                # Gray for strike
                styles[STRIKE_IDX] = 'background-color: #e5e7eb' # Tailwind gray-200
                
            return styles

        # Prepare DataFrame for styling and display
        display_df = df.drop(columns=['Is_ITM_Call', 'Is_ITM_Put'], errors='ignore')
        
        # Reorder columns for traditional display: CALLS...STRIKE...PUTS
        # The columns fetched are: ['Strike', 'CALL_Volume', 'CALL_Open Interest', 'CALL_Bid', 'CALL_Ask', 'PUT_Volume', 'PUT_Open Interest', 'PUT_Bid', 'PUT_Ask']
        # We need to map yfinance columns to a cleaner display set

        # Renaming for display
        display_df.rename(columns={
            'CALL_Volume': 'C-Vol', 
            'CALL_Open Interest': 'C-OI',
            'PUT_Volume': 'P-Vol', 
            'PUT_Open Interest': 'P-OI',
        }, inplace=True)
        
        # Column order for display
        DISPLAY_COLUMNS = [
            'C-Vol', 'C-OI', 'CALL_Bid', 'CALL_Ask', 
            'Strike', 
            'PUT_Bid', 'PUT_Ask', 'P-OI', 'P-Vol'
        ]
        
        # Filter and reorder
        display_df = display_df[DISPLAY_COLUMNS]

        # Apply the styling (Note: Streamlit styling is limited compared to custom CSS)
        # Apply the styling logic to the original DF for accurate ITM indexing, then select the display columns.
        styled_df = df.drop(columns=['PUT_Volume', 'PUT_Open Interest', 'CALL_Volume', 'CALL_Open Interest'], errors='ignore').style.apply(highlight_itm, axis=1)

        # Use markdown to create the grouped headers (Calls/Puts)
        st.markdown("""
            <style>
            .stDataFrame table {
                border-collapse: separate;
            }
            .stDataFrame th {
                text-align: center !important;
            }
            /* The 5th column is Strike */
            .stDataFrame table th:nth-child(5) {
                background-color: #9ca3af !important; 
                color: #1f2937 !important; 
                font-weight: bold !important;
            }
            </style>
            <div style='text-align: center; display: flex; width: 100%; border-radius: 0.5rem; overflow: hidden;'>
                <div style='flex: 4; background-color: #d1d5db; color: #1f2937; padding: 5px; border-right: 1px solid #9ca3af;'>CALLS</div>
                <div style='flex: 1; background-color: #4b5563; color: white; padding: 5px; font-weight: bold;'>STRIKE</div>
                <div style='flex: 4; background-color: #d1d5db; color: #1f2937; padding: 5px; border-left: 1px solid #9ca3af;'>PUTS</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display the styled dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No options chain data available for the selected ticker and expiration.")


# --- INITIAL LOAD ---

# This fetches the initial data when the app first starts or is cleared
if st.session_state.data_summary is None and st.session_state.ticker:
     # Run fetch and update state, then rerun
     st.session_state.data_summary, st.session_state.data_chain = fetch_options_data(st.session_state.ticker)
     if st.session_state.data_summary:
        st.rerun()
