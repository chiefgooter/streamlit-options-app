import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import plotly.express as px
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes.greeks.analytical import delta, theta, vega, gamma

# --- CONFIGURATION ---
RISK_FREE_RATE = 0.01  # 1% risk-free rate assumption

# --- DATA FETCHING AND CALCULATION WITH YFINANCE ---

@st.cache_data(ttl=600) # Cache data for 10 minutes to avoid hitting API limits
def fetch_options_data(ticker_symbol, selected_expiration):
    """
    Fetches stock data, options chain, and calculates Greeks using yfinance and py_vollib.
    """
    try:
        if not ticker_symbol:
            return None, None, None, None

        ticker = yf.Ticker(ticker_symbol)
        
        # 1. Get Stock Summary
        info = ticker.info
        last_price = info.get('currentPrice') or info.get('previousClose')
        
        if last_price is None or last_price == 0:
            st.error(f"Could not find current price data for ticker: {ticker_symbol.upper()}. Check the ticker symbol.")
            return None, None, None, None
            
        prev_close = info.get('previousClose', last_price)
        change_percent = ((last_price - prev_close) / prev_close) * 100 if prev_close and prev_close != 0 else 0

        stock_summary = {
            'ticker': ticker_symbol.upper(),
            'lastPrice': last_price,
            'changePercent': round(change_percent, 2),
            'volume': round(info.get('volume', 0) / 1000000, 1), 
            'expiration': selected_expiration
        }

        # 2. Get Options Expirations List
        all_expirations = ticker.options
        if not all_expirations:
            st.warning(f"No options chain found for {ticker_symbol.upper()}.")
            return stock_summary, None, [], None

        # If we are only fetching the list of expirations, return early
        if not selected_expiration:
            return stock_summary, None, all_expirations, None

        # 3. Time to Expiration (t) calculation
        expiry_date = datetime.strptime(selected_expiration, '%Y-%m-%d')
        today = datetime.now()
        time_to_exp = (expiry_date - today).days / 365.0
        
        # 4. Fetch Chain Data
        option_chain = ticker.option_chain(selected_expiration)
        calls = option_chain.calls
        puts = option_chain.puts
        
        # 5. Process and Merge DataFrames
        
        def process_option_side(df, flag):
            # Calculate implied volatility using bid/ask average as price
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            
            # Use 'lastPrice' from yfinance for IV calculation, if available, otherwise use mid_price
            price_for_iv = df['lastPrice'].fillna(df['mid_price'])

            df['impliedVolatility'] = price_for_iv.apply(
                lambda p: implied_volatility(
                    p, last_price, df['strike'], 
                    time_to_exp, RISK_FREE_RATE, flag
                )
            )

            # Filter out extreme/invalid IVs (e.g., > 300% or < 0.01%)
            df['impliedVolatility'] = np.where(
                (df['impliedVolatility'] > 3.0) | (df['impliedVolatility'] < 0.01), 
                np.nan, df['impliedVolatility']
            )
            # Use median IV for missing values (common practice)
            median_iv = df['impliedVolatility'].median() or 0.5 
            df['IV'] = df['impliedVolatility'].fillna(median_iv)

            # Calculate Greeks using Black-Scholes model
            df['Delta'] = df.apply(
                lambda row: delta(flag, last_price, row['strike'], time_to_exp, RISK_FREE_RATE, row['IV']), axis=1
            )
            df['Theta'] = df.apply(
                lambda row: theta(flag, last_price, row['strike'], time_to_exp, RISK_FREE_RATE, row['IV']), axis=1
            ) / 365.0 # Theta per day
            
            # Select and rename columns
            prefix = 'CALL_' if flag == 'c' else 'PUT_'
            df_cols = df[['strike', 'volume', 'openInterest', 'bid', 'ask', 'Delta', 'Theta', 'IV']].copy()
            df_cols.rename(columns={
                'strike': 'Strike',
                'volume': prefix + 'Volume',
                'openInterest': prefix + 'Open Interest',
                'bid': prefix + 'Bid',
                'ask': prefix + 'Ask',
                'Delta': prefix + 'Delta',
                'Theta': prefix + 'Theta',
                'IV': prefix + 'IV'
            }, inplace=True)
            return df_cols

        # Add current stock price to be used in IV/Greeks calculation
        calls_processed = process_option_side(calls, 'c')
        puts_processed = process_option_side(puts, 'p')

        # Merge DataFrames on the Strike price
        merged_df = pd.merge(calls_processed, puts_processed, on='Strike', how='outer')
        
        # Add ITM/OTM Flags for conditional styling
        merged_df['Is_ITM_Call'] = merged_df['Strike'] < last_price
        merged_df['Is_ITM_Put'] = merged_df['Strike'] > last_price
        
        # Sort and clean
        merged_df.sort_values(by='Strike', inplace=True)
        merged_df.fillna(0, inplace=True)
        
        # Calculate summary metrics (Flow Indicators)
        total_call_vol = merged_df['CALL_Volume'].sum()
        total_put_vol = merged_df['PUT_Volume'].sum()
        total_call_oi = merged_df['CALL_Open Interest'].sum()
        total_put_oi = merged_df['PUT_Open Interest'].sum()

        flow_indicators = {
            'PCR_Volume': total_put_vol / (total_call_vol if total_call_vol else 1),
            'PCR_OI': total_put_oi / (total_call_oi if total_call_oi else 1),
            'Total_Call_Volume': total_call_vol,
            'Total_Put_Volume': total_put_vol,
        }

        return stock_summary, merged_df, all_expirations, flow_indicators

    except Exception as e:
        st.error(f"An error occurred while fetching data or calculating Greeks: {e}")
        st.warning("Please ensure the ticker symbol is correct. API limits may have been reached or options data might be temporarily unavailable.")
        return None, None, None, None

# --- STREAMLIT UI LAYOUT AND LOGIC ---

# Set wide layout and title
st.set_page_config(layout="wide", page_title="Advanced Options Flow & Greeks Analyzer")

st.title("Advanced Options Flow & Greeks Analyzer")
st.markdown("Real data (often delayed) with calculated Black-Scholes Greeks.")

# Initialize session state 
if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None
if 'data_chain' not in st.session_state:
    st.session_state.data_chain = None
if 'expirations' not in st.session_state:
    st.session_state.expirations = []
if 'selected_exp' not in st.session_state:
    st.session_state.selected_exp = None
if 'flow_indicators' not in st.session_state:
    st.session_state.flow_indicators = None
if 'strike_min' not in st.session_state:
    st.session_state.strike_min = 0
if 'strike_max' not in st.session_state:
    st.session_state.strike_max = 99999

# --- INPUTS AND FETCH BUTTON ---
with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.session_state.ticker = st.text_input(
            "Stock Ticker",
            value=st.session_state.ticker,
            placeholder="e.g., AAPL",
            key="ticker_input_box"
        ).upper().strip()

    with col2:
        # Determine current index for the select box
        if st.session_state.expirations and st.session_state.selected_exp in st.session_state.expirations:
            default_index = st.session_state.expirations.index(st.session_state.selected_exp)
        else:
            default_index = 0
            
        st.session_state.selected_exp = st.selectbox(
            "Select Expiration Date",
            options=st.session_state.expirations,
            index=default_index,
            disabled=not st.session_state.expirations,
            key="exp_select_box"
        )

    with col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Fetch Options Data", type="primary", use_container_width=True):
            # Fetch data based on selected ticker and expiration
            with st.spinner(f"Fetching data for {st.session_state.ticker}..."):
                summary, chain, all_expirations, flow = fetch_options_data(st.session_state.ticker, st.session_state.selected_exp)
                
                # Update state
                st.session_state.data_summary = summary
                st.session_state.data_chain = chain
                st.session_state.expirations = all_expirations if all_expirations else []
                st.session_state.flow_indicators = flow

                # If the list was just fetched and an expiration wasn't explicitly selected, set the first one
                if all_expirations and not st.session_state.selected_exp:
                    st.session_state.selected_exp = all_expirations[0]
                
            st.rerun() 

# --- INITIAL LOAD ---
if st.session_state.data_summary is None and st.session_state.ticker and not st.session_state.expirations:
     # Run initial fetch to populate expirations list
     with st.spinner(f"Loading available expirations for {st.session_state.ticker}..."):
        summary, chain, all_expirations, flow = fetch_options_data(st.session_state.ticker, None)
     
     if all_expirations:
        st.session_state.expirations = all_expirations
        st.session_state.selected_exp = all_expirations[0]
        # Re-run fetch with the first expiration selected to populate chain data
        with st.spinner(f"Fetching chain for {st.session_state.selected_exp}..."):
            summary, chain, all_expirations, flow = fetch_options_data(st.session_state.ticker, st.session_state.selected_exp)
        
        st.session_state.data_summary = summary
        st.session_state.data_chain = chain
        st.session_state.flow_indicators = flow
        st.rerun()


# --- DISPLAY RESULTS (Organized with TABS) ---
if st.session_state.data_summary is not None and st.session_state.data_chain is not None:
    summary = st.session_state.data_summary
    df = st.session_state.data_chain
    flow = st.session_state.flow_indicators

    # --- TABBED INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["📊 Summary & Flow", "📜 Options Chain & Greeks", "📈 Visualization"])

    with tab1:
        st.subheader(f"Current Quote for {summary['ticker']}")
        
        # Summary Metrics
        col_price, col_change, col_vol, col_exp = st.columns(4)
        
        change_color = "inverse" 

        col_price.metric("Last Price", f"${summary['lastPrice']:.2f}")
        col_change.metric("Change (%)", f"{summary['changePercent']:.2f}%", delta=f"{summary['changePercent']:.2f}%", delta_color=change_color)
        col_vol.metric("Daily Volume (M)", f"{summary['volume']:.1f}")
        col_exp.metric("Expiration Date", summary['expiration'])
        
        st.markdown("---")
        st.subheader("Market Sentiment (Put/Call Ratios)")
        st.markdown("Ratios > 1.0 are typically bearish; Ratios < 1.0 are typically bullish.")
        
        # Flow Metrics
        col_pcr_v, col_pcr_oi, col_vol_c, col_vol_p = st.columns(4)

        col_pcr_v.metric(
            "Volume P/C Ratio", 
            f"{flow['PCR_Volume']:.2f}",
            delta="Bearish" if flow['PCR_Volume'] > 1.05 else "Bullish" if flow['PCR_Volume'] < 0.95 else "Neutral",
            delta_color="inverse" if flow['PCR_Volume'] < 0.95 else "off"
        )
        col_pcr_oi.metric(
            "Open Interest P/C Ratio", 
            f"{flow['PCR_OI']:.2f}",
            delta="Bearish" if flow['PCR_OI'] > 1.05 else "Bullish" if flow['PCR_OI'] < 0.95 else "Neutral",
            delta_color="inverse" if flow['PCR_OI'] < 0.95 else "off"
        )
        col_vol_c.metric("Total Call Volume", f"{flow['Total_Call_Volume']:,}")
        col_vol_p.metric("Total Put Volume", f"{flow['Total_Put_Volume']:,}")

    with tab2:
        st.subheader(f"Options Chain and Greeks (Exp: {summary['expiration']})")
        st.write("Greeks (Delta, Theta) are calculated using the Black-Scholes model. Theta is displayed per day.")

        # --- Strike Filtering Controls ---
        col_f1, col_f2, col_f3 = st.columns([1, 1, 3])
        
        min_strike_default = df['Strike'].min()
        max_strike_default = df['Strike'].max()
        
        st.session_state.strike_min = col_f1.number_input(
            "Min Strike", 
            value=min_strike_default, 
            min_value=0.0, 
            key='min_strike'
        )
        st.session_state.strike_max = col_f2.number_input(
            "Max Strike", 
            value=max_strike_default, 
            min_value=0.0, 
            key='max_strike'
        )
        
        # Apply filtering
        filtered_df = df[
            (df['Strike'] >= st.session_state.strike_min) & 
            (df['Strike'] <= st.session_state.strike_max)
        ].copy()

        # Custom function to apply conditional formatting based on ITM/OTM
        def highlight_itm(s):
            is_call_itm = s['Is_ITM_Call']
            is_put_itm = s['Is_ITM_Put']
            
            # The columns in the DF are: [Strike, CALL_Volume, CALL_Open Interest, CALL_Bid, CALL_Ask, CALL_Delta, CALL_Theta, CALL_IV, PUT_Volume, PUT_Open Interest, PUT_Bid, PUT_Ask, PUT_Delta, PUT_Theta, PUT_IV, Is_ITM_Call, Is_ITM_Put]
            
            # Initialize styles array based on number of columns
            styles = [''] * len(s) 
            
            # Column Index Map (excluding temp columns)
            # Strike is index 0
            CALL_START = 1
            CALL_END = 7 # CALL_IV
            PUT_START = 8
            PUT_END = 14 # PUT_IV

            # Call ITM - Highlight Call Columns
            if is_call_itm:
                # Highlight call columns
                for i in range(CALL_START, CALL_END + 1):
                    styles[i] = 'background-color: #ecfdf5' # Light Green
                # Gray for strike
                styles[0] = 'background-color: #e5e7eb' # Light Gray
                
            # Put ITM - Highlight Put Columns
            elif is_put_itm:
                # Highlight put columns
                for i in range(PUT_START, PUT_END + 1):
                    styles[i] = 'background-color: #fef2f2' # Light Red
                # Gray for strike
                styles[0] = 'background-color: #e5e7eb' # Light Gray
                
            return styles

        # Prepare DataFrame for styling and display
        display_df_raw = filtered_df.drop(columns=['Is_ITM_Call', 'Is_ITM_Put'], errors='ignore')

        # Renaming and reordering columns for user-friendly view
        display_df_raw.rename(columns={
            'CALL_Volume': 'C-Vol', 'CALL_Open Interest': 'C-OI', 'CALL_Bid': 'C-Bid', 'CALL_Ask': 'C-Ask', 'CALL_IV': 'C-IV',
            'PUT_Volume': 'P-Vol', 'PUT_Open Interest': 'P-OI', 'PUT_Bid': 'P-Bid', 'PUT_Ask': 'P-Ask', 'PUT_IV': 'P-IV',
        }, inplace=True)
        
        # Display columns in the preferred order
        DISPLAY_COLUMNS = [
            'C-Vol', 'C-OI', 'CALL_Delta', 'CALL_Theta', 'C-Bid', 'C-Ask', 'C-IV',
            'Strike', 
            'P-IV', 'P-Bid', 'P-Ask', 'PUT_Delta', 'PUT_Theta', 'P-OI', 'P-Vol'
        ]
        
        # Filter and reorder
        display_df = display_df_raw[DISPLAY_COLUMNS]

        # Apply formatting to values
        display_df = display_df.style.format({
            'CALL_Delta': "{:.3f}", 
            'PUT_Delta': "{:.3f}",
            'CALL_Theta': "{:.4f}", 
            'PUT_Theta': "{:.4f}",
            'C-IV': "{:.2%}",
            'P-IV': "{:.2%}",
            'Strike': "{:.2f}",
            'C-Bid': "{:.2f}", 'C-Ask': "{:.2f}",
            'P-Bid': "{:.2f}", 'P-Ask': "{:.2f}",
        }).apply(highlight_itm, axis=1)


        # Custom header to visually group the columns
        st.markdown("""
            <style>
            /* Specific styling for the header rows to create the Calls/Strike/Puts visual separation */
            .stDataFrame table th:nth-child(8) {
                background-color: #4b5563 !important; 
                color: white !important; 
                font-weight: bold !important;
            }
            </style>
            <div style='text-align: center; display: flex; width: 100%; border-radius: 0.5rem; overflow: hidden; font-weight: bold;'>
                <div style='flex: 7; background-color: #d1d5db; color: #1f2937; padding: 5px; border-right: 1px solid #9ca3af;'>CALLS</div>
                <div style='flex: 1; background-color: #4b5563; color: white; padding: 5px;'>STRIKE</div>
                <div style='flex: 7; background-color: #d1d5db; color: #1f2937; padding: 5px; border-left: 1px solid #9ca3af;'>PUTS</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


    with tab3:
        st.subheader("Volume and Open Interest Distribution by Strike")
        
        # Prepare data for visualization
        # We use the filtered DataFrame for the chart
        chart_df = filtered_df.copy()
        
        # Melt DataFrame for Plotly (better for multi-series bar charts)
        melted_df = pd.melt(
            chart_df, 
            id_vars=['Strike'], 
            value_vars=['CALL_Volume', 'PUT_Volume', 'CALL_Open Interest', 'PUT_Open Interest'],
            var_name='Metric', 
            value_name='Value'
        )
        
        # Define colors for better contrast
        color_map = {
            'CALL_Volume': '#10b981',  # Emerald 500
            'PUT_Volume': '#ef4444',     # Red 500
            'CALL_Open Interest': '#34d399', # Emerald 300
            'PUT_Open Interest': '#f87171'  # Red 300
        }

        # Plotly Bar Chart: Volume vs OI
        fig = px.bar(
            melted_df, 
            x='Strike', 
            y='Value', 
            color='Metric',
            color_discrete_map=color_map,
            barmode='group',
            height=550,
            title=f"Options Flow by Strike ({summary['expiration']})",
            labels={'Value': 'Total Contracts', 'Strike': 'Strike Price', 'Metric': 'Flow Metric'},
            hover_data={'Value': True, 'Metric': False}
        )
        
        # Add a vertical line for the current stock price (ATM)
        fig.add_vline(
            x=summary['lastPrice'], 
            line_width=3, 
            line_dash="dash", 
            line_color="#2563eb", 
            annotation_text=f"Stock Price ${summary['lastPrice']:.2f}",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig, use_container_width=True)
