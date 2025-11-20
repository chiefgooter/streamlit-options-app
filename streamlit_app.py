import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import plotly.express as px
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes.greeks.analytical import delta, theta, vega, gamma

# --- CONFIGURATION ---
RISK_FREE_RATE = 0.01  # 1% risk-free rate assumption (used for Greeks calculation)

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
        
        # Guard against zero time to expiry (if the expiration is today or past)
        if time_to_exp <= 0:
            time_to_exp = 1/365.0 # Set minimum non-zero time

        # 4. Fetch Chain Data
        option_chain = ticker.option_chain(selected_expiration)
        calls = option_chain.calls
        puts = option_chain.puts
        
        # 5. Process and Merge DataFrames
        
        def process_option_side(df, flag):
            
            # --- Robust Data Cleaning ---
            # 1. Calculate mid_price and ensure all necessary columns are numeric
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            df = df.astype({'strike': float, 'mid_price': float, 'volume': float, 'openInterest': float})
            
            # 2. Set an option price for calculation: use mid_price.
            option_price = df['mid_price']
            
            # --- Implied Volatility (IV) Calculation ---
            
            # 3. Define the vectorized IV function
            def safe_implied_volatility(price, strike):
                try:
                    # 'k' is the strike price, 's' is the current stock price (last_price)
                    return implied_volatility(price, last_price, strike, time_to_exp, RISK_FREE_RATE, flag)
                except Exception:
                    return np.nan # Return NaN if calculation fails

            vectorized_iv = np.vectorize(safe_implied_volatility)

            valid_iv = np.where(
                (option_price > 0.01) & pd.notna(option_price) & (df['strike'] > 0),
                vectorized_iv(option_price, df['strike']),
                np.nan
            )
            # Store the computed IV
            df['impliedVolatility'] = valid_iv
            
            # 4. Filter out extreme/invalid IVs
            df['impliedVolatility'] = np.where(
                (df['impliedVolatility'] > 3.0) | (df['impliedVolatility'] < 0.01), 
                np.nan, df['impliedVolatility']
            )
            
            # Use median IV for missing values (crucial for Greeks calculation)
            median_iv = df['impliedVolatility'].median() 
            # If median is also NaN, fall back to a reasonable 50% IV
            df['IV'] = df['impliedVolatility'].fillna(median_iv if pd.notna(median_iv) else 0.5) 
            
            # 5. Set a minimum IV floor (e.g., 1%) to prevent zero-division in Greeks model
            df['IV'] = np.where(df['IV'] < 0.01, 0.01, df['IV']) 

            # --- Greeks Calculation ---
            # Calculate Greeks using Black-Scholes model
            df['Delta'] = df.apply(
                lambda row: delta(flag, last_price, row['strike'], time_to_exp, RISK_FREE_RATE, row['IV']), axis=1
            )
            df['Theta'] = df.apply(
                # Theta result needs to be divided by 365 to get daily decay
                lambda row: theta(flag, last_price, row['strike'], time_to_exp, RISK_FREE_RATE, row['IV']), axis=1
            ) / 365.0 
            
            # Select and rename columns
            prefix = 'CALL_' if flag == 'c' else 'PUT_'
            df_cols = df[['strike', 'volume', 'openInterest', 'bid', 'ask', 'Delta', 'Theta', 'IV']].copy()
            
            # Rename all columns for consistency before merge
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

        # Process Calls and Puts
        calls_processed = process_option_side(calls, 'c')
        puts_processed = process_option_side(puts, 'p')

        # Merge DataFrames
        merged_df = pd.merge(calls_processed, puts_processed, on='Strike', how='outer')
        
        # Add ITM/OTM Flags
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
        # Crucial for debugging: display the error in the app
        st.error(f"A runtime error occurred during data fetching or calculation. Please try another ticker or expiration.")
        st.code(f"Detailed Error: {e}", language='text')
        return None, None, None, None

# --- STREAMLIT UI LAYOUT AND LOGIC ---

# Set wide layout and title
st.set_page_config(layout="wide", page_title="Advanced Options Flow & Greeks Analyzer")

st.title("Advanced Options Flow & Greeks Analyzer")
st.markdown("Real data (often delayed) with calculated Black-Scholes Greeks.")

# Initialize session state 
if 'ticker' not in st.session_state: st.session_state.ticker = "AAPL"
if 'data_summary' not in st.session_state: st.session_state.data_summary = None
if 'data_chain' not in st.session_state: st.session_state.data_chain = None
if 'expirations' not in st.session_state: st.session_state.expirations = []
if 'selected_exp' not in st.session_state: st.session_state.selected_exp = None
if 'flow_indicators' not in st.session_state: st.session_state.flow_indicators = None
if 'strike_min' not in st.session_state: st.session_state.strike_min = 0
if 'strike_max' not in st.session_state: st.session_state.strike_max = 99999

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
        # Determine the default index for the select box safely
        current_exp_list = st.session_state.expirations
        current_selected = st.session_state.selected_exp
        
        default_index = 0
        if current_selected and current_selected in current_exp_list:
            default_index = current_exp_list.index(current_selected)
        elif current_exp_list:
            # If nothing selected yet, select the first one
            st.session_state.selected_exp = current_exp_list[0]
            
        st.session_state.selected_exp = st.selectbox(
            "Select Expiration Date",
            options=current_exp_list,
            index=default_index,
            disabled=not current_exp_list,
            key="exp_select_box"
        )

    with col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Fetch Options Data", type="primary", use_container_width=True):
            # Clear cache to force fresh data load
            st.cache_data.clear()

            with st.spinner(f"Fetching data for {st.session_state.ticker}..."):
                # Step 1: Get Expirations List
                _, _, all_expirations, _ = fetch_options_data(st.session_state.ticker, None)
                st.session_state.expirations = all_expirations if all_expirations else []
                
                # If the selected_exp is now invalid (e.g., we changed the ticker), reset it
                if st.session_state.selected_exp not in st.session_state.expirations and st.session_state.expirations:
                     st.session_state.selected_exp = st.session_state.expirations[0]

                # Step 2: Fetch the full chain if we have a valid expiration selected
                if st.session_state.selected_exp and st.session_state.selected_exp in st.session_state.expirations:
                    summary, chain, _, flow = fetch_options_data(st.session_state.ticker, st.session_state.selected_exp)
                    st.session_state.data_summary = summary
                    st.session_state.data_chain = chain
                    st.session_state.flow_indicators = flow
                else:
                    st.session_state.data_summary = None
                    st.session_state.data_chain = None
                    st.session_state.flow_indicators = None

            st.rerun() 

# --- INITIAL LOAD (To automatically fetch expirations on first load) ---
if not st.session_state.expirations and st.session_state.ticker:
     with st.spinner(f"Loading available expirations for {st.session_state.ticker}..."):
        # Step 1: Get Expirations List
        summary, chain, all_expirations, flow = fetch_options_data(st.session_state.ticker, None)
     
     if all_expirations:
        st.session_state.expirations = all_expirations
        st.session_state.selected_exp = all_expirations[0]
        
        # Step 2: Fetch Chain Data for the first expiration
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

    tab1, tab2, tab3 = st.tabs(["📊 Summary & Flow", "📜 Options Chain & Greeks", "📈 Visualization"])

    with tab1:
        st.subheader(f"Current Quote for {summary['ticker']}")
        
        col_price, col_change, col_vol, col_exp = st.columns(4)
        
        # Use 'inverse' color for delta/change since positive change (green) means bullish (green), but 
        # for P/C ratio, high (red) is bearish, low (green) is bullish.
        change_color = "inverse" 

        col_price.metric("Last Price", f"${summary['lastPrice']:.2f}")
        col_change.metric("Change (%)", f"{summary['changePercent']:.2f}%", delta=f"{summary['changePercent']:.2f}%", delta_color=change_color)
        col_vol.metric("Daily Volume (M)", f"{summary['volume']:.1f}")
        col_exp.metric("Expiration Date", summary['expiration'])
        
        st.markdown("---")
        st.subheader("Market Sentiment (Put/Call Ratios)")
        
        col_pcr_v, col_pcr_oi, col_vol_c, col_vol_p = st.columns(4)
        
        # P/C Ratio logic: High P/C (above 1) is Bearish (Red). Low P/C (below 1) is Bullish (Green).
        pcr_v_sentiment = "Neutral"
        pcr_v_color = "off"
        if flow['PCR_Volume'] > 1.05:
            pcr_v_sentiment = "Bearish"
        elif flow['PCR_Volume'] < 0.95:
            pcr_v_sentiment = "Bullish"
            pcr_v_color = "inverse" # Inverse color makes negative (less than 1) green
            
        pcr_oi_sentiment = "Neutral"
        pcr_oi_color = "off"
        if flow['PCR_OI'] > 1.05:
            pcr_oi_sentiment = "Bearish"
        elif flow['PCR_OI'] < 0.95:
            pcr_oi_sentiment = "Bullish"
            pcr_oi_color = "inverse" # Inverse color makes negative (less than 1) green


        col_pcr_v.metric(
            "Volume P/C Ratio", 
            f"{flow['PCR_Volume']:.2f}",
            delta=pcr_v_sentiment,
            delta_color=pcr_v_color
        )
        col_pcr_oi.metric(
            "Open Interest P/C Ratio", 
            f"{flow['PCR_OI']:.2f}",
            delta=pcr_oi_sentiment,
            delta_color=pcr_oi_color
        )
        col_vol_c.metric("Total Call Volume", f"{flow['Total_Call_Volume']:,}")
        col_vol_p.metric("Total Put Volume", f"{flow['Total_Put_Volume']:,}")

    with tab2:
        st.subheader(f"Options Chain and Greeks (Exp: {summary['expiration']})")
        st.write("Greeks (Delta, Theta) are calculated using the Black-Scholes model. Theta is displayed per day.")

        col_f1, col_f2, col_f3 = st.columns([1, 1, 3])
        
        # Determine initial values for strike filters
        if not df.empty:
            min_strike_default = df['Strike'].min()
            max_strike_default = df['Strike'].max()
        else:
            min_strike_default = 0
            max_strike_default = summary['lastPrice'] * 2 if summary['lastPrice'] else 100
        
        st.session_state.strike_min = col_f1.number_input(
            "Min Strike", 
            value=float(min_strike_default), 
            min_value=0.0, 
            key='min_strike_t2'
        )
        st.session_state.strike_max = col_f2.number_input(
            "Max Strike", 
            value=float(max_strike_default), 
            min_value=0.0, 
            key='max_strike_t2'
        )
        
        # Filter the main DataFrame
        filtered_df = df[
            (df['Strike'] >= st.session_state.strike_min) & 
            (df['Strike'] <= st.session_state.strike_max)
        ].copy()

        if filtered_df.empty:
            st.warning("No options strikes found within the selected minimum and maximum strike price range. Please widen your filter range.")
        else:
            
            # 1. Define Rename Map
            RENAME_MAP = {
                'CALL_Volume': 'C-Vol', 'CALL_Open Interest': 'C-OI', 'CALL_Bid': 'C-Bid', 'CALL_Ask': 'C-Ask', 'CALL_IV': 'C-IV',
                'PUT_Volume': 'P-Vol', 'PUT_Open Interest': 'P-OI', 'PUT_Bid': 'P-Bid', 'PUT_Ask': 'P-Ask', 'PUT_IV': 'P-IV',
                'CALL_Delta': 'C-Delta', 'CALL_Theta': 'C-Theta',
                'PUT_Delta': 'P-Delta', 'PUT_Theta': 'P-Theta',
            }

            # Apply rename to a copy (includes Delta/Theta/Strike and ITM flags)
            df_renamed = filtered_df.rename(columns=RENAME_MAP).copy()

            # Define the final columns for display (excluding the ITM flag columns)
            DISPLAY_COLUMNS = [
                'C-Vol', 'C-OI', 'C-Delta', 'C-Theta', 'C-Bid', 'C-Ask', 'C-IV',
                'Strike', 
                'P-IV', 'P-Bid', 'P-Ask', 'P-Delta', 'P-Theta', 'P-OI', 'P-Vol'
            ]
            
            # The data that the styler will operate on (includes flags)
            df_for_styler = df_renamed.copy()

            # --- Styling Logic ---
            
            # Define new dark colors for better contrast in dark mode
            ITM_CALL_COLOR = '#14532d'  # Dark Forest Green
            ITM_PUT_COLOR = '#7f1d1d'   # Dark Maroon Red
            ATM_STRIKE_COLOR = '#374151' # Darker Gray for Strike Column

            def highlight_itm(s):
                """Applies color formatting based on ITM flags for each row."""
                is_call_itm = s['Is_ITM_Call']
                is_put_itm = s['Is_ITM_Put']
                
                # Create styles dictionary mapping column names to styles
                styles = {}
                
                if is_call_itm:
                    # Highlight Call side columns with DARK GREEN
                    for col in ['C-Vol', 'C-OI', 'C-Delta', 'C-Theta', 'C-Bid', 'C-Ask', 'C-IV']:
                        styles[col] = f'background-color: {ITM_CALL_COLOR}'
                    styles['Strike'] = f'background-color: {ATM_STRIKE_COLOR}'
                elif is_put_itm:
                    # Highlight Put side columns with DARK RED
                    for col in ['P-IV', 'P-Bid', 'P-Ask', 'P-Delta', 'P-Theta', 'P-OI', 'P-Vol']:
                        styles[col] = f'background-color: {ITM_PUT_COLOR}'
                    styles['Strike'] = f'background-color: {ATM_STRIKE_COLOR}'
                    
                # Return array matching the columns in df_for_styler
                style_array = []
                for col_name in df_for_styler.columns:
                    style_array.append(styles.get(col_name, ''))
                return style_array


            # 2. Apply styling and formatting (no chaining)
            # Apply styling
            styled_df = df_for_styler.style.apply(highlight_itm, axis=1)
            
            # Apply formatting
            styled_df = styled_df.format({
                'C-Delta': "{:.3f}", 
                'P-Delta': "{:.3f}",
                'C-Theta': "{:.4f}", 
                'P-Theta': "{:.4f}",
                'C-IV': "{:.2%}",
                'P-IV': "{:.2%}",
                'Strike': "{:.2f}",
                'C-Bid': "{:.2f}", 'C-Ask': "{:.2f}",
                'P-Bid': "{:.2f}", 'P-Ask': "{:.2f}",
            })

            # 3. Final Display
            df_display = styled_df.set_properties(**{'border-color': '#444'}).set_table_attributes('style="width:100%"')

            # Custom header for the options chain
            st.markdown("""
                <style>
                /* Style the strike column header */
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
            
            # Display the final styled DataFrame using the subset of columns defined earlier
            st.dataframe(
                df_display,
                column_order=DISPLAY_COLUMNS, # Explicitly tell Streamlit which columns to show
                use_container_width=True,
                hide_index=True
            )


    with tab3:
        st.subheader("Volume and Open Interest Distribution by Strike")
        
        if filtered_df.empty:
            st.warning("Cannot generate visualization: No options data available for the selected range.")
        else:
            chart_df = filtered_df.copy()
            
            # Rename columns for better plot labels
            chart_df.rename(columns={
                'CALL_Volume': 'Call Volume', 'PUT_Volume': 'Put Volume', 
                'CALL_Open Interest': 'Call Open Interest', 'PUT_Open Interest': 'Put Open Interest'
            }, inplace=True)

            melted_df = pd.melt(
                chart_df, 
                id_vars=['Strike'], 
                value_vars=['Call Volume', 'Put Volume', 'Call Open Interest', 'Put Open Interest'],
                var_name='Metric', 
                value_name='Value'
            )
            
            color_map = {
                'Call Volume': '#10b981', 
                'Put Volume': '#ef4444', 
                'Call Open Interest': '#34d399', 
                'Put Open Interest': '#f87171'
            }

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
            
            fig.add_vline(
                x=summary['lastPrice'], 
                line_width=3, 
                line_dash="dash", 
                line_color="#2563eb", 
                annotation_text=f"Stock Price ${summary['lastPrice']:.2f}",
                annotation_position="top right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
