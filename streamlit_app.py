#!/usr/bin/env python3
"""
Streamlit Dashboard for Unified Trading Bot
A modern, browser-accessible UI for market data visualization
"""

import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Unified Trading Bot Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Market symbols configuration (matching backend structure)
MARKET_SYMBOLS = {
    "indices": {
        "SPY": {"name": "S&P 500 ETF", "description": "SPDR S&P 500 ETF Trust"},
        "QQQ": {"name": "NASDAQ 100", "description": "Invesco QQQ Trust ETF"},
        "DIA": {"name": "Dow Jones", "description": "SPDR Dow Jones Industrial Average ETF"},
        "IWM": {"name": "Russell 2000", "description": "iShares Russell 2000 ETF"}
    },
    "crypto": {
        "BTC-USD": {"name": "Bitcoin", "description": "Bitcoin USD"},
        "ETH-USD": {"name": "Ethereum", "description": "Ethereum USD"},
        "DOGE-USD": {"name": "Dogecoin", "description": "Dogecoin USD"},
        "XRP-USD": {"name": "XRP", "description": "XRP USD"},
        "SOL-USD": {"name": "Solana", "description": "Solana USD"}
    }
}

# API Configuration
API_BASE_URL = "http://localhost:8000"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_market_data(category: str = "all", period: str = "5d", interval: str = "1h") -> Optional[Dict[str, Any]]:
    """
    Fetch market data from the FastAPI backend
    """
    try:
        url = f"{API_BASE_URL}/market-data"
        params = {"category": category, "period": period, "interval": interval}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data from API: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def generate_fallback_data(symbol: str, base_price: float = 100, days: int = 5) -> pd.DataFrame:
    """
    Generate fallback sample data when API is unavailable
    (Reused from dashboard.py)
    """
    # Generate hourly data for the specified number of days
    periods = days * 24
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        periods=periods
    )
    
    # Generate realistic price movement using random walk
    np.random.seed(hash(symbol) % 1000)  # Consistent seed per symbol
    
    prices = []
    current_price = base_price
    
    for i in range(periods):
        # Random walk with slight upward bias
        change_pct = np.random.normal(0.001, 0.02)  # 0.1% bias, 2% volatility
        current_price *= (1 + change_pct)
        prices.append(current_price)
    
    # Generate OHLC data
    df = pd.DataFrame({
        'datetime': dates,
        'close': prices
    })
    
    # Generate open, high, low from close prices
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(df)))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(df)))
    df['volume'] = np.random.randint(1000000, 10000000, len(df))
    
    return df

def get_fallback_market_data() -> dict:
    """
    Generate fallback market data when API is unavailable
    """
    # Base prices for different assets
    base_prices = {
        "SPY": 475, "QQQ": 385, "DIA": 340, "IWM": 195,
        "BTC-USD": 43000, "ETH-USD": 2600, "DOGE-USD": 0.08, "XRP-USD": 0.52, "SOL-USD": 98
    }
    
    market_data = {}
    
    for category, symbols in MARKET_SYMBOLS.items():
        for symbol in symbols:
            df = generate_fallback_data(symbol, base_prices.get(symbol, 100))
            
            # Calculate price change
            latest_price = df['close'].iloc[-1]
            previous_price = df['close'].iloc[-2] if len(df) > 1 else latest_price
            price_change = latest_price - previous_price
            price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
            
            market_data[symbol] = {
                "symbol": symbol,
                "latest_price": latest_price,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "is_positive": price_change >= 0,
                "data": df.to_dict('records')
            }
    
    return {"data": market_data}

def process_api_data(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process API response data into the format expected by the dashboard
    """
    processed_data = {}
    
    if not api_response or "data" not in api_response:
        return {}
    
    for symbol, data in api_response["data"].items():
        if data and "data" in data and data["data"] and not data.get("error"):
            # Convert API data to DataFrame for easier processing
            df = pd.DataFrame(data["data"])
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            processed_data[symbol] = {
                "symbol": symbol,
                "latest_price": data.get("latest_price", 0),
                "price_change": data.get("price_change", 0),
                "price_change_pct": data.get("price_change_pct", 0),
                "is_positive": data.get("is_positive", True),
                "data": df
            }
    
    return processed_data

def create_price_chart(symbol_data: Dict[str, Any], symbol: str, chart_type: str = "candlestick") -> go.Figure:
    """Create a time series price chart for a symbol"""
    
    # Get symbol info
    symbol_info = None
    for category in MARKET_SYMBOLS.values():
        if symbol in category:
            symbol_info = category[symbol]
            break
    
    name = symbol_info["name"] if symbol_info else symbol
    
    # Handle both DataFrame and dict data
    if isinstance(symbol_data["data"], pd.DataFrame):
        df = symbol_data["data"]
    else:
        df = pd.DataFrame(symbol_data["data"])
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
    
    fig = go.Figure()
    
    if chart_type == "candlestick" and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol,
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ))
    else:
        # Fallback to line chart
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['close'],
            mode='lines',
            name=symbol,
            line=dict(color='#00ff88', width=2)
        ))
    
    fig.update_layout(
        title=f"{symbol} - {name}",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=400,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_summary_card(symbol_data: dict, symbol: str):
    """Create a summary card for a symbol using Streamlit components"""
    
    # Get symbol info
    symbol_info = None
    for category in MARKET_SYMBOLS.values():
        if symbol in category:
            symbol_info = category[symbol]
            break
    
    name = symbol_info["name"] if symbol_info else symbol
    latest_price = symbol_data["latest_price"]
    price_change = symbol_data["price_change"]
    price_change_pct = symbol_data["price_change_pct"]
    is_positive = symbol_data["is_positive"]
    
    # Format price based on asset type
    if symbol.endswith("-USD") and latest_price < 1:
        price_text = f"${latest_price:.4f}"
    elif symbol.endswith("-USD"):
        price_text = f"${latest_price:,.2f}"
    else:
        price_text = f"${latest_price:.2f}"
    
    change_color = "green" if is_positive else "red"
    change_icon = "↗" if is_positive else "↘"
    
    # Create the card using Streamlit metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**{symbol}**")
        st.caption(name)
    
    with col2:
        st.metric(
            label="Price",
            value=price_text,
            delta=f"{change_icon} {price_change:+.2f} ({price_change_pct:+.2f}%)"
        )

def create_overview_charts(market_data: Dict[str, Any]) -> go.Figure:
    """Create overview charts for all markets"""
    performance_data = []
    
    for symbol, data in market_data.items():
        symbol_info = None
        category = "Other"
        
        # Find category
        if symbol in MARKET_SYMBOLS["indices"]:
            category = "Indices"
            symbol_info = MARKET_SYMBOLS["indices"][symbol]
        elif symbol in MARKET_SYMBOLS["crypto"]:
            category = "Crypto"
            symbol_info = MARKET_SYMBOLS["crypto"][symbol]
        
        name = symbol_info["name"] if symbol_info else symbol
        
        performance_data.append({
            "Symbol": symbol,
            "Name": name,
            "Change (%)": data["price_change_pct"],
            "Category": category
        })
    
    perf_df = pd.DataFrame(performance_data)
    
    # Create bar chart
    fig = px.bar(
        perf_df,
        x="Symbol",
        y="Change (%)",
        color="Change (%)",
        color_continuous_scale=["#ff4444", "#888888", "#00ff88"],
        title="Price Change Comparison",
        template="plotly_dark",
        hover_data=["Name", "Category"]
    )
    
    fig.update_layout(height=400)
    
    return fig

def main():
    """Main Streamlit app"""
    
    # Title and header
    st.title("🚀 Unified Trading Bot Dashboard")
    st.markdown("Real-time market data visualization and trading insights")
    
    # Add refresh button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        data_source = st.selectbox("Data Source", ["Live API", "Sample Data"], index=0)
    
    # Fetch data
    if data_source == "Live API":
        with st.spinner("Fetching live market data..."):
            api_response = fetch_market_data()
            if api_response:
                market_data = process_api_data(api_response)
                if market_data:
                    st.success("✅ Live data loaded successfully")
                else:
                    st.warning("⚠️ Live API returned no valid data, using sample data")
                    market_data = process_api_data(get_fallback_market_data())
            else:
                st.warning("⚠️ API unavailable, using sample data")
                market_data = process_api_data(get_fallback_market_data())
    else:
        market_data = process_api_data(get_fallback_market_data())
        st.info("📊 Using sample data for demonstration")
    
    if not market_data:
        st.error("❌ No data available")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Market Overview", "📊 Global Indices", "₿ Cryptocurrencies"])
    
    with tab1:
        st.header("📊 Market Summary")
        
        # Create summary cards in a grid
        cols = st.columns(3)
        col_idx = 0
        
        for symbol in market_data:
            with cols[col_idx % 3]:
                create_summary_card(market_data[symbol], symbol)
            col_idx += 1
        
        st.header("📈 Performance Overview")
        if market_data:
            fig = create_overview_charts(market_data)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📊 Global Indices")
        st.markdown("Major global stock market indices and ETFs")
        
        indices_data = {k: v for k, v in market_data.items() if k in MARKET_SYMBOLS["indices"]}
        
        if indices_data:
            for symbol, data in indices_data.items():
                st.subheader(f"{symbol} - {MARKET_SYMBOLS['indices'][symbol]['name']}")
                fig = create_price_chart(data, symbol)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No indices data available")
    
    with tab3:
        st.header("₿ Cryptocurrencies")
        st.markdown("Popular cryptocurrency pairs vs USD")
        
        crypto_data = {k: v for k, v in market_data.items() if k in MARKET_SYMBOLS["crypto"]}
        
        if crypto_data:
            for symbol, data in crypto_data.items():
                st.subheader(f"{symbol} - {MARKET_SYMBOLS['crypto'][symbol]['name']}")
                fig = create_price_chart(data, symbol)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No cryptocurrency data available")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit for modern market data visualization")

if __name__ == "__main__":
    main()