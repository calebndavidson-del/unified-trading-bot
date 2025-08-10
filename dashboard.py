#!/usr/bin/env python3
"""
Dash Dashboard for Unified Trading Bot
A data-rich, browser-accessible UI for market data visualization
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Unified Trading Bot Dashboard"

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

def generate_sample_data(symbol: str, base_price: float = 100, days: int = 5) -> pd.DataFrame:
    """
    Generate sample time series data for demonstration
    This will be replaced with real market data integration
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

def get_sample_market_data() -> dict:
    """
    Generate sample market data for all symbols
    This mimics the structure returned by the FastAPI backend
    """
    # Base prices for different assets
    base_prices = {
        "SPY": 475, "QQQ": 385, "DIA": 340, "IWM": 195,
        "BTC-USD": 43000, "ETH-USD": 2600, "DOGE-USD": 0.08, "XRP-USD": 0.52, "SOL-USD": 98
    }
    
    market_data = {}
    
    for category, symbols in MARKET_SYMBOLS.items():
        for symbol in symbols:
            df = generate_sample_data(symbol, base_prices.get(symbol, 100))
            
            # Calculate price change
            latest_price = df['close'].iloc[-1]
            previous_price = df['close'].iloc[-2] if len(df) > 1 else latest_price
            price_change = latest_price - previous_price
            price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
            
            market_data[symbol] = {
                "symbol": symbol,
                "name": symbols[symbol]["name"],
                "description": symbols[symbol]["description"],
                "latest_price": latest_price,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "is_positive": price_change >= 0,
                "data": df
            }
    
    return market_data

# Generate sample data
sample_data = get_sample_market_data()

def create_price_chart(symbol_data: dict, symbol: str) -> go.Figure:
    """Create a time series price chart for a symbol"""
    df = symbol_data["data"]
    
    fig = go.Figure()
    
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
    
    fig.update_layout(
        title=f"{symbol} - {symbol_data['name']}",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=400,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_summary_card(symbol_data: dict, symbol: str) -> html.Div:
    """Create a summary card for a symbol"""
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
    
    change_color = "#00ff88" if is_positive else "#ff4444"
    change_icon = "↗" if is_positive else "↘"
    
    return html.Div([
        html.Div([
            html.H4(symbol, style={"margin": "0", "color": "white"}),
            html.P(symbol_data["name"], style={"margin": "0", "fontSize": "12px", "color": "#aaa"})
        ], style={"marginBottom": "10px"}),
        
        html.Div([
            html.H3(price_text, style={"margin": "0", "color": "white"}),
            html.P([
                f"{change_icon} {price_change:+.2f} ({price_change_pct:+.2f}%)"
            ], style={"margin": "0", "color": change_color, "fontSize": "14px"})
        ])
    ], style={
        "backgroundColor": "#2c2c2c",
        "padding": "15px",
        "borderRadius": "8px",
        "border": f"2px solid {change_color}",
        "margin": "5px"
    })

def create_overview_charts() -> html.Div:
    """Create overview charts for all markets"""
    # Performance comparison chart
    symbols = list(sample_data.keys())
    performance_data = []
    
    for symbol in symbols:
        data = sample_data[symbol]
        performance_data.append({
            "Symbol": symbol,
            "Name": data["name"],
            "Change (%)": data["price_change_pct"],
            "Category": "Indices" if symbol in MARKET_SYMBOLS["indices"] else "Crypto"
        })
    
    perf_df = pd.DataFrame(performance_data)
    
    # Create bar chart
    fig = px.bar(
        perf_df,
        x="Symbol",
        y="Change (%)",
        color="Change (%)",
        color_continuous_scale=["#ff4444", "#888888", "#00ff88"],
        title="Price Change Comparison (Last Hour)",
        template="plotly_dark"
    )
    
    fig.update_layout(height=400)
    
    return html.Div([
        dcc.Graph(figure=fig),
        html.P(
            "📊 Sample data for demonstration. Connect to live data feeds for real-time updates.",
            style={"textAlign": "center", "color": "#888", "fontStyle": "italic", "marginTop": "10px"}
        )
    ])

# Define the app layout
app.layout = html.Div([
    html.Div([
        html.H1("🚀 Unified Trading Bot Dashboard", style={"color": "white", "textAlign": "center"}),
        html.P("Real-time market data visualization and trading insights", 
               style={"color": "#aaa", "textAlign": "center", "marginBottom": "30px"})
    ], style={"marginBottom": "20px"}),
    
    dcc.Tabs(id="main-tabs", value="overview", children=[
        dcc.Tab(label="📈 Market Overview", value="overview", style={"backgroundColor": "#1e1e1e", "color": "white"}),
        dcc.Tab(label="📊 US Indices", value="indices", style={"backgroundColor": "#1e1e1e", "color": "white"}),
        dcc.Tab(label="₿ Cryptocurrencies", value="crypto", style={"backgroundColor": "#1e1e1e", "color": "white"}),
    ], style={"marginBottom": "20px"}),
    
    html.Div(id="tab-content")
    
], style={
    "backgroundColor": "#1a1a1a",
    "minHeight": "100vh",
    "padding": "20px",
    "fontFamily": "Arial, sans-serif"
})

@app.callback(Output("tab-content", "children"), [Input("main-tabs", "value")])
def render_tab_content(active_tab):
    """Render content based on selected tab"""
    
    if active_tab == "overview":
        # Overview tab with summary cards and performance chart
        summary_cards = []
        for symbol in sample_data:
            summary_cards.append(create_summary_card(sample_data[symbol], symbol))
        
        return html.Div([
            html.H3("📊 Market Summary", style={"color": "white", "marginBottom": "20px"}),
            html.Div(summary_cards, style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
                "gap": "10px",
                "marginBottom": "30px"
            }),
            html.H3("📈 Performance Overview", style={"color": "white", "marginBottom": "20px"}),
            create_overview_charts()
        ])
    
    elif active_tab == "indices":
        # US Indices tab
        charts = []
        for symbol in MARKET_SYMBOLS["indices"]:
            if symbol in sample_data:
                charts.append(html.Div([
                    dcc.Graph(figure=create_price_chart(sample_data[symbol], symbol))
                ], style={"marginBottom": "20px"}))
        
        return html.Div([
            html.H3("📊 US Market Indices", style={"color": "white", "marginBottom": "20px"}),
            html.P("Major US stock market indices and ETFs", style={"color": "#aaa", "marginBottom": "20px"}),
            html.Div(charts)
        ])
    
    elif active_tab == "crypto":
        # Cryptocurrency tab
        charts = []
        for symbol in MARKET_SYMBOLS["crypto"]:
            if symbol in sample_data:
                charts.append(html.Div([
                    dcc.Graph(figure=create_price_chart(sample_data[symbol], symbol))
                ], style={"marginBottom": "20px"}))
        
        return html.Div([
            html.H3("₿ Cryptocurrencies", style={"color": "white", "marginBottom": "20px"}),
            html.P("Popular cryptocurrency pairs vs USD", style={"color": "#aaa", "marginBottom": "20px"}),
            html.Div(charts)
        ])
    
    return html.Div([
        html.H3("Welcome to the Trading Dashboard", style={"color": "white"}),
        html.P("Select a tab to view market data", style={"color": "#aaa"})
    ])

if __name__ == "__main__":
    print("🚀 Starting Unified Trading Bot Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    print("🔧 This is sample data. For live data, integrate with the FastAPI backend.")
    print("⏹  Press Ctrl+C to stop the server")
    
    app.run(
        debug=True,
        host="0.0.0.0",
        port=8050,
        dev_tools_ui=False,
        dev_tools_props_check=False
    )