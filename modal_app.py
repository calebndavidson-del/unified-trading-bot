#!/usr/bin/env python3
"""
Modal-hosted Dash Dashboard for Unified Trading Bot
A cloud-based, browser-accessible UI for live market data visualization
"""

import modal
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Modal app
app = modal.App("unified-trading-bot-dashboard")

# Define the Modal image with required dependencies
image = modal.Image.debian_slim().pip_install([
    "dash>=2.14.0",
    "plotly>=5.15.0", 
    "pandas>=2.0.0",
    "yfinance>=0.2.18",
    "requests>=2.31.0"
])

def fetch_aapl_data(period: str = "5d", interval: str = "1h") -> Optional[Dict[str, Any]]:
    """
    Fetch live AAPL stock data using yfinance
    
    Args:
        period: Time period (5d for 5 days)
        interval: Data interval (1h for hourly)
    
    Returns:
        Dictionary with time series data and metadata
    """
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            logger.warning("No data found for AAPL")
            return None
        
        # Convert to JSON-friendly format
        data = {
            "symbol": "AAPL",
            "timestamp": datetime.now().isoformat(),
            "period": period,
            "interval": interval,
            "data": []
        }
        
        # Convert pandas DataFrame to list of dictionaries
        for index, row in hist.iterrows():
            data_point = {
                "datetime": index.isoformat(),
                "timestamp": int(index.timestamp() * 1000),
                "open": float(row['Open']) if not pd.isna(row['Open']) else None,
                "high": float(row['High']) if not pd.isna(row['High']) else None,
                "low": float(row['Low']) if not pd.isna(row['Low']) else None,
                "close": float(row['Close']) if not pd.isna(row['Close']) else None,
                "volume": int(row['Volume']) if not pd.isna(row['Volume']) else None
            }
            data["data"].append(data_point)
        
        # Add latest price and change information
        if len(data["data"]) >= 2:
            latest = data["data"][-1]
            previous = data["data"][-2]
            
            if latest["close"] and previous["close"]:
                price_change = latest["close"] - previous["close"]
                price_change_pct = (price_change / previous["close"]) * 100
                
                data["latest_price"] = latest["close"]
                data["price_change"] = price_change
                data["price_change_pct"] = price_change_pct
                data["is_positive"] = price_change >= 0
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching AAPL data: {str(e)}")
        return None

def create_dash_app():
    """Create and configure the Dash application"""
    
    # Initialize Dash app
    dash_app = dash.Dash(__name__)
    dash_app.title = "Unified Trading Bot - AAPL Dashboard"
    
    # Define the layout
    dash_app.layout = html.Div([
        html.Div([
            html.H1("🚀 Unified Trading Bot", 
                   style={'text-align': 'center', 'color': '#2c3e50', 'margin-bottom': '10px'}),
            html.H2("Live AAPL Stock Data Dashboard", 
                   style={'text-align': 'center', 'color': '#34495e', 'margin-bottom': '30px'}),
        ]),
        
        # Real-time data summary
        html.Div(id='summary-cards', style={'margin-bottom': '30px'}),
        
        # Main chart
        html.Div([
            html.H3("AAPL Stock Price (5 Days, 1 Hour Intervals)", 
                   style={'text-align': 'center', 'color': '#2c3e50'}),
            dcc.Graph(id='aapl-chart'),
        ], style={'margin-bottom': '30px'}),
        
        # Volume chart
        html.Div([
            html.H3("AAPL Trading Volume", 
                   style={'text-align': 'center', 'color': '#2c3e50'}),
            dcc.Graph(id='volume-chart'),
        ], style={'margin-bottom': '30px'}),
        
        # Auto-refresh interval (every 5 minutes)
        dcc.Interval(
            id='interval-component',
            interval=5*60*1000,  # 5 minutes in milliseconds
            n_intervals=0
        ),
        
        # Footer
        html.Div([
            html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", 
                  style={'text-align': 'center', 'color': '#7f8c8d', 'margin-top': '50px'})
        ])
    ], style={'padding': '20px', 'max-width': '1200px', 'margin': '0 auto'})
    
    # Callback for updating charts
    @dash_app.callback(
        [Output('summary-cards', 'children'),
         Output('aapl-chart', 'figure'),
         Output('volume-chart', 'figure')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        """Update dashboard with live AAPL data"""
        
        # Fetch live data
        data = fetch_aapl_data()
        
        if not data or not data.get('data'):
            # Fallback error message
            error_div = html.Div([
                html.H4("⚠️ Unable to fetch live data", style={'color': 'red', 'text-align': 'center'})
            ])
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available")
            return error_div, empty_fig, empty_fig
        
        # Create DataFrame from data
        df = pd.DataFrame(data['data'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Summary cards
        latest_price = data.get('latest_price', 0)
        price_change = data.get('price_change', 0)
        price_change_pct = data.get('price_change_pct', 0)
        is_positive = data.get('is_positive', True)
        
        change_color = '#27ae60' if is_positive else '#e74c3c'
        change_symbol = '+' if is_positive else ''
        
        summary_cards = html.Div([
            html.Div([
                html.H4("Current Price", style={'margin': '0', 'color': '#2c3e50'}),
                html.H2(f"${latest_price:.2f}", style={'margin': '10px 0', 'color': '#2c3e50'})
            ], style={'background-color': '#ecf0f1', 'padding': '20px', 'border-radius': '10px', 
                     'text-align': 'center', 'width': '200px', 'display': 'inline-block', 'margin': '0 10px'}),
            
            html.Div([
                html.H4("Price Change", style={'margin': '0', 'color': '#2c3e50'}),
                html.H2(f"{change_symbol}${price_change:.2f}", 
                       style={'margin': '10px 0', 'color': change_color})
            ], style={'background-color': '#ecf0f1', 'padding': '20px', 'border-radius': '10px', 
                     'text-align': 'center', 'width': '200px', 'display': 'inline-block', 'margin': '0 10px'}),
            
            html.Div([
                html.H4("Percentage Change", style={'margin': '0', 'color': '#2c3e50'}),
                html.H2(f"{change_symbol}{price_change_pct:.2f}%", 
                       style={'margin': '10px 0', 'color': change_color})
            ], style={'background-color': '#ecf0f1', 'padding': '20px', 'border-radius': '10px', 
                     'text-align': 'center', 'width': '200px', 'display': 'inline-block', 'margin': '0 10px'})
        ], style={'text-align': 'center'})
        
        # Price chart (candlestick)
        price_fig = go.Figure(data=go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='AAPL'
        ))
        
        price_fig.update_layout(
            title="AAPL Stock Price - Candlestick Chart",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        # Volume chart
        volume_fig = go.Figure(data=go.Bar(
            x=df['datetime'],
            y=df['volume'],
            name='Volume',
            marker_color='rgba(52, 152, 219, 0.7)'
        ))
        
        volume_fig.update_layout(
            title="AAPL Trading Volume",
            xaxis_title="Time",
            yaxis_title="Volume",
            template="plotly_white",
            height=300,
            showlegend=False
        )
        
        return summary_cards, price_fig, volume_fig
    
    return dash_app

@app.function(image=image)
@modal.fastapi_endpoint()
def web():
    """Modal web endpoint that serves the Dash application"""
    dash_app = create_dash_app()
    return dash_app.server

if __name__ == "__main__":
    # For local development
    dash_app = create_dash_app()
    dash_app.run(debug=True, host='0.0.0.0', port=8050)