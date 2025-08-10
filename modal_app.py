#!/usr/bin/env python3

"""
Modal deployment file for Unified Trading Bot.
Includes both FastAPI backend and Dash dashboard deployment to Modal cloud platform.
"""

import modal

# --- FastAPI Backend Deployment ---

fastapi_app_modal = modal.App("unified-trading-bot")
fastapi_image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

@fastapi_app_modal.function(
    image=fastapi_image,
    schedule=modal.Cron("0 */6 * * *"),
    timeout=3600
)
def keepalive():
    print("Keeping Modal app alive...")

@fastapi_app_modal.function(
    image=fastapi_image,
    allow_concurrent_inputs=100,
    timeout=300
)
@modal.asgi_app()
def fastapi_app():
    from backend.main import app as fastapi_app
    return fastapi_app

@fastapi_app_modal.function(
    image=fastapi_image,
    schedule=modal.Cron("0 9 * * 1-5"),
    timeout=1800
)
def daily_market_update():
    print("Running daily market data update...")

# --- Dash Dashboard Deployment ---

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dash_app_modal = modal.App("unified-trading-bot-dashboard")
dash_image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

def fetch_aapl_data(period: str = "5d", interval: str = "1h") -> Optional[Dict[str, Any]]:
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            logger.warning("No data found for AAPL")
            return None

        data = {
            "symbol": "AAPL",
            "timestamp": datetime.now().isoformat(),
            "period": period,
            "interval": interval,
            "data": []
        }
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
    dash_app = dash.Dash(__name__)
    dash_app.title = "Unified Trading Bot - AAPL Dashboard"
    dash_app.layout = html.Div([
        html.Div([
            html.H1("🚀 Unified Trading Bot", style={'text-align': 'center', 'color': '#2c3e50', 'margin-bottom': '10px'}),
            html.H2("Live AAPL Stock Data Dashboard", style={'text-align': 'center', 'color': '#34495e', 'margin-bottom': '30px'}),
        ]),
        html.Div(id='summary-cards', style={'margin-bottom': '30px'}),
        html.Div([
            html.H3("AAPL Stock Price (5 Days, 1 Hour Intervals)", style={'text-align': 'center', 'color': '#2c3e50'}),
            dcc.Graph(id='aapl-chart'),
        ], style={'margin-bottom': '30px'}),
        html.Div([
            html.H3("AAPL Trading Volume", style={'text-align': 'center', 'color': '#2c3e50'}),
            dcc.Graph(id='volume-chart'),
        ], style={'margin-bottom': '30px'}),
        dcc.Interval(
            id='interval-component',
            interval=5*60*1000,
            n_intervals=0
        ),
        html.Div([
            html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}", style={'text-align': 'center', 'color': '#7f8c8d', 'margin-top': '50px'})
        ])
    ], style={'padding': '20px', 'max-width': '1200px', 'margin': '0 auto'})
    @dash_app.callback(
        [Output('summary-cards', 'children'),
         Output('aapl-chart', 'figure'),
         Output('volume-chart', 'figure')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_dashboard(n):
        data = fetch_aapl_data()
        if not data or not data.get('data'):
            error_div = html.Div([
                html.H4("⚠️ Unable to fetch live data", style={'color': 'red', 'text-align': 'center'})
            ])
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data available")
            return error_div, empty_fig, empty_fig
        df = pd.DataFrame(data['data'])
        df['datetime'] = pd.to_datetime(df['datetime'])
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

@dash_app_modal.function(image=dash_image)
@modal.fastapi_endpoint()
def web():
    dash_app = create_dash_app()
    return dash_app.server

if __name__ == "__main__":
    dash_app = create_dash_app()
    dash_app.run(debug=True, host='0.0.0.0', port=8050)