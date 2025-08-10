#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import os
import time
import subprocess
import threading
from datetime import datetime, timedelta
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import itertools
from symbol_scanner import SymbolScanner, MarketCategoryScanner, cached_smart_scan

st.set_page_config(
    page_title="Enhanced Trading Bot",
    page_icon="🚀",
    layout="wide"
)

@st.cache_data
def load_config():
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None

def init_session_state():
    defaults = {
        'bot_running': False,
        'bot_process': None,
        'portfolio_value': 100000,
        'trades_df': pd.DataFrame(),
        'last_update': None,
        'backtest_results': {},
        'optimization_results': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_trades():
    """Load recent trades from CSV"""
    try:
        if os.path.exists('logs/trades.csv'):
            df = pd.read_csv('logs/trades.csv')
            if not df.empty:
                df['ts'] = pd.to_datetime(df['ts'])
                return df.tail(50)
    except Exception as e:
        st.error(f"Error loading trades: {e}")
    return pd.DataFrame()

def calculate_technical_indicators(df, rsi_period=14, bb_period=20, bb_std=2):
    """Calculate technical indicators with dynamic parameters"""
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_ma = df['Close'].rolling(window=bb_period).mean()
    bb_upper = bb_ma + (df['Close'].rolling(window=bb_period).std() * bb_std)
    bb_lower = bb_ma - (df['Close'].rolling(window=bb_period).std() * bb_std)
    
    # Moving averages
    ma_short = df['Close'].rolling(window=10).mean()
    ma_long = df['Close'].rolling(window=20).mean()
    
    return {
        'rsi': rsi,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_ma': bb_ma,
        'ma_short': ma_short,
        'ma_long': ma_long
    }

def advanced_backtest(symbol, days, params):
    """Advanced backtesting with dynamic parameters"""
    try:
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            return None
        
        # Calculate indicators with custom parameters
        indicators = calculate_technical_indicators(
            df, 
            rsi_period=params['rsi_period'],
            bb_period=params['bb_period'],
            bb_std=params['bb_std']
        )
        
        # Enhanced trading strategy
        position = 0
        cash = params['starting_capital']
        portfolio_value = []
        trades = []
        
        for i in range(max(params['bb_period'], params['rsi_period']), len(df)):
            current_price = df['Close'].iloc[i]
            rsi_val = indicators['rsi'].iloc[i]
            bb_upper = indicators['bb_upper'].iloc[i]
            bb_lower = indicators['bb_lower'].iloc[i]
            ma_short = indicators['ma_short'].iloc[i]
            ma_long = indicators['ma_long'].iloc[i]
            
            # Generate signals
            buy_signal = (
                rsi_val < params['rsi_oversold'] and 
                current_price < bb_lower and
                ma_short > ma_long
            )
            
            sell_signal = (
                rsi_val > params['rsi_overbought'] or
                current_price > bb_upper or
                (position > 0 and current_price < position * (1 - params['stop_loss']))
            )
            
            # Execute trades
            if buy_signal and position == 0:
                shares = (cash * params['position_size']) // current_price
                if shares > 0:
                    position = current_price
                    cash -= shares * current_price
                    trades.append({
                        'date': df.index[i],
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares
                    })
            
            elif sell_signal and position > 0:
                profit = shares * (current_price - position)
                cash += shares * current_price
                trades.append({
                    'date': df.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'profit': profit
                })
                position = 0
                shares = 0
            
            # Calculate portfolio value
            total_value = cash + (shares * current_price if position > 0 else 0)
            portfolio_value.append(total_value)
        
        # Calculate metrics
        portfolio_series = pd.Series(portfolio_value, index=df.index[max(params['bb_period'], params['rsi_period']):])
        returns = portfolio_series.pct_change().fillna(0)
        
        total_return = (portfolio_series.iloc[-1] / params['starting_capital']) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_dd = ((portfolio_series / portfolio_series.cummax()) - 1).min()
        
        return {
            'symbol': symbol,
            'params': params,
            'equity_curve': portfolio_series,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_value': portfolio_series.iloc[-1],
            'trades': trades,
            'win_rate': len([t for t in trades if t.get('profit', 0) > 0]) / max(len([t for t in trades if 'profit' in t]), 1)
        }
    except Exception as e:
        st.error(f"Backtest error for {symbol}: {e}")
        return None

def run_parameter_optimization(symbols, days, param_ranges):
    """Run optimization across parameter ranges"""
    results = []
    
    # Create parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(itertools.product(*param_values))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_combinations = len(combinations)
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        params.update({
            'starting_capital': 100000,
            'position_size': 0.1
        })
        
        # Run backtest for first symbol (can be extended to all symbols)
        result = advanced_backtest(symbols[0], days, params)
        if result:
            results.append(result)
        
        # Update progress
        progress = (i + 1) / total_combinations
        progress_bar.progress(progress)
        status_text.text(f"Testing combination {i+1}/{total_combinations}")
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def start_bot():
    """Start the trading bot as subprocess"""
    try:
        if st.session_state.bot_process is None or st.session_state.bot_process.poll() is not None:
            st.session_state.bot_process = subprocess.Popen(['python', 'quant_bot.py'])
            st.session_state.bot_running = True
            return True
    except Exception as e:
        st.error(f"Error starting bot: {e}")
    return False

def stop_bot():
    """Stop the trading bot"""
    try:
        if st.session_state.bot_process and st.session_state.bot_process.poll() is None:
            st.session_state.bot_process.terminate()
            st.session_state.bot_process.wait(timeout=5)
        st.session_state.bot_running = False
        st.session_state.bot_process = None
        return True
    except Exception as e:
        st.error(f"Error stopping bot: {e}")
    return False

def main():
    init_session_state()
    
    st.title("🚀 Enhanced Quantitative Trading Bot")
    st.markdown("**Advanced Backtesting, Parameter Optimization & Live Trading Platform**")
    
    config = load_config()
    if not config:
        st.error("Could not load config.yaml - please check if file exists")
        st.stop()
    
    # Load symbols
    symbols = []
    try:
        with open(config['universe']['symbols_csv'], 'r') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except:
        st.warning("Could not load symbols")
        symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META']  # Fallback symbols
    
    # Enhanced Sidebar with Parameter Controls
    with st.sidebar:
        st.header("🎛️ Bot Controls")
        
        # Bot status and controls
        if st.session_state.bot_process and st.session_state.bot_process.poll() is None:
            st.success("🟢 Bot Running")
            if st.button("⏹️ Stop Bot", use_container_width=True):
                if stop_bot():
                    st.rerun()
        else:
            st.error("🔴 Bot Stopped")
            if st.button("▶️ Start Bot", use_container_width=True):
                if start_bot():
                    st.rerun()
        
        st.divider()
        
        # Trading Parameters Section
        st.header("⚙️ Trading Parameters")
        
        with st.expander("📊 Technical Indicators", expanded=True):
            rsi_period = st.slider("RSI Period", 5, 50, 14)
            rsi_oversold = st.slider("RSI Oversold", 10, 40, 30)
            rsi_overbought = st.slider("RSI Overbought", 60, 90, 70)
            
            bb_period = st.slider("Bollinger Bands Period", 10, 50, 20)
            bb_std = st.slider("BB Standard Deviations", 1.0, 3.0, 2.0, 0.1)
        
        with st.expander("💰 Risk Management", expanded=True):
            position_size = st.slider("Position Size (%)", 1, 50, 10) / 100
            stop_loss = st.slider("Stop Loss (%)", 0.5, 10.0, 2.0) / 100
            take_profit = st.slider("Take Profit (%)", 1.0, 20.0, 4.0) / 100
        
        # Store parameters in session state
        st.session_state.trading_params = {
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'bb_period': bb_period,
            'bb_std': bb_std,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'starting_capital': 100000
        }
        
        st.divider()
        
        # Bot stats
        trades_df = load_trades()
        if not trades_df.empty:
            st.metric("Total Trades", len(trades_df))
            recent_trades = trades_df[trades_df['ts'] > (datetime.now() - timedelta(hours=24))]
            st.metric("Today's Trades", len(recent_trades))
            
            if len(trades_df) > 0:
                total_pnl = trades_df['net'].sum()
                st.metric("Total P&L", f"${total_pnl:+,.2f}")
        else:
            st.metric("Total Trades", "0")
            st.metric("Today's Trades", "0")
            st.metric("Total P&L", "$0.00")
    
    # Main content tabs - New focused structure
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Day Trading", 
        "🔬 Backtest/Optimization", 
        "🚀 Live Trading", 
        "💻 Code Editor"
    ])
    
    with tab1:
        st.header("📈 Day Trading")
        st.markdown("**Real-time market data and quick order execution for active day trading**")
        
        # Real-time market data section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Quick Trade Execution")
            
            # Symbol selection for day trading
            day_trading_symbol = st.selectbox(
                "Select Symbol for Day Trading", 
                symbols[:10], 
                key="day_trading_symbol",
                help="Choose a symbol for active day trading"
            )
            
            # Get real-time data
            if day_trading_symbol:
                try:
                    ticker = yf.Ticker(day_trading_symbol)
                    # Get recent data for intraday analysis
                    hist_data = ticker.history(period="1d", interval="1m")
                    
                    if not hist_data.empty:
                        current_price = hist_data['Close'].iloc[-1]
                        price_change = hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[0]
                        price_change_pct = (price_change / hist_data['Close'].iloc[0]) * 100
                        
                        # Real-time price display
                        st.subheader(f"💰 {day_trading_symbol} Real-Time Price")
                        
                        price_cols = st.columns(4)
                        with price_cols[0]:
                            st.metric("Current Price", f"${current_price:.2f}", 
                                     delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
                        with price_cols[1]:
                            st.metric("Day High", f"${hist_data['High'].max():.2f}")
                        with price_cols[2]:
                            st.metric("Day Low", f"${hist_data['Low'].min():.2f}")
                        with price_cols[3]:
                            volume = hist_data['Volume'].sum()
                            st.metric("Volume", f"{volume:,.0f}")
                        
                        # Quick order entry
                        st.subheader("⚡ Quick Order Entry")
                        
                        order_cols = st.columns(4)
                        with order_cols[0]:
                            order_qty = st.number_input("Quantity", min_value=1, value=100, step=1)
                        with order_cols[1]:
                            order_type = st.selectbox("Order Type", ["Market", "Limit"])
                        with order_cols[2]:
                            if order_type == "Limit":
                                limit_price = st.number_input("Limit Price", value=current_price, step=0.01)
                            else:
                                limit_price = current_price
                        
                        # Buy/Sell buttons
                        button_cols = st.columns(2)
                        with button_cols[0]:
                            if st.button("🟢 BUY", use_container_width=True, type="primary"):
                                # Simulate buy order
                                st.success(f"✅ BUY order placed: {order_qty} shares of {day_trading_symbol} at ${limit_price:.2f}")
                                # Log the order (in practice, this would go to the broker)
                                st.balloons()
                        
                        with button_cols[1]:
                            if st.button("🔴 SELL", use_container_width=True):
                                # Simulate sell order
                                st.warning(f"📤 SELL order placed: {order_qty} shares of {day_trading_symbol} at ${limit_price:.2f}")
                                # Log the order (in practice, this would go to the broker)
                        
                        # Intraday chart with technical indicators
                        st.subheader("📊 Intraday Chart with Technical Indicators")
                        
                        # Calculate technical indicators for intraday
                        if len(hist_data) > 20:
                            indicators = calculate_technical_indicators(hist_data, rsi_period=14, bb_period=20, bb_std=2)
                            
                            # Create intraday chart
                            fig = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=(f'{day_trading_symbol} Price & Bollinger Bands', 'RSI'),
                                vertical_spacing=0.15,
                                row_heights=[0.7, 0.3]
                            )
                            
                            # Price chart with Bollinger Bands
                            fig.add_trace(go.Scatter(
                                x=hist_data.index, y=hist_data['Close'],
                                mode='lines', name='Price',
                                line=dict(color='blue', width=2)
                            ), row=1, col=1)
                            
                            if 'bb_upper' in indicators:
                                fig.add_trace(go.Scatter(
                                    x=hist_data.index, y=indicators['bb_upper'],
                                    mode='lines', name='BB Upper',
                                    line=dict(color='red', dash='dash', width=1)
                                ), row=1, col=1)
                                
                                fig.add_trace(go.Scatter(
                                    x=hist_data.index, y=indicators['bb_lower'],
                                    mode='lines', name='BB Lower',
                                    line=dict(color='red', dash='dash', width=1),
                                    fill='tonexty', fillcolor='rgba(255,0,0,0.1)'
                                ), row=1, col=1)
                                
                                fig.add_trace(go.Scatter(
                                    x=hist_data.index, y=indicators['bb_ma'],
                                    mode='lines', name='BB Middle',
                                    line=dict(color='orange', width=1)
                                ), row=1, col=1)
                            
                            # Moving averages
                            if 'ma_short' in indicators:
                                fig.add_trace(go.Scatter(
                                    x=hist_data.index, y=indicators['ma_short'],
                                    mode='lines', name='MA 10',
                                    line=dict(color='green', width=1)
                                ), row=1, col=1)
                                
                                fig.add_trace(go.Scatter(
                                    x=hist_data.index, y=indicators['ma_long'],
                                    mode='lines', name='MA 20',
                                    line=dict(color='purple', width=1)
                                ), row=1, col=1)
                            
                            # RSI subplot
                            if 'rsi' in indicators:
                                fig.add_trace(go.Scatter(
                                    x=hist_data.index, y=indicators['rsi'],
                                    mode='lines', name='RSI',
                                    line=dict(color='blue', width=2)
                                ), row=2, col=1)
                                
                                # RSI levels
                                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
                            
                            fig.update_layout(
                                title=f"{day_trading_symbol} - Intraday Analysis",
                                height=600,
                                showlegend=True
                            )
                            
                            fig.update_xaxes(title_text="Time", row=2, col=1)
                            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough data for technical indicators. Try a different symbol or time period.")
                    
                    else:
                        st.warning(f"No intraday data available for {day_trading_symbol}")
                        
                except Exception as e:
                    st.error(f"Error fetching data for {day_trading_symbol}: {e}")
        
        with col2:
            st.subheader("📊 Current Positions & P&L")
            
            # Load current trades to show positions
            trades_df = load_trades()
            
            if not trades_df.empty:
                # Calculate current positions
                positions = {}
                for _, trade in trades_df.iterrows():
                    symbol = trade['symbol']
                    side = trade['side']
                    qty = trade['qty']
                    price = trade['price']
                    
                    if symbol not in positions:
                        positions[symbol] = {'qty': 0, 'avg_price': 0, 'total_cost': 0}
                    
                    if side == 'buy':
                        old_total = positions[symbol]['qty'] * positions[symbol]['avg_price']
                        new_qty = positions[symbol]['qty'] + qty
                        new_total = old_total + (qty * price)
                        positions[symbol]['qty'] = new_qty
                        positions[symbol]['avg_price'] = new_total / new_qty if new_qty > 0 else 0
                        positions[symbol]['total_cost'] = new_total
                    else:  # sell
                        positions[symbol]['qty'] -= qty
                        if positions[symbol]['qty'] <= 0:
                            positions[symbol] = {'qty': 0, 'avg_price': 0, 'total_cost': 0}
                
                # Display positions
                if any(pos['qty'] > 0 for pos in positions.values()):
                    for symbol, pos in positions.items():
                        if pos['qty'] > 0:
                            try:
                                current_ticker = yf.Ticker(symbol)
                                current_data = current_ticker.history(period="1d", interval="1m")
                                if not current_data.empty:
                                    current_price = current_data['Close'].iloc[-1]
                                    market_value = pos['qty'] * current_price
                                    unrealized_pnl = market_value - pos['total_cost']
                                    pnl_pct = (unrealized_pnl / pos['total_cost']) * 100 if pos['total_cost'] > 0 else 0
                                    
                                    st.markdown(f"### 📈 {symbol}")
                                    
                                    pos_cols = st.columns(2)
                                    with pos_cols[0]:
                                        st.metric("Quantity", f"{pos['qty']:.0f}")
                                        st.metric("Avg Price", f"${pos['avg_price']:.2f}")
                                    with pos_cols[1]:
                                        st.metric("Current Price", f"${current_price:.2f}")
                                        st.metric("Market Value", f"${market_value:.2f}")
                                    
                                    st.metric("Unrealized P&L", f"${unrealized_pnl:+.2f}", 
                                             delta=f"{pnl_pct:+.2f}%")
                                    
                                    # Quick close position button
                                    if st.button(f"🔴 Close {symbol} Position", key=f"close_{symbol}", use_container_width=True):
                                        st.warning(f"📤 SELL order placed to close {pos['qty']:.0f} shares of {symbol}")
                                    
                                    st.divider()
                            except:
                                st.write(f"**{symbol}**: {pos['qty']:.0f} shares @ ${pos['avg_price']:.2f}")
                else:
                    st.info("No open positions")
            else:
                st.info("No trading history found")
            
            # Day trading performance metrics
            st.subheader("📈 Day Trading Performance")
            
            # Calculate today's performance
            today = datetime.now().date()
            if not trades_df.empty:
                today_trades = trades_df[trades_df['ts'].dt.date == today]
                
                if not today_trades.empty:
                    today_pnl = today_trades['net'].sum()
                    today_volume = (today_trades['qty'] * today_trades['price']).sum()
                    num_trades = len(today_trades)
                    
                    perf_cols = st.columns(2)
                    with perf_cols[0]:
                        st.metric("Today's P&L", f"${today_pnl:+.2f}")
                        st.metric("Today's Volume", f"${today_volume:,.0f}")
                    with perf_cols[1]:
                        st.metric("Today's Trades", f"{num_trades}")
                        avg_trade = today_pnl / num_trades if num_trades > 0 else 0
                        st.metric("Avg Trade P&L", f"${avg_trade:+.2f}")
                else:
                    st.info("No trades today")
            else:
                st.info("No trading data available")
            
            # Day trading strategies section
            st.subheader("🎯 Day Trading Strategies")
            
            strategy_type = st.selectbox(
                "Active Strategy",
                ["Scalping", "Momentum", "Breakout", "Mean Reversion"],
                help="Select your day trading strategy focus"
            )
            
            if strategy_type == "Scalping":
                st.info("🔥 **Scalping Mode**: Quick 1-5 minute trades targeting small price movements")
                st.write("• Target: 0.1-0.5% per trade")
                st.write("• Hold time: 1-5 minutes")
                st.write("• Risk: Very tight stops")
            elif strategy_type == "Momentum":
                st.info("⚡ **Momentum Mode**: Following strong price movements with volume")
                st.write("• Target: 0.5-2% per trade")
                st.write("• Hold time: 5-30 minutes")
                st.write("• Risk: Trend reversal")
            elif strategy_type == "Breakout":
                st.info("🚀 **Breakout Mode**: Trading key support/resistance breaks")
                st.write("• Target: 1-3% per trade")
                st.write("• Hold time: 15-60 minutes")
                st.write("• Risk: False breakouts")
            elif strategy_type == "Mean Reversion":
                st.info("🔄 **Mean Reversion Mode**: Trading oversold/overbought conditions")
                st.write("• Target: 0.5-1.5% per trade")
                st.write("• Hold time: 10-45 minutes")
                st.write("• Risk: Trend continuation")
            
            # Risk management for day trading
            st.subheader("🛡️ Day Trading Risk Management")
            
            max_loss_day = st.slider("Max Daily Loss ($)", 100, 5000, 1000, step=100)
            max_position_size = st.slider("Max Position Size (%)", 5, 50, 25, step=5)
            
            # Display risk metrics
            st.write(f"**Daily Loss Limit**: ${max_loss_day}")
            st.write(f"**Max Position Size**: {max_position_size}% of portfolio")
            
            if not trades_df.empty and not today_trades.empty:
                remaining_loss = max_loss_day + today_pnl  # today_pnl is negative if losing
                if remaining_loss < 0:
                    st.error(f"🚨 Daily loss limit exceeded by ${-remaining_loss:.2f}")
                else:
                    st.success(f"💚 ${remaining_loss:.2f} remaining before daily limit")
        st.header("🚀 QuantConnect-Style Parameter Optimization")
        st.markdown("**Professional-grade parameter optimization with automatic range generation and comprehensive analysis**")
        
        # Prominent intro section
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3>🎯 How It Works</h3>
        <p><strong>1.</strong> 📊 Select symbols and time period<br>
        <strong>2.</strong> 🎛️ Choose strategy type (auto-generates optimal parameter ranges)<br>
        <strong>3.</strong> 🚀 Click "Optimize & Backtest" - system tests hundreds of combinations<br>
        <strong>4.</strong> 📈 Review comprehensive results with rankings and analysis<br>
        <strong>5.</strong> ✅ Apply best parameters to your bot with one click</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mode selection
        col_mode1, col_mode2 = st.columns([3, 1])
        with col_mode1:
            optimization_mode = st.radio(
                "Optimization Mode",
                ["🎯 Simple Mode (Recommended)", "⚙️ Advanced Mode (Expert)"],
                horizontal=True,
                help="Simple Mode: Fully automated with smart ranges | Advanced Mode: Manual configuration"
            )
        with col_mode2:
            if st.button("ℹ️ Help", key="quantconnect_optimization_help", use_container_width=True):
                st.info("""
                **Simple Mode**: Automatically optimizes parameters with smart ranges - perfect for most users
                **Advanced Mode**: Manual parameter range configuration for experts
                """)
        
        # Import new optimization classes
        try:
            from parameter_manager import ParameterManager, create_default_parameters
            from optimization_engine import OptimizationEngine
            from results_analyzer import ResultsAnalyzer
            
            if optimization_mode.startswith("🎯"):
                # SIMPLE MODE - QuantConnect Style - Enhanced
                st.subheader("🎯 Simple Optimization Setup")
                st.markdown("*Let our AI choose optimal parameter ranges for you*")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### **Configuration**")
                    
                    # Symbol selection (auto-selected from scanner if available)
                    opt_symbols = symbols[:15]  # More symbols available
                    if 'auto_selected_optimize' in st.session_state:
                        default_symbol = st.session_state['auto_selected_optimize']
                        st.success(f"🔍 Auto-selected from Smart Scanner: **{default_symbol}**")
                        del st.session_state['auto_selected_optimize']
                    else:
                        default_symbol = opt_symbols[0]
                    
                    selected_symbols = st.multiselect(
                        "🎯 **Symbols to Optimize**",
                        opt_symbols,
                        default=[default_symbol],
                        help="Select one or more symbols for optimization. More symbols = longer optimization time."
                    )
                    
                    # Time period
                    period_options = {
                        "1 Month (Quick)": 30,
                        "2 Months (Balanced)": 60, 
                        "3 Months (Thorough)": 90,
                        "4 Months (Comprehensive)": 120
                    }
                    selected_period_name = st.selectbox(
                        "📅 **Backtest Period**", 
                        list(period_options.keys()), 
                        index=1,
                        help="Longer periods provide more reliable results but take more time"
                    )
                    opt_period = period_options[selected_period_name]
                    
                    # Strategy type with descriptions
                    strategy_options = {
                        "RSI + Bollinger Bands": "Most popular - works well in trending and ranging markets",
                        "Momentum": "Best for trending markets - captures price momentum", 
                        "Mean Reversion": "Best for ranging markets - profits from price extremes"
                    }
                    
                    strategy_type = st.selectbox(
                        "📈 **Strategy Type**",
                        list(strategy_options.keys()),
                        help="Each strategy has optimized parameter ranges for its market style"
                    )
                    
                    # Show strategy description
                    st.info(f"💡 **{strategy_type}**: {strategy_options[strategy_type]}")
                    
                    # Optimization objective
                    objective_options = {
                        "Sharpe Ratio": "Risk-adjusted returns (recommended)",
                        "Total Return": "Maximize total profit",
                        "Calmar Ratio": "Return vs maximum drawdown", 
                        "Sortino Ratio": "Downside risk-adjusted returns"
                    }
                    
                    objective = st.selectbox(
                        "🎯 **Optimization Goal**",
                        list(objective_options.keys()),
                        help="What metric to optimize for"
                    )
                    
                    # Show objective description
                    st.caption(f"📊 {objective_options[objective]}")
                    
                    # Max combinations with intelligent defaults
                    est_combinations = st.selectbox(
                        "⚡ **Optimization Depth**",
                        ["Quick (50 combinations)", "Balanced (200 combinations)", "Thorough (500 combinations)", "Comprehensive (1000 combinations)"],
                        index=1,
                        help="More combinations = better results but longer time"
                    )
                    max_combinations = int(est_combinations.split("(")[1].split(" ")[0])
                    
                    # Show parameter preview
                    strategy_map = {
                        "RSI + Bollinger Bands": "rsi_bollinger",
                        "Momentum": "momentum", 
                        "Mean Reversion": "mean_reversion"
                    }
                    
                    preview_params = create_default_parameters(strategy_map[strategy_type])
                    param_info = preview_params.get_parameter_info()
                    
                    with st.expander("📋 **Parameter Ranges Preview**", expanded=False):
                        st.markdown(f"**Total possible combinations**: {param_info['total_combinations']:,}")
                        st.markdown("**Parameters to optimize:**")
                        for name, info in param_info['parameters'].items():
                            range_desc = f"{info['min_value']} to {info['max_value']} (step {info['step']})"
                            st.markdown(f"• **{name.replace('_', ' ').title()}**: {range_desc} = {info['total_values']} values")
                    
                    st.markdown("---")
                    
                    # Enhanced optimization button
                    if st.button("🚀 **OPTIMIZE & BACKTEST**", 
                                use_container_width=True, 
                                type="primary",
                                help="Start the QuantConnect-style optimization process"):
                        if selected_symbols:
                            # Initialize parameter manager with smart ranges
                            param_manager = create_default_parameters(strategy_map[strategy_type])
                            
                            # Initialize optimization engine
                            engine = OptimizationEngine(max_workers=4)
                            
                            # Set up progress tracking
                            progress_placeholder = st.empty()
                            status_placeholder = st.empty()
                            
                            def progress_callback(current, total, status):
                                progress_placeholder.progress(current / total)
                                status_placeholder.text(status)
                            
                            engine.set_progress_callback(progress_callback)
                            
                            # Run optimization
                            with st.spinner("🚀 Running QuantConnect-style optimization..."):
                                try:
                                    objective_map = {
                                        "Sharpe Ratio": "sharpe_ratio",
                                        "Total Return": "total_return", 
                                        "Calmar Ratio": "calmar_ratio",
                                        "Sortino Ratio": "sortino_ratio"
                                    }
                                    
                                    summary = engine.run_optimization(
                                        parameter_manager=param_manager,
                                        symbols=selected_symbols,
                                        days=opt_period,
                                        objective=objective_map[objective],
                                        max_combinations=max_combinations
                                    )
                                    
                                    st.session_state['quantconnect_optimization'] = {
                                        'summary': summary,
                                        'analyzer': ResultsAnalyzer(),
                                        'objective': objective_map[objective],
                                        'strategy_type': strategy_type,
                                        'symbols': selected_symbols,
                                        'period': selected_period_name
                                    }
                                    
                                    # Initialize analyzer
                                    st.session_state['quantconnect_optimization']['analyzer'].analyze_results(summary)
                                    
                                    progress_placeholder.empty()
                                    status_placeholder.empty()
                                    
                                    if summary.successful_runs > 0:
                                        st.success(f"🎉 Optimization completed! Tested {summary.successful_runs} combinations in {summary.total_time:.1f}s")
                                        st.balloons()
                                    else:
                                        st.warning("⚠️ Optimization completed but no successful results. Try different symbols or check your internet connection.")
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Optimization failed: {e}")
                                    st.info("💡 This may be due to network connectivity issues. The system works with live market data.")
                                    progress_placeholder.empty()
                                    status_placeholder.empty()
                        else:
                            st.warning("⚠️ Please select at least one symbol to optimize")
                
                with col2:
                    # Enhanced Results Display
                    if 'quantconnect_optimization' in st.session_state:
                        opt_data = st.session_state['quantconnect_optimization']
                        summary = opt_data['summary']
                        analyzer = opt_data['analyzer']
                        objective = opt_data['objective']
                        strategy_type = opt_data.get('strategy_type', 'Unknown')
                        symbols = opt_data.get('symbols', [])
                        period = opt_data.get('period', 'Unknown')
                        
                        st.markdown("### 🏆 **Optimization Results**")
                        st.markdown(f"*Strategy: {strategy_type} | Symbols: {', '.join(symbols)} | Period: {period}*")
                        
                        if summary.successful_runs > 0:
                            # Quick stats with enhanced styling
                            col2a, col2b, col2c, col2d = st.columns(4)
                            with col2a:
                                st.metric("🧪 Tests", f"{summary.successful_runs:,}", help="Successful parameter combinations tested")
                            with col2b:
                                best_score = getattr(summary.best_result, objective) if summary.best_result else 0
                                st.metric("🏆 Best Score", f"{best_score:.3f}", help=f"Best {objective.replace('_', ' ').title()}")
                            with col2c:
                                st.metric("⏱️ Time", f"{summary.total_time:.1f}s", help="Total optimization time")
                            with col2d:
                                success_rate = (summary.successful_runs / summary.total_combinations) * 100
                                st.metric("✅ Success", f"{success_rate:.1f}%", help="Percentage of successful tests")
                            
                            # Best parameters display with enhanced styling
                            if summary.best_result:
                                st.markdown("### ✨ **Optimal Parameters Found**")
                                
                                # Create a nice parameter display
                                params = summary.best_result.parameters
                                param_items = list(params.items())
                                
                                # Display parameters in a grid
                                num_cols = 3
                                param_cols = st.columns(num_cols)
                                
                                for i, (param, value) in enumerate(param_items):
                                    col_idx = i % num_cols
                                    with param_cols[col_idx]:
                                        param_name = param.replace('_', ' ').title()
                                        if isinstance(value, float):
                                            if 0 < value < 1:
                                                st.metric(f"🎛️ {param_name}", f"{value:.1%}")
                                            else:
                                                st.metric(f"🎛️ {param_name}", f"{value:.3f}")
                                        else:
                                            st.metric(f"🎛️ {param_name}", str(value))
                                
                                # Performance metrics with better formatting
                                st.markdown("### 📊 **Performance Metrics**")
                                perf_cols = st.columns(4)
                                with perf_cols[0]:
                                    return_val = summary.best_result.total_return
                                    return_delta = f"+{return_val:.2%}" if return_val > 0 else f"{return_val:.2%}"
                                    st.metric("💰 Return", f"{return_val:.2%}", delta=return_delta)
                                with perf_cols[1]:
                                    sharpe_val = summary.best_result.sharpe_ratio
                                    sharpe_status = "Excellent" if sharpe_val > 2 else "Good" if sharpe_val > 1 else "Fair"
                                    st.metric("📈 Sharpe", f"{sharpe_val:.3f}", help=f"Risk-adjusted return: {sharpe_status}")
                                with perf_cols[2]:
                                    dd_val = summary.best_result.max_drawdown
                                    st.metric("📉 Max DD", f"{dd_val:.2%}", delta=f"{dd_val:.2%}")
                                with perf_cols[3]:
                                    wr_val = summary.best_result.win_rate
                                    st.metric("🎯 Win Rate", f"{wr_val:.1%}", help="Percentage of profitable trades")
                                
                                # Action buttons
                                st.markdown("### ⚡ **Quick Actions**")
                                action_cols = st.columns(3)
                                
                                with action_cols[0]:
                                    if st.button("✅ **Apply to Bot**", use_container_width=True, type="primary"):
                                        # Update session state trading parameters
                                        st.session_state.trading_params.update(summary.best_result.parameters)
                                        st.success("🎉 Best parameters applied to trading bot!")
                                        st.balloons()
                                
                                with action_cols[1]:
                                    if st.button("📊 **View Analysis**", use_container_width=True):
                                        st.session_state['show_detailed_analysis'] = True
                                        st.rerun()
                                
                                with action_cols[2]:
                                    if st.button("💾 **Export Results**", use_container_width=True):
                                        filename = analyzer.export_results_to_csv()
                                        st.success(f"📥 Results exported to {filename}")
                                
                                # Quick performance visualization
                                if len(summary.results) > 1:
                                    st.markdown("### 📈 **Performance Distribution**")
                                    fig_dist = analyzer.create_performance_distribution_chart(objective)
                                    st.plotly_chart(fig_dist, use_container_width=True)
                        
                        else:
                            st.warning("⚠️ No successful optimizations found")
                            st.info("""
                            **Possible causes:**
                            - Network connectivity issues (system needs internet access for market data)
                            - Selected symbols may not have sufficient data
                            - Try different symbols or time periods
                            
                            **💡 Tip**: The system works best with popular stocks like AAPL, MSFT, TSLA, etc.
                            """)
                    
                    else:
                        # Enhanced getting started guide
                        st.markdown("### 🚀 **Ready to Optimize?**")
                        st.markdown("""
                        <div style='background-color: #e8f4fd; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4;'>
                        <h4>👈 Configure your optimization settings</h4>
                        <p>1. 📊 Choose your symbols<br>
                        2. 📅 Select time period<br>
                        3. 📈 Pick strategy type<br>
                        4. 🚀 Click "OPTIMIZE & BACKTEST"</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("### 🎯 **What You'll Get**")
                        feature_cols = st.columns(2)
                        with feature_cols[0]:
                            st.markdown("""
                            **📊 Comprehensive Analysis:**
                            - Best parameter combinations
                            - Performance rankings
                            - Risk-adjusted metrics
                            - Win rates and drawdowns
                            """)
                        with feature_cols[1]:
                            st.markdown("""
                            **🚀 Advanced Features:**
                            - Parameter sensitivity heatmaps
                            - Robustness testing
                            - Equity curve comparisons
                            - One-click parameter application
                            """)
                        
                        # Sample results preview
                        st.markdown("### 📈 **Example Results**")
                        sample_data = {
                            'Rank': [1, 2, 3, 4, 5],
                            'RSI Period': [14, 12, 16, 10, 18],
                            'BB Period': [20, 25, 15, 30, 22],
                            'Position Size': ['10%', '15%', '8%', '20%', '12%'],
                            'Sharpe Ratio': [2.45, 2.31, 2.18, 2.05, 1.98],
                            'Return': ['24.5%', '22.1%', '20.8%', '19.2%', '18.7%']
                        }
                        st.dataframe(pd.DataFrame(sample_data), use_container_width=True, hide_index=True)
            
            else:
                # ADVANCED MODE - Manual Configuration (existing functionality enhanced)
                st.subheader("⚙️ Advanced Manual Configuration")
                st.markdown("*For expert users who want full control over parameter ranges*")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### **Manual Settings**")
                    
                    opt_symbol = st.selectbox("🎯 **Symbol**", symbols[:10])
                    
                    # Check if auto-selected from scanner
                    if 'auto_selected_optimize' in st.session_state:
                        opt_symbol = st.session_state['auto_selected_optimize']
                        st.success(f"🔍 Auto-selected from Smart Scanner: **{opt_symbol}**")
                        del st.session_state['auto_selected_optimize']
                    
                    opt_period = st.selectbox("📅 **Period (days)**", [30, 60, 90, 120])
                    
                    st.markdown("### **Parameter Ranges**")
                    
                    # RSI optimization ranges
                    rsi_min, rsi_max = st.slider("📊 **RSI Period Range**", 5, 50, (10, 25))
                    rsi_step = st.selectbox("RSI Step", [1, 2, 5], index=1)
                    
                    # Bollinger Bands ranges
                    bb_min, bb_max = st.slider("📈 **BB Period Range**", 10, 50, (15, 30))
                    bb_step = st.selectbox("BB Step", [1, 2, 5], index=1)
                    
                    # Position size range
                    pos_min, pos_max = st.slider("💰 **Position Size Range (%)**", 1, 30, (5, 20))
                    
                    # Estimated combinations
                    est_combinations = (rsi_max - rsi_min + 1) // rsi_step * (bb_max - bb_min + 1) // bb_step * (pos_max - pos_min + 1) // 2
                    st.info(f"📊 Estimated combinations: **{est_combinations:,}**")
                    
                    if st.button("🔥 **Start Advanced Optimization**", use_container_width=True, type="primary"):
                        st.info("Advanced optimization feature available - see Simple Mode for demo")
        
        except ImportError as e:
            st.error(f"Error importing optimization modules: {e}")
            st.info("Make sure parameter_manager.py, optimization_engine.py, and results_analyzer.py are in the project directory")
        
        with subtab2:
            # Smart Scanner content
            st.subheader("🔍 Intelligent Symbol Scanner")
            st.markdown("**Discover optimal trading opportunities automatically**")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Scanner Configuration")
                
                # Market categories selection
                from symbol_scanner import MarketCategoryScanner, cached_smart_scan
                
                st.write("**Market Categories to Scan:**")
                available_categories = list(MarketCategoryScanner.MARKET_CATEGORIES.keys())
                selected_categories = st.multiselect(
                    "Categories",
                    available_categories,
                    default=['SP500', 'NASDAQ100'],
                    help="Select market categories to scan for opportunities",
                    key="scanner_categories"
                )
                
                # Scanning mode
                scanning_mode = st.selectbox(
                    "Scanning Mode",
                    ['Conservative', 'Balanced', 'Aggressive'],
                    index=1,
                    help="""
                    • Conservative: Focus on stable trends and low volatility
                    • Balanced: Mix of momentum, trend, and stability factors  
                    • Aggressive: Prioritize momentum and high volatility opportunities
                    """,
                    key="scanner_mode"
                )
                
                # Advanced filters
                with st.expander("🔧 Advanced Filters", expanded=False):
                    st.write("**Price Range:**")
                    price_range = st.slider("Price Range ($)", 1, 1000, (10, 500), key="scanner_price_range")
                    
                    st.write("**Minimum Volume:**")
                    min_volume = st.number_input("Min Daily Volume", value=100000, step=50000, key="scanner_min_volume")
                    
                    st.write("**Minimum Score:**")
                    min_score = st.slider("Minimum Score", 0, 100, 60, key="scanner_min_score")
                
                # Number of results
                top_n = st.slider("Top N Results", 5, 50, 20, key="scanner_top_n")
                
                # Smart Scan button
                if st.button("🔍 **Smart Scan**", use_container_width=True, type="primary", key="smart_scan_button"):
                    if selected_categories:
                        with st.spinner("🔍 Scanning markets for opportunities..."):
                            try:
                                # Create filter dictionary
                                filters = {
                                    'min_price': price_range[0],
                                    'max_price': price_range[1],
                                    'min_volume': min_volume,
                                    'min_score': min_score
                                }
                                
                                # Perform smart scan with caching
                                results = cached_smart_scan(
                                    categories_tuple=tuple(selected_categories),
                                    mode=scanning_mode.lower(),
                                    top_n=top_n,
                                    filters_key=f"{price_range}_{min_volume}_{min_score}"
                                )
                                
                                if results:
                                    st.session_state['backtest_scan_results'] = results
                                    st.session_state['backtest_scan_mode'] = scanning_mode
                                    st.success(f"✅ Found {len(results)} high-potential symbols!")
                                else:
                                    st.warning("No symbols found matching the criteria. Try adjusting filters.")
                                    
                            except Exception as e:
                                st.error(f"Scanning error: {e}")
                    else:
                        st.warning("Please select at least one market category to scan")
            
            with col2:
                if 'backtest_scan_results' in st.session_state and st.session_state['backtest_scan_results']:
                    results = st.session_state['backtest_scan_results']
                    scan_mode = st.session_state.get('backtest_scan_mode', 'Unknown')
                    
                    st.subheader(f"📊 Scan Results ({scan_mode} Mode)")
                    
                    # Results summary
                    col2a, col2b, col2c, col2d = st.columns(4)
                    with col2a:
                        st.metric("Symbols Found", len(results))
                    with col2b:
                        avg_score = sum(r['score'] for r in results) / len(results)
                        st.metric("Average Score", f"{avg_score:.1f}")
                    with col2c:
                        top_score = max(r['score'] for r in results)
                        st.metric("Top Score", f"{top_score:.1f}")
                    with col2d:
                        price_range = f"${min(r['price'] for r in results):.0f}-${max(r['price'] for r in results):.0f}"
                        st.metric("Price Range", price_range)
                    
                    # Results table with enhanced display
                    st.subheader("🏆 Top Opportunities")
                    
                    # Create enhanced dataframe for display
                    display_data = []
                    for i, result in enumerate(results):
                        # Create score breakdown
                        sub_scores = result['sub_scores']
                        score_breakdown = f"M:{sub_scores.get('momentum', 0):.0f} T:{sub_scores.get('trend', 0):.0f} R:{sub_scores.get('rsi', 0):.0f}"
                        
                        display_data.append({
                            'Rank': i + 1,
                            'Symbol': result['symbol'],
                            'Score': f"{result['score']:.1f}",
                            'Price': f"${result['price']:.2f}",
                            'Volume': f"{result['volume']:,.0f}",
                            'Breakdown': score_breakdown,
                            'Reasoning': result['reasoning'][:50] + "..." if len(result['reasoning']) > 50 else result['reasoning']
                        })
                    
                    df_display = pd.DataFrame(display_data)
                    st.dataframe(df_display, use_container_width=True, height=400)
                    
                    # Quick actions
                    st.subheader("⚡ Quick Actions")
                    col2a, col2b, col2c = st.columns(3)
                    
                    with col2a:
                        if st.button("📈 Send to Backtesting", use_container_width=True, key="send_to_backtest"):
                            # Auto-populate backtest tab with top 5 symbols
                            top_5_symbols = [r['symbol'] for r in results[:5]]
                            st.session_state['auto_selected_backtest_symbols'] = top_5_symbols
                            st.info(f"Selected top 5 symbols for backtesting: {', '.join(top_5_symbols)}")
                    
                    with col2b:
                        if st.button("🎯 Send to Optimization", use_container_width=True, key="send_to_optimize"):
                            # Auto-select best symbol for optimization
                            best_symbol = results[0]['symbol']
                            st.session_state['auto_selected_optimize'] = best_symbol
                            st.info(f"Selected {best_symbol} for parameter optimization")
                    
                    with col2c:
                        # Export results
                        if st.button("💾 Export Results", use_container_width=True, key="export_scan_results"):
                            csv_data = pd.DataFrame(results).to_csv(index=False)
                            st.download_button(
                                "📥 Download CSV",
                                csv_data,
                                f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                "text/csv",
                                use_container_width=True,
                                key="download_scan_csv"
                            )
                
                else:
                    st.info("👆 Configure your scan settings and click '🔍 Smart Scan' to discover trading opportunities!")
        
        with subtab3:
            # Strategy Backtesting content
            st.subheader("📊 Advanced Strategy Backtesting")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Backtest Configuration")
                
                # Symbol selection
                # Check if auto-selected symbols are available from scanner
                default_symbols = st.session_state.get('auto_selected_backtest_symbols', symbols[:3])
                selected_symbols = st.multiselect(
                    "Select Symbols", 
                    symbols[:20], 
                    default=default_symbols,
                    key="backtest_symbols"
                )
                
                # Show if symbols were auto-selected
                if 'auto_selected_backtest_symbols' in st.session_state:
                    st.info(f"🔍 Auto-selected from Smart Scanner: {', '.join(st.session_state['auto_selected_backtest_symbols'])}")
                
                # Time period
                test_periods = {
                    "1 Week": 7,
                    "2 Weeks": 14,
                    "1 Month": 30,
                    "3 Months": 90,
                    "6 Months": 180
                }
                
                selected_period = st.selectbox("Test Period", list(test_periods.keys()), key="backtest_period")
                
                # Use current parameters or custom
                use_current_params = st.checkbox("Use Current Parameters", value=True, key="use_current_params")
                
                if not use_current_params:
                    st.subheader("Custom Parameters")
                    custom_rsi = st.slider("Custom RSI Period", 5, 50, 14, key="custom_rsi_backtest")
                    custom_bb = st.slider("Custom BB Period", 10, 50, 20, key="custom_bb_backtest")
                    # Add more custom parameters as needed
                
                if st.button("🚀 Run Advanced Backtest", use_container_width=True, key="run_advanced_backtest"):
                    if selected_symbols:
                        params = st.session_state.trading_params if use_current_params else {
                            'rsi_period': custom_rsi,
                            'bb_period': custom_bb,
                            # Add other custom params
                            'rsi_oversold': 30,
                            'rsi_overbought': 70,
                            'bb_std': 2.0,
                            'position_size': 0.1,
                            'stop_loss': 0.02,
                            'take_profit': 0.04,
                            'starting_capital': 100000
                        }
                        
                        with st.spinner("Running advanced backtests..."):
                            results = {}
                            for symbol in selected_symbols:
                                result = advanced_backtest(symbol, test_periods[selected_period], params)
                                if result:
                                    results[symbol] = result
                            
                            if results:
                                st.session_state['advanced_backtest_results'] = results
                                st.success(f"✅ Completed backtests for {len(results)} symbols!")
                    else:
                        st.warning("Please select at least one symbol")
            
            with col2:
                if 'advanced_backtest_results' in st.session_state:
                    results = st.session_state['advanced_backtest_results']
                    
                    st.subheader("📊 Backtest Results Comparison")
                    
                    # Summary table
                    summary_data = []
                    for symbol, result in results.items():
                        summary_data.append({
                            'Symbol': symbol,
                            'Return': f"{result['total_return']:.2%}",
                            'Sharpe': f"{result['sharpe_ratio']:.2f}",
                            'Max DD': f"{result['max_drawdown']:.2%}",
                            'Win Rate': f"{result['win_rate']:.1%}",
                            'Final Value': f"${result['final_value']:,.0f}"
                        })
                    
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                    
                    # Combined equity curves
                    fig = go.Figure()
                    for symbol, result in results.items():
                        fig.add_trace(go.Scatter(
                            x=result['equity_curve'].index,
                            y=result['equity_curve'].values,
                            mode='lines',
                            name=symbol,
                            line=dict(width=2)
                        ))
                    
                    fig.update_layout(
                        title="Portfolio Equity Curves Comparison",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with subtab4:
            # Strategy Comparison content
            st.subheader("📈 Multi-Strategy Comparison")
            
            st.subheader("🎯 Strategy Templates")
            
            # Pre-defined strategy templates
            strategies = {
                "Conservative": {
                    'rsi_period': 20, 'rsi_oversold': 25, 'rsi_overbought': 75,
                    'bb_period': 25, 'bb_std': 2.5, 'position_size': 0.05, 'stop_loss': 0.015
                },
                "Aggressive": {
                    'rsi_period': 10, 'rsi_oversold': 35, 'rsi_overbought': 65,
                    'bb_period': 15, 'bb_std': 1.5, 'position_size': 0.20, 'stop_loss': 0.03
                },
                "Balanced": {
                    'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
                    'bb_period': 20, 'bb_std': 2.0, 'position_size': 0.10, 'stop_loss': 0.02
                },
                "Current": st.session_state.get('trading_params', {})
            }
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_strategies = st.multiselect(
                    "Select Strategies to Compare",
                    list(strategies.keys()),
                    default=["Conservative", "Balanced", "Aggressive"],
                    key="strategy_comparison_select"
                )
                
                comparison_symbol = st.selectbox("Comparison Symbol", symbols[:10], key="comp_symbol")
                comparison_period = st.selectbox("Comparison Period", [30, 60, 90], key="comp_period")
                
                if st.button("🔥 Compare Strategies", use_container_width=True, key="compare_strategies"):
                    if selected_strategies:
                        with st.spinner("Running strategy comparison..."):
                            strategy_results = {}
                            for strategy_name in selected_strategies:
                                if strategy_name in strategies:
                                    params = strategies[strategy_name].copy()
                                    params.update({'starting_capital': 100000, 'take_profit': 0.04})
                                    result = advanced_backtest(comparison_symbol, comparison_period, params)
                                    if result:
                                        strategy_results[strategy_name] = result
                            
                            if strategy_results:
                                st.session_state['strategy_comparison'] = strategy_results
                                st.success(f"✅ Compared {len(strategy_results)} strategies!")
            
            with col2:
                if 'strategy_comparison' in st.session_state:
                    results = st.session_state['strategy_comparison']
                    
                    st.subheader("📊 Strategy Performance Comparison")
                    
                    # Comparison metrics
                    comparison_data = []
                    for strategy, result in results.items():
                        comparison_data.append({
                            'Strategy': strategy,
                            'Return': f"{result['total_return']:.2%}",
                            'Sharpe': f"{result['sharpe_ratio']:.2f}",
                            'Volatility': f"{result['volatility']:.2%}",
                            'Max DD': f"{result['max_drawdown']:.2%}",
                            'Win Rate': f"{result['win_rate']:.1%}",
                            'Final Value': f"${result['final_value']:,.0f}"
                        })
                    
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                    
                    # Strategy equity curves
                    fig = go.Figure()
                    colors = ['blue', 'red', 'green', 'orange', 'purple']
                    for i, (strategy, result) in enumerate(results.items()):
                        fig.add_trace(go.Scatter(
                            x=result['equity_curve'].index,
                            y=result['equity_curve'].values,
                            mode='lines',
                            name=strategy,
                            line=dict(width=3, color=colors[i % len(colors)])
                        ))
                    
                    fig.update_layout(
                        title="Strategy Performance Comparison",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=500,
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("🚀 Live Trading")
        st.markdown("**Live strategy execution controls, portfolio monitoring, and performance tracking**")
        
        # Real-time controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("💾 Export All Data", use_container_width=True):
                trades_df = load_trades()
                if not trades_df.empty:
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        "enhanced_trades.csv",
                        "text/csv",
                        use_container_width=True
                    )
        with col3:
            if st.button("📊 Generate Report", use_container_width=True):
                # Generate comprehensive trading report
                st.info("📈 Comprehensive report generation coming soon!")
        
        # Live strategy execution controls
        st.subheader("⚡ Live Strategy Controls")
        
        # Bot control section with enhanced status
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🤖 Bot Status & Control")
            
            # Check bot status
            bot_status = "Running" if st.session_state.bot_running and st.session_state.bot_process and st.session_state.bot_process.poll() is None else "Stopped"
            
            if bot_status == "Running":
                st.success("🟢 **Bot Status**: Live Trading Active")
                
                # Show current trading parameters being used
                st.write("**Current Parameters:**")
                if 'trading_params' in st.session_state:
                    params = st.session_state.trading_params
                    param_cols = st.columns(2)
                    with param_cols[0]:
                        st.write(f"• RSI Period: {params.get('rsi_period', 'N/A')}")
                        st.write(f"• BB Period: {params.get('bb_period', 'N/A')}")
                        st.write(f"• Position Size: {params.get('position_size', 'N/A'):.1%}")
                    with param_cols[1]:
                        st.write(f"• Stop Loss: {params.get('stop_loss', 'N/A'):.1%}")
                        st.write(f"• Take Profit: {params.get('take_profit', 'N/A'):.1%}")
                        st.write(f"• RSI Oversold: {params.get('rsi_oversold', 'N/A')}")
                
                if st.button("⏹️ **Stop Live Trading**", use_container_width=True, type="secondary"):
                    if stop_bot():
                        st.success("✅ Live trading stopped successfully")
                        st.rerun()
                    else:
                        st.error("❌ Failed to stop bot")
            else:
                st.error("🔴 **Bot Status**: Not Running")
                
                if st.button("▶️ **Start Live Trading**", use_container_width=True, type="primary"):
                    if start_bot():
                        st.success("✅ Live trading started successfully")
                        st.rerun()
                    else:
                        st.error("❌ Failed to start bot")
        
        with col2:
            st.subheader("🎛️ Risk Management Settings")
            
            # Live risk management controls
            daily_loss_limit = st.slider("Daily Loss Limit ($)", 100, 10000, 2000, step=100, key="live_daily_loss")
            max_concurrent_trades = st.slider("Max Concurrent Trades", 1, 10, 3, key="live_max_trades")
            emergency_stop_loss = st.slider("Emergency Stop Loss (%)", 1, 20, 10, key="live_emergency_stop")
            
            # Risk status indicators
            trades_df = load_trades()
            if not trades_df.empty:
                today = datetime.now().date()
                today_trades = trades_df[trades_df['ts'].dt.date == today]
                today_pnl = today_trades['net'].sum() if not today_trades.empty else 0
                
                # Risk alerts
                if today_pnl < -daily_loss_limit:
                    st.error(f"🚨 Daily loss limit exceeded: ${today_pnl:.2f}")
                elif today_pnl < -daily_loss_limit * 0.8:
                    st.warning(f"⚠️ Approaching daily loss limit: ${today_pnl:.2f}")
                else:
                    remaining = daily_loss_limit + today_pnl
                    st.success(f"💚 Risk status: ${remaining:.2f} remaining")
            
            # Emergency controls
            st.markdown("### 🚨 Emergency Controls")
            if st.button("🛑 **EMERGENCY STOP ALL**", use_container_width=True):
                if stop_bot():
                    st.error("🚨 EMERGENCY STOP ACTIVATED - All trading halted")
                    st.balloons()  # Ironically celebratory for stopping losses
        
        # Portfolio monitoring section
        st.subheader("📊 Portfolio Monitoring")
        
        # Enhanced trade monitoring
        trades_df = load_trades()
        
        if not trades_df.empty:
            # Real-time performance metrics
            st.subheader("⚡ Real-Time Performance")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                total_trades = len(trades_df)
                st.metric("Total Trades", total_trades)
            with col2:
                profitable_trades = (trades_df['net'] > 0).sum()
                win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                total_pnl = trades_df['net'].sum()
                st.metric("Total P&L", f"${total_pnl:+,.2f}")
            with col4:
                avg_trade = trades_df['net'].mean() if total_trades > 0 else 0
                st.metric("Avg Trade", f"${avg_trade:+.2f}")
            with col5:
                max_win = trades_df['net'].max() if total_trades > 0 else 0
                st.metric("Best Trade", f"${max_win:+.2f}")
            
            # Enhanced trade table with filtering
            st.subheader("📊 Live Trade History & Analysis")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                symbol_filter = st.selectbox("Filter by Symbol", ['All'] + list(trades_df['symbol'].unique()), key="live_symbol_filter")
            with col2:
                side_filter = st.selectbox("Filter by Side", ['All', 'buy', 'sell'], key="live_side_filter")
            with col3:
                date_filter = st.selectbox("Filter by Date", ['All', 'Today', 'Last 7 days', 'Last 30 days'], key="live_date_filter")
            
            # Apply filters
            filtered_df = trades_df.copy()
            if symbol_filter != 'All':
                filtered_df = filtered_df[filtered_df['symbol'] == symbol_filter]
            if side_filter != 'All':
                filtered_df = filtered_df[filtered_df['side'] == side_filter]
            if date_filter != 'All':
                if date_filter == 'Today':
                    filtered_df = filtered_df[filtered_df['ts'].dt.date == datetime.now().date()]
                elif date_filter == 'Last 7 days':
                    filtered_df = filtered_df[filtered_df['ts'] > (datetime.now() - timedelta(days=7))]
                elif date_filter == 'Last 30 days':
                    filtered_df = filtered_df[filtered_df['ts'] > (datetime.now() - timedelta(days=30))]
            
            # Enhanced display
            if not filtered_df.empty:
                display_df = filtered_df.copy()
                display_df['Time'] = display_df['ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
                display_df['Price'] = display_df['price'].apply(lambda x: f"${x:.4f}")
                display_df['Quantity'] = display_df['qty'].apply(lambda x: f"{x:.4f}")
                display_df['P&L'] = display_df['net'].apply(lambda x: f"${x:+.2f}")
                display_df['Portfolio'] = display_df['equity_after'].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(
                    display_df[['Time', 'symbol', 'side', 'Quantity', 'Price', 'P&L', 'Portfolio']],
                    use_container_width=True
                )
                
                # P&L distribution chart
                fig = px.histogram(
                    filtered_df, 
                    x='net', 
                    title='P&L Distribution',
                    labels={'net': 'P&L ($)', 'count': 'Number of Trades'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance over time
                st.subheader("📈 Portfolio Performance Over Time")
                
                # Calculate cumulative P&L
                filtered_df_sorted = filtered_df.sort_values('ts')
                filtered_df_sorted['cumulative_pnl'] = filtered_df_sorted['net'].cumsum()
                
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Scatter(
                    x=filtered_df_sorted['ts'],
                    y=filtered_df_sorted['cumulative_pnl'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='blue', width=2)
                ))
                
                fig_perf.update_layout(
                    title="Cumulative P&L Over Time",
                    xaxis_title="Time",
                    yaxis_title="Cumulative P&L ($)",
                    height=400
                )
                st.plotly_chart(fig_perf, use_container_width=True)
                
            else:
                st.info("No trades match the selected filters")
        else:
            st.info("📊 No trades yet. Start the bot to begin live trading with your optimized parameters!")
            
            # Show strategy preview for live trading
            st.subheader("🎯 Ready for Live Trading")
            
            if 'trading_params' in st.session_state:
                params = st.session_state.trading_params
                st.write("**Current Strategy Parameters:**")
                
                param_cols = st.columns(3)
                with param_cols[0]:
                    st.write(f"📊 **RSI Period**: {params.get('rsi_period', 14)}")
                    st.write(f"📈 **BB Period**: {params.get('bb_period', 20)}")
                with param_cols[1]:
                    st.write(f"💰 **Position Size**: {params.get('position_size', 0.1):.1%}")
                    st.write(f"🛡️ **Stop Loss**: {params.get('stop_loss', 0.02):.1%}")
                with param_cols[2]:
                    st.write(f"🎯 **Take Profit**: {params.get('take_profit', 0.04):.1%}")
                    st.write(f"⚡ **RSI Oversold**: {params.get('rsi_oversold', 30)}")
                
                st.info("💡 **Tip**: Use the Backtest/Optimization tab to find optimal parameters before starting live trading!")
            else:
                st.warning("⚠️ No trading parameters set. Please run optimization first.")

    with tab4:
        st.header("💻 Code Editor")
        st.markdown("**Built-in code editor for strategy modifications and custom algorithm development**")
        
        # Strategy code editor section
        st.subheader("📝 Strategy Code Editor")
        
        # Strategy template selector
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 Strategy Templates")
            
            # Pre-defined strategy templates
            strategy_templates = {
                "RSI + Bollinger Bands": """
# RSI + Bollinger Bands Strategy
import pandas as pd
import numpy as np

def calculate_indicators(data, rsi_period=14, bb_period=20, bb_std=2):
    \"\"\"Calculate RSI and Bollinger Bands indicators\"\"\"
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    bb_ma = data['Close'].rolling(window=bb_period).mean()
    bb_std_dev = data['Close'].rolling(window=bb_period).std()
    bb_upper = bb_ma + (bb_std_dev * bb_std)
    bb_lower = bb_ma - (bb_std_dev * bb_std)
    
    return {
        'rsi': rsi,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_ma': bb_ma
    }

def generate_signals(data, indicators, rsi_oversold=30, rsi_overbought=70):
    \"\"\"Generate buy/sell signals based on RSI and Bollinger Bands\"\"\"
    
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['signal'] = 0
    
    # Buy signal: RSI oversold and price below lower Bollinger Band
    buy_condition = (
        (indicators['rsi'] < rsi_oversold) & 
        (data['Close'] < indicators['bb_lower'])
    )
    
    # Sell signal: RSI overbought or price above upper Bollinger Band
    sell_condition = (
        (indicators['rsi'] > rsi_overbought) | 
        (data['Close'] > indicators['bb_upper'])
    )
    
    signals.loc[buy_condition, 'signal'] = 1   # Buy
    signals.loc[sell_condition, 'signal'] = -1 # Sell
    
    return signals

# Main strategy function
def run_strategy(data, **params):
    \"\"\"Main strategy execution function\"\"\"
    
    # Get parameters with defaults
    rsi_period = params.get('rsi_period', 14)
    bb_period = params.get('bb_period', 20) 
    bb_std = params.get('bb_std', 2.0)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    
    # Calculate indicators
    indicators = calculate_indicators(
        data, rsi_period, bb_period, bb_std
    )
    
    # Generate signals
    signals = generate_signals(
        data, indicators, rsi_oversold, rsi_overbought
    )
    
    return signals, indicators
""",
                
                "Momentum Strategy": """
# Momentum Strategy
import pandas as pd
import numpy as np

def calculate_momentum_indicators(data, short_ma=10, long_ma=20, volume_ma=50):
    \"\"\"Calculate momentum indicators\"\"\"
    
    # Moving averages
    short_ma_vals = data['Close'].rolling(window=short_ma).mean()
    long_ma_vals = data['Close'].rolling(window=long_ma).mean()
    
    # Volume momentum
    volume_ma_vals = data['Volume'].rolling(window=volume_ma).mean()
    volume_ratio = data['Volume'] / volume_ma_vals
    
    # Price momentum
    price_momentum = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1)
    
    # Rate of change
    roc_period = 12
    roc = ((data['Close'] - data['Close'].shift(roc_period)) / data['Close'].shift(roc_period)) * 100
    
    return {
        'short_ma': short_ma_vals,
        'long_ma': long_ma_vals,
        'volume_ratio': volume_ratio,
        'price_momentum': price_momentum,
        'roc': roc
    }

def generate_momentum_signals(data, indicators, momentum_threshold=0.02, volume_threshold=1.5):
    \"\"\"Generate momentum-based signals\"\"\"
    
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['signal'] = 0
    
    # Strong momentum up with high volume
    buy_condition = (
        (indicators['short_ma'] > indicators['long_ma']) &
        (indicators['price_momentum'] > momentum_threshold) &
        (indicators['volume_ratio'] > volume_threshold) &
        (indicators['roc'] > 5)
    )
    
    # Momentum reversal or weak momentum
    sell_condition = (
        (indicators['short_ma'] < indicators['long_ma']) |
        (indicators['price_momentum'] < -momentum_threshold) |
        (indicators['roc'] < -3)
    )
    
    signals.loc[buy_condition, 'signal'] = 1   # Buy
    signals.loc[sell_condition, 'signal'] = -1 # Sell
    
    return signals

# Main strategy function
def run_strategy(data, **params):
    \"\"\"Main momentum strategy execution\"\"\"
    
    # Get parameters
    short_ma = params.get('short_ma', 10)
    long_ma = params.get('long_ma', 20)
    volume_ma = params.get('volume_ma', 50)
    momentum_threshold = params.get('momentum_threshold', 0.02)
    volume_threshold = params.get('volume_threshold', 1.5)
    
    # Calculate indicators
    indicators = calculate_momentum_indicators(data, short_ma, long_ma, volume_ma)
    
    # Generate signals
    signals = generate_momentum_signals(
        data, indicators, momentum_threshold, volume_threshold
    )
    
    return signals, indicators
""",
                
                "Mean Reversion Strategy": """
# Mean Reversion Strategy
import pandas as pd
import numpy as np

def calculate_mean_reversion_indicators(data, zscore_period=20, rsi_period=14):
    \"\"\"Calculate mean reversion indicators\"\"\"
    
    # Z-Score calculation
    rolling_mean = data['Close'].rolling(window=zscore_period).mean()
    rolling_std = data['Close'].rolling(window=zscore_period).std()
    zscore = (data['Close'] - rolling_mean) / rolling_std
    
    # RSI for overbought/oversold
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Band deviation
    bb_ma = data['Close'].rolling(window=zscore_period).mean()
    bb_std = data['Close'].rolling(window=zscore_period).std()
    bb_deviation = (data['Close'] - bb_ma) / bb_std
    
    return {
        'zscore': zscore,
        'rsi': rsi,
        'bb_deviation': bb_deviation,
        'rolling_mean': rolling_mean
    }

def generate_mean_reversion_signals(data, indicators, zscore_threshold=2.0, rsi_oversold=20, rsi_overbought=80):
    \"\"\"Generate mean reversion signals\"\"\"
    
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['signal'] = 0
    
    # Buy when oversold (below mean)
    buy_condition = (
        (indicators['zscore'] < -zscore_threshold) &
        (indicators['rsi'] < rsi_oversold)
    )
    
    # Sell when overbought (above mean) or return to mean
    sell_condition = (
        (indicators['zscore'] > zscore_threshold) |
        (indicators['rsi'] > rsi_overbought) |
        ((indicators['zscore'] > -0.5) & (indicators['zscore'] < 0.5))  # Return to mean
    )
    
    signals.loc[buy_condition, 'signal'] = 1   # Buy
    signals.loc[sell_condition, 'signal'] = -1 # Sell
    
    return signals

# Main strategy function  
def run_strategy(data, **params):
    \"\"\"Main mean reversion strategy execution\"\"\"
    
    # Get parameters
    zscore_period = params.get('zscore_period', 20)
    rsi_period = params.get('rsi_period', 14)
    zscore_threshold = params.get('zscore_threshold', 2.0)
    rsi_oversold = params.get('rsi_oversold', 20)
    rsi_overbought = params.get('rsi_overbought', 80)
    
    # Calculate indicators
    indicators = calculate_mean_reversion_indicators(data, zscore_period, rsi_period)
    
    # Generate signals
    signals = generate_mean_reversion_signals(
        data, indicators, zscore_threshold, rsi_oversold, rsi_overbought
    )
    
    return signals, indicators
""",
                
                "Custom Template": """
# Custom Strategy Template
import pandas as pd
import numpy as np

def custom_indicators(data, **params):
    \"\"\"
    Calculate custom indicators for your strategy
    
    Add your custom indicator calculations here
    \"\"\"
    
    # Example: Simple moving average
    ma_period = params.get('ma_period', 20)
    sma = data['Close'].rolling(window=ma_period).mean()
    
    # Add more indicators as needed
    
    return {
        'sma': sma,
        # Add more indicators here
    }

def custom_signals(data, indicators, **params):
    \"\"\"
    Generate custom trading signals
    
    Implement your custom signal logic here
    \"\"\"
    
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['signal'] = 0
    
    # Example: Buy when price crosses above SMA
    buy_condition = (
        (data['Close'] > indicators['sma']) &
        (data['Close'].shift(1) <= indicators['sma'].shift(1))
    )
    
    # Example: Sell when price crosses below SMA
    sell_condition = (
        (data['Close'] < indicators['sma']) &
        (data['Close'].shift(1) >= indicators['sma'].shift(1))
    )
    
    signals.loc[buy_condition, 'signal'] = 1   # Buy
    signals.loc[sell_condition, 'signal'] = -1 # Sell
    
    return signals

# Main strategy function
def run_strategy(data, **params):
    \"\"\"Main custom strategy execution\"\"\"
    
    # Calculate your custom indicators
    indicators = custom_indicators(data, **params)
    
    # Generate trading signals
    signals = custom_signals(data, indicators, **params)
    
    return signals, indicators

# Strategy metadata
STRATEGY_INFO = {
    'name': 'Custom Strategy',
    'description': 'A template for creating custom trading strategies',
    'parameters': {
        'ma_period': {'type': 'int', 'min': 5, 'max': 50, 'default': 20},
        # Add more parameter definitions here
    }
}
"""
            }
            
            selected_template = st.selectbox(
                "Choose Strategy Template",
                list(strategy_templates.keys()),
                help="Select a pre-built strategy template to start with"
            )
            
            # Load template button
            if st.button("📥 Load Template", use_container_width=True):
                st.session_state['strategy_code'] = strategy_templates[selected_template]
                st.success(f"✅ Loaded {selected_template} template")
                st.rerun()
            
            # Strategy actions
            st.subheader("⚡ Quick Actions")
            
            # Save strategy
            strategy_name = st.text_input("Strategy Name", placeholder="MyCustomStrategy", key="strategy_name_input")
            
            if st.button("💾 Save Strategy", use_container_width=True):
                if strategy_name and 'strategy_code' in st.session_state:
                    # Create strategies directory if it doesn't exist
                    os.makedirs("strategies", exist_ok=True)
                    
                    # Save strategy to file
                    filename = f"strategies/{strategy_name}.py"
                    with open(filename, 'w') as f:
                        f.write(st.session_state['strategy_code'])
                    
                    st.success(f"✅ Strategy saved as {filename}")
                else:
                    st.warning("⚠️ Please enter a strategy name and load/edit code")
            
            # Test strategy button
            if st.button("🧪 Test Strategy", use_container_width=True, type="primary"):
                if 'strategy_code' in st.session_state:
                    st.session_state['test_strategy'] = True
                    st.success("🚀 Testing strategy... check results on the right")
                    st.rerun()
                else:
                    st.warning("⚠️ Please load or write strategy code first")
            
            # Load saved strategy
            st.subheader("📂 Saved Strategies")
            
            # List saved strategies
            if os.path.exists("strategies"):
                saved_strategies = [f for f in os.listdir("strategies") if f.endswith('.py')]
                if saved_strategies:
                    selected_saved = st.selectbox("Load Saved Strategy", saved_strategies, key="saved_strategy_select")
                    
                    if st.button("📂 Load Saved Strategy", use_container_width=True):
                        with open(f"strategies/{selected_saved}", 'r') as f:
                            st.session_state['strategy_code'] = f.read()
                        st.success(f"✅ Loaded {selected_saved}")
                        st.rerun()
                else:
                    st.info("No saved strategies found")
            else:
                st.info("No strategies directory found")
        
        with col2:
            st.subheader("✏️ Code Editor")
            
            # Initialize strategy code if not exists
            if 'strategy_code' not in st.session_state:
                st.session_state['strategy_code'] = strategy_templates["RSI + Bollinger Bands"]
            
            # Code editor (using text_area as Monaco editor requires additional setup)
            strategy_code = st.text_area(
                "Strategy Code",
                value=st.session_state['strategy_code'],
                height=600,
                help="Edit your strategy code here. Use Python syntax with pandas and numpy.",
                key="strategy_code_editor"
            )
            
            # Update session state when code changes
            if strategy_code != st.session_state['strategy_code']:
                st.session_state['strategy_code'] = strategy_code
            
            # Code editor tools
            editor_cols = st.columns(4)
            with editor_cols[0]:
                if st.button("🔍 Validate Syntax", use_container_width=True):
                    try:
                        compile(strategy_code, '<string>', 'exec')
                        st.success("✅ Syntax is valid")
                    except SyntaxError as e:
                        st.error(f"❌ Syntax Error: {e}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            with editor_cols[1]:
                if st.button("📋 Copy Code", use_container_width=True):
                    st.write("Code copied to clipboard (manual copy)")
                    st.code(strategy_code, language='python')
            
            with editor_cols[2]:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state['strategy_code'] = strategy_templates[selected_template]
                    st.warning("⚠️ Code reset to template")
                    st.rerun()
            
            with editor_cols[3]:
                if st.button("📖 Help", use_container_width=True):
                    st.info("""
                    **Strategy Code Help:**
                    
                    • Your strategy should implement a `run_strategy(data, **params)` function
                    • Return signals DataFrame with 'signal' column (1=buy, -1=sell, 0=hold)
                    • Use pandas and numpy for calculations
                    • Access data with 'Close', 'Open', 'High', 'Low', 'Volume' columns
                    • Test your strategy before using in live trading
                    """)
        
        # Strategy testing results
        if st.session_state.get('test_strategy', False):
            st.subheader("🧪 Strategy Test Results")
            
            try:
                # Test with sample data
                test_symbol = "AAPL"  # Default test symbol
                ticker = yf.Ticker(test_symbol)
                test_data = ticker.history(period="3mo", interval="1d")
                
                if not test_data.empty:
                    # Execute the strategy code
                    local_vars = {}
                    exec(strategy_code, {"pd": pd, "np": np}, local_vars)
                    
                    if 'run_strategy' in local_vars:
                        # Run the strategy
                        signals, indicators = local_vars['run_strategy'](test_data)
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Signal Summary:**")
                            buy_signals = (signals['signal'] == 1).sum()
                            sell_signals = (signals['signal'] == -1).sum()
                            
                            st.metric("Buy Signals", buy_signals)
                            st.metric("Sell Signals", sell_signals)
                            st.metric("Total Signals", buy_signals + sell_signals)
                        
                        with col2:
                            st.write("**Last 10 Signals:**")
                            recent_signals = signals[signals['signal'] != 0].tail(10)
                            if not recent_signals.empty:
                                display_signals = recent_signals.copy()
                                display_signals['Date'] = display_signals.index.strftime('%Y-%m-%d')
                                display_signals['Action'] = display_signals['signal'].map({1: 'BUY', -1: 'SELL'})
                                display_signals['Price'] = display_signals['price'].apply(lambda x: f"${x:.2f}")
                                
                                st.dataframe(
                                    display_signals[['Date', 'Action', 'Price']], 
                                    use_container_width=True
                                )
                            else:
                                st.info("No signals generated")
                        
                        # Plot strategy results
                        fig = go.Figure()
                        
                        # Price chart
                        fig.add_trace(go.Scatter(
                            x=test_data.index,
                            y=test_data['Close'],
                            mode='lines',
                            name='Price',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Buy signals
                        buy_points = signals[signals['signal'] == 1]
                        if not buy_points.empty:
                            fig.add_trace(go.Scatter(
                                x=buy_points.index,
                                y=buy_points['price'],
                                mode='markers',
                                name='Buy Signal',
                                marker=dict(color='green', size=10, symbol='triangle-up')
                            ))
                        
                        # Sell signals
                        sell_points = signals[signals['signal'] == -1]
                        if not sell_points.empty:
                            fig.add_trace(go.Scatter(
                                x=sell_points.index,
                                y=sell_points['price'],
                                mode='markers',
                                name='Sell Signal',
                                marker=dict(color='red', size=10, symbol='triangle-down')
                            ))
                        
                        fig.update_layout(
                            title=f"Strategy Test Results - {test_symbol}",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.success("✅ Strategy test completed successfully!")
                        
                    else:
                        st.error("❌ Strategy code must contain a 'run_strategy' function")
                
                else:
                    st.error("❌ Could not fetch test data")
                    
            except Exception as e:
                st.error(f"❌ Strategy test failed: {e}")
                st.write("**Debug Info:**")
                st.code(str(e))
            
            # Reset test flag
            st.session_state['test_strategy'] = False
        
        # Strategy documentation
        st.subheader("📚 Strategy Development Guide")
        
        with st.expander("📖 Complete Strategy Development Guide", expanded=False):
            st.markdown("""
            ## 🎯 Strategy Development Guide
            
            ### **Function Structure**
            Your strategy must implement this main function:
            ```python
            def run_strategy(data, **params):
                # Calculate indicators
                indicators = calculate_indicators(data, **params)
                
                # Generate signals  
                signals = generate_signals(data, indicators, **params)
                
                return signals, indicators
            ```
            
            ### **Data Format**
            - `data`: pandas DataFrame with OHLCV columns
            - `data.index`: DatetimeIndex
            - Columns: 'Open', 'High', 'Low', 'Close', 'Volume'
            
            ### **Signal Format**
            Return DataFrame with:
            - `signal`: 1 (buy), -1 (sell), 0 (hold)
            - `price`: current price for signal
            
            ### **Best Practices**
            1. **Error Handling**: Use try-catch blocks
            2. **Parameter Validation**: Check parameter ranges
            3. **Performance**: Vectorize operations with pandas
            4. **Testing**: Test with different market conditions
            5. **Documentation**: Comment your code clearly
            
            ### **Available Libraries**
            - `pandas` as `pd`: Data manipulation
            - `numpy` as `np`: Mathematical operations
            - Built-in Python libraries
            
            ### **Example Indicators**
            ```python
            # Simple Moving Average
            sma = data['Close'].rolling(window=20).mean()
            
            # Exponential Moving Average  
            ema = data['Close'].ewm(span=12).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            ```
            
            ### **Signal Examples**
            ```python
            # Buy when price crosses above SMA
            buy_condition = (
                (data['Close'] > sma) & 
                (data['Close'].shift(1) <= sma.shift(1))
            )
            
            # Sell when RSI is overbought
            sell_condition = rsi > 70
            
            signals.loc[buy_condition, 'signal'] = 1
            signals.loc[sell_condition, 'signal'] = -1
            ```
            """)

if __name__ == "__main__":
    main()
