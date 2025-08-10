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
    
    # Main content tabs - Enhanced
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🔄 Advanced Backtesting", 
        "🚀 Parameter Optimization",
        "📈 Multi-Strategy Comparison",
        "⚡ Live Trading"
    ])
    
    with tab1:
        st.header("📊 Enhanced Trading Dashboard")
        
        # Current parameter display
        st.subheader("🎯 Current Strategy Parameters")
        params = st.session_state.get('trading_params', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RSI Period", params.get('rsi_period', 14))
            st.metric("RSI Oversold", params.get('rsi_oversold', 30))
        with col2:
            st.metric("BB Period", params.get('bb_period', 20))
            st.metric("BB Std Dev", f"{params.get('bb_std', 2.0):.1f}")
        with col3:
            st.metric("Position Size", f"{params.get('position_size', 0.1):.1%}")
            st.metric("Stop Loss", f"{params.get('stop_loss', 0.02):.1%}")
        with col4:
            st.metric("Take Profit", f"{params.get('take_profit', 0.04):.1%}")
            st.metric("Starting Capital", f"${params.get('starting_capital', 100000):,.0f}")
        
        # Quick backtest with current parameters
        st.subheader("⚡ Quick Strategy Test")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            test_symbol = st.selectbox("Test Symbol", symbols[:10])
            test_days = st.selectbox("Test Period", [5, 15, 30, 60])
            
            if st.button("🚀 Test Current Parameters", use_container_width=True):
                with st.spinner("Running quick test..."):
                    result = advanced_backtest(test_symbol, test_days, st.session_state.trading_params)
                    if result:
                        st.session_state['quick_test'] = result
        
        with col2:
            if 'quick_test' in st.session_state:
                result = st.session_state['quick_test']
                
                # Quick metrics
                col2a, col2b, col2c, col2d = st.columns(4)
                with col2a:
                    st.metric("Return", f"{result['total_return']:.2%}")
                with col2b:
                    st.metric("Sharpe", f"{result['sharpe_ratio']:.2f}")
                with col2c:
                    st.metric("Max DD", f"{result['max_drawdown']:.2%}")
                with col2d:
                    st.metric("Win Rate", f"{result['win_rate']:.1%}")
                
                # Quick equity curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=result['equity_curve'].index,
                    y=result['equity_curve'].values,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='blue', width=2)
                ))
                fig.update_layout(
                    title=f"Quick Test: {result['symbol']} ({test_days} days)",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🔄 Advanced Backtesting Engine")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Backtest Configuration")
            
            # Symbol selection
            selected_symbols = st.multiselect(
                "Select Symbols", 
                symbols[:20], 
                default=symbols[:3]
            )
            
            # Time period
            test_periods = {
                "1 Week": 7,
                "2 Weeks": 14,
                "1 Month": 30,
                "3 Months": 90,
                "6 Months": 180
            }
            
            selected_period = st.selectbox("Test Period", list(test_periods.keys()))
            
            # Use current parameters or custom
            use_current_params = st.checkbox("Use Current Parameters", value=True)
            
            if not use_current_params:
                st.subheader("Custom Parameters")
                custom_rsi = st.slider("Custom RSI Period", 5, 50, 14, key="custom_rsi")
                custom_bb = st.slider("Custom BB Period", 10, 50, 20, key="custom_bb")
                # Add more custom parameters as needed
            
            if st.button("🚀 Run Advanced Backtest", use_container_width=True):
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
    
    with tab3:
        st.header("🚀 Parameter Optimization Engine")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Optimization Settings")
            
            opt_symbol = st.selectbox("Optimization Symbol", symbols[:10])
            opt_period = st.selectbox("Optimization Period", [30, 60, 90])
            
            st.subheader("Parameter Ranges")
            
            # RSI optimization ranges
            rsi_min, rsi_max = st.slider("RSI Period Range", 5, 50, (10, 25))
            rsi_step = st.selectbox("RSI Step", [1, 2, 5], index=1)
            
            # Bollinger Bands ranges
            bb_min, bb_max = st.slider("BB Period Range", 10, 50, (15, 30))
            bb_step = st.selectbox("BB Step", [1, 2, 5], index=1)
            
            # Position size range
            pos_min, pos_max = st.slider("Position Size Range (%)", 1, 30, (5, 20))
            
            if st.button("🔥 Start Optimization", use_container_width=True):
                param_ranges = {
                    'rsi_period': list(range(rsi_min, rsi_max + 1, rsi_step)),
                    'bb_period': list(range(bb_min, bb_max + 1, bb_step)),
                    'position_size': [x/100 for x in range(pos_min, pos_max + 1, 2)],
                    'rsi_oversold': [30],  # Fixed for now
                    'rsi_overbought': [70],  # Fixed for now
                    'bb_std': [2.0],  # Fixed for now
                    'stop_loss': [0.02],  # Fixed for now
                    'take_profit': [0.04]  # Fixed for now
                }
                
                with st.spinner("Running parameter optimization..."):
                    opt_results = run_parameter_optimization([opt_symbol], opt_period, param_ranges)
                    if opt_results:
                        st.session_state['optimization_results'] = opt_results
                        st.success(f"✅ Optimization completed! Tested {len(opt_results)} combinations")
        
        with col2:
            if 'optimization_results' in st.session_state:
                results = st.session_state['optimization_results']
                
                st.subheader("🏆 Optimization Results")
                
                # Sort by Sharpe ratio
                sorted_results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
                
                # Best parameters
                best_result = sorted_results[0]
                st.success("🎯 **Optimal Parameters Found:**")
                
                col2a, col2b, col2c = st.columns(3)
                with col2a:
                    st.metric("Best RSI Period", best_result['params']['rsi_period'])
                    st.metric("Best BB Period", best_result['params']['bb_period'])
                with col2b:
                    st.metric("Best Position Size", f"{best_result['params']['position_size']:.1%}")
                    st.metric("Best Return", f"{best_result['total_return']:.2%}")
                with col2c:
                    st.metric("Best Sharpe Ratio", f"{best_result['sharpe_ratio']:.2f}")
                    st.metric("Max Drawdown", f"{best_result['max_drawdown']:.2%}")
                
                # Performance heatmap
                if len(results) > 1:
                    st.subheader("📊 Parameter Performance Heatmap")
                    
                    # Create heatmap data
                    heatmap_data = []
                    for result in results:
                        heatmap_data.append({
                            'RSI_Period': result['params']['rsi_period'],
                            'BB_Period': result['params']['bb_period'],
                            'Position_Size': result['params']['position_size'],
                            'Sharpe_Ratio': result['sharpe_ratio'],
                            'Return': result['total_return']
                        })
                    
                    heatmap_df = pd.DataFrame(heatmap_data)
                    
                    # Pivot for heatmap
                    if len(heatmap_df) > 1:
                        pivot_df = heatmap_df.pivot_table(
                            values='Sharpe_Ratio', 
                            index='RSI_Period', 
                            columns='BB_Period', 
                            aggfunc='mean'
                        )
                        
                        fig = px.imshow(
                            pivot_df,
                            labels=dict(x="BB Period", y="RSI Period", color="Sharpe Ratio"),
                            x=pivot_df.columns,
                            y=pivot_df.index,
                            color_continuous_scale="RdYlGn"
                        )
                        fig.update_layout(title="Sharpe Ratio Heatmap", height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Top 10 results table
                st.subheader("🏅 Top 10 Parameter Combinations")
                top_10 = sorted_results[:10]
                top_10_data = []
                for i, result in enumerate(top_10):
                    top_10_data.append({
                        'Rank': i + 1,
                        'RSI': result['params']['rsi_period'],
                        'BB': result['params']['bb_period'],
                        'Pos Size': f"{result['params']['position_size']:.1%}",
                        'Return': f"{result['total_return']:.2%}",
                        'Sharpe': f"{result['sharpe_ratio']:.2f}",
                        'Max DD': f"{result['max_drawdown']:.2%}"
                    })
                
                st.dataframe(pd.DataFrame(top_10_data), use_container_width=True)
    
    with tab4:
        st.header("📈 Multi-Strategy Comparison")
        
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
                default=["Conservative", "Balanced", "Aggressive"]
            )
            
            comparison_symbol = st.selectbox("Comparison Symbol", symbols[:10], key="comp_symbol")
            comparison_period = st.selectbox("Comparison Period", [30, 60, 90], key="comp_period")
            
            if st.button("🔥 Compare Strategies", use_container_width=True):
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
                
                # Performance radar chart
                metrics = ['total_return', 'sharpe_ratio', 'win_rate']
                fig_radar = go.Figure()
                
                for strategy, result in results.items():
                    values = [
                        result['total_return'] * 100,  # Convert to percentage
                        result['sharpe_ratio'] * 10,   # Scale for visibility
                        result['win_rate'] * 100       # Convert to percentage
                    ]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=['Return (%)', 'Sharpe (×10)', 'Win Rate (%)'],
                        fill='toself',
                        name=strategy
                    ))
                
                fig_radar.update_layout(
                    title="Strategy Performance Radar",
                    polar=dict(radialaxis=dict(visible=True)),
                    height=400
                )
                st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab5:
        st.header("⚡ Enhanced Live Trading Monitor")
        
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
        
        # Enhanced trade monitoring
        trades_df = load_trades()
        
        if not trades_df.empty:
            # Real-time performance metrics
            st.subheader("⚡ Real-Time Performance")
            
            col1, col2, col3, col4 = st.columns(4)
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
            
            # Enhanced trade table with filtering
            st.subheader("📊 Trade History & Analysis")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                symbol_filter = st.selectbox("Filter by Symbol", ['All'] + list(trades_df['symbol'].unique()))
            with col2:
                side_filter = st.selectbox("Filter by Side", ['All', 'buy', 'sell'])
            with col3:
                date_filter = st.selectbox("Filter by Date", ['All', 'Today', 'Last 7 days', 'Last 30 days'])
            
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
            else:
                st.info("No trades match the selected filters")
        else:
            st.info("📊 No trades yet. Start the bot to begin trading with your optimized parameters!")

if __name__ == "__main__":
    main()
