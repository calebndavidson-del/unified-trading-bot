#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yaml
import os
import time
import subprocess
import threading
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Unified Trading Bot",
    page_icon="🤖",
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
        'last_update': None
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

def run_simple_backtest(symbols, days):
    """Run a simple backtest"""
    try:
        import yfinance as yf
        
        if not symbols:
            return None
            
        ticker = yf.Ticker(symbols[0])
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            return None
        
        returns = df['Close'].pct_change().fillna(0)
        equity = (1 + returns).cumprod() * 100000
        
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_dd = ((equity / equity.cummax()) - 1).min()
        
        return {
            'equity_curve': equity,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_value': equity.iloc[-1]
        }
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return None

def main():
    init_session_state()
    
    st.title("🤖 Unified Trading Bot")
    st.markdown("Complete backtesting and live trading interface")
    
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
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Bot Controls")
        
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
        
        st.divider()
        
        # Config info
        st.subheader("Configuration")
        if config:
            st.metric("Universe Size", len(symbols))
            st.metric("Loop Interval", f"{config.get('general', {}).get('loop_seconds', 300)}s")
            st.metric("Starting Equity", f"${config.get('broker', {}).get('starting_equity', 100000):,.0f}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔄 Backtesting", "📈 Live Trading"])
    
    with tab1:
        st.header("📊 Trading Dashboard")
        
        # Load and display current portfolio
        trades_df = load_trades()
        
        if not trades_df.empty and config:
            current_equity = config.get('broker', {}).get('starting_equity', 100000) + trades_df['net'].sum()
            total_return = (current_equity / config.get('broker', {}).get('starting_equity', 100000) - 1) * 100
        else:
            current_equity = 100000
            total_return = 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio Value", f"${current_equity:,.2f}")
        with col2:
            st.metric("Total Return", f"{total_return:+.2f}%")
        with col3:
            if not trades_df.empty:
                profitable = (trades_df['net'] > 0).sum()
                win_rate = profitable / len(trades_df) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            else:
                st.metric("Win Rate", "0.0%")
        with col4:
            positions = 0
            if not trades_df.empty:
                recent_trades = trades_df.tail(20)
                buy_trades = recent_trades[recent_trades['side'] == 'buy']
                sell_trades = recent_trades[recent_trades['side'] == 'sell']
                positions = max(0, len(buy_trades) - len(sell_trades))
            st.metric("Open Positions", positions)
        
        # Portfolio chart
        if not trades_df.empty and config:
            trades_df['cumulative_pnl'] = trades_df['net'].cumsum()
            starting_equity = config.get('broker', {}).get('starting_equity', 100000)
            trades_df['portfolio_value'] = starting_equity + trades_df['cumulative_pnl']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trades_df['ts'],
                y=trades_df['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Portfolio Performance",
                xaxis_title="Time",
                yaxis_title="Value ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📈 Start the bot to see portfolio performance")
    
    with tab2:
        st.header("🔄 Backtesting Engine")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Backtest Settings")
            
            test_periods = {
                "5 Days": 5,
                "30 Days": 30,
                "90 Days": 90
            }
            
            selected_period = st.selectbox("Test Period", list(test_periods.keys()))
            max_symbols = st.slider("Max Symbols", 1, min(20, len(symbols)), 5)
            
            if st.button("🚀 Run Backtest", use_container_width=True):
                with st.spinner("Running backtest..."):
                    result = run_simple_backtest(symbols[:max_symbols], test_periods[selected_period])
                    
                    if result:
                        st.session_state['backtest_result'] = result
                        st.success("✅ Backtest completed!")
                    else:
                        st.error("❌ Backtest failed")
        
        with col2:
            if 'backtest_result' in st.session_state:
                result = st.session_state['backtest_result']
                
                st.subheader("Backtest Results")
                
                # Metrics
                col2a, col2b, col2c, col2d = st.columns(4)
                with col2a:
                    st.metric("Total Return", f"{result['total_return']:.2%}")
                with col2b:
                    st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                with col2c:
                    st.metric("Max Drawdown", f"{result['max_drawdown']:.2%}")
                with col2d:
                    st.metric("Final Value", f"${result['final_value']:,.2f}")
                
                # Equity curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=result['equity_curve'].index,
                    y=result['equity_curve'].values,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='green', width=2)
                ))
                fig.update_layout(
                    title="Backtest Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("👆 Run a backtest to see results")
    
    with tab3:
        st.header("📈 Live Trading Monitor")
        
        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("💾 Export Trades", use_container_width=True):
                trades_df = load_trades()
                if not trades_df.empty:
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        "trades.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No trades to export")
        with col3:
            if st.button("🗑️ Clear Logs", use_container_width=True):
                try:
                    if os.path.exists('logs/trades.csv'):
                        pd.DataFrame(columns=["ts","symbol","side","qty","price","net","equity_after"]).to_csv('logs/trades.csv', index=False)
                    st.success("Logs cleared")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing logs: {e}")
        
        # Trade history
        st.subheader("Trade History")
        trades_df = load_trades()
        
        if not trades_df.empty:
            display_df = trades_df.copy()
            display_df['Time'] = display_df['ts'].dt.strftime('%H:%M:%S')
            display_df['Price'] = display_df['price'].apply(lambda x: f"${x:.4f}")
            display_df['Quantity'] = display_df['qty'].apply(lambda x: f"{x:.4f}")
            display_df['P&L'] = display_df['net'].apply(lambda x: f"${x:+.2f}")
            
            st.dataframe(
                display_df[['Time', 'symbol', 'side', 'Quantity', 'Price', 'P&L']].tail(20),
                use_container_width=True
            )
            
            # Trade statistics
            st.subheader("Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", len(trades_df))
            with col2:
                profitable = (trades_df['net'] > 0).sum()
                st.metric("Profitable Trades", profitable)
            with col3:
                total_pnl = trades_df['net'].sum()
                st.metric("Total P&L", f"${total_pnl:+,.2f}")
            with col4:
                avg_trade = trades_df['net'].mean() if len(trades_df) > 0 else 0
                st.metric("Avg Trade", f"${avg_trade:+.2f}")
        else:
            st.info("📊 No trades yet. Start the bot to begin trading.")

if __name__ == "__main__":
    main()