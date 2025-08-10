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
        
        # Simplified QuantConnect-Style Interface
        st.header("🚀 QuantConnect-Style Trading")
        st.markdown("**Automated parameter optimization for optimal performance**")
        
        # Get optimal parameters from optimization if available
        if 'quantconnect_optimization' in st.session_state:
            opt_data = st.session_state['quantconnect_optimization']
            if opt_data['summary'].best_result:
                st.success("✅ Using Optimized Parameters")
                best_params = opt_data['summary'].best_result.parameters
                current_params = {
                    'rsi_period': best_params.get('rsi_period', 14),
                    'rsi_oversold': best_params.get('rsi_oversold', 30),
                    'rsi_overbought': best_params.get('rsi_overbought', 70),
                    'bb_period': best_params.get('bb_period', 20),
                    'bb_std': best_params.get('bb_std', 2.0),
                    'position_size': best_params.get('position_size', 0.1),
                    'stop_loss': best_params.get('stop_loss', 0.02),
                    'take_profit': best_params.get('take_profit', 0.04),
                    'starting_capital': 100000
                }
                
                # Show key optimized metrics
                with st.expander("📊 Current Optimized Settings", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sharpe Ratio", f"{opt_data['summary'].best_result.sharpe_ratio:.2f}")
                        st.metric("Total Return", f"{opt_data['summary'].best_result.total_return:.2%}")
                    with col2:
                        st.metric("Win Rate", f"{opt_data['summary'].best_result.win_rate:.1%}")
                        st.metric("Max Drawdown", f"{opt_data['summary'].best_result.max_drawdown:.1%}")
                        
                st.info("💡 **Go to Parameter Optimization tab** to run new optimizations or adjust settings")
            else:
                current_params = {
                    'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
                    'bb_period': 20, 'bb_std': 2.0, 'position_size': 0.1,
                    'stop_loss': 0.02, 'take_profit': 0.04, 'starting_capital': 100000
                }
                st.warning("⚠️ Using default parameters - **optimize for better performance!**")
        else:
            # Default parameters when no optimization has been run
            current_params = {
                'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
                'bb_period': 20, 'bb_std': 2.0, 'position_size': 0.1,
                'stop_loss': 0.02, 'take_profit': 0.04, 'starting_capital': 100000
            }
            st.warning("⚠️ Using default parameters")
            st.info("🚀 **Run optimization** to find optimal settings automatically!")
        
        # Store parameters in session state
        st.session_state.trading_params = current_params
        
        # Advanced manual controls (collapsed by default)
        with st.expander("⚙️ Advanced Manual Override", expanded=False):
            st.warning("⚠️ **Manual parameters override optimization results**")
            st.markdown("*Use optimization instead for scientifically-backed parameters*")
            
            manual_mode = st.checkbox("Enable Manual Parameter Control", 
                                    help="Override optimized parameters with manual settings")
            
            if manual_mode:
                st.markdown("**📊 Technical Indicators**")
                col1, col2 = st.columns(2)
                with col1:
                    rsi_period = st.slider("RSI Period", 5, 50, current_params['rsi_period'])
                    rsi_oversold = st.slider("RSI Oversold", 10, 40, int(current_params['rsi_oversold']))
                    rsi_overbought = st.slider("RSI Overbought", 60, 90, int(current_params['rsi_overbought']))
                with col2:
                    bb_period = st.slider("BB Period", 10, 50, current_params['bb_period'])
                    bb_std = st.slider("BB Std Dev", 1.0, 3.0, current_params['bb_std'], 0.1)
                
                st.markdown("**💰 Risk Management**")
                col1, col2 = st.columns(2)
                with col1:
                    position_size = st.slider("Position Size (%)", 1, 50, int(current_params['position_size']*100)) / 100
                    stop_loss = st.slider("Stop Loss (%)", 0.5, 10.0, current_params['stop_loss']*100) / 100
                with col2:
                    take_profit = st.slider("Take Profit (%)", 1.0, 20.0, current_params['take_profit']*100) / 100
                
                # Update parameters if manual mode is enabled
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
                st.success("Manual parameters applied")
        
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
    
    # Main content tabs - Enhanced with Smart Scanner
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🔍 Smart Scanner",
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
        st.header("🔍 Intelligent Symbol Scanner")
        st.markdown("**Discover optimal trading opportunities automatically**")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Scanner Configuration")
            
            # Market categories selection
            st.write("**Market Categories to Scan:**")
            available_categories = list(MarketCategoryScanner.MARKET_CATEGORIES.keys())
            selected_categories = st.multiselect(
                "Categories",
                available_categories,
                default=['SP500', 'NASDAQ100'],
                help="Select market categories to scan for opportunities"
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
                """
            )
            
            # Advanced filters
            with st.expander("🔧 Advanced Filters", expanded=False):
                st.write("**Price Range:**")
                price_range = st.slider("Price Range ($)", 1, 1000, (10, 500))
                
                st.write("**Minimum Volume:**")
                min_volume = st.number_input("Min Daily Volume", value=100000, step=50000)
                
                st.write("**Minimum Score:**")
                min_score = st.slider("Minimum Score", 0, 100, 60)
            
            # Number of results
            top_n = st.slider("Top N Results", 5, 50, 20)
            
            # Smart Scan button
            if st.button("🔍 **Smart Scan**", use_container_width=True, type="primary"):
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
                                st.session_state['scan_results'] = results
                                st.session_state['scan_mode'] = scanning_mode
                                st.success(f"✅ Found {len(results)} high-potential symbols!")
                            else:
                                st.warning("No symbols found matching the criteria. Try adjusting filters.")
                                
                        except Exception as e:
                            st.error(f"Scanning error: {e}")
                else:
                    st.warning("Please select at least one market category to scan")
            
            # Quick scan buttons
            st.write("**Quick Scan Presets:**")
            col1a, col1b = st.columns(2)
            with col1a:
                if st.button("🛡️ Safe Plays", use_container_width=True):
                    # Trigger conservative scan of SP500
                    results = cached_smart_scan(
                        categories_tuple=('SP500',),
                        mode='conservative',
                        top_n=15,
                        filters_key="conservative_preset"
                    )
                    if results:
                        st.session_state['scan_results'] = results
                        st.session_state['scan_mode'] = 'Conservative'
                        st.rerun()
            
            with col1b:
                if st.button("⚡ High Growth", use_container_width=True):
                    # Trigger aggressive scan of growth stocks
                    results = cached_smart_scan(
                        categories_tuple=('POPULAR_GROWTH', 'NASDAQ100'),
                        mode='aggressive',
                        top_n=15,
                        filters_key="aggressive_preset"
                    )
                    if results:
                        st.session_state['scan_results'] = results
                        st.session_state['scan_mode'] = 'Aggressive'
                        st.rerun()
        
        with col2:
            if 'scan_results' in st.session_state and st.session_state['scan_results']:
                results = st.session_state['scan_results']
                scan_mode = st.session_state.get('scan_mode', 'Unknown')
                
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
                
                # Score distribution chart
                st.subheader("📈 Score Distribution")
                scores = [r['score'] for r in results]
                symbols = [r['symbol'] for r in results[:15]]  # Top 15 for readability
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=symbols,
                    y=scores[:15],
                    marker_color='lightblue',
                    text=[f"{s:.1f}" for s in scores[:15]],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Top 15 Symbol Scores",
                    xaxis_title="Symbol",
                    yaxis_title="Score",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Quick actions
                st.subheader("⚡ Quick Actions")
                col2a, col2b, col2c = st.columns(3)
                
                with col2a:
                    if st.button("📈 Backtest Top 5", use_container_width=True):
                        # Auto-populate backtest tab with top 5 symbols
                        top_5_symbols = [r['symbol'] for r in results[:5]]
                        st.session_state['auto_selected_symbols'] = top_5_symbols
                        st.info(f"Selected top 5 symbols for backtesting: {', '.join(top_5_symbols)}")
                
                with col2b:
                    if st.button("🎯 Optimize Best", use_container_width=True):
                        # Auto-select best symbol for optimization
                        best_symbol = results[0]['symbol']
                        st.session_state['auto_selected_optimize'] = best_symbol
                        st.info(f"Selected {best_symbol} for parameter optimization")
                
                with col2c:
                    # Export results
                    if st.button("💾 Export Results", use_container_width=True):
                        csv_data = pd.DataFrame(results).to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv_data,
                            f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                
                # Detailed breakdown for top symbol
                if results:
                    st.subheader(f"🔍 Detailed Analysis: {results[0]['symbol']}")
                    top_result = results[0]
                    
                    col2a, col2b = st.columns(2)
                    with col2a:
                        st.write(f"**Overall Score:** {top_result['score']:.1f}/100")
                        st.write(f"**Price:** ${top_result['price']:.2f}")
                        st.write(f"**Volume:** {top_result['volume']:,.0f}")
                        st.write(f"**Reasoning:** {top_result['reasoning']}")
                    
                    with col2b:
                        # Sub-score breakdown chart
                        sub_scores = top_result['sub_scores']
                        fig_radar = go.Figure()
                        
                        categories = list(sub_scores.keys())
                        values = list(sub_scores.values())
                        
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name=top_result['symbol'],
                            line_color='blue'
                        ))
                        
                        fig_radar.update_layout(
                            title=f"{top_result['symbol']} Score Breakdown",
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100]
                                )
                            ),
                            height=300
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
            
            else:
                st.info("👆 Configure your scan settings and click '🔍 Smart Scan' to discover trading opportunities!")
                
                # Show available categories preview
                st.subheader("📋 Available Market Categories")
                for category, symbols in MarketCategoryScanner.MARKET_CATEGORIES.items():
                    with st.expander(f"{category} ({len(symbols)} symbols)"):
                        st.write(", ".join(symbols[:10]) + ("..." if len(symbols) > 10 else ""))
    
    with tab3:
        st.header("🔄 Advanced Backtesting Engine")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Backtest Configuration")
            
            # Symbol selection
            # Check if auto-selected symbols are available from scanner
            default_symbols = st.session_state.get('auto_selected_symbols', symbols[:3])
            selected_symbols = st.multiselect(
                "Select Symbols", 
                symbols[:20], 
                default=default_symbols
            )
            
            # Show if symbols were auto-selected
            if 'auto_selected_symbols' in st.session_state:
                st.info(f"🔍 Auto-selected from Smart Scanner: {', '.join(st.session_state['auto_selected_symbols'])}")
            
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
    
    with tab4:
        st.header("🚀 QuantConnect-Style Parameter Optimization")
        st.markdown("**Professional-grade parameter optimization with automatic range generation and comprehensive analysis**")
        
        # Mode selection
        col_mode1, col_mode2 = st.columns([3, 1])
        with col_mode1:
            optimization_mode = st.radio(
                "Optimization Mode",
                ["🎯 Simple Mode (Automated)", "⚙️ Advanced Mode (Manual)"],
                horizontal=True
            )
        with col_mode2:
            if st.button("ℹ️ Help", use_container_width=True):
                st.info("""
                **Simple Mode**: Automatically optimizes parameters with smart ranges
                **Advanced Mode**: Manual parameter range configuration
                """)
        
        # Import new optimization classes
        try:
            from parameter_manager import ParameterManager, create_default_parameters
            from optimization_engine import OptimizationEngine
            from results_analyzer import ResultsAnalyzer
            
            if optimization_mode.startswith("🎯"):
                # SIMPLE MODE - QuantConnect Style
                st.subheader("🎯 Simple Optimization Setup")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Basic Settings**")
                    
                    # Symbol selection (auto-selected from scanner if available)
                    opt_symbols = symbols[:10]
                    if 'auto_selected_optimize' in st.session_state:
                        default_symbol = st.session_state['auto_selected_optimize']
                        st.info(f"🔍 Auto-selected from Smart Scanner: {default_symbol}")
                        del st.session_state['auto_selected_optimize']
                    else:
                        default_symbol = opt_symbols[0]
                    
                    selected_symbols = st.multiselect(
                        "Symbols to Optimize",
                        opt_symbols,
                        default=[default_symbol],
                        help="Select one or more symbols for optimization"
                    )
                    
                    # Time period
                    opt_period = st.selectbox("Backtest Period (days)", [30, 60, 90, 120], index=1)
                    
                    # Strategy type
                    strategy_type = st.selectbox(
                        "Strategy Type",
                        ["RSI + Bollinger Bands", "Momentum", "Mean Reversion"],
                        help="Different strategies have optimized parameter ranges"
                    )
                    
                    # Optimization objective
                    objective = st.selectbox(
                        "Optimization Objective",
                        ["Sharpe Ratio", "Total Return", "Calmar Ratio", "Sortino Ratio"],
                        help="Metric to optimize for best results"
                    )
                    
                    # Max combinations (to prevent overwhelming computation)
                    max_combinations = st.slider(
                        "Max Combinations to Test",
                        50, 1000, 200,
                        help="Limits computation time for large parameter spaces"
                    )
                    
                    # Show parameter preview
                    strategy_map = {
                        "RSI + Bollinger Bands": "rsi_bollinger",
                        "Momentum": "momentum", 
                        "Mean Reversion": "mean_reversion"
                    }
                    
                    preview_params = create_default_parameters(strategy_map[strategy_type])
                    param_info = preview_params.get_parameter_info()
                    
                    with st.expander("📋 Parameter Ranges Preview", expanded=False):
                        st.write(f"**Total combinations**: {param_info['total_combinations']:,}")
                        for name, info in param_info['parameters'].items():
                            st.write(f"• **{name}**: {info['min_value']} to {info['max_value']} (step {info['step']}) = {info['total_values']} values")
                    
                    # Single optimization button
                    if st.button("🚀 **Optimize & Backtest**", use_container_width=True, type="primary"):
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
                                        'objective': objective_map[objective]
                                    }
                                    
                                    # Initialize analyzer
                                    st.session_state['quantconnect_optimization']['analyzer'].analyze_results(summary)
                                    
                                    progress_placeholder.empty()
                                    status_placeholder.empty()
                                    
                                    st.success(f"✅ Optimization completed! Tested {summary.successful_runs} combinations in {summary.total_time:.1f}s")
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"Optimization failed: {e}")
                                    progress_placeholder.empty()
                                    status_placeholder.empty()
                        else:
                            st.warning("Please select at least one symbol")
                
                with col2:
                    # Results Display
                    if 'quantconnect_optimization' in st.session_state:
                        opt_data = st.session_state['quantconnect_optimization']
                        summary = opt_data['summary']
                        analyzer = opt_data['analyzer']
                        objective = opt_data['objective']
                        
                        st.subheader("📊 Optimization Results")
                        
                        # Quick stats
                        col2a, col2b, col2c, col2d = st.columns(4)
                        with col2a:
                            st.metric("Combinations", f"{summary.successful_runs:,}")
                        with col2b:
                            best_score = getattr(summary.best_result, objective) if summary.best_result else 0
                            st.metric("Best Score", f"{best_score:.3f}")
                        with col2c:
                            st.metric("Time", f"{summary.total_time:.1f}s")
                        with col2d:
                            success_rate = (summary.successful_runs / summary.total_combinations) * 100
                            st.metric("Success Rate", f"{success_rate:.1f}%")
                        
                        # Best parameters display
                        if summary.best_result:
                            st.success("🏆 **Optimal Parameters Found**")
                            
                            best_params_cols = st.columns(3)
                            params = summary.best_result.parameters
                            param_items = list(params.items())
                            
                            for i, (param, value) in enumerate(param_items):
                                col_idx = i % 3
                                with best_params_cols[col_idx]:
                                    if isinstance(value, float):
                                        if 0 < value < 1:
                                            st.metric(param.replace('_', ' ').title(), f"{value:.1%}")
                                        else:
                                            st.metric(param.replace('_', ' ').title(), f"{value:.3f}")
                                    else:
                                        st.metric(param.replace('_', ' ').title(), str(value))
                            
                            # Performance metrics
                            st.write("**Performance Metrics:**")
                            perf_cols = st.columns(4)
                            with perf_cols[0]:
                                st.metric("Return", f"{summary.best_result.total_return:.2%}")
                            with perf_cols[1]:
                                st.metric("Sharpe", f"{summary.best_result.sharpe_ratio:.3f}")
                            with perf_cols[2]:
                                st.metric("Max DD", f"{summary.best_result.max_drawdown:.2%}")
                            with perf_cols[3]:
                                st.metric("Win Rate", f"{summary.best_result.win_rate:.1%}")
                            
                            # Apply best parameters button
                            col_apply1, col_apply2 = st.columns([2, 1])
                            with col_apply1:
                                if st.button("✅ **Apply Best Parameters to Bot**", use_container_width=True, type="primary"):
                                    # Update session state trading parameters
                                    st.session_state.trading_params.update(summary.best_result.parameters)
                                    st.success("✅ Best parameters applied to trading bot! Check sidebar for updated settings.")
                                    st.rerun()  # Refresh to show updated parameters in sidebar
                            with col_apply2:
                                if st.button("📊 Detailed Analysis", use_container_width=True):
                                    st.session_state['show_detailed_analysis'] = True
                                    st.rerun()
                        
                        # Parameter Sensitivity Analysis
                        if st.session_state.get('show_detailed_analysis', False):
                            st.subheader("🔬 Parameter Sensitivity Analysis")
                            
                            # Get parameter impact analysis
                            param_sensitivity = analyzer.analyze_parameter_sensitivity()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Parameter Impact Ranking:**")
                                for i, (param, impact) in enumerate(param_sensitivity.items()):
                                    if i < 5:  # Show top 5 most impactful parameters
                                        impact_bar = "█" * int(impact * 20) + "░" * (20 - int(impact * 20))
                                        st.write(f"{param}: {impact_bar} {impact:.1%}")
                            
                            with col2:
                                st.write("**Robustness Analysis:**")
                                robustness = analyzer.analyze_robustness()
                                st.metric("Robustness Score", f"{robustness.robustness_score:.1%}")
                                st.metric("Overfitting Risk", f"{robustness.overfitting_risk:.1%}")
                                
                                if robustness.robustness_score > 0.7:
                                    st.success("✅ High robustness - parameters are reliable")
                                elif robustness.robustness_score > 0.5:
                                    st.warning("⚠️ Moderate robustness - additional validation recommended")
                                else:
                                    st.error("❌ Low robustness - high overfitting risk")
                            
                            # Hide detailed analysis button
                            if st.button("🔼 Hide Detailed Analysis"):
                                st.session_state['show_detailed_analysis'] = False
                                st.rerun()
                        
                        # Quick charts
                        if len(summary.results) > 1:
                            # Performance distribution
                            fig_dist = analyzer.create_performance_distribution_chart(objective)
                            st.plotly_chart(fig_dist, use_container_width=True)
                    
                    else:
                        st.info("👆 Configure settings and click '🚀 Optimize & Backtest' to discover optimal parameters automatically!")
                        
                        # Show strategy explanation
                        st.subheader("📈 How It Works")
                        st.write("""
                        **QuantConnect-Style Optimization:**
                        1. **Smart Parameter Ranges**: Automatically defined based on strategy type
                        2. **Grid Search**: Tests all parameter combinations systematically  
                        3. **Parallel Processing**: Fast execution using multiple CPU cores
                        4. **Comprehensive Analysis**: Performance metrics, robustness testing
                        5. **One-Click Application**: Apply best parameters instantly
                        """)
            
            else:
                # ADVANCED MODE - Manual Configuration (existing functionality enhanced)
                st.subheader("⚙️ Advanced Manual Configuration")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Optimization Settings")
                    
                    opt_symbol = st.selectbox("Optimization Symbol", symbols[:10])
                    
                    # Check if auto-selected from scanner
                    if 'auto_selected_optimize' in st.session_state:
                        opt_symbol = st.session_state['auto_selected_optimize']
                        st.info(f"🔍 Auto-selected from Smart Scanner: {opt_symbol}")
                        del st.session_state['auto_selected_optimize']
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
                    
                    if st.button("🔥 Start Advanced Optimization", use_container_width=True):
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
                        
                        # SAFE sorting with multiple fallbacks
                        if results is None:
                            st.error("❌ Optimization failed to generate results")
                        elif not results:
                            st.warning("⚠️ No optimization results found")
                        else:
                            try:
                                # Safe sorting with fallback keys
                                sorted_results = sorted(
                                    results, 
                                    key=lambda x: x.get('sharpe_ratio', x.get('total_return', 0)), 
                                    reverse=True
                                )
                            except (KeyError, TypeError, AttributeError) as e:
                                st.error(f"Error sorting results: {e}")
                                sorted_results = results  # Use unsorted results as fallback
                            
                            # Best parameters
                            if sorted_results:
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
            
            # Advanced Results Analysis (available in both modes)
            if 'quantconnect_optimization' in st.session_state:
                opt_data = st.session_state['quantconnect_optimization']
                summary = opt_data['summary']
                analyzer = opt_data['analyzer']
                
                st.divider()
                st.subheader("📈 Advanced Results Analysis")
                
                analysis_tabs = st.tabs([
                    "📊 Results Grid", 
                    "🔥 Heatmaps", 
                    "📈 Equity Curves",
                    "🎯 Sensitivity Analysis",
                    "🛡️ Robustness Testing",
                    "📋 Full Report"
                ])
                
                with analysis_tabs[0]:
                    # Results grid
                    st.write("**Top Parameter Combinations**")
                    results_grid = analyzer.create_results_grid(top_n=20, sort_by=opt_data['objective'])
                    st.dataframe(results_grid, use_container_width=True, height=400)
                    
                    # Export button
                    if st.button("💾 Export Results to CSV"):
                        filename = analyzer.export_results_to_csv()
                        st.success(f"✅ Results exported to {filename}")
                
                with analysis_tabs[1]:
                    # Parameter heatmaps
                    if analyzer.results_df is not None and len(analyzer.results_df) > 5:
                        param_cols = [col.replace('param_', '') for col in analyzer.results_df.columns if col.startswith('param_')]
                        
                        if len(param_cols) >= 2:
                            col_heat1, col_heat2 = st.columns(2)
                            with col_heat1:
                                param_x = st.selectbox("X-axis Parameter", param_cols, key="heat_x")
                            with col_heat2:
                                param_y = st.selectbox("Y-axis Parameter", param_cols, index=1, key="heat_y")
                            
                            if param_x != param_y:
                                heatmap_fig = analyzer.create_parameter_heatmap(param_x, param_y, opt_data['objective'])
                                st.plotly_chart(heatmap_fig, use_container_width=True)
                            
                            # Correlation matrix
                            st.write("**Parameter Correlation Matrix**")
                            corr_fig = analyzer.create_parameter_correlation_matrix()
                            st.plotly_chart(corr_fig, use_container_width=True)
                        else:
                            st.info("Need at least 2 parameters for heatmap analysis")
                    else:
                        st.info("Not enough data points for heatmap analysis")
                
                with analysis_tabs[2]:
                    # Equity curves comparison
                    st.write("**Top 5 Equity Curves Comparison**")
                    equity_fig = analyzer.create_equity_curves_comparison(top_n=5)
                    st.plotly_chart(equity_fig, use_container_width=True)
                    
                    # Optimization progress
                    st.write("**Optimization Progress**")
                    progress_fig = analyzer.create_optimization_progress_chart()
                    st.plotly_chart(progress_fig, use_container_width=True)
                
                with analysis_tabs[3]:
                    # Parameter sensitivity analysis
                    st.write("**Parameter Sensitivity Analysis**")
                    sensitivities = analyzer.create_parameter_sensitivity_analysis(opt_data['objective'])
                    
                    if sensitivities:
                        sens_data = []
                        for sens in sensitivities:
                            sens_data.append({
                                'Parameter': sens.parameter_name.replace('_', ' ').title(),
                                'Correlation': f"{sens.correlation_with_objective:.3f}",
                                'Importance': f"{sens.parameter_importance:.3f}",
                                'Sensitivity Score': f"{sens.sensitivity_score:.3f}",
                                'Optimal Range': f"{sens.optimal_range[0]:.3f} - {sens.optimal_range[1]:.3f}"
                            })
                        
                        st.dataframe(pd.DataFrame(sens_data), use_container_width=True)
                        
                        # Most sensitive parameters
                        st.info(f"**Most Sensitive Parameter**: {sensitivities[0].parameter_name} (Score: {sensitivities[0].sensitivity_score:.3f})")
                    else:
                        st.info("No sensitivity analysis available")
                
                with analysis_tabs[4]:
                    # Robustness testing
                    st.write("**Robustness Analysis**")
                    robustness = analyzer.analyze_robustness(opt_data['objective'])
                    
                    # Robustness dashboard
                    robustness_fig = analyzer.create_robustness_dashboard(robustness)
                    st.plotly_chart(robustness_fig, use_container_width=True)
                    
                    # Robustness metrics
                    rob_col1, rob_col2, rob_col3 = st.columns(3)
                    with rob_col1:
                        st.metric("Robustness Score", f"{robustness.robustness_score:.2%}")
                    with rob_col2:
                        st.metric("Performance Consistency", f"{robustness.performance_consistency:.2%}")
                    with rob_col3:
                        st.metric("Overfitting Risk", f"{robustness.overfitting_risk:.2%}")
                    
                    # Recommendations
                    if robustness.robustness_score > 0.8:
                        st.success("✅ **High Robustness**: Parameters are stable and reliable for live trading")
                    elif robustness.robustness_score > 0.6:
                        st.warning("⚠️ **Moderate Robustness**: Consider additional validation before live trading")
                    else:
                        st.error("❌ **Low Robustness**: High risk of overfitting, use walk-forward analysis")
                
                with analysis_tabs[5]:
                    # Full report
                    st.write("**Comprehensive Optimization Report**")
                    report = analyzer.generate_optimization_report()
                    st.text(report)
                    
                    # Download report
                    if st.button("💾 Download Full Report"):
                        st.download_button(
                            "📥 Download Report",
                            report,
                            f"optimization_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                            "text/plain"
                        )
        
        except ImportError as e:
            st.error(f"Error importing optimization modules: {e}")
            st.info("Make sure parameter_manager.py, optimization_engine.py, and results_analyzer.py are in the project directory")
    
    with tab5:
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
    
    with tab6:
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
