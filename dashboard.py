#!/usr/bin/env python3
"""
Main Streamlit App for Quantitative Trading Bot
Advanced trading dashboard with market analysis, signals, and risk management
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from features.candlestick import extract_all_candlestick_features, CandlestickPatternExtractor
from features.earnings import create_comprehensive_earnings_features
from features.market_trend import create_comprehensive_trend_features
from features.backtesting import BacktestEngine, TechnicalAnalysisStrategy, MeanReversionStrategy, MomentumStrategy, PatternRecognitionStrategy
from utils.visualization import create_dashboard_charts, CandlestickPlotter, TrendVisualization
from utils.risk import calculate_comprehensive_risk_metrics, PositionSizing, StopLossManager
from utils.asset_universe import AssetUniverseManager, AssetInfo, AssetUniverse
from utils.backtesting_metrics import BacktestingMetrics
from model_config import TradingBotConfig, load_config

# Page configuration
st.set_page_config(
    page_title="Quantitative Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ff88;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00ff88;
    }
    .bearish-card {
        border-left-color: #ff6b6b !important;
    }
    .sidebar-info {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class TradingDashboard:
    """Main trading dashboard class"""
    
    def __init__(self):
        self.config = self._load_config()
        self.plotter = CandlestickPlotter()
        self.trend_viz = TrendVisualization()
        self.pattern_extractor = CandlestickPatternExtractor()
        self.stop_manager = StopLossManager()
        
        # Initialize asset universe manager
        self.asset_manager = AssetUniverseManager()
        
        # Get symbols from asset universe or fallback to default
        if self.config.data.use_asset_universe:
            universe_symbols = self.asset_manager.get_universe().get_all_symbols()
            self.default_symbols = universe_symbols if universe_symbols else [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ',
                'BTC-USD', 'ETH-USD', 'SOL-USD'
            ]
        else:
            # Default symbols
            self.default_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ',
                'BTC-USD', 'ETH-USD', 'SOL-USD'
            ]
    
    def _load_config(self) -> TradingBotConfig:
        """Load trading bot configuration"""
        try:
            return load_config()
        except:
            return TradingBotConfig()
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_market_data(_self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch market data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                st.error(f"No data available for {symbol}")
                return pd.DataFrame()
            
            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def create_enhanced_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create enhanced features for analysis"""
        if data.empty:
            return data
        
        try:
            # Add candlestick patterns
            data = extract_all_candlestick_features(data, include_patterns=True, include_signals=True)
            
            # Add technical indicators and trend analysis
            data = create_comprehensive_trend_features(data)
            
            # Add earnings features (simplified for demo)
            # data = create_comprehensive_earnings_features(symbol, data)
            
            return data
        except Exception as e:
            st.warning(f"Some features could not be calculated: {str(e)}")
            return data
    
    def _refresh_symbols(self):
        """Refresh symbol list from asset universe"""
        if self.config.data.use_asset_universe:
            universe_symbols = self.asset_manager.get_universe().get_all_symbols()
            if universe_symbols:
                self.default_symbols = universe_symbols
            else:
                # Fallback to default if universe is empty
                self.default_symbols = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ',
                    'BTC-USD', 'ETH-USD', 'SOL-USD'
                ]

    def render_sidebar(self) -> Dict:
        """Render sidebar controls"""
        # Refresh symbols from asset universe
        self._refresh_symbols()
        
        st.sidebar.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.sidebar.title("📈 Unified Trading Bot")
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Navigation info
        st.sidebar.markdown("### 🧭 Navigation")
        st.sidebar.info("Use the tabs above to switch between Trading, Analysis, and Settings")
        
        # Asset universe status
        if self.config.data.use_asset_universe:
            universe = self.asset_manager.get_universe()
            total_assets = len(universe.get_all_symbols())
            st.sidebar.markdown("### 🌐 Asset Universe")
            st.sidebar.metric("Active Assets", total_assets)
            if total_assets == 0:
                st.sidebar.warning("No assets in universe! Add some in the Asset Universe tab.")
        
        # Symbol selection
        if not self.default_symbols:
            st.sidebar.error("No symbols available. Please add assets to your universe.")
            symbol = None
        else:
            symbol = st.sidebar.selectbox(
                "Select Symbol",
                options=self.default_symbols,
                index=0,
                help="Choose a symbol to analyze"
            )
        
        # Time period selection
        period = st.sidebar.selectbox(
            "Time Period",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Select the time period for analysis"
        )
        
        # Interval selection
        interval = st.sidebar.selectbox(
            "Data Interval",
            options=["1d", "1wk", "1mo"],
            index=0,
            help="Select the data interval"
        )
        
        st.sidebar.markdown("---")
        
        # Risk management settings
        st.sidebar.subheader("🛡️ Risk Management")
        
        position_size = st.sidebar.slider(
            "Position Size (%)",
            min_value=1,
            max_value=25,
            value=10,
            help="Maximum position size as percentage of portfolio"
        )
        
        stop_loss = st.sidebar.slider(
            "Stop Loss (%)",
            min_value=1,
            max_value=20,
            value=5,
            help="Stop loss percentage"
        )
        
        risk_free_rate = st.sidebar.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Risk-free rate for calculations"
        ) / 100
        
        margin_requirement = st.sidebar.slider(
            "Margin Requirement (%)",
            min_value=10,
            max_value=100,
            value=30,
            help="Percentage of portfolio value required as margin for available cash calculations"
        )
        
        return {
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'position_size': position_size / 100,
            'stop_loss': stop_loss / 100,
            'risk_free_rate': risk_free_rate,
            'margin_requirement': margin_requirement / 100
        }
    
    def render_main_metrics(self, data: pd.DataFrame, symbol: str, settings: Dict):
        """Render main performance metrics"""
        if data.empty:
            return
        
        st.markdown('<h2 class="main-header">📊 Market Analysis Dashboard</h2>', unsafe_allow_html=True)
        
        # Calculate basic metrics
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        # Calculate returns and risk metrics
        returns = data['Close'].pct_change().dropna()
        risk_metrics = calculate_comprehensive_risk_metrics(returns, risk_free_rate=settings['risk_free_rate'])
        
        # Display key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Current Price",
                value=f"${current_price:.2f}",
                delta=f"{price_change_pct:+.2f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Volatility (Ann.)",
                value=f"{risk_metrics['volatility']:.1%}",
                help="Annualized volatility"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Sharpe Ratio",
                value=f"{risk_metrics['sharpe_ratio']:.2f}",
                help="Risk-adjusted return"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Max Drawdown",
                value=f"{risk_metrics['max_drawdown']:.1%}",
                help="Maximum peak-to-trough decline"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="VaR (95%)",
                value=f"{risk_metrics['var_95']:.1%}",
                help="Value at Risk (95% confidence)"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_charts(self, data: pd.DataFrame, symbol: str):
        """Render main charts"""
        if data.empty:
            st.warning("No data available for charting")
            return
        
        # Create tabs for different chart types
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Technical Analysis", "🕯️ Candlestick Patterns", "📊 Trend Analysis", "⚠️ Risk Analysis"])
        
        with tab1:
            st.subheader(f"Technical Analysis - {symbol}")
            
            # Main candlestick chart with indicators
            if len(data) > 0:
                chart = self.plotter.create_candlestick_with_indicators(
                    data,
                    indicators=['sma', 'bb', 'rsi', 'macd'],
                    title=f"{symbol} - Technical Analysis"
                )
                st.plotly_chart(chart, width='stretch')
        
        with tab2:
            st.subheader(f"Candlestick Pattern Analysis - {symbol}")
            
            # Pattern analysis
            if 'pattern_doji' in data.columns:
                # Show pattern summary
                pattern_cols = [col for col in data.columns if col.startswith('pattern_')]
                if pattern_cols:
                    pattern_summary = {}
                    for col in pattern_cols[:10]:  # Show top 10 patterns
                        pattern_name = col.replace('pattern_', '').replace('_', ' ').title()
                        bullish_count = (data[col] > 0).sum()
                        bearish_count = (data[col] < 0).sum()
                        if bullish_count > 0 or bearish_count > 0:
                            pattern_summary[pattern_name] = {
                                'Bullish': bullish_count,
                                'Bearish': bearish_count
                            }
                    
                    if pattern_summary:
                        pattern_df = pd.DataFrame(pattern_summary).T.fillna(0)
                        st.dataframe(pattern_df, width='stretch')
                    
                    # Pattern strength chart
                    if 'bullish_pattern_score' in data.columns and 'bearish_pattern_score' in data.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['bullish_pattern_score'],
                            mode='lines',
                            name='Bullish Patterns',
                            line=dict(color='green')
                        ))
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=-data['bearish_pattern_score'],
                            mode='lines',
                            name='Bearish Patterns',
                            line=dict(color='red')
                        ))
                        fig.update_layout(
                            title=f"{symbol} - Pattern Strength Over Time",
                            template='plotly_dark',
                            height=400
                        )
                        st.plotly_chart(fig, width='stretch')
        
        with tab3:
            st.subheader(f"Trend Analysis - {symbol}")
            
            # Trend strength visualization
            if 'trend_strength' in data.columns:
                trend_chart = self.trend_viz.plot_trend_strength(data, title=f"{symbol} - Trend Strength")
                st.plotly_chart(trend_chart, width='stretch')
            
            # Signal analysis
            if 'composite_signal' in data.columns:
                signal_chart = self.trend_viz.plot_signal_analysis(data, title=f"{symbol} - Trading Signals")
                st.plotly_chart(signal_chart, width='stretch')
        
        with tab4:
            st.subheader(f"Risk Analysis - {symbol}")
            
            # Returns analysis
            returns = data['Close'].pct_change().dropna()
            
            if len(returns) > 0:
                # Drawdown chart
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.3)',
                    line=dict(color='#ff6b6b')
                ))
                fig.update_layout(
                    title=f"{symbol} - Drawdown Analysis",
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, width='stretch')
    
    def render_trading_signals(self, data: pd.DataFrame, symbol: str, settings: Dict):
        """Render trading signals and recommendations"""
        if data.empty:
            return
        
        st.subheader("🎯 Trading Signals & Recommendations")
        
        # Get latest signals
        latest_data = data.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Current Signals")
            
            signals = []
            
            # RSI signal
            if 'rsi' in latest_data:
                rsi = latest_data['rsi']
                if rsi < 30:
                    signals.append(("RSI Oversold", "BUY", "green"))
                elif rsi > 70:
                    signals.append(("RSI Overbought", "SELL", "red"))
                else:
                    signals.append(("RSI Neutral", "HOLD", "gray"))
            
            # MACD signal
            if 'macd_bullish_cross' in latest_data and latest_data['macd_bullish_cross']:
                signals.append(("MACD Bullish Cross", "BUY", "green"))
            elif 'macd_bearish_cross' in latest_data and latest_data['macd_bearish_cross']:
                signals.append(("MACD Bearish Cross", "SELL", "red"))
            
            # Trend signal
            if 'trend_strength' in latest_data:
                trend = latest_data['trend_strength']
                if trend > 0.3:
                    signals.append(("Strong Uptrend", "BUY", "green"))
                elif trend < -0.3:
                    signals.append(("Strong Downtrend", "SELL", "red"))
                else:
                    signals.append(("Sideways Trend", "HOLD", "gray"))
            
            # Display signals
            for signal_name, action, color in signals:
                st.markdown(f"**{signal_name}**: <span style='color: {color}'>{action}</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Position Sizing Recommendation")
            
            current_price = latest_data['Close']
            
            # Calculate stop loss level
            stop_loss_price = current_price * (1 - settings['stop_loss'])
            
            # Portfolio value for calculation
            portfolio_value = st.number_input(
                "Portfolio Value ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=1000
            )
            
            # Calculate position size using risk-based method
            risk_amount = portfolio_value * 0.02  # 2% risk per trade
            risk_per_share = abs(current_price - stop_loss_price)
            
            if risk_per_share > 0:
                suggested_shares = int(risk_amount / risk_per_share)
                suggested_value = suggested_shares * current_price
                position_pct = (suggested_value / portfolio_value) * 100
                
                st.metric("Suggested Shares", f"{suggested_shares:,}")
                st.metric("Position Value", f"${suggested_value:,.2f}")
                st.metric("Portfolio %", f"{position_pct:.1f}%")
                st.metric("Stop Loss", f"${stop_loss_price:.2f}")
    
    def render_trading_tab(self, settings: Dict):
        """Render the Trading tab with paper trading controls"""
        st.markdown('<div class="main-header">🤖 Trading Interface</div>', unsafe_allow_html=True)
        
        # Initialize session state for trading
        if 'trading_session_active' not in st.session_state:
            st.session_state.trading_session_active = False
        if 'paper_portfolio_value' not in st.session_state:
            st.session_state.paper_portfolio_value = 100000.0
        if 'trades_log' not in st.session_state:
            st.session_state.trades_log = []
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Create three columns for the main trading interface
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("🔧 Trading Controls")
            
            # Paper Trading Session Controls
            st.markdown("### Paper Trading Session")
            session_status = "🟢 Active" if st.session_state.trading_session_active else "🔴 Inactive"
            st.markdown(f"**Status:** {session_status}")
            
            col1a, col1b = st.columns(2)
            with col1a:
                if st.button("▶️ Start Session", disabled=st.session_state.trading_session_active):
                    st.session_state.trading_session_active = True
                    st.session_state.messages.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'type': 'success',
                        'message': 'Paper trading session started'
                    })
                    st.rerun()
            
            with col1b:
                if st.button("⏹️ Stop Session", disabled=not st.session_state.trading_session_active):
                    st.session_state.trading_session_active = False
                    st.session_state.messages.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'type': 'info',
                        'message': 'Paper trading session stopped'
                    })
                    st.rerun()
            
            st.markdown("---")
            
            # Strategy/Model Selection
            st.markdown("### Strategy & Model")
            strategy = st.selectbox(
                "Trading Strategy",
                ["Technical Analysis", "Mean Reversion", "Momentum", "Pattern Recognition"],
                help="Select the trading strategy to use"
            )
            
            model = st.selectbox(
                "ML Model",
                ["LSTM Neural Network", "Random Forest", "Support Vector Machine", "Ensemble"],
                help="Select the machine learning model"
            )
            
            confidence_threshold = st.slider(
                "Signal Confidence Threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.75,
                step=0.05,
                help="Minimum confidence level for trade signals"
            )
            
            st.markdown("---")
            
            # Risk Management
            st.markdown("### Risk Management")
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=25,
                value=int(settings.get('position_size', 0.1) * 100),
                help="Maximum position size as percentage of portfolio"
            )
            
            stop_loss_pct = st.slider(
                "Stop Loss (%)",
                min_value=1,
                max_value=20,
                value=int(settings.get('stop_loss', 0.05) * 100),
                help="Stop loss percentage"
            )
            
            take_profit_pct = st.slider(
                "Take Profit (%)",
                min_value=5,
                max_value=50,
                value=15,
                help="Take profit percentage"
            )
        
        with col2:
            st.subheader("📊 Portfolio Metrics")
            
            # Portfolio Overview
            portfolio_value = st.session_state.paper_portfolio_value
            daily_pnl = 0.0  # Calculate from trades
            total_pnl = portfolio_value - 100000.0
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
                st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl/100000*100:+.1f}%")
            
            with col2b:
                st.metric("Daily P&L", f"${daily_pnl:,.2f}")
                st.metric("Available Cash", f"${portfolio_value * (1 - settings.get('margin_requirement', 0.3)):,.2f}")
            
            # Current positions (placeholder)
            st.markdown("### Current Positions")
            if st.session_state.trading_session_active:
                st.info("No open positions")
            else:
                st.warning("Start trading session to view positions")
            
            # Live Trading Section (Disabled)
            st.markdown("---")
            st.markdown("### 🚫 Live Trading")
            st.warning("Live trading is currently disabled. This feature is under development.")
            col2c, col2d = st.columns(2)
            with col2c:
                st.button("▶️ Start Live", disabled=True)
            with col2d:
                st.button("⏹️ Stop Live", disabled=True)
        
        with col3:
            st.subheader("📝 Trade Log & Messages")
            
            # Trade Log
            st.markdown("### Recent Trades")
            if st.session_state.trades_log:
                for trade in st.session_state.trades_log[-5:]:  # Show last 5 trades
                    with st.expander(f"{trade['symbol']} - {trade['action']} - {trade['time']}"):
                        st.write(f"**Price:** ${trade['price']:.2f}")
                        st.write(f"**Quantity:** {trade['quantity']}")
                        st.write(f"**P&L:** ${trade.get('pnl', 0):.2f}")
            else:
                st.info("No trades executed yet")
            
            # Messages/Notifications
            st.markdown("### Messages & Notifications")
            message_container = st.container()
            
            with message_container:
                if st.session_state.messages:
                    for msg in st.session_state.messages[-10:]:  # Show last 10 messages
                        icon = "✅" if msg['type'] == 'success' else "ℹ️" if msg['type'] == 'info' else "⚠️"
                        st.markdown(f"**{msg['time']}** {icon} {msg['message']}")
                else:
                    st.info("No messages yet")
            
            # Manual trading controls (for paper trading)
            if st.session_state.trading_session_active:
                st.markdown("---")
                st.markdown("### Manual Trading")
                
                # Use the same symbol list as the sidebar for consistency
                symbol_list = getattr(self, "default_symbols", st.session_state.get("available_symbols", ['AAPL', 'MSFT', 'GOOGL', 'TSLA']))
                manual_symbol = st.selectbox("Symbol", symbol_list, key="manual_symbol")
                manual_action = st.selectbox("Action", ["BUY", "SELL"], key="manual_action")
                manual_quantity = st.number_input("Quantity", min_value=1, value=10, key="manual_quantity")
                
                if st.button("Execute Trade"):
                    # Fetch real-time price for the selected symbol
                    try:
                        ticker = yf.Ticker(manual_symbol)
                        price_data = ticker.history(period="1d")
                        if not price_data.empty:
                            current_price = price_data['Close'].iloc[-1]
                        else:
                            current_price = None
                    except Exception as e:
                        current_price = None

                    if current_price is not None:
                        new_trade = {
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'symbol': manual_symbol,
                            'action': manual_action,
                            'quantity': manual_quantity,
                            'price': float(current_price),
                            'pnl': 0.0
                        }
                        st.session_state.trades_log.append(new_trade)
                        st.session_state.messages.append({
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'type': 'success',
                            'message': f'{manual_action} {manual_quantity} shares of {manual_symbol} at ${current_price:.2f}'
                        })
                        st.rerun()
                    else:
                        st.session_state.messages.append({
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'type': 'error',
                            'message': f'Failed to fetch price for {manual_symbol}. Trade not executed.'
                        })
    def render_analysis_tab(self, settings: Dict):
        """Render the Analysis tab with existing functionality"""
        # Fetch data
        with st.spinner(f"Fetching data for {settings['symbol']}..."):
            data = self.fetch_market_data(
                settings['symbol'],
                settings['period'],
                settings['interval']
            )
        
        if not data.empty:
            # Create enhanced features
            with st.spinner("Calculating technical indicators..."):
                enhanced_data = self.create_enhanced_features(data, settings['symbol'])
            
            # Render main metrics
            self.render_main_metrics(enhanced_data, settings['symbol'], settings)
            
            # Render charts
            self.render_charts(enhanced_data, settings['symbol'])
            
            # Render trading signals
            self.render_trading_signals(enhanced_data, settings['symbol'], settings)
            
            # Data export option
            st.subheader("📥 Data Export")
            if st.button("Download Enhanced Data"):
                csv = enhanced_data.to_csv()
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{settings['symbol']}_enhanced_data.csv",
                    mime="text/csv"
                )
        else:
            st.error("Unable to fetch data. Please try a different symbol or check your internet connection.")

    def render_settings_tab(self):
        """Render the Settings tab"""
        st.markdown('<div class="main-header">⚙️ Settings</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 General Settings")
            
            # Theme settings
            st.markdown("### Display Preferences")
            chart_theme = st.selectbox("Chart Theme", ["Dark", "Light"], index=0)
            auto_refresh = st.checkbox("Auto-refresh data", value=False)
            refresh_interval = st.slider("Refresh interval (minutes)", 1, 60, 5)
            
            # Data settings
            st.markdown("### Data Settings")
            default_period = st.selectbox("Default time period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
            default_interval = st.selectbox("Default interval", ["1d", "1wk", "1mo"], index=0)
            
            # Alert settings
            st.markdown("### Notifications")
            enable_alerts = st.checkbox("Enable trading alerts", value=True)
            email_notifications = st.checkbox("Email notifications", value=False)
            sound_alerts = st.checkbox("Sound alerts", value=True)
        
        with col2:
            st.subheader("🛡️ Risk Management Defaults")
            
            # Default risk settings
            default_position_size = st.slider("Default position size (%)", 1, 25, 10)
            default_stop_loss = st.slider("Default stop loss (%)", 1, 20, 5)
            default_take_profit = st.slider("Default take profit (%)", 5, 50, 15)
            default_margin_requirement = st.slider("Default margin requirement (%)", 10, 100, 30)
            
            # Advanced settings
            st.markdown("### Advanced")
            max_concurrent_trades = st.number_input("Max concurrent trades", 1, 20, 5)
            min_trade_amount = st.number_input("Minimum trade amount ($)", 100, 10000, 1000)
            
            # API settings
            st.markdown("### API Configuration")
            st.info("API settings are configured via environment variables")
            
            # Save settings
            if st.button("💾 Save Settings"):
                st.success("Settings saved successfully!")

    def render_asset_universe_tab(self):
        """Render asset universe management tab"""
        st.markdown('<h1 class="main-header">🌐 Asset Universe Management</h1>', unsafe_allow_html=True)
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔍 Search & Add Assets")
            
            # Search functionality
            search_query = st.text_input("🔎 Search for assets", placeholder="Enter symbol or company name...")
            
            asset_type_filter = st.selectbox(
                "Filter by Asset Type",
                options=["all", "stock", "etf", "crypto", "index"],
                index=0
            )
            
            if search_query:
                # Search for assets
                search_results = self.asset_manager.search_assets(
                    search_query, 
                    None if asset_type_filter == "all" else asset_type_filter
                )
                
                if search_results:
                    st.markdown("**Search Results:**")
                    for asset in search_results[:10]:  # Show top 10 results
                        col_symbol, col_name, col_add = st.columns([1, 2, 1])
                        with col_symbol:
                            st.text(asset.symbol)
                        with col_name:
                            st.text(f"{asset.name} ({asset.asset_type})")
                        with col_add:
                            if st.button(f"➕", key=f"add_{asset.symbol}"):
                                if self.asset_manager.add_to_universe(asset.symbol, asset.asset_type):
                                    self.asset_manager.save_universe()
                                    st.success(f"Added {asset.symbol} to universe!")
                                    st.rerun()
                else:
                    st.info("No assets found. Try a different search term.")
            
            # Manual symbol addition
            st.markdown("---")
            st.subheader("➕ Add Custom Symbol")
            
            custom_symbol = st.text_input("Symbol", placeholder="e.g., AAPL, BTC-USD")
            if custom_symbol:
                if st.button("🔍 Validate & Add"):
                    is_valid, message, asset_info = self.asset_manager.validate_symbol(custom_symbol)
                    if is_valid and asset_info:
                        if self.asset_manager.add_to_universe(asset_info.symbol, asset_info.asset_type):
                            self.asset_manager.save_universe()
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error("Failed to add symbol to universe")
                    else:
                        st.error(f"❌ {message}")
        
        with col2:
            st.subheader("📊 Current Universe")
            
            universe = self.asset_manager.get_universe()
            all_symbols = universe.get_all_symbols()
            
            if all_symbols:
                st.metric("Total Assets", len(all_symbols))
                
                # Show breakdown by category
                categories = [
                    ("📈 Stocks", universe.stocks),
                    ("💰 Cryptocurrencies", universe.crypto),
                    ("📊 ETFs", universe.etfs),
                    ("📉 Indexes", universe.indexes),
                    ("🎯 Custom", universe.custom)
                ]
                
                for cat_name, cat_symbols in categories:
                    if cat_symbols:
                        with st.expander(f"{cat_name} ({len(cat_symbols)})"):
                            for symbol in sorted(cat_symbols):
                                col_sym, col_remove = st.columns([3, 1])
                                with col_sym:
                                    st.text(symbol)
                                with col_remove:
                                    if st.button("❌", key=f"remove_{symbol}_{cat_name.replace(' ', '_')}"):
                                        if self.asset_manager.remove_from_universe(symbol):
                                            self.asset_manager.save_universe()
                                            st.success(f"Removed {symbol}")
                                            st.rerun()
                
                # Clear all button
                if st.button("🗑️ Clear All Assets", type="secondary"):
                    if st.button("⚠️ Confirm Clear All", type="primary"):
                        universe.stocks.clear()
                        universe.crypto.clear()
                        universe.etfs.clear()
                        universe.indexes.clear()
                        universe.custom.clear()
                        self.asset_manager.save_universe()
                        st.success("Cleared all assets from universe!")
                        st.rerun()
            else:
                st.info("No assets in universe. Add some assets to get started!")
        
        # Preloaded lists section
        st.markdown("---")
        st.subheader("📚 Preloaded Asset Lists")
        
        preloaded_lists = self.asset_manager.get_preloaded_lists()
        
        # Create tabs for different preloaded lists
        list_tabs = st.tabs([
            "🏢 Top 250 US Stocks",
            "📊 Top 50 ETFs", 
            "🌍 Top 10 Global Indexes",
            "💎 Top 10 Cryptocurrencies"
        ])
        
        list_names = [
            'top_250_us_stocks',
            'top_50_etfs', 
            'top_10_global_indexes',
            'top_10_crypto'
        ]
        
        for tab, list_name in zip(list_tabs, list_names):
            with tab:
                assets = preloaded_lists[list_name]
                
                # Bulk selection
                col1_bulk, col2_bulk = st.columns([3, 1])
                with col1_bulk:
                    st.markdown(f"**{len(assets)} assets available**")
                with col2_bulk:
                    if st.button(f"➕ Add All", key=f"add_all_{list_name}"):
                        count = self.asset_manager.bulk_add_to_universe(assets)
                        self.asset_manager.save_universe()
                        st.success(f"Added {count} assets to universe!")
                        st.rerun()
                
                # Show assets in a table format
                display_data = []
                for asset in assets[:20]:  # Show first 20 for performance
                    display_data.append({
                        'Symbol': asset.symbol,
                        'Name': asset.name,
                        'Type': asset.asset_type,
                        'Sector': asset.sector or 'N/A',
                        'Industry': asset.industry or 'N/A'
                    })
                
                if display_data:
                    df = pd.DataFrame(display_data)
                    st.dataframe(df, width='stretch')
                    
                    if len(assets) > 20:
                        st.info(f"Showing first 20 of {len(assets)} assets. Use 'Add All' to add complete list.")
                
                # Individual add buttons for visible assets
                st.markdown("**Quick Add:**")
                cols = st.columns(4)
                for i, asset in enumerate(assets[:8]):  # Show first 8 for quick add
                    with cols[i % 4]:
                        if st.button(f"➕ {asset.symbol}", key=f"quick_add_{asset.symbol}_{list_name}"):
                            if self.asset_manager.add_to_universe(asset.symbol, asset.asset_type):
                                self.asset_manager.save_universe()
                                st.success(f"Added {asset.symbol}!")
                                st.rerun()

    def render_backtesting_tab(self, settings: Dict):
        """Render the Backtesting tab"""
        st.markdown('<h1 class="main-header">🔍 Strategy Backtesting</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Test your trading strategies with historical data from the current year
        Select assets, strategy, and model to simulate trading performance with real market data.
        """)
        
        # Create columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Backtest Configuration")
            
            # Asset Selection
            st.markdown("**Asset Selection**")
            
            # Get available symbols from asset universe or fallback
            if hasattr(self, 'asset_manager') and self.asset_manager:
                universe = self.asset_manager.get_universe()
                available_symbols = universe.get_all_symbols()
                if not available_symbols:
                    available_symbols = self.default_symbols
            else:
                available_symbols = self.default_symbols
            
            # Multi-select for symbols
            selected_symbols = st.multiselect(
                "Select Assets to Backtest",
                options=available_symbols,
                default=available_symbols[:5] if len(available_symbols) >= 5 else available_symbols,
                help="Choose the assets you want to include in the backtest"
            )
            
            if not selected_symbols:
                st.warning("Please select at least one asset to backtest.")
                return
            
            # Strategy Selection
            st.markdown("**Strategy Configuration**")
            strategy_options = [
                "Technical Analysis",
                "Mean Reversion", 
                "Momentum",
                "Pattern Recognition"
            ]
            
            selected_strategy = st.selectbox(
                "Trading Strategy",
                options=strategy_options,
                help="Choose the trading strategy to test"
            )
            
            # Model Selection (placeholder for future ML integration)
            model_options = [
                "LSTM Neural Network",
                "Random Forest",
                "Support Vector Machine",
                "Ensemble"
            ]
            
            selected_model = st.selectbox(
                "ML Model (Future Integration)",
                options=model_options,
                help="Machine learning model for signal enhancement"
            )
            
            # Backtest Period Selection
            st.markdown("**Backtest Period**")
            
            period_options = [
                ("1mo", "1 Month"),
                ("6mo", "6 Months"), 
                ("1y", "1 Year"),
                ("3y", "3 Years"),
                ("5y", "5 Years")
            ]
            
            selected_period = st.selectbox(
                "Lookback Period",
                options=[option[0] for option in period_options],
                format_func=lambda x: next(option[1] for option in period_options if option[0] == x),
                index=2,  # Default to "1 Year"
                help="How far back the backtest should go. Longer periods provide more data but may include different market conditions."
            )
            
            # Risk Management Settings
            st.markdown("**Risk Management**")
            
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=1000,
                help="Starting portfolio value"
            )
            
            confidence_threshold = st.slider(
                "Signal Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                help="Minimum signal strength required to execute trades"
            )
            
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Maximum percentage of portfolio per position"
            ) / 100
            
            commission_rate = st.number_input(
                "Commission Rate (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Trading commission as percentage of trade value"
            ) / 100
            
            # Strategy-specific parameters
            if selected_strategy == "Technical Analysis":
                st.markdown("**Technical Analysis Parameters**")
                
                col_ta1, col_ta2 = st.columns(2)
                with col_ta1:
                    rsi_oversold = st.number_input("RSI Oversold", min_value=10, max_value=40, value=30)
                    ma_short = st.number_input("Short MA Period", min_value=5, max_value=30, value=10)
                
                with col_ta2:
                    rsi_overbought = st.number_input("RSI Overbought", min_value=60, max_value=90, value=70)
                    ma_long = st.number_input("Long MA Period", min_value=20, max_value=100, value=50)
                
                strategy_config = {
                    'rsi_oversold': rsi_oversold,
                    'rsi_overbought': rsi_overbought,
                    'ma_short': ma_short,
                    'ma_long': ma_long
                }
            
            elif selected_strategy == "Mean Reversion":
                st.markdown("**Mean Reversion Parameters**")
                
                col_mr1, col_mr2 = st.columns(2)
                with col_mr1:
                    bb_period = st.number_input("Bollinger Period", min_value=10, max_value=50, value=20)
                with col_mr2:
                    bb_std = st.number_input("Bollinger Std Dev", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
                
                strategy_config = {
                    'bb_period': bb_period,
                    'bb_std': bb_std
                }
            
            elif selected_strategy == "Momentum":
                st.markdown("**Momentum Parameters**")
                
                col_mom1, col_mom2 = st.columns(2)
                with col_mom1:
                    macd_fast = st.number_input("MACD Fast", min_value=5, max_value=20, value=12)
                    macd_signal = st.number_input("MACD Signal", min_value=5, max_value=15, value=9)
                with col_mom2:
                    macd_slow = st.number_input("MACD Slow", min_value=20, max_value=50, value=26)
                
                strategy_config = {
                    'ma_fast': macd_fast,
                    'ma_slow': macd_slow,
                    'signal_line': macd_signal
                }
            
            else:  # Pattern Recognition
                st.markdown("**Pattern Recognition Parameters**")
                pattern_weight = st.slider(
                    "Pattern Weight",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Weight for candlestick pattern signals"
                )
                
                strategy_config = {
                    'pattern_weight': pattern_weight
                }
            
            # Run Backtest Button
            run_backtest = st.button(
                "🚀 Run Backtest",
                type="primary",
                width='stretch',
                help="Execute the backtest with selected parameters"
            )
        
        with col2:
            if run_backtest:
                # Initialize backtesting engine
                import copy
                config = copy.deepcopy(self.config)
                config.risk.initial_capital = initial_capital
                config.risk.max_position_size = max_position_size
                
                backtest_engine = BacktestEngine(config)
                backtest_engine.commission_rate = commission_rate
                
                # Update strategy configuration
                if selected_strategy == "Technical Analysis":
                    backtest_engine.strategies[selected_strategy] = TechnicalAnalysisStrategy(strategy_config)
                elif selected_strategy == "Mean Reversion":
                    backtest_engine.strategies[selected_strategy] = MeanReversionStrategy(strategy_config)
                elif selected_strategy == "Momentum":
                    backtest_engine.strategies[selected_strategy] = MomentumStrategy(strategy_config)
                elif selected_strategy == "Pattern Recognition":
                    backtest_engine.strategies[selected_strategy] = PatternRecognitionStrategy(strategy_config)
                
                # Run backtest
                with st.spinner("Running backtest... This may take a moment."):
                    try:
                        results = backtest_engine.run_backtest(
                            symbols=selected_symbols,
                            strategy_name=selected_strategy,
                            model_name=selected_model,
                            confidence_threshold=confidence_threshold,
                            backtest_period=selected_period
                        )
                        
                        if "error" in results:
                            st.error(f"Backtest failed: {results['error']}")
                            return
                        
                        # Store results in session state
                        st.session_state.backtest_results = results
                        st.session_state.backtest_engine = backtest_engine
                        
                    except Exception as e:
                        st.error(f"Error running backtest: {str(e)}")
                        return
            
            # Display results if available
            if hasattr(st.session_state, 'backtest_results') and st.session_state.backtest_results:
                results = st.session_state.backtest_results
                backtest_engine = st.session_state.backtest_engine
                
                st.subheader("📊 Backtest Results")
                
                # Key Metrics Row
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric(
                        "Total Return",
                        f"{results['total_return_pct']:.2f}%",
                        delta=f"{results['total_return_pct']:.2f}%"
                    )
                
                with metric_cols[1]:
                    st.metric(
                        "Sharpe Ratio",
                        f"{results['sharpe_ratio']:.3f}",
                        delta=f"{results['sharpe_ratio']:.3f}"
                    )
                
                with metric_cols[2]:
                    st.metric(
                        "Max Drawdown",
                        f"{results['max_drawdown_pct']:.2f}%",
                        delta=f"{results['max_drawdown_pct']:.2f}%",
                        delta_color="inverse"
                    )
                
                with metric_cols[3]:
                    st.metric(
                        "Win Rate",
                        f"{results['win_rate_pct']:.1f}%",
                        delta=f"{results['win_rate_pct']:.1f}%"
                    )
                
                # Additional Metrics
                st.markdown("---")
                detail_cols = st.columns(3)
                
                with detail_cols[0]:
                    st.markdown("**📈 Performance**")
                    st.write(f"Initial Capital: ${results['initial_capital']:,.2f}")
                    st.write(f"Final Value: ${results['final_value']:,.2f}")
                    st.write(f"Volatility: {results['volatility_pct']:.2f}%")
                    st.write(f"Total Days: {results['total_days']}")
                
                with detail_cols[1]:
                    st.markdown("**📊 Trading Stats**")
                    st.write(f"Total Trades: {results['total_trades']}")
                    st.write(f"Winning Trades: {results['winning_trades']}")
                    st.write(f"Losing Trades: {results['losing_trades']}")
                    st.write(f"Profit Factor: {results['profit_factor']:.2f}")
                
                with detail_cols[2]:
                    st.markdown("**⚙️ Configuration**")
                    st.write(f"Strategy: {results['strategy']}")
                    st.write(f"Model: {results['model']}")
                    st.write(f"Assets: {len(results['symbols'])}")
                    st.write(f"Period: {results['start_date']} to {results['end_date']}")
                
                # Charts Section
                st.markdown("---")
                st.subheader("📈 Performance Charts")
                
                chart_tabs = st.tabs(["Equity Curve", "Trade Analysis", "Performance Metrics"])
                
                with chart_tabs[0]:
                    # Equity curve chart
                    if 'portfolio_history' in results and not results['portfolio_history'].empty:
                        equity_chart = BacktestingMetrics.create_equity_curve_chart(results['portfolio_history'])
                        st.plotly_chart(equity_chart, width='stretch')
                    else:
                        st.warning("No portfolio history data available for chart.")
                
                with chart_tabs[1]:
                    # Trade analysis
                    trade_details = backtest_engine.get_trade_details()
                    if not trade_details.empty:
                        # Trade analysis chart
                        trade_chart = BacktestingMetrics.create_trade_analysis_chart(trade_details)
                        st.plotly_chart(trade_chart, width='stretch')
                        
                        # Trade log table
                        st.markdown("**Trade Log**")
                        st.dataframe(trade_details, width='stretch')
                    else:
                        st.info("No completed trades to display.")
                
                with chart_tabs[2]:
                    # Performance metrics
                    if 'portfolio_history' in results and not results['portfolio_history'].empty:
                        advanced_metrics = BacktestingMetrics.calculate_advanced_metrics(results['portfolio_history'])
                        
                        # Metrics radar chart
                        metrics_chart = BacktestingMetrics.create_metrics_summary_chart(advanced_metrics)
                        st.plotly_chart(metrics_chart, width='stretch')
                        
                        # Monthly returns heatmap
                        monthly_chart = BacktestingMetrics.create_monthly_returns_heatmap(results['portfolio_history'])
                        st.plotly_chart(monthly_chart, width='stretch')
                    else:
                        st.warning("No portfolio history data available for metrics.")
                
                # Export Results
                st.markdown("---")
                st.subheader("💾 Export Results")
                
                export_cols = st.columns(3)
                
                with export_cols[0]:
                    if st.button("📄 Download Report", width='stretch'):
                        trade_details = backtest_engine.get_trade_details()
                        report = BacktestingMetrics.generate_performance_report(
                            results, trade_details, results.get('portfolio_history', pd.DataFrame())
                        )
                        st.download_button(
                            label="📄 Download Report",
                            data=report,
                            file_name=f"backtest_report_{results['strategy'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            mime="text/plain"
                        )
                
                with export_cols[1]:
                    if st.button("📊 Download Trade Log", width='stretch'):
                        trade_details = backtest_engine.get_trade_details()
                        if not trade_details.empty:
                            csv = trade_details.to_csv(index=False)
                            st.download_button(
                                label="📊 Download Trade Log",
                                data=csv,
                                file_name=f"trade_log_{results['strategy'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No trades to export.")
                
                with export_cols[2]:
                    if st.button("📈 Download Portfolio Data", width='stretch'):
                        if 'portfolio_history' in results and not results['portfolio_history'].empty:
                            csv = results['portfolio_history'].to_csv()
                            st.download_button(
                                label="📈 Download Portfolio Data",
                                data=csv,
                                file_name=f"portfolio_history_{results['strategy'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No portfolio data to export.")
            
            else:
                st.info("👆 Configure your backtest parameters and click 'Run Backtest' to see results.")

    def run(self):
        """Run the main dashboard"""
        # Main navigation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 Trading", "📊 Analysis", "🔍 Backtesting", "⚙️ Settings", "🌐 Asset Universe"])
        
        # Render sidebar (context-aware based on selected tab)
        settings = self.render_sidebar()
        
        with tab1:
            self.render_trading_tab(settings)
        
        with tab2:
            self.render_analysis_tab(settings)
        
        with tab3:
            self.render_backtesting_tab(settings)
        
        with tab4:
            self.render_settings_tab()
            
        with tab5:
            self.render_asset_universe_tab()


# Initialize and run the dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()
