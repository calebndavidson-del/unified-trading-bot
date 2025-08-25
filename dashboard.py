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
from utils.visualization import create_dashboard_charts, CandlestickPlotter, TrendVisualization
from utils.risk import calculate_comprehensive_risk_metrics, PositionSizing, StopLossManager
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
    
    def render_sidebar(self) -> Dict:
        """Render sidebar controls"""
        st.sidebar.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.sidebar.title("📈 Trading Bot Controls")
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Symbol selection
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
        
        return {
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'position_size': position_size / 100,
            'stop_loss': stop_loss / 100,
            'risk_free_rate': risk_free_rate
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
                st.plotly_chart(chart, use_container_width=True)
        
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
                        st.dataframe(pattern_df, use_container_width=True)
                    
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
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader(f"Trend Analysis - {symbol}")
            
            # Trend strength visualization
            if 'trend_strength' in data.columns:
                trend_chart = self.trend_viz.plot_trend_strength(data, title=f"{symbol} - Trend Strength")
                st.plotly_chart(trend_chart, use_container_width=True)
            
            # Signal analysis
            if 'composite_signal' in data.columns:
                signal_chart = self.trend_viz.plot_signal_analysis(data, title=f"{symbol} - Trading Signals")
                st.plotly_chart(signal_chart, use_container_width=True)
        
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
                st.plotly_chart(fig, use_container_width=True)
    
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
    
    def run(self):
        """Run the main dashboard"""
        # Render sidebar
        settings = self.render_sidebar()
        
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


# Initialize and run the dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()
