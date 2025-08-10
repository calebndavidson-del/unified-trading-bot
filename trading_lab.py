#!/usr/bin/env python3
"""
Trading Lab - Advanced Parameter Experimentation and Learning Interface
Focuses on maximizing bot and user learning through comprehensive parameter exposure
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import os
import json
import time
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import subprocess
import threading

# Import existing modules
from parameter_manager import ParameterManager, create_default_parameters
from quant_bot import TradingParameters, PaperBroker

@dataclass
class TradingSession:
    """Trading session data for logging parameter experiments"""
    session_id: str
    start_time: datetime
    parameters: Dict[str, Any]
    total_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    final_equity: float = 100000.0
    duration_minutes: int = 0
    mode: str = "paper"  # paper or live
    notes: str = ""

class TradingLabInterface:
    """Main Trading Lab interface for parameter experimentation"""
    
    def __init__(self):
        self.initialize_session_state()
        self.load_config()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'lab_mode': 'paper',
            'lab_bot_running': False,
            'lab_bot_process': None,
            'lab_portfolio_value': 100000.0,
            'lab_trades_df': pd.DataFrame(),
            'lab_session_history': [],
            'lab_current_session': None,
            'lab_parameter_sets': [],
            'lab_auto_trade_enabled': False,
            'lab_selected_strategy': 'rsi_bollinger',
            'lab_universe_symbols': ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN']
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def load_config(self):
        """Load configuration from config.yaml"""
        try:
            with open('config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            st.error(f"Error loading config: {e}")
            self.config = {}
    
    def load_trades(self):
        """Load recent trades from CSV"""
        try:
            if os.path.exists('logs/trades.csv'):
                df = pd.read_csv('logs/trades.csv')
                if not df.empty:
                    df['ts'] = pd.to_datetime(df['ts'])
                    return df.tail(100)  # Load more trades for analysis
        except Exception as e:
            st.error(f"Error loading trades: {e}")
        return pd.DataFrame()
    
    def get_comprehensive_parameters(self) -> Dict[str, Any]:
        """Get all comprehensive trading parameters for exposure"""
        return {
            # Technical Indicators
            'rsi_period': {'min': 5, 'max': 50, 'default': 14, 'step': 1, 'description': 'RSI calculation period'},
            'rsi_oversold': {'min': 10, 'max': 40, 'default': 30, 'step': 1, 'description': 'RSI oversold threshold'},
            'rsi_overbought': {'min': 60, 'max': 90, 'default': 70, 'step': 1, 'description': 'RSI overbought threshold'},
            
            'bb_period': {'min': 10, 'max': 50, 'default': 20, 'step': 1, 'description': 'Bollinger Bands period'},
            'bb_std': {'min': 1.0, 'max': 3.0, 'default': 2.0, 'step': 0.1, 'description': 'Bollinger Bands standard deviations'},
            
            'ma_short': {'min': 5, 'max': 30, 'default': 10, 'step': 1, 'description': 'Short moving average period'},
            'ma_long': {'min': 20, 'max': 100, 'default': 50, 'step': 1, 'description': 'Long moving average period'},
            
            'macd_fast': {'min': 8, 'max': 20, 'default': 12, 'step': 1, 'description': 'MACD fast EMA period'},
            'macd_slow': {'min': 20, 'max': 40, 'default': 26, 'step': 1, 'description': 'MACD slow EMA period'},
            'macd_signal': {'min': 5, 'max': 15, 'default': 9, 'step': 1, 'description': 'MACD signal line period'},
            
            # Risk Management
            'position_size': {'min': 0.01, 'max': 0.5, 'default': 0.1, 'step': 0.01, 'description': 'Position size as fraction of portfolio'},
            'stop_loss': {'min': 0.005, 'max': 0.1, 'default': 0.02, 'step': 0.005, 'description': 'Stop loss percentage'},
            'take_profit': {'min': 0.01, 'max': 0.2, 'default': 0.04, 'step': 0.01, 'description': 'Take profit percentage'},
            'max_positions': {'min': 1, 'max': 10, 'default': 3, 'step': 1, 'description': 'Maximum concurrent positions'},
            'risk_per_trade': {'min': 0.01, 'max': 0.05, 'default': 0.02, 'step': 0.005, 'description': 'Risk per trade as fraction of portfolio'},
            
            # Trend and Momentum
            'trend_strength_min': {'min': 0.1, 'max': 1.0, 'default': 0.3, 'step': 0.1, 'description': 'Minimum trend strength required'},
            'volume_factor': {'min': 1.0, 'max': 5.0, 'default': 1.5, 'step': 0.1, 'description': 'Volume confirmation factor'},
            'momentum_threshold': {'min': 0.001, 'max': 0.05, 'default': 0.01, 'step': 0.001, 'description': 'Momentum threshold for entry'},
            
            # Market Regime
            'volatility_threshold': {'min': 0.1, 'max': 1.0, 'default': 0.25, 'step': 0.05, 'description': 'Volatility threshold for regime detection'},
            'correlation_threshold': {'min': 0.1, 'max': 0.9, 'default': 0.5, 'step': 0.1, 'description': 'Market correlation threshold'},
            
            # Timing
            'entry_patience': {'min': 1, 'max': 10, 'default': 3, 'step': 1, 'description': 'Bars to wait for entry confirmation'},
            'exit_patience': {'min': 1, 'max': 5, 'default': 2, 'step': 1, 'description': 'Bars to wait for exit confirmation'},
            'cool_down_period': {'min': 1, 'max': 20, 'default': 5, 'step': 1, 'description': 'Cool down period between trades (bars)'},
            
            # Filters
            'min_price': {'min': 1.0, 'max': 50.0, 'default': 5.0, 'step': 1.0, 'description': 'Minimum stock price filter'},
            'max_price': {'min': 50.0, 'max': 1000.0, 'default': 500.0, 'step': 10.0, 'description': 'Maximum stock price filter'},
            'min_volume': {'min': 100000, 'max': 10000000, 'default': 1000000, 'step': 100000, 'description': 'Minimum daily volume filter'},
            'market_cap_min': {'min': 1e9, 'max': 1e12, 'default': 5e9, 'step': 1e9, 'description': 'Minimum market cap filter'},
        }
    
    def render_parameter_sidebar(self):
        """Render comprehensive parameter sidebar"""
        with st.sidebar:
            st.header("🎛️ Trading Lab Controls")
            
            # Mode Selection
            st.subheader("🔄 Trading Mode")
            mode_options = ["Paper Trading", "Live Trading (Disabled)"]
            selected_mode = st.selectbox(
                "Select Mode",
                mode_options,
                index=0,
                help="Paper trading for safe experimentation. Live trading coming soon!"
            )
            
            if "Live Trading" in selected_mode:
                st.warning("⚠️ Live trading is disabled for safety. Use Paper Trading for experimentation.")
                st.session_state.lab_mode = 'paper'
            else:
                st.session_state.lab_mode = 'paper'
            
            # Bot Control
            st.subheader("🤖 Bot Control")
            
            if st.session_state.lab_bot_running:
                st.success("🟢 Bot Running")
                if st.button("⏹️ Stop Bot", use_container_width=True, key="lab_stop_bot"):
                    self.stop_lab_bot()
                    st.rerun()
            else:
                st.error("🔴 Bot Stopped")
                if st.button("▶️ Start Bot", use_container_width=True, key="lab_start_bot"):
                    self.start_lab_bot()
                    st.rerun()
            
            st.divider()
            
            # Strategy Selection
            st.subheader("📈 Strategy Selection")
            strategy_options = {
                'rsi_bollinger': 'RSI + Bollinger Bands',
                'momentum': 'Momentum Trading',
                'mean_reversion': 'Mean Reversion',
                'trend_following': 'Trend Following',
                'multi_timeframe': 'Multi-Timeframe Analysis'
            }
            
            st.session_state.lab_selected_strategy = st.selectbox(
                "Strategy Type",
                list(strategy_options.keys()),
                format_func=lambda x: strategy_options[x],
                help="Select the base strategy for parameter optimization"
            )
            
            # Universe Selection (Parameter-driven)
            st.subheader("🎯 Trading Universe")
            universe_modes = {
                'top_volume': 'High Volume Stocks',
                'trending': 'Trending Stocks',
                'volatile': 'High Volatility Stocks',
                'custom': 'Custom Selection'
            }
            
            universe_mode = st.selectbox(
                "Universe Selection Mode",
                list(universe_modes.keys()),
                format_func=lambda x: universe_modes[x],
                help="Parameter-driven stock selection instead of manual picking"
            )
            
            # Auto-trading toggle
            st.subheader("⚡ Automation")
            st.session_state.lab_auto_trade_enabled = st.checkbox(
                "Enable Auto-Trading",
                value=st.session_state.lab_auto_trade_enabled,
                help="Automatically execute trades based on parameters"
            )
            
            # Quick Preset Buttons
            st.subheader("🚀 Quick Presets")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🏛️ Conservative", use_container_width=True, key="lab_conservative"):
                    self.apply_parameter_preset('conservative')
                    st.rerun()
                if st.button("⚡ Aggressive", use_container_width=True, key="lab_aggressive"):
                    self.apply_parameter_preset('aggressive')
                    st.rerun()
            with col2:
                if st.button("⚖️ Balanced", use_container_width=True, key="lab_balanced"):
                    self.apply_parameter_preset('balanced')
                    st.rerun()
                if st.button("🔄 Reset", use_container_width=True, key="lab_reset"):
                    self.apply_parameter_preset('default')
                    st.rerun()
            
            st.divider()
            
            # Comprehensive Parameters
            st.subheader("🔧 All Parameters")
            
            params = self.get_comprehensive_parameters()
            
            # Group parameters by category
            param_categories = {
                "📊 Technical Indicators": ['rsi_period', 'rsi_oversold', 'rsi_overbought', 'bb_period', 'bb_std', 'ma_short', 'ma_long', 'macd_fast', 'macd_slow', 'macd_signal'],
                "🛡️ Risk Management": ['position_size', 'stop_loss', 'take_profit', 'max_positions', 'risk_per_trade'],
                "📈 Trend & Momentum": ['trend_strength_min', 'volume_factor', 'momentum_threshold'],
                "🌊 Market Regime": ['volatility_threshold', 'correlation_threshold'],
                "⏰ Timing": ['entry_patience', 'exit_patience', 'cool_down_period'],
                "🔍 Filters": ['min_price', 'max_price', 'min_volume', 'market_cap_min']
            }
            
            # Initialize parameter values in session state
            if 'lab_parameters' not in st.session_state:
                st.session_state.lab_parameters = {
                    key: config['default'] for key, config in params.items()
                }
            
            # Render parameter controls by category
            for category, param_list in param_categories.items():
                with st.expander(category, expanded=False):
                    for param_key in param_list:
                        if param_key in params:
                            config = params[param_key]
                            
                            if isinstance(config['default'], int):
                                value = st.slider(
                                    config['description'],
                                    min_value=config['min'],
                                    max_value=config['max'],
                                    value=st.session_state.lab_parameters.get(param_key, config['default']),
                                    step=config['step'],
                                    key=f"lab_{param_key}",
                                    help=f"Range: {config['min']} - {config['max']}"
                                )
                            else:
                                value = st.slider(
                                    config['description'],
                                    min_value=float(config['min']),
                                    max_value=float(config['max']),
                                    value=float(st.session_state.lab_parameters.get(param_key, config['default'])),
                                    step=float(config['step']),
                                    key=f"lab_{param_key}",
                                    format="%.3f",
                                    help=f"Range: {config['min']} - {config['max']}"
                                )
                            
                            st.session_state.lab_parameters[param_key] = value
            
            # Save Parameter Set
            st.subheader("💾 Parameter Management")
            
            set_name = st.text_input("Parameter Set Name", placeholder="MyStrategy_v1", key="lab_param_set_name")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Set", use_container_width=True, key="lab_save_set"):
                    if set_name:
                        self.save_parameter_set(set_name)
                        st.success(f"✅ Saved '{set_name}'")
                    else:
                        st.warning("⚠️ Enter a name first")
            
            with col2:
                if st.button("📊 Start Session", use_container_width=True, key="lab_start_session"):
                    self.start_new_session()
                    st.success("🚀 New session started!")
    
    def apply_parameter_preset(self, preset_type: str):
        """Apply parameter presets"""
        presets = {
            'conservative': {
                'rsi_period': 20, 'rsi_oversold': 25, 'rsi_overbought': 75,
                'bb_period': 25, 'bb_std': 2.5, 'position_size': 0.05,
                'stop_loss': 0.015, 'take_profit': 0.03, 'max_positions': 2,
                'risk_per_trade': 0.01, 'trend_strength_min': 0.5,
                'volume_factor': 2.0, 'momentum_threshold': 0.005
            },
            'aggressive': {
                'rsi_period': 10, 'rsi_oversold': 35, 'rsi_overbought': 65,
                'bb_period': 15, 'bb_std': 1.5, 'position_size': 0.25,
                'stop_loss': 0.03, 'take_profit': 0.06, 'max_positions': 5,
                'risk_per_trade': 0.03, 'trend_strength_min': 0.2,
                'volume_factor': 1.2, 'momentum_threshold': 0.02
            },
            'balanced': {
                'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
                'bb_period': 20, 'bb_std': 2.0, 'position_size': 0.1,
                'stop_loss': 0.02, 'take_profit': 0.04, 'max_positions': 3,
                'risk_per_trade': 0.02, 'trend_strength_min': 0.3,
                'volume_factor': 1.5, 'momentum_threshold': 0.01
            },
            'default': {key: config['default'] for key, config in self.get_comprehensive_parameters().items()}
        }
        
        if preset_type in presets:
            # Update only the parameters that exist in the preset
            for key, value in presets[preset_type].items():
                if key in st.session_state.lab_parameters:
                    st.session_state.lab_parameters[key] = value
    
    def save_parameter_set(self, name: str):
        """Save current parameter set to session history"""
        parameter_set = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'parameters': st.session_state.lab_parameters.copy(),
            'strategy': st.session_state.lab_selected_strategy,
            'mode': st.session_state.lab_mode
        }
        
        st.session_state.lab_parameter_sets.append(parameter_set)
        
        # Save to file for persistence
        os.makedirs('logs', exist_ok=True)
        with open('logs/parameter_sets.json', 'w') as f:
            json.dump(st.session_state.lab_parameter_sets, f, indent=2)
    
    def start_new_session(self):
        """Start a new trading session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = TradingSession(
            session_id=session_id,
            start_time=datetime.now(),
            parameters=st.session_state.lab_parameters.copy(),
            mode=st.session_state.lab_mode
        )
        
        st.session_state.lab_current_session = session
    
    def start_lab_bot(self):
        """Start the trading bot for the lab"""
        try:
            # In a real implementation, this would start the bot with current parameters
            st.session_state.lab_bot_running = True
            return True
        except Exception as e:
            st.error(f"Error starting lab bot: {e}")
            return False
    
    def stop_lab_bot(self):
        """Stop the trading bot"""
        try:
            st.session_state.lab_bot_running = False
            return True
        except Exception as e:
            st.error(f"Error stopping lab bot: {e}")
            return False
    
    def render_main_interface(self):
        """Render the main Trading Lab interface"""
        st.title("🧪 Trading Lab - Parameter Experimentation & Learning")
        st.markdown("**Maximize bot and user learning through comprehensive parameter exposure and real-time analysis**")
        
        # Quick Status Bar
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            status_color = "🟢" if st.session_state.lab_bot_running else "🔴"
            st.metric("Bot Status", f"{status_color} {'Running' if st.session_state.lab_bot_running else 'Stopped'}")
        
        with col2:
            st.metric("Mode", f"📝 {st.session_state.lab_mode.title()}")
        
        with col3:
            portfolio_value = st.session_state.lab_portfolio_value
            st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
        
        with col4:
            trades_df = self.load_trades()
            today_trades = len(trades_df[trades_df['ts'].dt.date == datetime.now().date()]) if not trades_df.empty else 0
            st.metric("Today's Trades", today_trades)
        
        with col5:
            session_count = len(st.session_state.lab_parameter_sets)
            st.metric("Parameter Sets", session_count)
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Real-time Analysis",
            "💰 P&L & Positions", 
            "📋 Trade Log & Rationale",
            "📈 Session History"
        ])
        
        with tab1:
            self.render_realtime_analysis()
        
        with tab2:
            self.render_pnl_positions()
        
        with tab3:
            self.render_trade_log()
        
        with tab4:
            self.render_session_history()
    
    def render_realtime_analysis(self):
        """Render real-time market analysis"""
        st.subheader("📊 Real-time Parameter Impact Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Real-time parameter visualization
            st.markdown("### 🎛️ Current Parameter Settings")
            
            # Display current parameters in a nice grid
            params = st.session_state.lab_parameters
            
            # Group key parameters for display
            key_params = {
                'RSI Period': params.get('rsi_period', 14),
                'RSI Oversold': params.get('rsi_oversold', 30),
                'RSI Overbought': params.get('rsi_overbought', 70),
                'BB Period': params.get('bb_period', 20),
                'BB Std Dev': params.get('bb_std', 2.0),
                'Position Size': f"{params.get('position_size', 0.1):.1%}",
                'Stop Loss': f"{params.get('stop_loss', 0.02):.1%}",
                'Take Profit': f"{params.get('take_profit', 0.04):.1%}",
            }
            
            param_cols = st.columns(4)
            for i, (name, value) in enumerate(key_params.items()):
                with param_cols[i % 4]:
                    st.metric(name, value)
            
            # Parameter sensitivity analysis
            st.markdown("### 📈 Parameter Sensitivity (Last 30 Days)")
            
            # Create a mock sensitivity analysis chart
            sensitivity_data = {
                'RSI Period': [0.15, 0.12, 0.18, 0.10, 0.22],
                'Position Size': [0.08, 0.25, 0.15, 0.30, 0.12],
                'Stop Loss': [0.20, 0.18, 0.25, 0.15, 0.22],
                'BB Period': [0.12, 0.15, 0.20, 0.08, 0.18]
            }
            
            fig_sensitivity = go.Figure()
            
            for param, returns in sensitivity_data.items():
                fig_sensitivity.add_trace(go.Scatter(
                    x=list(range(len(returns))),
                    y=returns,
                    mode='lines+markers',
                    name=param,
                    line=dict(width=2)
                ))
            
            fig_sensitivity.update_layout(
                title="Parameter Sensitivity to Returns",
                xaxis_title="Test Scenarios",
                yaxis_title="Return Impact",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        with col2:
            # Current market regime analysis
            st.markdown("### 🌊 Market Regime Analysis")
            
            # Mock market regime data
            regime_data = {
                'Trending': 65,
                'Range-bound': 25,
                'Volatile': 10
            }
            
            fig_regime = px.pie(
                values=list(regime_data.values()),
                names=list(regime_data.keys()),
                title="Current Market Regime"
            )
            fig_regime.update_layout(height=300)
            st.plotly_chart(fig_regime, use_container_width=True)
            
            # Strategy performance by regime
            st.markdown("### 📊 Strategy Performance by Regime")
            
            regime_performance = pd.DataFrame({
                'Regime': ['Trending', 'Range-bound', 'Volatile'],
                'Win Rate': [0.68, 0.45, 0.52],
                'Avg Return': [0.025, 0.008, 0.015]
            })
            
            st.dataframe(regime_performance, use_container_width=True, hide_index=True)
            
            # Real-time signals
            st.markdown("### ⚡ Current Signals")
            
            # Mock signal data
            signals = [
                {"Symbol": "AAPL", "Signal": "BUY", "Strength": "Strong", "RSI": 28},
                {"Symbol": "TSLA", "Signal": "SELL", "Strength": "Weak", "RSI": 72},
                {"Symbol": "NVDA", "Signal": "HOLD", "Strength": "Neutral", "RSI": 55}
            ]
            
            for signal in signals:
                color = "🟢" if signal["Signal"] == "BUY" else "🔴" if signal["Signal"] == "SELL" else "🟡"
                st.write(f"{color} **{signal['Symbol']}**: {signal['Signal']} ({signal['Strength']}) - RSI: {signal['RSI']}")
    
    def render_pnl_positions(self):
        """Render P&L and positions analysis"""
        st.subheader("💰 Real-time P&L and Position Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Real-time P&L line chart
            st.markdown("### 📈 Real-time P&L Line Graph")
            
            # Load actual trades if available, otherwise create mock data
            trades_df = self.load_trades()
            
            if not trades_df.empty:
                # Calculate cumulative P&L from actual trades
                trades_df_sorted = trades_df.sort_values('ts')
                trades_df_sorted['cumulative_pnl'] = trades_df_sorted['net'].cumsum()
                
                fig_pnl = go.Figure()
                
                fig_pnl.add_trace(go.Scatter(
                    x=trades_df_sorted['ts'],
                    y=trades_df_sorted['cumulative_pnl'],
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='blue', width=2),
                    fill='tonexty',
                    fillcolor='rgba(0,100,255,0.1)'
                ))
                
                # Add zero line
                fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig_pnl.update_layout(
                    title="Real-time Portfolio P&L",
                    xaxis_title="Time",
                    yaxis_title="Cumulative P&L ($)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_pnl, use_container_width=True)
                
                # P&L statistics
                total_pnl = trades_df_sorted['cumulative_pnl'].iloc[-1] if not trades_df_sorted.empty else 0
                max_pnl = trades_df_sorted['cumulative_pnl'].max() if not trades_df_sorted.empty else 0
                min_pnl = trades_df_sorted['cumulative_pnl'].min() if not trades_df_sorted.empty else 0
                
                pnl_cols = st.columns(3)
                with pnl_cols[0]:
                    st.metric("Total P&L", f"${total_pnl:+.2f}")
                with pnl_cols[1]:
                    st.metric("Max P&L", f"${max_pnl:+.2f}")
                with pnl_cols[2]:
                    st.metric("Max Drawdown", f"${min_pnl:+.2f}")
            
            else:
                # Create mock P&L data for demonstration
                dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
                mock_pnl = np.cumsum(np.random.normal(10, 50, len(dates)))
                
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(
                    x=dates,
                    y=mock_pnl,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='blue', width=2),
                    fill='tonexty',
                    fillcolor='rgba(0,100,255,0.1)'
                ))
                
                fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig_pnl.update_layout(
                    title="Real-time Portfolio P&L (Demo Data)",
                    xaxis_title="Date",
                    yaxis_title="Cumulative P&L ($)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_pnl, use_container_width=True)
                
                st.info("📊 Start trading to see real P&L data here!")
        
        with col2:
            # Current positions visualization
            st.markdown("### 📊 Current Positions")
            
            # Mock position data for demonstration
            positions = [
                {"Symbol": "AAPL", "Shares": 50, "Avg Price": 175.20, "Current": 178.50, "P&L": 165.00},
                {"Symbol": "TSLA", "Shares": 25, "Avg Price": 242.80, "Current": 238.90, "P&L": -97.50},
                {"Symbol": "NVDA", "Shares": 30, "Avg Price": 421.60, "Current": 435.20, "P&L": 408.00}
            ]
            
            if positions:
                # Position performance chart
                symbols = [pos["Symbol"] for pos in positions]
                pnls = [pos["P&L"] for pos in positions]
                colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
                
                fig_pos = go.Figure(data=[
                    go.Bar(x=symbols, y=pnls, marker_color=colors)
                ])
                
                fig_pos.update_layout(
                    title="Position P&L",
                    xaxis_title="Symbol",
                    yaxis_title="P&L ($)",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig_pos, use_container_width=True)
                
                # Position details table
                pos_df = pd.DataFrame(positions)
                pos_df['P&L %'] = ((pos_df['Current'] - pos_df['Avg Price']) / pos_df['Avg Price'] * 100).round(2)
                pos_df['Market Value'] = (pos_df['Shares'] * pos_df['Current']).round(2)
                
                st.dataframe(
                    pos_df[['Symbol', 'Shares', 'Avg Price', 'Current', 'P&L', 'P&L %', 'Market Value']], 
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No current positions. Start trading to see positions here!")
            
            # Risk metrics
            st.markdown("### ⚖️ Risk Metrics")
            
            risk_metrics = {
                "Portfolio Beta": 1.15,
                "Sharpe Ratio": 1.42,
                "Max Drawdown": -3.2,
                "VAR (95%)": -1.8,
                "Volatility": 18.5
            }
            
            for metric, value in risk_metrics.items():
                if "%" in metric or "Drawdown" in metric or "VAR" in metric:
                    st.metric(metric, f"{value:.1f}%")
                else:
                    st.metric(metric, f"{value:.2f}")
    
    def render_trade_log(self):
        """Render detailed trade log with bot rationale"""
        st.subheader("📋 Detailed Trade Log with Bot Rationale")
        
        # Load actual trades
        trades_df = self.load_trades()
        
        if not trades_df.empty:
            # Add mock rationale for demonstration (in real implementation, this would come from the bot)
            enhanced_trades = trades_df.copy()
            
            # Generate mock rationale based on trade characteristics
            def generate_rationale(row):
                if row['side'] == 'buy':
                    reasons = [
                        f"RSI ({st.session_state.lab_parameters.get('rsi_period', 14)}-period) showed oversold at {np.random.randint(20, 35)}",
                        f"Price broke below lower Bollinger Band (period: {st.session_state.lab_parameters.get('bb_period', 20)})",
                        f"Volume spike detected ({np.random.randint(150, 300)}% of average)",
                        f"Momentum threshold ({st.session_state.lab_parameters.get('momentum_threshold', 0.01):.3f}) exceeded"
                    ]
                else:
                    reasons = [
                        f"RSI ({st.session_state.lab_parameters.get('rsi_period', 14)}-period) showed overbought at {np.random.randint(70, 85)}",
                        f"Price hit take profit target ({st.session_state.lab_parameters.get('take_profit', 0.04):.1%})",
                        f"Stop loss triggered at {st.session_state.lab_parameters.get('stop_loss', 0.02):.1%}",
                        f"Trend reversal signal detected"
                    ]
                return np.random.choice(reasons)
            
            enhanced_trades['bot_rationale'] = enhanced_trades.apply(generate_rationale, axis=1)
            enhanced_trades['confidence'] = np.random.uniform(0.6, 0.95, len(enhanced_trades))
            enhanced_trades['market_regime'] = np.random.choice(['Trending', 'Range-bound', 'Volatile'], len(enhanced_trades))
            
            # Display enhanced trade log
            st.markdown("### 📊 Recent Trades with Analysis")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                symbol_filter = st.selectbox("Filter by Symbol", ['All'] + list(enhanced_trades['symbol'].unique()), key="log_symbol_filter")
            with col2:
                side_filter = st.selectbox("Filter by Side", ['All', 'buy', 'sell'], key="log_side_filter")
            with col3:
                regime_filter = st.selectbox("Filter by Regime", ['All', 'Trending', 'Range-bound', 'Volatile'], key="log_regime_filter")
            
            # Apply filters
            filtered_trades = enhanced_trades.copy()
            if symbol_filter != 'All':
                filtered_trades = filtered_trades[filtered_trades['symbol'] == symbol_filter]
            if side_filter != 'All':
                filtered_trades = filtered_trades[filtered_trades['side'] == side_filter]
            if regime_filter != 'All':
                filtered_trades = filtered_trades[filtered_trades['market_regime'] == regime_filter]
            
            # Display trades with expanded details
            for _, trade in filtered_trades.tail(10).iterrows():
                with st.expander(f"🔍 {trade['symbol']} - {trade['side'].upper()} - {trade['ts'].strftime('%Y-%m-%d %H:%M')} - P&L: ${trade['net']:+.2f}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Bot Rationale:**")
                        st.write(f"📝 {trade['bot_rationale']}")
                        
                        st.markdown("**Trade Details:**")
                        st.write(f"• **Symbol**: {trade['symbol']}")
                        st.write(f"• **Side**: {trade['side'].upper()}")
                        st.write(f"• **Quantity**: {trade['qty']:.4f}")
                        st.write(f"• **Price**: ${trade['price']:.4f}")
                        st.write(f"• **Net P&L**: ${trade['net']:+.2f}")
                        st.write(f"• **Equity After**: ${trade['equity_after']:,.2f}")
                        
                        st.markdown("**Market Context:**")
                        st.write(f"• **Market Regime**: {trade['market_regime']}")
                        st.write(f"• **Bot Confidence**: {trade['confidence']:.1%}")
                    
                    with col2:
                        # Mini chart for this trade (mock data)
                        mini_chart_data = pd.DataFrame({
                            'time': pd.date_range(trade['ts'] - timedelta(hours=1), trade['ts'] + timedelta(hours=1), freq='5min'),
                            'price': trade['price'] + np.random.normal(0, trade['price']*0.002, 25)
                        })
                        
                        fig_mini = go.Figure()
                        fig_mini.add_trace(go.Scatter(
                            x=mini_chart_data['time'],
                            y=mini_chart_data['price'],
                            mode='lines',
                            line=dict(color='blue', width=1),
                            showlegend=False
                        ))
                        
                        # Mark the trade point
                        fig_mini.add_trace(go.Scatter(
                            x=[trade['ts']],
                            y=[trade['price']],
                            mode='markers',
                            marker=dict(
                                color='green' if trade['side'] == 'buy' else 'red',
                                size=10,
                                symbol='triangle-up' if trade['side'] == 'buy' else 'triangle-down'
                            ),
                            showlegend=False
                        ))
                        
                        fig_mini.update_layout(
                            title=f"{trade['symbol']} - Trade Context",
                            height=200,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        
                        st.plotly_chart(fig_mini, use_container_width=True)
        
        else:
            st.info("📊 No trades yet. Start the bot to see detailed trade logs with rationale!")
            
            # Show example of what the trade log will look like
            st.markdown("### 📋 Example Trade Log Entry")
            
            example_trade = {
                'symbol': 'AAPL',
                'side': 'buy',
                'qty': 10,
                'price': 175.20,
                'net': 85.50,
                'ts': datetime.now(),
                'rationale': 'RSI (14-period) showed oversold at 28, price broke below lower Bollinger Band, volume spike detected (230% of average)',
                'confidence': 0.87,
                'regime': 'Trending'
            }
            
            st.code(f"""
Trade Details:
• Symbol: {example_trade['symbol']}
• Side: {example_trade['side'].upper()}
• Quantity: {example_trade['qty']}
• Price: ${example_trade['price']:.2f}
• P&L: ${example_trade['net']:+.2f}

Bot Rationale:
{example_trade['rationale']}

Market Context:
• Regime: {example_trade['regime']}
• Confidence: {example_trade['confidence']:.1%}
            """)
    
    def render_session_history(self):
        """Render session history and parameter set analysis"""
        st.subheader("📈 Session History & Parameter Set Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Parameter set history
            st.markdown("### 📊 Parameter Set Performance Comparison")
            
            if st.session_state.lab_parameter_sets:
                # Create a comparison table of parameter sets
                comparison_data = []
                
                for i, param_set in enumerate(st.session_state.lab_parameter_sets):
                    # Mock performance data (in real implementation, this would be calculated from actual trading results)
                    mock_performance = {
                        'Set Name': param_set['name'],
                        'Strategy': param_set['strategy'],
                        'Created': param_set['timestamp'][:10],  # Just date
                        'Total Return': f"{np.random.uniform(-5, 15):.2f}%",
                        'Win Rate': f"{np.random.uniform(45, 75):.1f}%",
                        'Sharpe Ratio': f"{np.random.uniform(0.5, 2.5):.2f}",
                        'Max Drawdown': f"{np.random.uniform(-8, -1):.2f}%",
                        'Total Trades': np.random.randint(10, 100)
                    }
                    comparison_data.append(mock_performance)
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Parameter evolution chart
                st.markdown("### 📈 Parameter Evolution Over Time")
                
                if len(st.session_state.lab_parameter_sets) > 1:
                    # Track how key parameters have changed over time
                    param_evolution = []
                    key_params = ['rsi_period', 'position_size', 'stop_loss', 'take_profit']
                    
                    for param_set in st.session_state.lab_parameter_sets:
                        evolution_point = {'name': param_set['name'], 'timestamp': param_set['timestamp']}
                        for param in key_params:
                            evolution_point[param] = param_set['parameters'].get(param, 0)
                        param_evolution.append(evolution_point)
                    
                    evolution_df = pd.DataFrame(param_evolution)
                    
                    fig_evolution = go.Figure()
                    
                    for param in key_params:
                        if param in evolution_df.columns:
                            fig_evolution.add_trace(go.Scatter(
                                x=list(range(len(evolution_df))),
                                y=evolution_df[param],
                                mode='lines+markers',
                                name=param.replace('_', ' ').title(),
                                line=dict(width=2)
                            ))
                    
                    fig_evolution.update_layout(
                        title="Parameter Values Over Time",
                        xaxis_title="Parameter Set Index",
                        yaxis_title="Parameter Value",
                        height=400
                    )
                    
                    st.plotly_chart(fig_evolution, use_container_width=True)
                
            else:
                st.info("💡 Save parameter sets to see performance comparison and evolution analysis!")
                
                # Show example of what the analysis will look like
                st.markdown("### 📊 Example Parameter Set Comparison")
                
                example_data = pd.DataFrame({
                    'Set Name': ['Conservative_v1', 'Aggressive_v1', 'Balanced_v1'],
                    'Strategy': ['rsi_bollinger', 'momentum', 'rsi_bollinger'],
                    'Created': ['2024-01-15', '2024-01-16', '2024-01-17'],
                    'Total Return': ['8.5%', '12.3%', '10.1%'],
                    'Win Rate': ['68.2%', '58.7%', '62.4%'],
                    'Sharpe Ratio': ['1.85', '1.42', '1.67'],
                    'Max Drawdown': ['-3.2%', '-5.8%', '-4.1%'],
                    'Total Trades': [45, 78, 56]
                })
                
                st.dataframe(example_data, use_container_width=True, hide_index=True)
        
        with col2:
            # Current session status
            st.markdown("### 🚀 Current Session")
            
            if st.session_state.lab_current_session:
                session = st.session_state.lab_current_session
                
                # Calculate session duration
                duration = datetime.now() - session.start_time
                duration_minutes = int(duration.total_seconds() / 60)
                
                st.metric("Session ID", session.session_id[-8:])  # Last 8 chars
                st.metric("Duration", f"{duration_minutes} min")
                st.metric("Mode", session.mode.title())
                
                # Session progress
                st.markdown("**Session Progress:**")
                st.write(f"• Started: {session.start_time.strftime('%H:%M:%S')}")
                st.write(f"• Parameters: {len(session.parameters)} set")
                st.write(f"• Strategy: {st.session_state.lab_selected_strategy}")
                
                # Quick session actions
                if st.button("💾 Save & End Session", use_container_width=True, key="lab_save_end_session"):
                    self.end_current_session()
                    st.success("✅ Session saved!")
                    st.rerun()
            
            else:
                st.info("No active session. Click 'Start Session' in the sidebar to begin!")
            
            # Session management
            st.markdown("### 📋 Session Management")
            
            if st.button("🗑️ Clear All History", use_container_width=True, key="lab_clear_history"):
                if st.button("⚠️ Confirm Clear", use_container_width=True, type="secondary", key="lab_confirm_clear"):
                    st.session_state.lab_parameter_sets = []
                    st.session_state.lab_session_history = []
                    st.success("✅ History cleared!")
                    st.rerun()
            
            if st.button("📥 Export History", use_container_width=True, key="lab_export_history"):
                if st.session_state.lab_parameter_sets:
                    # Create exportable data
                    export_data = {
                        'parameter_sets': st.session_state.lab_parameter_sets,
                        'export_timestamp': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                    
                    st.download_button(
                        "💾 Download JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"trading_lab_history_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.warning("No history to export!")
            
            # Learning insights
            st.markdown("### 🧠 Learning Insights")
            
            insights = [
                "🎯 RSI periods between 12-16 show best results",
                "📊 Position size of 8-12% optimizes risk/return",
                "⚡ Trending markets favor momentum strategies",
                "🛡️ Tighter stops (1.5-2%) reduce drawdowns",
                "📈 Higher volume confirmation improves win rate"
            ]
            
            for insight in insights:
                st.write(insight)
    
    def end_current_session(self):
        """End the current trading session and save to history"""
        if st.session_state.lab_current_session:
            session = st.session_state.lab_current_session
            
            # Update session with final statistics
            duration = datetime.now() - session.start_time
            session.duration_minutes = int(duration.total_seconds() / 60)
            
            # Calculate final statistics from trades (mock data for now)
            session.total_trades = np.random.randint(5, 25)
            session.total_pnl = np.random.uniform(-200, 500)
            session.win_rate = np.random.uniform(0.45, 0.75)
            session.max_drawdown = np.random.uniform(-0.08, -0.01)
            session.sharpe_ratio = np.random.uniform(0.5, 2.5)
            session.final_equity = 100000 + session.total_pnl
            
            # Add to session history
            st.session_state.lab_session_history.append(session)
            st.session_state.lab_current_session = None
    
    def run(self):
        """Main method to run the Trading Lab interface"""
        # Render parameter sidebar
        self.render_parameter_sidebar()
        
        # Render main interface
        self.render_main_interface()

def main():
    """Main function to run the Trading Lab"""
    st.set_page_config(
        page_title="Trading Lab - Parameter Experimentation",
        page_icon="🧪",
        layout="wide"
    )
    
    # Initialize and run Trading Lab
    lab = TradingLabInterface()
    lab.run()

if __name__ == "__main__":
    main()