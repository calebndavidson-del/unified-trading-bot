#!/usr/bin/env python3
"""
Analysis Page for Automated Optimization Backtesting

This module provides the UI interface for running automated optimization
and viewing results in a leaderboard format.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from features.backtesting import AutomatedOptimizationBacktest, OptimizationConfig
from utils.asset_universe import AssetUniverseManager


class AnalysisPage:
    """Analysis page for optimization backtesting"""
    
    def __init__(self):
        self.asset_manager = AssetUniverseManager()
        
    def render(self):
        """Render the analysis page"""
        st.title("🔬 Automated Optimization Analysis")
        st.markdown("Automatically optimize trading strategies using Bayesian optimization")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content
        if 'optimization_running' not in st.session_state:
            st.session_state.optimization_running = False
            
        if 'optimization_results' not in st.session_state:
            st.session_state.optimization_results = None
        
        # Show different content based on state
        if st.session_state.optimization_running:
            self._render_optimization_progress()
        elif st.session_state.optimization_results:
            self._render_results()
        else:
            self._render_start_page()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🎯 Optimization Settings")
        
        # Symbol selection
        st.sidebar.subheader("📊 Portfolio Selection")
        
        # Option to use asset universe or manual selection
        use_universe = st.sidebar.checkbox("Use Asset Universe", value=True)
        
        if use_universe:
            # Get symbols from asset universe
            universe = self.asset_manager.get_universe()
            all_symbols = universe.get_all_symbols()
            
            if all_symbols:
                # Show distribution from universe
                stock_symbols = universe.get_symbols_by_type('stock')[:10]
                crypto_symbols = universe.get_symbols_by_type('crypto')[:5]
                etf_symbols = universe.get_symbols_by_type('etf')[:5]
                
                default_symbols = stock_symbols + crypto_symbols + etf_symbols
                
                selected_symbols = st.sidebar.multiselect(
                    "Select Symbols",
                    options=all_symbols,
                    default=default_symbols,
                    help="Choose symbols from the asset universe"
                )
            else:
                st.sidebar.warning("Asset universe is empty, using manual selection")
                selected_symbols = st.sidebar.multiselect(
                    "Select Symbols",
                    options=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BTC-USD', 'ETH-USD'],
                    default=['AAPL', 'MSFT', 'GOOGL']
                )
        else:
            # Manual symbol selection
            symbol_input = st.sidebar.text_area(
                "Enter Symbols (one per line)",
                value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA",
                help="Enter one symbol per line"
            )
            selected_symbols = [s.strip().upper() for s in symbol_input.split('\n') if s.strip()]
        
        # Optimization parameters
        st.sidebar.subheader("⚙️ Optimization Parameters")
        
        n_trials = st.sidebar.slider(
            "Number of Trials",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="More trials = better optimization but longer time"
        )
        
        objective_metric = st.sidebar.selectbox(
            "Optimization Objective",
            options=["sharpe_ratio", "total_return", "profit_factor"],
            index=0,
            help="Metric to optimize for"
        )
        
        timeout_minutes = st.sidebar.slider(
            "Timeout (minutes)",
            min_value=5,
            max_value=120,
            value=30,
            step=5,
            help="Maximum time for optimization"
        )
        
        # Store settings in session state
        st.session_state.optimization_config = {
            'symbols': selected_symbols,
            'n_trials': n_trials,
            'objective_metric': objective_metric,
            'timeout': timeout_minutes * 60,  # Convert to seconds
            'n_jobs': 1,
            'study_name': f"optimization_{datetime.now().strftime('%Y%m%d_%H%M')}"
        }
    
    def _render_start_page(self):
        """Render the initial start page"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🚀 Start Optimization")
            
            # Show current configuration
            config = st.session_state.optimization_config
            
            st.info(f"""
            **Current Configuration:**
            - **Symbols:** {', '.join(config['symbols'][:5])}{'...' if len(config['symbols']) > 5 else ''}
            - **Trials:** {config['n_trials']}
            - **Objective:** {config['objective_metric'].replace('_', ' ').title()}
            - **Timeout:** {config['timeout']//60} minutes
            """)
            
            # Validation
            if not config['symbols']:
                st.error("❌ Please select at least one symbol")
                return
            
            if len(config['symbols']) > 20:
                st.warning("⚠️ Using many symbols will significantly increase optimization time")
            
            # Start button
            if st.button("🚀 Start Automated Optimization", type="primary", use_container_width=True):
                self._start_optimization()
        
        with col2:
            st.subheader("📈 What This Does")
            st.markdown("""
            **Automated Optimization:**
            
            ✅ **Automatically selects:**
            - Model types (LSTM, Ensemble, Random Forest)
            - Trading strategies (Technical Analysis, Mean Reversion, Momentum, etc.)
            - All relevant parameters
            
            ✅ **Uses Bayesian Optimization** for efficient search
            
            ✅ **Tracks comprehensive metrics:**
            - Orders and trading signals
            - Sharpe ratio and P&L
            - Drawdown and win rate
            - Risk-adjusted returns
            
            ✅ **Shows best configurations** in leaderboard format
            """)
    
    def _start_optimization(self):
        """Start the optimization process"""
        config = st.session_state.optimization_config
        
        # Create optimization config
        opt_config = OptimizationConfig(
            n_trials=config['n_trials'],
            study_name=config['study_name'],
            direction="maximize",
            symbols=config['symbols'],
            timeout=config['timeout'],
            n_jobs=config['n_jobs'],
            objective_metric=config['objective_metric']
        )
        
        # Start optimization
        st.session_state.optimization_running = True
        st.session_state.optimization_config_used = config
        
        # Show progress and run optimization
        with st.spinner("🔍 Running automated optimization..."):
            optimizer = AutomatedOptimizationBacktest(opt_config)
            results = optimizer.optimize(symbols=config['symbols'], n_trials=config['n_trials'])
            
            st.session_state.optimization_results = results
            st.session_state.optimization_running = False
        
        # Trigger rerun to show results
        st.rerun()
    
    def _render_optimization_progress(self):
        """Render optimization progress (placeholder for now)"""
        st.header("🔄 Optimization in Progress")
        
        config = st.session_state.optimization_config_used
        
        st.info(f"""
        **Running optimization on {len(config['symbols'])} symbols with {config['n_trials']} trials...**
        
        This may take several minutes depending on your configuration.
        """)
    
    def _render_results(self):
        """Render optimization results"""
        results = st.session_state.optimization_results
        
        if "error" in results:
            st.error(f"❌ Optimization failed: {results['error']}")
            if st.button("🔄 Try Again"):
                st.session_state.optimization_results = None
                st.rerun()
            return
        
        st.header("🏆 Optimization Results")
        
        # Summary metrics
        self._render_summary_metrics(results)
        
        # Leaderboard
        self._render_leaderboard(results)
        
        # Actions
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Run New Optimization"):
                st.session_state.optimization_results = None
                st.rerun()
        
        with col2:
            if st.button("📊 Export Results"):
                self._export_results(results)
    
    def _render_summary_metrics(self, results: Dict[str, Any]):
        """Render summary metrics"""
        stats = results['optimization_stats']
        config = results['config']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Best Score",
                f"{stats['best_score']:.4f}",
                help=f"Best {config['objective_metric'].replace('_', ' ').title()}"
            )
        
        with col2:
            st.metric(
                "✅ Success Rate",
                f"{stats['successful_trials']}/{stats['total_trials']}",
                f"{stats['successful_trials']/stats['total_trials']*100:.1f}%"
            )
        
        with col3:
            st.metric(
                "⏱️ Total Time",
                f"{stats['optimization_time']:.1f}s",
                f"{stats['avg_trial_time']:.1f}s/trial"
            )
        
        with col4:
            st.metric(
                "📊 Avg Score",
                f"{stats['avg_score']:.4f}",
                f"±{stats['score_std']:.4f}"
            )
    
    def _render_leaderboard(self, results: Dict[str, Any]):
        """Render the results leaderboard"""
        st.subheader("🏆 Top Configurations")
        
        leaderboard = results['leaderboard']
        
        if not leaderboard:
            st.warning("No successful configurations found")
            return
        
        # Convert to DataFrame for display
        df = pd.DataFrame(leaderboard)
        
        # Format columns for display
        display_columns = {
            'rank': 'Rank',
            'score': 'Score',
            'model': 'Model',
            'strategy': 'Strategy',
            'total_return_pct': 'Return %',
            'sharpe_ratio': 'Sharpe',
            'max_drawdown_pct': 'Max DD %',
            'win_rate_pct': 'Win Rate %',
            'total_trades': 'Trades',
            'profit_factor': 'Profit Factor'
        }
        
        display_df = df[list(display_columns.keys())].copy()
        display_df.columns = list(display_columns.values())
        
        # Format numerical columns
        numerical_cols = ['Score', 'Return %', 'Sharpe', 'Max DD %', 'Win Rate %', 'Profit Factor']
        for col in numerical_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        st.dataframe(display_df, use_container_width=True, height=400)
    
    def _export_results(self, results: Dict[str, Any]):
        """Export optimization results"""
        try:
            # Create export data
            export_data = {
                'optimization_summary': results['optimization_stats'],
                'leaderboard': results['leaderboard'],
                'model_performance': results['model_performance'],
                'strategy_performance': results['strategy_performance'],
                'config': results['config']
            }
            
            # Convert to JSON
            json_str = json.dumps(export_data, indent=2, default=str)
            
            # Download button
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_str,
                file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            st.success("✅ Results prepared for download!")
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")


def render_analysis_page():
    """Main function to render the analysis page"""
    page = AnalysisPage()
    page.render()


# For standalone testing
if __name__ == "__main__":
    render_analysis_page()