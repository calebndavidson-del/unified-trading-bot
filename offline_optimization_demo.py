#!/usr/bin/env python3
"""
Offline QuantConnect-Style Parameter Optimization Demo
Works without internet connection using mock data for demonstration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from parameter_manager import create_default_parameters
from optimization_engine import OptimizationResult, OptimizationSummary
from results_analyzer import ResultsAnalyzer

def create_mock_market_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Create realistic mock market data for testing"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simulate realistic price movement
    np.random.seed(42)  # For reproducible results
    base_price = 150 + np.random.normal(0, 50)  # Random starting price
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    
    # Add some trend
    trend = np.linspace(-0.1, 0.1, len(dates)) * 0.001
    returns += trend
    
    # Calculate prices
    prices = base_price * np.cumprod(1 + returns)
    
    # Create OHLCV data
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
    volumes = np.random.randint(100000, 2000000, len(dates))
    
    df = pd.DataFrame({
        'Open': prices,
        'High': highs,
        'Low': lows, 
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    return df

def run_offline_optimization_demo():
    """Run offline optimization demo with mock data"""
    
    st.set_page_config(
        page_title="QuantConnect Optimization Demo",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 QuantConnect-Style Parameter Optimization Demo")
    st.markdown("**Professional-grade parameter optimization with mock data (works offline)**")
    
    # Demo introduction
    st.markdown("""
    <div style='background-color: #e8f4fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h3>🎯 Demo Features</h3>
    <p>This demo shows the full QuantConnect-style optimization workflow using realistic mock market data:</p>
    <ul>
    <li>📊 <strong>Smart Parameter Ranges</strong> - Automatically generated for different strategies</li>
    <li>🚀 <strong>Parallel Optimization</strong> - Tests hundreds of parameter combinations quickly</li>
    <li>📈 <strong>Comprehensive Analysis</strong> - Risk-adjusted metrics, robustness testing, heatmaps</li>
    <li>✅ <strong>One-Click Application</strong> - Apply best parameters instantly</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛️ Demo Configuration")
        
        # Symbol selection (mock symbols)
        mock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        selected_symbols = st.multiselect(
            "Select Symbols to Optimize",
            mock_symbols,
            default=['AAPL', 'MSFT'],
            help="Select symbols for optimization (using mock data)"
        )
        
        # Time period
        period_options = {
            "1 Month": 30,
            "2 Months": 60, 
            "3 Months": 90
        }
        selected_period = st.selectbox("Backtest Period", list(period_options.keys()), index=1)
        opt_period = period_options[selected_period]
        
        # Strategy type
        strategy_map = {
            "RSI + Bollinger Bands": "rsi_bollinger",
            "Momentum": "momentum", 
            "Mean Reversion": "mean_reversion"
        }
        strategy_type = st.selectbox("Strategy Type", list(strategy_map.keys()))
        
        # Optimization objective
        objective_options = {
            "Sharpe Ratio": "sharpe_ratio",
            "Total Return": "total_return",
            "Calmar Ratio": "calmar_ratio"
        }
        objective = st.selectbox("Optimization Objective", list(objective_options.keys()))
        
        # Max combinations
        max_combinations = st.slider("Max Combinations", 20, 200, 50)
        
        # Show parameter preview
        preview_params = create_default_parameters(strategy_map[strategy_type])
        param_info = preview_params.get_parameter_info()
        
        with st.expander("📋 Parameter Ranges Preview"):
            st.write(f"Total possible combinations: {param_info['total_combinations']:,}")
            for name, info in param_info['parameters'].items():
                st.write(f"• {name}: {info['min_value']} to {info['max_value']} (step {info['step']}) = {info['total_values']} values")
        
        # Main optimization button
        if st.button("🚀 Run Demo Optimization", use_container_width=True, type="primary"):
            if selected_symbols:
                with st.spinner("Running QuantConnect-style optimization with mock data..."):
                    
                    # Create mock optimization results
                    results = []
                    param_manager = create_default_parameters(strategy_map[strategy_type])
                    combinations = param_manager.get_parameter_combinations()[:max_combinations]
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, symbol in enumerate(selected_symbols):
                        # Create mock data for this symbol
                        df = create_mock_market_data(symbol, opt_period)
                        
                        # Test a subset of parameter combinations
                        symbol_results = []
                        for j, params in enumerate(combinations[:max_combinations//len(selected_symbols)]):
                            
                            # Update progress
                            overall_progress = (i * len(combinations) + j) / (len(selected_symbols) * len(combinations)) 
                            progress_bar.progress(overall_progress)
                            status_text.text(f"Testing {symbol} - combination {j+1}")
                            
                            # Simulate optimization result
                            result = create_mock_optimization_result(symbol, params, df)
                            if result:
                                symbol_results.append(result)
                        
                        results.extend(symbol_results)
                    
                    progress_bar.progress(1.0)
                    status_text.text("Optimization complete!")
                    
                    # Create optimization summary
                    summary = create_mock_optimization_summary(results)
                    
                    # Store results
                    st.session_state['demo_optimization'] = {
                        'summary': summary,
                        'analyzer': ResultsAnalyzer(),
                        'objective': objective_options[objective],
                        'strategy_type': strategy_type,
                        'symbols': selected_symbols,
                        'period': selected_period
                    }
                    
                    # Initialize analyzer
                    st.session_state['demo_optimization']['analyzer'].analyze_results(summary)
                    
                    st.success(f"🎉 Demo optimization completed! Tested {len(results)} combinations")
                    st.balloons()
                    st.rerun()
            else:
                st.warning("Please select at least one symbol")
    
    with col2:
        if 'demo_optimization' in st.session_state:
            opt_data = st.session_state['demo_optimization']
            summary = opt_data['summary']
            analyzer = opt_data['analyzer']
            objective_key = opt_data['objective']
            
            st.subheader("🏆 Demo Optimization Results")
            
            # Quick stats
            col2a, col2b, col2c, col2d = st.columns(4)
            with col2a:
                st.metric("Combinations", f"{summary.successful_runs:,}")
            with col2b:
                best_score = getattr(summary.best_result, objective_key) if summary.best_result else 0
                st.metric("Best Score", f"{best_score:.3f}")
            with col2c:
                st.metric("Time", f"{summary.total_time:.1f}s")
            with col2d:
                st.metric("Success Rate", "100%")
            
            # Best parameters display
            if summary.best_result:
                st.success("✨ Optimal Parameters Found")
                
                # Display parameters in a nice grid
                params = summary.best_result.parameters
                param_items = list(params.items())
                
                param_cols = st.columns(3)
                for i, (param, value) in enumerate(param_items):
                    col_idx = i % 3
                    with param_cols[col_idx]:
                        param_name = param.replace('_', ' ').title()
                        if isinstance(value, float):
                            if 0 < value < 1:
                                st.metric(f"🎛️ {param_name}", f"{value:.1%}")
                            else:
                                st.metric(f"🎛️ {param_name}", f"{value:.3f}")
                        else:
                            st.metric(f"🎛️ {param_name}", str(value))
                
                # Performance metrics
                st.markdown("### 📊 Performance Metrics")
                perf_cols = st.columns(4)
                with perf_cols[0]:
                    st.metric("💰 Return", f"{summary.best_result.total_return:.2%}")
                with perf_cols[1]:
                    st.metric("📈 Sharpe", f"{summary.best_result.sharpe_ratio:.3f}")
                with perf_cols[2]:
                    st.metric("📉 Max DD", f"{summary.best_result.max_drawdown:.2%}")
                with perf_cols[3]:
                    st.metric("🎯 Win Rate", f"{summary.best_result.win_rate:.1%}")
                
                # Performance distribution chart
                if len(summary.results) > 1:
                    fig_dist = analyzer.create_performance_distribution_chart(objective_key)
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                # Advanced analysis tabs
                st.markdown("### 📈 Advanced Analysis")
                analysis_tabs = st.tabs(["📊 Results Grid", "🔥 Heatmaps", "📈 Equity Curves"])
                
                with analysis_tabs[0]:
                    results_grid = analyzer.create_results_grid(top_n=15, sort_by=objective_key)
                    st.dataframe(results_grid, use_container_width=True)
                
                with analysis_tabs[1]:
                    if analyzer.results_df is not None and len(analyzer.results_df) > 5:
                        param_cols = [col.replace('param_', '') for col in analyzer.results_df.columns if col.startswith('param_')]
                        if len(param_cols) >= 2:
                            heatmap_fig = analyzer.create_parameter_heatmap(param_cols[0], param_cols[1], objective_key)
                            st.plotly_chart(heatmap_fig, use_container_width=True)
                
                with analysis_tabs[2]:
                    equity_fig = analyzer.create_equity_curves_comparison(top_n=3)
                    st.plotly_chart(equity_fig, use_container_width=True)
        
        else:
            st.info("👈 Configure your demo settings and click 'Run Demo Optimization' to see the system in action!")
            
            st.markdown("### 🌟 What This Demo Shows")
            st.markdown("""
            **Real QuantConnect-Style Features:**
            - ✅ Automatic parameter range generation
            - ✅ Grid search optimization
            - ✅ Parallel processing simulation
            - ✅ Comprehensive performance metrics
            - ✅ Parameter sensitivity analysis
            - ✅ Robustness testing
            - ✅ Interactive visualizations
            - ✅ One-click parameter application
            
            **In Production:**
            - 🌐 Uses real market data from Yahoo Finance
            - ⚡ Actual parallel processing with ThreadPoolExecutor
            - 💾 Results caching for performance
            - 📊 Live data quality scoring
            - 🔄 Real-time progress updates
            """)

def create_mock_optimization_result(symbol: str, params: dict, df: pd.DataFrame) -> OptimizationResult:
    """Create a realistic mock optimization result"""
    
    # Add some randomness based on parameters
    np.random.seed(hash(str(params)) % 2**32)
    
    # Simulate strategy performance based on parameters
    performance_factor = 1.0
    if 'rsi_period' in params:
        performance_factor += (params['rsi_period'] - 15) * 0.001
    if 'position_size' in params:
        performance_factor += params['position_size'] * 0.5
    
    # Generate realistic returns
    base_return = np.random.normal(0.001 * performance_factor, 0.015)
    total_return = base_return * len(df) + np.random.normal(0, 0.1)
    
    # Create equity curve
    daily_returns = np.random.normal(base_return, 0.015, len(df))
    equity_values = 100000 * np.cumprod(1 + daily_returns)
    equity_curve = pd.Series(equity_values, index=df.index)
    
    # Calculate metrics
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe_ratio = (np.mean(daily_returns) * 252) / volatility if volatility > 0 else 0
    max_drawdown = ((equity_curve / equity_curve.cummax()) - 1).min()
    
    # Generate mock trades
    trades = []
    for i in range(0, len(df), 5):
        if i < len(df) - 1:
            trades.append({
                'date': df.index[i],
                'action': 'BUY' if i % 10 == 0 else 'SELL',
                'price': df['Close'].iloc[i],
                'shares': 100,
                'profit': np.random.normal(20, 50) if i % 10 != 0 else None
            })
    
    win_trades = len([t for t in trades if t.get('profit', 0) > 0])
    total_trades = len([t for t in trades if 'profit' in t and t['profit'] is not None])
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    
    return OptimizationResult(
        symbol=symbol,
        parameters=params,
        total_return=total_return,
        annualized_return=total_return * (252 / len(df)),
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sharpe_ratio * 1.1,
        max_drawdown=max_drawdown,
        calmar_ratio=total_return / abs(max_drawdown) if max_drawdown != 0 else 0,
        total_trades=total_trades,
        win_rate=win_rate,
        avg_win=30.0,
        avg_loss=25.0,
        profit_factor=1.2,
        final_value=equity_values[-1],
        equity_curve=equity_curve,
        trades_list=trades,
        optimization_time=0.1,
        backtest_start=str(df.index[0]),
        backtest_end=str(df.index[-1]),
        data_quality_score=95.0
    )

def create_mock_optimization_summary(results: list) -> OptimizationSummary:
    """Create optimization summary from mock results"""
    
    if not results:
        return OptimizationSummary(0, 0, 0, None, None, {}, 1.0, [])
    
    # Sort by Sharpe ratio
    results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
    
    best_result = results[0]
    worst_result = results[-1]
    
    # Calculate average metrics
    avg_metrics = {
        'total_return': np.mean([r.total_return for r in results]),
        'sharpe_ratio': np.mean([r.sharpe_ratio for r in results]),
        'max_drawdown': np.mean([r.max_drawdown for r in results]),
        'win_rate': np.mean([r.win_rate for r in results])
    }
    
    return OptimizationSummary(
        total_combinations=len(results),
        successful_runs=len(results),
        failed_runs=0,
        best_result=best_result,
        worst_result=worst_result,
        average_metrics=avg_metrics,
        total_time=len(results) * 0.1,
        results=results
    )

if __name__ == "__main__":
    run_offline_optimization_demo()