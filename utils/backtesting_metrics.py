#!/usr/bin/env python3
"""
Backtesting Performance Metrics and Visualization
Comprehensive analysis tools for backtesting results
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta


class BacktestingMetrics:
    """Calculate and visualize backtesting performance metrics"""
    
    @staticmethod
    def calculate_advanced_metrics(portfolio_df: pd.DataFrame, 
                                 benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        returns = portfolio_df['returns'] if 'returns' in portfolio_df.columns else portfolio_df['portfolio_value'].pct_change().fillna(0)
        
        metrics = {}
        
        # Basic return metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        
        metrics['total_return'] = total_return
        metrics['annualized_return'] = annualized_return
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        metrics['volatility'] = volatility
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate / 252
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        metrics['sharpe_ratio'] = sharpe
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        metrics['sortino_ratio'] = sortino
        
        # Maximum drawdown
        portfolio_values = portfolio_df['portfolio_value']
        running_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        metrics['max_drawdown'] = max_drawdown
        
        # Calmar ratio (annualized return / max drawdown)
        calmar = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
        metrics['calmar_ratio'] = calmar
        
        # VaR and CVaR (95% confidence)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        metrics['var_95'] = var_95
        metrics['cvar_95'] = cvar_95
        
        # Win rate and profit metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        metrics['win_rate'] = win_rate
        metrics['avg_win'] = avg_win
        metrics['avg_loss'] = avg_loss
        metrics['profit_factor'] = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Consistency metrics
        metrics['daily_win_rate'] = win_rate
        
        # Rolling metrics
        rolling_sharpe = returns.rolling(window=252).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        metrics['avg_rolling_sharpe'] = rolling_sharpe.mean()
        
        # Beta and Alpha (if benchmark provided)
        if benchmark_returns is not None:
            # Align dates
            aligned_returns = returns.reindex(benchmark_returns.index, fill_value=0)
            aligned_benchmark = benchmark_returns.reindex(returns.index, fill_value=0)
            
            # Calculate beta
            covariance = np.cov(aligned_returns, aligned_benchmark)[0][1]
            benchmark_variance = np.var(aligned_benchmark)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Calculate alpha
            alpha = annualized_return - risk_free_rate - beta * (aligned_benchmark.mean() * 252 - risk_free_rate)
            
            metrics['beta'] = beta
            metrics['alpha'] = alpha
            
            # Information ratio
            active_returns = aligned_returns - aligned_benchmark
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
            metrics['information_ratio'] = information_ratio
        
        return metrics
    
    @staticmethod
    def create_equity_curve_chart(portfolio_df: pd.DataFrame, 
                                benchmark_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create interactive equity curve chart"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Portfolio Value', 'Daily Returns', 'Drawdown'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['portfolio_value'],
                name='Portfolio Value',
                line=dict(color='#00ff88', width=2),
                hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add benchmark if provided
        if benchmark_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_data['Close'],
                    name='Benchmark',
                    line=dict(color='#ff6b6b', width=2, dash='dash'),
                    hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Daily returns
        returns = portfolio_df['returns'] if 'returns' in portfolio_df.columns else portfolio_df['portfolio_value'].pct_change().fillna(0)
        colors = ['#00ff88' if r >= 0 else '#ff6b6b' for r in returns]
        
        fig.add_trace(
            go.Bar(
                x=portfolio_df.index,
                y=returns * 100,
                name='Daily Returns',
                marker_color=colors,
                opacity=0.7,
                hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Drawdown
        portfolio_values = portfolio_df['portfolio_value']
        running_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=drawdowns,
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(255, 107, 107, 0.3)',
                line=dict(color='#ff6b6b', width=2),
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Backtesting Results - Equity Curve Analysis',
            height=800,
            showlegend=True,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
    
    @staticmethod
    def create_metrics_summary_chart(metrics: Dict[str, float]) -> go.Figure:
        """Create metrics summary visualization"""
        # Prepare data for radar chart
        categories = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Win Rate', 'Profit Factor']
        values = [
            min(metrics.get('sharpe_ratio', 0) / 3, 1),  # Normalize Sharpe (good if >3)
            min(metrics.get('sortino_ratio', 0) / 3, 1),  # Normalize Sortino
            min(metrics.get('calmar_ratio', 0) / 5, 1),   # Normalize Calmar
            metrics.get('win_rate', 0),                    # Already 0-1
            min(metrics.get('profit_factor', 0) / 2, 1)    # Normalize profit factor
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(0, 255, 136, 0.3)',
            line=dict(color='#00ff88', width=3),
            name='Performance Metrics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=['Poor', 'Below Avg', 'Average', 'Good', 'Very Good', 'Excellent']
                )
            ),
            title="Performance Metrics Radar Chart",
            template='plotly_dark',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_monthly_returns_heatmap(portfolio_df: pd.DataFrame) -> go.Figure:
        """Create monthly returns heatmap"""
        returns = portfolio_df['returns'] if 'returns' in portfolio_df.columns else portfolio_df['portfolio_value'].pct_change().fillna(0)
        
        # Resample to monthly returns
        monthly_returns = (1 + returns).resample('M').prod() - 1
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_data = monthly_returns.reset_index()
        monthly_data['Year'] = monthly_data['index'].dt.year
        monthly_data['Month'] = monthly_data['index'].dt.strftime('%b')
        
        pivot_table = monthly_data.pivot(index='Year', columns='Month', values='returns')
        
        # Reorder months
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table = pivot_table.reindex(columns=month_order)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values * 100,  # Convert to percentage
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_table.values * 100, 2),
            texttemplate='%{text}%',
            textfont={"size": 10},
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Month',
            yaxis_title='Year',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_trade_analysis_chart(trades_df: pd.DataFrame) -> go.Figure:
        """Create trade analysis visualization"""
        if trades_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No trades to display",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                font=dict(size=20, color='white'),
                showarrow=False
            )
            fig.update_layout(
                template='plotly_dark',
                height=400,
                title="Trade Analysis"
            )
            return fig
        
        # Parse P&L column
        trades_df['PnL_numeric'] = trades_df['P&L'].str.replace('$', '').str.replace(',', '').astype(float)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('P&L Distribution', 'Trades Over Time', 'Win/Loss by Symbol', 'Return Distribution'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # P&L Distribution
        fig.add_trace(
            go.Histogram(
                x=trades_df['PnL_numeric'],
                name='P&L Distribution',
                marker_color='#00ff88',
                opacity=0.7,
                nbinsx=20
            ),
            row=1, col=1
        )
        
        # Trades over time
        trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date'])
        cumulative_pnl = trades_df['PnL_numeric'].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=trades_df['Entry Date'],
                y=cumulative_pnl,
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#00ff88', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # Win/Loss by symbol
        symbol_stats = trades_df.groupby('Symbol').agg({
            'PnL_numeric': ['sum', 'count']
        }).round(2)
        symbol_stats.columns = ['Total_PnL', 'Trade_Count']
        
        colors = ['#00ff88' if pnl >= 0 else '#ff6b6b' for pnl in symbol_stats['Total_PnL']]
        
        fig.add_trace(
            go.Bar(
                x=symbol_stats.index,
                y=symbol_stats['Total_PnL'],
                name='P&L by Symbol',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # Return distribution
        trades_df['Return_pct'] = trades_df['Return %'].str.replace('%', '').astype(float)
        
        fig.add_trace(
            go.Histogram(
                x=trades_df['Return_pct'],
                name='Return %',
                marker_color='#ff6b6b',
                opacity=0.7,
                nbinsx=20
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Trade Analysis Dashboard',
            height=600,
            template='plotly_dark',
            showlegend=False
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Total P&L ($)", row=2, col=1)
        fig.update_xaxes(title_text="Symbol", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        fig.update_xaxes(title_text="Return (%)", row=2, col=2)
        
        return fig
    
    @staticmethod
    def format_metrics_for_display(metrics: Dict[str, float]) -> Dict[str, str]:
        """Format metrics for display in Streamlit"""
        formatted = {}
        
        # Percentage metrics
        percentage_keys = [
            'total_return', 'annualized_return', 'volatility', 'max_drawdown',
            'win_rate', 'daily_win_rate', 'var_95', 'cvar_95'
        ]
        
        for key, value in metrics.items():
            if key in percentage_keys:
                formatted[key] = f"{value * 100:.2f}%"
            elif key in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'alpha', 'beta', 'information_ratio']:
                formatted[key] = f"{value:.3f}"
            elif key in ['avg_win', 'avg_loss']:
                formatted[key] = f"${value:.4f}"
            elif key == 'profit_factor':
                if value == float('inf'):
                    formatted[key] = "∞"
                else:
                    formatted[key] = f"{value:.2f}"
            else:
                formatted[key] = f"{value:.4f}"
        
        return formatted
    
    @staticmethod
    def generate_performance_report(metrics: Dict[str, float], 
                                  trades_df: pd.DataFrame,
                                  portfolio_df: pd.DataFrame) -> str:
        """Generate a comprehensive performance report"""
        report = []
        report.append("# BACKTESTING PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("## SUMMARY STATISTICS")
        report.append(f"Total Return: {metrics.get('total_return', 0) * 100:.2f}%")
        report.append(f"Annualized Return: {metrics.get('annualized_return', 0) * 100:.2f}%")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        report.append(f"Maximum Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%")
        report.append(f"Volatility: {metrics.get('volatility', 0) * 100:.2f}%")
        report.append("")
        
        # Trade statistics
        if not trades_df.empty:
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['P&L'].str.contains(r'^\$[0-9]', regex=True)])
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            report.append("## TRADE STATISTICS")
            report.append(f"Total Trades: {total_trades}")
            report.append(f"Winning Trades: {winning_trades}")
            report.append(f"Losing Trades: {total_trades - winning_trades}")
            report.append(f"Win Rate: {win_rate:.2f}%")
            report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            report.append("")
        
        # Risk metrics
        report.append("## RISK METRICS")
        report.append(f"Value at Risk (95%): {metrics.get('var_95', 0) * 100:.2f}%")
        report.append(f"Conditional VaR (95%): {metrics.get('cvar_95', 0) * 100:.2f}%")
        report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
        report.append("")
        
        # Portfolio evolution
        start_date = portfolio_df.index[0].strftime('%Y-%m-%d')
        end_date = portfolio_df.index[-1].strftime('%Y-%m-%d')
        trading_days = len(portfolio_df)
        
        report.append("## PORTFOLIO EVOLUTION")
        report.append(f"Start Date: {start_date}")
        report.append(f"End Date: {end_date}")
        report.append(f"Trading Days: {trading_days}")
        report.append(f"Initial Capital: ${portfolio_df['portfolio_value'].iloc[0]:,.2f}")
        report.append(f"Final Value: ${portfolio_df['portfolio_value'].iloc[-1]:,.2f}")
        report.append("")
        
        return "\n".join(report)