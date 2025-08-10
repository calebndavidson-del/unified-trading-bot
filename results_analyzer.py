#!/usr/bin/env python3
"""
Advanced Results Analyzer for Trading Bot Optimization
Provides comprehensive analysis and visualization of optimization results
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import streamlit as st
from optimization_engine import OptimizationResult, OptimizationSummary


@dataclass
class ParameterSensitivity:
    """Parameter sensitivity analysis result"""
    parameter_name: str
    correlation_with_objective: float
    parameter_importance: float
    optimal_range: Tuple[float, float]
    sensitivity_score: float


@dataclass
class RobustnessMetrics:
    """Robustness analysis metrics"""
    parameter_stability: Dict[str, float]
    performance_consistency: float
    overfitting_risk: float
    robustness_score: float
    recommended_parameters: Dict[str, Any]


class ResultsAnalyzer:
    """Advanced analyzer for optimization results with comprehensive visualization"""
    
    def __init__(self):
        self.summary: Optional[OptimizationSummary] = None
        self.results_df: Optional[pd.DataFrame] = None
    
    def analyze_results(self, summary: OptimizationSummary) -> None:
        """Analyze optimization results and prepare for visualization"""
        self.summary = summary
        
        # Convert results to DataFrame for easier analysis
        self.results_df = self._create_results_dataframe(summary.results)
    
    def _create_results_dataframe(self, results: List[OptimizationResult]) -> pd.DataFrame:
        """Convert optimization results to pandas DataFrame"""
        data = []
        
        for result in results:
            row = {
                'symbol': result.symbol,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'calmar_ratio': result.calmar_ratio,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'final_value': result.final_value,
                'data_quality_score': result.data_quality_score
            }
            
            # Add parameter values
            for param_name, param_value in result.parameters.items():
                row[f'param_{param_name}'] = param_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def create_results_grid(self, top_n: int = 20, sort_by: str = 'sharpe_ratio') -> pd.DataFrame:
        """Create formatted results grid for display"""
        if self.results_df is None or self.results_df.empty:
            return pd.DataFrame()
        
        # Get top N results
        top_results = self.results_df.nlargest(top_n, sort_by).copy()
        
        # Format for display
        display_df = pd.DataFrame()
        display_df['Rank'] = range(1, len(top_results) + 1)
        display_df['Symbol'] = top_results['symbol']
        
        # Add parameter columns
        param_cols = [col for col in top_results.columns if col.startswith('param_')]
        for col in param_cols:
            param_name = col.replace('param_', '').replace('_', ' ').title()
            if top_results[col].dtype in ['int64', 'float64']:
                if top_results[col].apply(lambda x: x == int(x) if pd.notna(x) else True).all():
                    display_df[param_name] = top_results[col].astype(int)
                else:
                    display_df[param_name] = top_results[col].round(3)
            else:
                display_df[param_name] = top_results[col]
        
        # Add performance metrics
        display_df['Return'] = top_results['total_return'].apply(lambda x: f"{x:.2%}")
        display_df['Sharpe'] = top_results['sharpe_ratio'].round(3)
        display_df['Max DD'] = top_results['max_drawdown'].apply(lambda x: f"{x:.2%}")
        display_df['Win Rate'] = top_results['win_rate'].apply(lambda x: f"{x:.1%}")
        display_df['Trades'] = top_results['total_trades'].astype(int)
        
        return display_df
    
    def create_parameter_heatmap(self, param_x: str, param_y: str, 
                                 objective: str = 'sharpe_ratio') -> go.Figure:
        """Create parameter performance heatmap"""
        if self.results_df is None:
            return go.Figure()
        
        param_x_col = f'param_{param_x}'
        param_y_col = f'param_{param_y}'
        
        if param_x_col not in self.results_df.columns or param_y_col not in self.results_df.columns:
            return go.Figure()
        
        # Create pivot table
        pivot_df = self.results_df.pivot_table(
            values=objective,
            index=param_y_col,
            columns=param_x_col,
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = px.imshow(
            pivot_df,
            labels=dict(x=param_x.replace('_', ' ').title(), 
                       y=param_y.replace('_', ' ').title(), 
                       color=objective.replace('_', ' ').title()),
            title=f"{objective.replace('_', ' ').title()} Heatmap: {param_x.title()} vs {param_y.title()}",
            color_continuous_scale="RdYlGn"
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_parameter_sensitivity_analysis(self, objective: str = 'sharpe_ratio') -> List[ParameterSensitivity]:
        """Analyze parameter sensitivity to objective function"""
        if self.results_df is None:
            return []
        
        param_cols = [col for col in self.results_df.columns if col.startswith('param_')]
        sensitivities = []
        
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            
            # Calculate correlation with objective
            correlation = self.results_df[param_col].corr(self.results_df[objective])
            
            # Calculate parameter importance (range of objective values)
            param_groups = self.results_df.groupby(param_col)[objective].agg(['mean', 'std', 'min', 'max'])
            importance = (param_groups['max'] - param_groups['min']).mean()
            
            # Find optimal range (top 25% of results)
            top_quartile = self.results_df.nlargest(len(self.results_df) // 4, objective)
            optimal_min = top_quartile[param_col].min()
            optimal_max = top_quartile[param_col].max()
            
            # Calculate sensitivity score
            sensitivity_score = abs(correlation) * importance
            
            sensitivities.append(ParameterSensitivity(
                parameter_name=param_name,
                correlation_with_objective=correlation,
                parameter_importance=importance,
                optimal_range=(optimal_min, optimal_max),
                sensitivity_score=sensitivity_score
            ))
        
        # Sort by sensitivity score
        sensitivities.sort(key=lambda x: x.sensitivity_score, reverse=True)
        return sensitivities
    
    def create_performance_distribution_chart(self, metric: str = 'sharpe_ratio') -> go.Figure:
        """Create performance distribution histogram"""
        if self.results_df is None:
            return go.Figure()
        
        fig = px.histogram(
            self.results_df,
            x=metric,
            nbins=30,
            title=f"Distribution of {metric.replace('_', ' ').title()}",
            labels={metric: metric.replace('_', ' ').title()}
        )
        
        # Add mean line
        mean_value = self.results_df[metric].mean()
        fig.add_vline(x=mean_value, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_value:.3f}")
        
        # Add best result line
        best_value = self.results_df[metric].max()
        fig.add_vline(x=best_value, line_dash="dash", line_color="green",
                      annotation_text=f"Best: {best_value:.3f}")
        
        fig.update_layout(height=400)
        return fig
    
    def create_parameter_correlation_matrix(self) -> go.Figure:
        """Create correlation matrix of all parameters"""
        if self.results_df is None:
            return go.Figure()
        
        # Get parameter columns
        param_cols = [col for col in self.results_df.columns if col.startswith('param_')]
        
        if len(param_cols) < 2:
            return go.Figure()
        
        # Calculate correlation matrix
        corr_matrix = self.results_df[param_cols].corr()
        
        # Clean up parameter names for display
        clean_names = [col.replace('param_', '').replace('_', ' ').title() for col in param_cols]
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            title="Parameter Correlation Matrix",
            color_continuous_scale="RdBu",
            text_auto=True,
            x=clean_names,
            y=clean_names
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_optimization_progress_chart(self) -> go.Figure:
        """Create optimization progress chart showing best result over time"""
        if self.results_df is None:
            return go.Figure()
        
        # Sort by assumed order (could be enhanced with timestamp)
        sorted_df = self.results_df.copy()
        sorted_df['cumulative_best'] = sorted_df['sharpe_ratio'].cummax()
        sorted_df['iteration'] = range(len(sorted_df))
        
        fig = go.Figure()
        
        # Add all results
        fig.add_trace(go.Scatter(
            x=sorted_df['iteration'],
            y=sorted_df['sharpe_ratio'],
            mode='markers',
            name='All Results',
            opacity=0.6,
            marker=dict(size=5, color='lightblue')
        ))
        
        # Add cumulative best
        fig.add_trace(go.Scatter(
            x=sorted_df['iteration'],
            y=sorted_df['cumulative_best'],
            mode='lines+markers',
            name='Best So Far',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title="Optimization Progress",
            xaxis_title="Iteration",
            yaxis_title="Sharpe Ratio",
            height=400
        )
        
        return fig
    
    def create_equity_curves_comparison(self, top_n: int = 5) -> go.Figure:
        """Create equity curves comparison for top N results"""
        if self.summary is None or not self.summary.results:
            return go.Figure()
        
        # Get top N results
        top_results = sorted(self.summary.results, key=lambda x: x.sharpe_ratio, reverse=True)[:top_n]
        
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, result in enumerate(top_results):
            fig.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                mode='lines',
                name=f"Rank {i+1} (Sharpe: {result.sharpe_ratio:.3f})",
                line=dict(width=2, color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title=f"Top {top_n} Equity Curves Comparison",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=500
        )
        
        return fig
    
    def analyze_robustness(self, objective: str = 'sharpe_ratio') -> RobustnessMetrics:
        """Analyze parameter robustness and stability"""
        if self.results_df is None:
            return RobustnessMetrics({}, 0, 0, 0, {})
        
        param_cols = [col for col in self.results_df.columns if col.startswith('param_')]
        
        # Parameter stability analysis
        parameter_stability = {}
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            
            # Get top 25% results
            top_quartile = self.results_df.nlargest(len(self.results_df) // 4, objective)
            
            # Calculate coefficient of variation for this parameter in top results
            param_values = top_quartile[param_col]
            cv = param_values.std() / param_values.mean() if param_values.mean() != 0 else 1
            stability = 1 / (1 + cv)  # Higher stability = lower variation
            
            parameter_stability[param_name] = stability
        
        # Performance consistency
        performance_std = self.results_df[objective].std()
        performance_mean = abs(self.results_df[objective].mean())
        performance_consistency = 1 / (1 + performance_std / performance_mean) if performance_mean != 0 else 0
        
        # Overfitting risk (high performance variance indicates potential overfitting)
        top_10_percent = self.results_df.nlargest(len(self.results_df) // 10, objective)
        overfitting_risk = top_10_percent[objective].std() / top_10_percent[objective].mean() if len(top_10_percent) > 0 else 1
        
        # Overall robustness score
        avg_param_stability = np.mean(list(parameter_stability.values()))
        robustness_score = (avg_param_stability + performance_consistency + (1 - min(overfitting_risk, 1))) / 3
        
        # Recommended parameters (mode of top quartile for each parameter)
        top_quartile = self.results_df.nlargest(len(self.results_df) // 4, objective)
        recommended_parameters = {}
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            recommended_parameters[param_name] = top_quartile[param_col].median()
        
        return RobustnessMetrics(
            parameter_stability=parameter_stability,
            performance_consistency=performance_consistency,
            overfitting_risk=overfitting_risk,
            robustness_score=robustness_score,
            recommended_parameters=recommended_parameters
        )
    
    def create_robustness_dashboard(self, robustness: RobustnessMetrics) -> go.Figure:
        """Create robustness analysis dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Parameter Stability', 'Robustness Metrics', 
                           'Risk Assessment', 'Recommended vs Current'],
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        # Parameter stability bar chart
        if robustness.parameter_stability:
            params = list(robustness.parameter_stability.keys())
            stability_scores = list(robustness.parameter_stability.values())
            
            fig.add_trace(go.Bar(
                x=params,
                y=stability_scores,
                name="Stability",
                marker_color='lightblue'
            ), row=1, col=1)
        
        # Robustness score gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=robustness.robustness_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Robustness Score"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 80], 'color': "yellow"},
                       {'range': [80, 100], 'color': "green"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}
        ), row=1, col=2)
        
        # Overfitting risk gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=robustness.overfitting_risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overfitting Risk"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkred"},
                   'steps': [
                       {'range': [0, 30], 'color': "green"},
                       {'range': [30, 70], 'color': "yellow"},
                       {'range': [70, 100], 'color': "red"}],
                   'threshold': {'line': {'color': "black", 'width': 4},
                                'thickness': 0.75, 'value': 50}}
        ), row=2, col=1)
        
        fig.update_layout(height=600, title_text="Robustness Analysis Dashboard")
        return fig
    
    def export_results_to_csv(self, filename: str = None) -> str:
        """Export results to CSV file"""
        if self.results_df is None:
            return ""
        
        if filename is None:
            filename = f"optimization_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        self.results_df.to_csv(filename, index=False)
        return filename
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        if self.results_df is None or self.summary is None:
            return {}
        
        return {
            'total_combinations_tested': len(self.results_df),
            'success_rate': len(self.results_df) / self.summary.total_combinations * 100,
            'best_sharpe_ratio': self.results_df['sharpe_ratio'].max(),
            'worst_sharpe_ratio': self.results_df['sharpe_ratio'].min(),
            'median_sharpe_ratio': self.results_df['sharpe_ratio'].median(),
            'best_return': self.results_df['total_return'].max(),
            'worst_return': self.results_df['total_return'].min(),
            'median_return': self.results_df['total_return'].median(),
            'average_trades_per_test': self.results_df['total_trades'].mean(),
            'average_win_rate': self.results_df['win_rate'].mean(),
            'optimization_time_seconds': self.summary.total_time
        }
    
    def generate_optimization_report(self) -> str:
        """Generate a comprehensive text report of optimization results"""
        if self.results_df is None or self.summary is None:
            return "No optimization results available."
        
        stats = self.get_summary_statistics()
        best_result = self.summary.best_result
        robustness = self.analyze_robustness()
        
        report = f"""
📊 OPTIMIZATION RESULTS REPORT
{'='*50}

📈 SUMMARY STATISTICS
• Total combinations tested: {stats['total_combinations_tested']:,}
• Success rate: {stats['success_rate']:.1f}%
• Optimization time: {stats['optimization_time_seconds']:.1f} seconds

🏆 PERFORMANCE METRICS
• Best Sharpe Ratio: {stats['best_sharpe_ratio']:.3f}
• Median Sharpe Ratio: {stats['median_sharpe_ratio']:.3f}
• Best Return: {stats['best_return']:.2%}
• Median Return: {stats['median_return']:.2%}

🎯 BEST PARAMETER COMBINATION
Symbol: {best_result.symbol}
Sharpe Ratio: {best_result.sharpe_ratio:.3f}
Total Return: {best_result.total_return:.2%}
Max Drawdown: {best_result.max_drawdown:.2%}
Win Rate: {best_result.win_rate:.1%}

Parameters:
"""
        
        for param, value in best_result.parameters.items():
            if isinstance(value, float):
                report += f"  • {param}: {value:.4f}\n"
            else:
                report += f"  • {param}: {value}\n"
        
        report += f"""
🔍 ROBUSTNESS ANALYSIS
• Robustness Score: {robustness.robustness_score:.2%}
• Performance Consistency: {robustness.performance_consistency:.2%}
• Overfitting Risk: {robustness.overfitting_risk:.2%}

💡 RECOMMENDATIONS
"""
        
        if robustness.robustness_score > 0.8:
            report += "✅ High robustness - parameters are stable and reliable\n"
        elif robustness.robustness_score > 0.6:
            report += "⚠️  Moderate robustness - consider additional validation\n"
        else:
            report += "❌ Low robustness - high risk of overfitting\n"
        
        if robustness.overfitting_risk < 0.3:
            report += "✅ Low overfitting risk - results are likely generalizable\n"
        else:
            report += "⚠️  High overfitting risk - use walk-forward analysis\n"
        
        return report
    
    def analyze_parameter_sensitivity(self, objective: str = 'sharpe_ratio') -> Dict[str, float]:
        """Analyze parameter sensitivity - how much each parameter impacts performance"""
        if self.results_df is None or len(self.results_df) < 2:
            return {}
        
        param_cols = [col for col in self.results_df.columns if col.startswith('param_')]
        sensitivity_scores = {}
        
        for param_col in param_cols:
            param_name = param_col.replace('param_', '')
            
            # Calculate correlation between parameter values and objective
            correlation = abs(self.results_df[param_col].corr(self.results_df[objective]))
            
            # If correlation is NaN, use a different approach
            if pd.isna(correlation):
                # Use range-based sensitivity (larger impact if parameter range affects performance)
                param_values = self.results_df[param_col].unique()
                if len(param_values) > 1:
                    performance_by_param = {}
                    for val in param_values:
                        subset = self.results_df[self.results_df[param_col] == val]
                        performance_by_param[val] = subset[objective].mean()
                    
                    # Sensitivity is the range of performance across parameter values
                    performance_range = max(performance_by_param.values()) - min(performance_by_param.values())
                    overall_range = self.results_df[objective].max() - self.results_df[objective].min()
                    sensitivity = performance_range / overall_range if overall_range != 0 else 0
                else:
                    sensitivity = 0
            else:
                sensitivity = correlation
            
            sensitivity_scores[param_name] = min(sensitivity, 1.0)  # Cap at 1.0
        
        return sensitivity_scores


# Example usage
if __name__ == "__main__":
    # This would typically be called with real optimization results
    print("ResultsAnalyzer module loaded successfully")
    print("Use with OptimizationSummary from optimization_engine.py")