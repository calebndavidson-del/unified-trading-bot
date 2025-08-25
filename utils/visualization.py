#!/usr/bin/env python3
"""
Candlestick and Trend Plotting Utilities for Dashboard
Advanced visualization components for trading analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# import matplotlib.pyplot as plt  # Not available
# import seaborn as sns  # Not available
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class CandlestickPlotter:
    """Advanced candlestick chart plotting with technical indicators"""
    
    def __init__(self, theme: str = 'plotly_dark'):
        self.theme = theme
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff6b6b',
            'volume': 'rgba(128, 128, 128, 0.5)',
            'ma_short': '#ffd93d',
            'ma_long': '#6bcf7f',
            'rsi': '#ff9f43',
            'macd': '#74b9ff',
            'bb_upper': 'rgba(255, 255, 255, 0.3)',
            'bb_lower': 'rgba(255, 255, 255, 0.3)',
            'support': '#ff7675',
            'resistance': '#74b9ff'
        }
    
    def create_basic_candlestick(self, df: pd.DataFrame, title: str = "Candlestick Chart") -> go.Figure:
        """Create basic candlestick chart"""
        fig = go.Figure(data=go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color=self.colors['bullish'],
            decreasing_line_color=self.colors['bearish'],
            name="Price"
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_candlestick_with_volume(self, df: pd.DataFrame, 
                                     title: str = "Price and Volume") -> go.Figure:
        """Create candlestick chart with volume subplot"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Volume bars
        if 'Volume' in df.columns:
            colors = [self.colors['bullish'] if close >= open else self.colors['bearish'] 
                     for close, open in zip(df['Close'], df['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    marker_color=colors,
                    name="Volume",
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_candlestick_with_indicators(self, df: pd.DataFrame,
                                         indicators: List[str] = ['sma', 'bb', 'rsi'],
                                         title: str = "Technical Analysis") -> go.Figure:
        """Create comprehensive candlestick chart with technical indicators"""
        
        # Determine number of subplots based on indicators
        subplot_count = 1  # Main price chart
        if 'rsi' in indicators or 'macd' in indicators or 'stoch' in indicators:
            subplot_count += 1
        if len([ind for ind in indicators if ind in ['rsi', 'macd', 'stoch']]) > 1:
            subplot_count += 1
        
        # Create subplots
        if subplot_count == 1:
            fig = go.Figure()
        else:
            row_heights = [0.6] + [0.4 / (subplot_count - 1)] * (subplot_count - 1)
            fig = make_subplots(
                rows=subplot_count, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=row_heights
            )
        
        # Main candlestick chart
        candlestick = go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color=self.colors['bullish'],
            decreasing_line_color=self.colors['bearish'],
            name="Price"
        )
        
        if subplot_count == 1:
            fig.add_trace(candlestick)
        else:
            fig.add_trace(candlestick, row=1, col=1)
        
        # Add moving averages
        if 'sma' in indicators:
            if 'sma_20' in df.columns:
                trace = go.Scatter(
                    x=df.index,
                    y=df['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color=self.colors['ma_short'], width=1)
                )
                if subplot_count == 1:
                    fig.add_trace(trace)
                else:
                    fig.add_trace(trace, row=1, col=1)
            
            if 'sma_50' in df.columns:
                trace = go.Scatter(
                    x=df.index,
                    y=df['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color=self.colors['ma_long'], width=1)
                )
                if subplot_count == 1:
                    fig.add_trace(trace)
                else:
                    fig.add_trace(trace, row=1, col=1)
        
        # Add Bollinger Bands
        if 'bb' in indicators and all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            # Upper band
            trace_upper = go.Scatter(
                x=df.index,
                y=df['bb_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color=self.colors['bb_upper'], width=1),
                fill=None
            )
            
            # Lower band
            trace_lower = go.Scatter(
                x=df.index,
                y=df['bb_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color=self.colors['bb_lower'], width=1),
                fill='tonexty',
                fillcolor='rgba(255, 255, 255, 0.1)'
            )
            
            # Middle band
            trace_middle = go.Scatter(
                x=df.index,
                y=df['bb_middle'],
                mode='lines',
                name='BB Middle',
                line=dict(color='rgba(255, 255, 255, 0.5)', width=1, dash='dot')
            )
            
            if subplot_count == 1:
                fig.add_trace(trace_upper)
                fig.add_trace(trace_lower)
                fig.add_trace(trace_middle)
            else:
                fig.add_trace(trace_upper, row=1, col=1)
                fig.add_trace(trace_lower, row=1, col=1)
                fig.add_trace(trace_middle, row=1, col=1)
        
        # Add support and resistance levels
        if 'support_resistance' in indicators:
            if 'resistance' in df.columns:
                fig.add_hline(
                    y=df['resistance'].iloc[-1],
                    line_dash="dash",
                    line_color=self.colors['resistance'],
                    annotation_text="Resistance"
                )
            
            if 'support' in df.columns:
                fig.add_hline(
                    y=df['support'].iloc[-1],
                    line_dash="dash",
                    line_color=self.colors['support'],
                    annotation_text="Support"
                )
        
        # Add RSI subplot
        current_row = 2
        if 'rsi' in indicators and 'rsi' in df.columns and subplot_count > 1:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color=self.colors['rsi'], width=2)
                ),
                row=current_row, col=1
            )
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
            
            fig.update_yaxes(title_text="RSI", row=current_row, col=1)
            current_row += 1
        
        # Add MACD subplot
        if 'macd' in indicators and all(col in df.columns for col in ['macd', 'macd_signal']) and subplot_count > 1:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(color=self.colors['macd'], width=2)
                ),
                row=current_row, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd_signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='orange', width=1)
                ),
                row=current_row, col=1
            )
            
            if 'macd_histogram' in df.columns:
                colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['macd_histogram'],
                        name='Histogram',
                        marker_color=colors,
                        opacity=0.6
                    ),
                    row=current_row, col=1
                )
            
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=200 * subplot_count + 200,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig


class TrendVisualization:
    """Trend analysis visualization tools"""
    
    def __init__(self, theme: str = 'plotly_dark'):
        self.theme = theme
        self.colors = {
            'uptrend': '#00ff88',
            'downtrend': '#ff6b6b',
            'sideways': '#ffd93d',
            'strong_uptrend': '#00cc66',
            'strong_downtrend': '#ff4444'
        }
    
    def plot_trend_strength(self, df: pd.DataFrame, title: str = "Trend Strength Analysis") -> go.Figure:
        """Plot trend strength over time"""
        fig = go.Figure()
        
        if 'trend_strength' in df.columns:
            # Color mapping based on trend strength
            colors = []
            for strength in df['trend_strength']:
                if strength > 0.6:
                    colors.append(self.colors['strong_uptrend'])
                elif strength > 0.2:
                    colors.append(self.colors['uptrend'])
                elif strength < -0.6:
                    colors.append(self.colors['strong_downtrend'])
                elif strength < -0.2:
                    colors.append(self.colors['downtrend'])
                else:
                    colors.append(self.colors['sideways'])
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['trend_strength'],
                    mode='lines+markers',
                    name='Trend Strength',
                    line=dict(width=2),
                    marker=dict(color=colors, size=4)
                )
            )
            
            # Add reference lines
            fig.add_hline(y=0.6, line_dash="dash", line_color="green", annotation_text="Strong Bullish")
            fig.add_hline(y=0.2, line_dash="dot", line_color="lightgreen", annotation_text="Bullish")
            fig.add_hline(y=0, line_dash="solid", line_color="gray", annotation_text="Neutral")
            fig.add_hline(y=-0.2, line_dash="dot", line_color="lightcoral", annotation_text="Bearish")
            fig.add_hline(y=-0.6, line_dash="dash", line_color="red", annotation_text="Strong Bearish")
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="Date",
            yaxis_title="Trend Strength",
            height=400
        )
        
        return fig
    
    def plot_market_regime(self, df: pd.DataFrame, title: str = "Market Regime Analysis") -> go.Figure:
        """Plot market regime classification"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Price with Regime', 'Hurst Exponent'),
            row_heights=[0.7, 0.3]
        )
        
        # Price chart with regime coloring
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='white', width=2)
            ),
            row=1, col=1
        )
        
        # Add regime background colors
        if 'market_regime' in df.columns:
            trending_mask = df['market_regime'] == 'trending'
            mean_reverting_mask = df['market_regime'] == 'mean_reverting'
            
            # Add shapes for trending periods
            for start, end in self._get_consecutive_periods(df.index[trending_mask]):
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor="green", opacity=0.1,
                    layer="below", line_width=0,
                    row=1, col=1
                )
            
            # Add shapes for mean-reverting periods
            for start, end in self._get_consecutive_periods(df.index[mean_reverting_mask]):
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor="red", opacity=0.1,
                    layer="below", line_width=0,
                    row=1, col=1
                )
        
        # Hurst exponent
        if 'hurst_exponent' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['hurst_exponent'],
                    mode='lines',
                    name='Hurst Exponent',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
            
            # Reference line at 0.5
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _get_consecutive_periods(self, dates):
        """Helper function to get consecutive date periods"""
        if len(dates) == 0:
            return []
        
        periods = []
        start = dates[0]
        prev = dates[0]
        
        for date in dates[1:]:
            if (date - prev).days > 1:  # Gap detected
                periods.append((start, prev))
                start = date
            prev = date
        
        periods.append((start, prev))
        return periods
    
    def plot_signal_analysis(self, df: pd.DataFrame, title: str = "Trading Signals") -> go.Figure:
        """Plot trading signals on price chart"""
        fig = go.Figure()
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='white', width=2)
            )
        )
        
        # Buy signals
        buy_signals = df[df.get('composite_signal', 0) > 0.3]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        color='green',
                        size=10,
                        symbol='triangle-up'
                    )
                )
            )
        
        # Sell signals
        sell_signals = df[df.get('composite_signal', 0) < -0.3]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='triangle-down'
                    )
                )
            )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="Date",
            yaxis_title="Price",
            height=500
        )
        
        return fig


class PerformanceVisualization:
    """Performance and risk visualization tools"""
    
    def __init__(self, theme: str = 'plotly_dark'):
        self.theme = theme
    
    def plot_cumulative_returns(self, returns: pd.Series, benchmark_returns: pd.Series = None,
                               title: str = "Cumulative Returns") -> go.Figure:
        """Plot cumulative returns comparison"""
        fig = go.Figure()
        
        # Portfolio cumulative returns
        cum_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cum_returns.index,
                y=cum_returns,
                mode='lines',
                name='Portfolio',
                line=dict(color='#00ff88', width=2)
            )
        )
        
        # Benchmark cumulative returns
        if benchmark_returns is not None:
            benchmark_cum = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cum.index,
                    y=benchmark_cum,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#ff6b6b', width=2)
                )
            )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=400
        )
        
        return fig
    
    def plot_drawdown(self, returns: pd.Series, title: str = "Drawdown Analysis") -> go.Figure:
        """Plot drawdown over time"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(255, 107, 107, 0.3)',
                line=dict(color='#ff6b6b', width=2)
            )
        )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis_title="Date",
            yaxis_title="Drawdown",
            height=400
        )
        
        return fig
    
    def plot_rolling_metrics(self, returns: pd.Series, window: int = 60,
                           title: str = "Rolling Risk Metrics") -> go.Figure:
        """Plot rolling risk metrics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rolling Sharpe', 'Rolling Volatility', 
                          'Rolling VaR', 'Rolling Beta'),
            vertical_spacing=0.15
        )
        
        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name='Sharpe'),
            row=1, col=1
        )
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol, name='Volatility'),
            row=1, col=2
        )
        
        # Rolling VaR
        rolling_var = returns.rolling(window).quantile(0.05)
        fig.add_trace(
            go.Scatter(x=rolling_var.index, y=rolling_var, name='VaR'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=600,
            showlegend=False
        )
        
        return fig


def create_dashboard_charts(data: pd.DataFrame, symbol: str) -> Dict[str, go.Figure]:
    """Create comprehensive dashboard charts"""
    plotter = CandlestickPlotter()
    trend_viz = TrendVisualization()
    perf_viz = PerformanceVisualization()
    
    charts = {}
    
    # Main candlestick chart with indicators
    charts['main_chart'] = plotter.create_candlestick_with_indicators(
        data, 
        indicators=['sma', 'bb', 'rsi', 'macd'],
        title=f"{symbol} - Technical Analysis"
    )
    
    # Volume chart
    charts['volume_chart'] = plotter.create_candlestick_with_volume(
        data,
        title=f"{symbol} - Price and Volume"
    )
    
    # Trend analysis
    if 'trend_strength' in data.columns:
        charts['trend_chart'] = trend_viz.plot_trend_strength(
            data,
            title=f"{symbol} - Trend Strength"
        )
    
    # Signal analysis
    if 'composite_signal' in data.columns:
        charts['signal_chart'] = trend_viz.plot_signal_analysis(
            data,
            title=f"{symbol} - Trading Signals"
        )
    
    # Performance analysis
    if 'Close' in data.columns:
        returns = data['Close'].pct_change().dropna()
        charts['returns_chart'] = perf_viz.plot_cumulative_returns(
            returns,
            title=f"{symbol} - Cumulative Returns"
        )
        
        charts['drawdown_chart'] = perf_viz.plot_drawdown(
            returns,
            title=f"{symbol} - Drawdown"
        )
    
    return charts


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    from features.market_trend import create_comprehensive_trend_features
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="6m")
    
    # Add technical indicators
    enhanced_data = create_comprehensive_trend_features(data)
    
    # Create visualizations
    plotter = CandlestickPlotter()
    
    # Basic candlestick chart
    basic_chart = plotter.create_basic_candlestick(enhanced_data, "AAPL - Basic Chart")
    
    # Advanced chart with indicators
    advanced_chart = plotter.create_candlestick_with_indicators(
        enhanced_data,
        indicators=['sma', 'bb', 'rsi', 'macd'],
        title="AAPL - Technical Analysis"
    )
    
    print("Visualization charts created successfully!")
    print(f"Data shape: {enhanced_data.shape}")
    print(f"Available indicators: {[col for col in enhanced_data.columns if any(x in col.lower() for x in ['sma', 'rsi', 'macd', 'bb'])]}")