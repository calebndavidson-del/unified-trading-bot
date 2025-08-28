# Automated Optimization Backtesting Framework

This document describes the new automated optimization backtesting framework that was added to the unified trading bot.

## Overview

The automated optimization framework automatically selects and optimizes:
- **Model types** (ML, ensemble, etc.)
- **Trading strategies** (Buy, Hold, Sell logic using all available data)
- **All relevant parameters** within pre-set fixed ranges

It uses **Bayesian Optimization** to efficiently search for the best configuration and provides a comprehensive leaderboard of results.

## Key Features

### 🤖 Fully Automated
- No manual parameter tuning required
- Automatically tries different model and strategy combinations
- Uses fixed, reasonable parameter ranges for all assets

### 🧠 Bayesian Optimization
- Efficient search using Optuna's TPE sampler
- Learns from previous trials to suggest better parameters
- Includes pruning for early stopping of poor configurations

### 📊 Comprehensive Tracking
- **Orders**: Buy, Hold, Sell signals with timestamps
- **Performance Metrics**: Sharpe ratio, P&L, drawdown, win rate
- **Risk Metrics**: Volatility, profit factor, maximum drawdown
- **Configuration Details**: Full model and strategy parameters

### 🏆 Leaderboard Results
- Displays only the best configurations after optimization
- Sortable by different metrics (Sharpe ratio, total return, profit factor)
- Detailed configuration for each top performer

## Architecture

### Core Components

1. **AutomatedOptimizationBacktest** (`features/backtesting.py`)
   - Main optimization engine
   - Manages Optuna study and trial execution
   - Calculates objective scores with penalties

2. **AutoModelSelector** (`features/models.py`)
   - Defines available models and strategies
   - Suggests parameter configurations for trials
   - Manages parameter ranges and constraints

3. **AnalysisPage** (`analysis/analysis_page.py`)
   - Streamlit UI for optimization interface
   - Configuration management and result display
   - Leaderboard and detailed analysis views

### Integration Points

- **Dashboard Integration**: New "🔬 Optimization" tab in main dashboard
- **Asset Universe**: Integrates with existing asset management system
- **Backtesting Engine**: Uses existing BacktestEngine for evaluation
- **Model System**: Leverages existing ensemble and strategy classes

## Usage

### Starting an Optimization

1. **Navigate** to the "🔬 Optimization" tab in the dashboard
2. **Configure** portfolio symbols (uses Asset Universe by default)
3. **Set parameters**:
   - Number of trials (10-500)
   - Optimization objective (Sharpe ratio, total return, profit factor)
   - Timeout in minutes (5-120)
4. **Click** "🚀 Start Automated Optimization"

### Viewing Results

After optimization completes, you'll see:

1. **Summary Metrics**:
   - Best score achieved
   - Success rate of trials
   - Total optimization time
   - Average performance across trials

2. **Leaderboard Table**:
   - Top 10 configurations ranked by objective
   - Key performance metrics for each
   - Model and strategy combinations

3. **Configuration Details**:
   - Expandable sections for top 3 performers
   - Full parameter configurations
   - Complete backtest results

### Exporting Results

- **JSON Export**: Download complete results with all configurations
- **Configuration Reuse**: Save best configuration for later use

## Model and Strategy Options

### Available Models

- **LSTM Neural Network**: Deep learning with ensemble capabilities
- **Ensemble ML**: Combination of multiple ML models
- **Random Forest**: Tree-based ensemble method

### Available Strategies

- **Technical Analysis**: RSI, moving averages, momentum indicators
- **Mean Reversion**: Bollinger Bands, statistical arbitrage
- **Momentum**: MACD, trend following, breakout strategies
- **Pattern Recognition**: Candlestick patterns, chart formations
- **Unified Strategy**: Combines all data sources and strategies

### Parameter Optimization

Each model and strategy has predefined parameter ranges:

- **Confidence thresholds**: 0.6 - 0.9
- **Position sizing**: 5% - 30% of portfolio
- **Stop losses**: 2% - 10%
- **Take profits**: 4% - 18%
- **Lookback windows**: 30 - 252 days
- **Model-specific parameters**: Regularization, ensemble methods, etc.

## Performance Optimization

### Objective Function Design

The framework uses a sophisticated objective function that:

1. **Primary metric**: Optimizes for the selected objective (Sharpe ratio by default)
2. **Penalty system**: Applies penalties for:
   - Low win rates (< 40%)
   - High drawdowns (> 20%)
   - Too few trades (< 5, potential overfitting)

### Efficient Search

- **Bayesian optimization** learns from previous trials
- **Early pruning** stops poor configurations quickly
- **Parallel execution** support for faster optimization
- **Caching** to avoid redundant calculations

## Testing

The framework includes comprehensive tests:

- **Unit tests**: Model selector, configuration validation, objective scoring
- **Integration tests**: End-to-end optimization with real data
- **Backward compatibility**: All existing tests continue to pass

Run tests with:
```bash
python test_optimization_framework.py
python test_backtesting.py  # Verify no breaking changes
```

## Future Enhancements

Potential improvements for future versions:

1. **Multi-objective optimization**: Optimize for multiple metrics simultaneously
2. **Advanced asset selection**: Automatically select optimal symbols
3. **Walk-forward analysis**: Out-of-sample validation with rolling windows
4. **Custom strategies**: User-defined strategy upload and optimization
5. **Real-time optimization**: Continuous parameter adjustment during live trading

## Configuration Examples

### Quick Optimization (Testing)
```python
config = OptimizationConfig(
    n_trials=10,
    symbols=['AAPL'],
    timeout=300,  # 5 minutes
    objective_metric='sharpe_ratio'
)
```

### Production Optimization
```python
config = OptimizationConfig(
    n_trials=200,
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    timeout=7200,  # 2 hours
    objective_metric='sharpe_ratio'
)
```

### Diversified Portfolio Optimization
```python
config = OptimizationConfig(
    n_trials=500,
    symbols=['SPY', 'QQQ', 'BTC-USD', 'ETH-USD', 'GLD', 'TLT'],
    timeout=10800,  # 3 hours
    objective_metric='profit_factor'
)
```

## Dependencies

The optimization framework requires:
- `optuna >= 4.0.0` - Bayesian optimization
- `streamlit >= 1.32.0` - Web interface
- `plotly >= 5.15.0` - Interactive charts
- `pandas >= 2.2.2` - Data manipulation
- `numpy >= 1.24.0` - Numerical computing

All dependencies are included in the existing `requirements.txt`.