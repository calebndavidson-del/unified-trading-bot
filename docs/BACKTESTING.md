# Backtesting Module Documentation

## Overview

The backtesting module allows users to test trading strategies against historical data from the current calendar year (January 1 to present date). It provides comprehensive performance analysis including portfolio simulation, trade execution, and risk metrics calculation.

## Features

### 🎯 Strategy Selection
- **Technical Analysis**: RSI and moving average crossover signals
- **Mean Reversion**: Bollinger Bands-based mean reversion
- **Momentum**: MACD-based momentum trading
- **Pattern Recognition**: Candlestick pattern-based signals

### 📊 Data Management
- **Current Year Data**: Automatically fetches data from Jan 1 to present
- **Multi-Asset Support**: Test strategies across multiple stocks, ETFs, and crypto
- **Data Validation**: Ensures data quality and completeness

### 💼 Portfolio Simulation
- **Position Sizing**: Configurable position sizing methods
- **Risk Management**: Stop-loss and take-profit orders
- **Commission Modeling**: Realistic transaction costs
- **Capital Management**: Initial capital and cash flow tracking

### 📈 Performance Metrics
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Volatility**: Portfolio volatility measures
- **Value at Risk (VaR)**: Risk exposure calculations

## Usage

### Web Interface

1. **Navigate to Backtesting Tab**: Click on "🔍 Backtesting" in the main dashboard
2. **Select Assets**: Choose from available symbols in your asset universe
3. **Configure Strategy**: Select strategy and adjust parameters
4. **Set Risk Parameters**: Configure capital, position sizing, and commission
5. **Run Backtest**: Execute the backtest and view results

### Programmatic Usage

```python
from features.backtesting import BacktestEngine, TechnicalAnalysisStrategy
from model_config import TradingBotConfig

# Initialize configuration
config = TradingBotConfig()
config.risk.initial_capital = 100000

# Create backtesting engine
engine = BacktestEngine(config)

# Run backtest
results = engine.run_backtest(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    strategy_name='Technical Analysis',
    confidence_threshold=0.75
)

# View results
print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
```

## Strategy Details

### Technical Analysis Strategy

Uses RSI and moving average signals:
- **Buy Signal**: RSI < oversold threshold AND short MA > long MA
- **Sell Signal**: RSI > overbought threshold AND short MA < long MA
- **Parameters**: RSI periods, MA periods, overbought/oversold levels

```python
strategy_config = {
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'ma_short': 10,
    'ma_long': 50
}
```

### Mean Reversion Strategy

Uses Bollinger Bands for mean reversion:
- **Buy Signal**: Price touches lower Bollinger Band
- **Sell Signal**: Price touches upper Bollinger Band
- **Parameters**: Bollinger period, standard deviation multiplier

```python
strategy_config = {
    'bb_period': 20,
    'bb_std': 2.0
}
```

### Momentum Strategy

Uses MACD for momentum detection:
- **Buy Signal**: MACD line crosses above signal line
- **Sell Signal**: MACD line crosses below signal line
- **Parameters**: Fast EMA, slow EMA, signal line periods

```python
strategy_config = {
    'ma_fast': 12,
    'ma_slow': 26,
    'signal_line': 9
}
```

### Pattern Recognition Strategy

Uses candlestick patterns:
- **Signals**: Based on detected bullish/bearish patterns
- **Patterns**: Doji, hammer, shooting star, engulfing, harami
- **Parameters**: Pattern weight for signal strength

```python
strategy_config = {
    'pattern_weight': 0.5
}
```

## Configuration Options

### Risk Management
```python
@dataclass
class BacktestingConfig:
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% max per position
    commission_rate: float = 0.001  # 0.1% commission
    confidence_threshold: float = 0.75
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.05  # 5%
    use_take_profit: bool = True
    take_profit_pct: float = 0.15  # 15%
```

### Data Settings
```python
# Current year only (default)
use_current_year_only: bool = True

# Custom date range (optional)
start_date: str = "2025-01-01"
end_date: str = "2025-08-26"
```

## Output and Visualization

### Interactive Charts
- **Equity Curve**: Portfolio value over time
- **Daily Returns**: Bar chart of daily returns
- **Drawdown Chart**: Portfolio drawdown visualization
- **Trade Analysis**: Trade distribution and cumulative P&L
- **Performance Radar**: Multi-metric performance overview
- **Monthly Heatmap**: Monthly returns visualization

### Downloadable Reports
- **Performance Report**: Comprehensive text report
- **Trade Log**: Detailed CSV of all trades
- **Portfolio Data**: Historical portfolio values

### Key Metrics Display
- Total Return (%)
- Sharpe Ratio
- Maximum Drawdown (%)
- Win Rate (%)
- Volatility (%)
- Total Trades
- Profit Factor

## Example Results

```
📊 Backtest Results
==================
Total Return: 15.32%
Sharpe Ratio: 1.247
Max Drawdown: -8.45%
Win Rate: 68.4%
Volatility: 18.3%
Total Trades: 47
Profit Factor: 1.89
Period: 2025-01-02 to 2025-08-25
```

## Performance Considerations

### Data Limitations
- **Current Year Only**: Limited to 2025 data (approximately 8 months)
- **Market Conditions**: Results specific to 2025 market environment
- **Sample Size**: Limited trading history may affect statistical significance

### Strategy Limitations
- **No Machine Learning**: Strategies use traditional technical analysis
- **No Fundamental Data**: Strategies don't incorporate earnings or financial metrics
- **Simplified Execution**: Perfect execution assumed (no slippage modeling)

### Recommended Use Cases
- **Strategy Comparison**: Compare different strategy performance
- **Parameter Optimization**: Test different strategy parameters
- **Risk Assessment**: Understand strategy risk characteristics
- **Educational**: Learn about trading strategy performance

## Extending the Framework

### Adding New Strategies

```python
from features.backtesting import TradingStrategy

class CustomStrategy(TradingStrategy):
    def __init__(self, config=None):
        super().__init__("Custom Strategy", config)
    
    def generate_signals(self, data):
        signals = pd.Series(0.0, index=data.index)
        # Implement your signal logic
        return signals
```

### Custom Performance Metrics

```python
from utils.backtesting_metrics import BacktestingMetrics

# Add custom metrics
def calculate_custom_metrics(portfolio_df):
    # Your custom calculations
    return custom_metrics
```

## Best Practices

1. **Parameter Testing**: Test multiple parameter combinations
2. **Risk Management**: Always include proper position sizing
3. **Out-of-Sample**: Consider forward testing on unseen data
4. **Statistical Significance**: Consider sample size limitations
5. **Market Conditions**: Understand current year market context
6. **Strategy Combination**: Consider combining multiple strategies

## Troubleshooting

### Common Issues

**No Data Available**
- Check internet connection
- Verify symbol validity
- Ensure market is open/has recent data

**Poor Performance**
- Adjust strategy parameters
- Lower confidence threshold
- Check commission settings
- Review position sizing

**No Trades Executed**
- Lower confidence threshold
- Adjust strategy parameters
- Check data availability
- Verify signal generation

### Support

For issues or questions:
1. Check the test script: `python test_backtesting.py`
2. Review configuration settings
3. Verify data availability for selected symbols
4. Check error logs in the dashboard

## Future Enhancements

- **Multi-Year Backtesting**: Extend to historical years
- **Machine Learning Integration**: Incorporate ML models
- **Advanced Order Types**: Limit orders, stop orders
- **Portfolio Optimization**: Modern portfolio theory integration
- **Real-time Paper Trading**: Live strategy testing
- **Strategy Ensemble**: Combine multiple strategies
- **Risk Parity**: Advanced position sizing methods