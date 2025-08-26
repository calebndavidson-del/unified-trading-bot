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
- **Data Validation**: Ensures data quality and completeness with detailed error reporting
- **Timezone Handling**: All timestamps normalized to UTC for consistent processing
- **Missing Data Handling**: Gracefully handles holidays, weekends, and missing data points
- **Error Recovery**: Robust error handling with detailed troubleshooting information

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

## Date and Timezone Handling

### Timezone Normalization
All historical data is automatically normalized to UTC timezone to ensure consistent processing:

```python
# Data fetching automatically converts timezones
data = engine.fetch_current_year_data(['AAPL', 'MSFT'])
# All timestamps are now in UTC regardless of source timezone
```

### Missing Data Handling
The backtest engine provides intelligent missing data handling with asset-type specific logic:

#### Asset Type Detection
The system automatically detects asset types and applies appropriate data validation:

- **Stocks/ETFs/Indexes**: Traditional assets that only trade on weekdays
- **Cryptocurrency**: Digital assets that trade 24/7, 365 days a year

```python
from features.backtesting import AssetTypeDetector

# Automatic asset type detection
asset_type = AssetTypeDetector.detect_asset_type('AAPL')      # → 'stock'
asset_type = AssetTypeDetector.detect_asset_type('BTC-USD')   # → 'crypto'
asset_type = AssetTypeDetector.detect_asset_type('SPY')       # → 'etf'
asset_type = AssetTypeDetector.detect_asset_type('^GSPC')     # → 'index'
```

#### Expected vs Unexpected Missing Data

**Expected Missing Data** (not logged as errors):
- **Weekends**: Saturday and Sunday for stocks/ETFs/indexes
- **Market Holidays**: US federal holidays when markets are closed
- **Crypto Weekend Gaps**: Small time gaps within tolerance for crypto assets

**Unexpected Missing Data** (logged with warnings):
- **Weekday Stock Gaps**: Missing data on trading days for traditional assets
- **Large Crypto Gaps**: Crypto data gaps exceeding tolerance threshold
- **Invalid Symbols**: Non-existent tickers or delisted securities

#### Crypto-Specific Tolerance Settings

For cryptocurrency assets, the system uses configurable tolerance thresholds:

```python
from features.backtesting import MissingDataConfig

config = MissingDataConfig()
config.crypto_daily_tolerance_hours = 6.0  # Max 6-hour gap for daily crypto data
config.strict_mode = False  # Whether to halt on excessive missing data
config.max_missing_data_ratio = 0.1  # 10% max missing data in strict mode
```

- **Tolerance Checking**: Crypto gaps under 6 hours (default) are considered acceptable
- **Violation Logging**: Only gaps exceeding tolerance are logged as warnings
- **Closest Date Logic**: Uses nearest available data within tolerance

#### Strict Mode

Enable strict mode to halt backtesting when missing data exceeds thresholds:

```python
# Configure strict mode
missing_data_config = MissingDataConfig()
missing_data_config.strict_mode = True
missing_data_config.max_missing_data_ratio = 0.05  # Halt if >5% data is missing

# Initialize engine with strict mode
engine = BacktestEngine(config, missing_data_config)
```

#### Missing Data Summary Report

At the end of each backtest, a comprehensive missing data summary is generated:

```
📊 Missing Data Summary Report
========================================
📈 Expected gaps (weekends/holidays): 76
⚠️ Unexpected gaps: 3
🔴 Crypto tolerance violations: 1

📊 Missing data by asset type:
  stock: 2 gaps
  crypto: 1 gaps

⚠️ Symbols with unexpected missing data:
  TSLA (stock): 2 unexpected gaps

🔴 Crypto tolerance violations (>6.0h gaps):
  BTC-USD: 1 violations
    2025-03-15: 8.2h gap
```

This report helps you understand:
- **Data Quality**: Overall completeness of your dataset
- **Expected vs Unexpected**: Distinguish between normal gaps and data issues
- **Asset-Specific Issues**: Identify problematic symbols or asset types
- **Tolerance Violations**: Crypto gaps that exceed acceptable thresholds

### Date Validation
Each date access is validated before processing:

```python
# Safe date access with validation
price = engine._validate_date_access(date, data, symbol)
if price is not None:
    # Process the valid price
    pass
else:
    # Skip this date/symbol combination
    continue
```

### Error Handling
Comprehensive error handling provides clear debugging information:

- **Timestamp Errors**: Clear messages when dates don't exist in datasets
- **Timezone Conflicts**: Automatic conversion between different timezone representations  
- **Data Quality Issues**: Warnings for NaN values or suspicious data
- **Progress Tracking**: Real-time progress updates during long backtests

## Troubleshooting

### Common Issues

**No Data Available**
- Check internet connection
- Verify symbol validity (use valid ticker symbols like 'AAPL', not company names)
- Ensure market is open/has recent data
- Check error messages for specific timezone or data issues

**Timestamp Errors**
- All data is automatically converted to UTC timezone
- Missing dates (holidays/weekends) are handled gracefully
- Check error logs for specific date issues
- Validate that symbols have data for the requested date range

**Poor Performance**
- Adjust strategy parameters
- Lower confidence threshold for more trades
- Check commission settings
- Review position sizing

**No Trades Executed**
- Lower confidence threshold (try 0.5 instead of 0.75)
- Adjust strategy parameters
- Check data availability for all symbols
- Verify signal generation is working

### Data Quality Monitoring
The backtest engine provides detailed data quality metrics and missing data analysis:

```python
results = engine.run_backtest(symbols, strategy_name)

# Basic metrics
print(f"Data quality: {results['data_quality']}")
print(f"Successful days: {results['successful_days']}")
print(f"Skipped days: {results['skipped_days']}")

# Missing data analysis
if 'missing_data_summary' in results:
    summary = results['missing_data_summary']
    print(f"Expected gaps: {summary['total_expected_gaps']}")
    print(f"Unexpected gaps: {summary['total_unexpected_gaps']}")
    print(f"Crypto violations: {summary['crypto_tolerance_violations']}")
    print(f"Status: {summary['status']}")
```

#### Interpreting Missing Data Reports

**Clean Status**: No unexpected missing data detected
```
📊 Missing Data Summary Report
========================================
✅ No missing data issues detected
```

**Issues Found**: Some unexpected gaps detected
```
📊 Missing Data Summary Report
========================================
📈 Expected gaps (weekends/holidays): 76
⚠️ Unexpected gaps: 5
🔴 Crypto tolerance violations: 1
```

**Common Missing Data Scenarios**:

1. **High Expected Gaps**: Normal for mixed stock/crypto backtests
   - Stocks: Only weekdays have data
   - Crypto: Should have data for all days
   - Expected gaps represent weekends/holidays for traditional assets

2. **Unexpected Stock Gaps**: May indicate:
   - Market holidays not in our calendar
   - Data provider issues
   - Symbol delisting or trading halts

3. **Crypto Tolerance Violations**: Indicate:
   - Data provider outages
   - Network connectivity issues
   - Exchange maintenance periods exceeding normal tolerance

**Troubleshooting Tips**:
- Review the summary report for patterns
- Check if unexpected gaps cluster around specific dates
- Verify internet connectivity during backtest period
- Consider adjusting crypto tolerance for less strict checking

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