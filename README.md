# Unified Trading Bot - Quantitative Market Analysis Dashboard

A comprehensive Streamlit-based trading dashboard with advanced technical analysis, candlestick pattern recognition, and risk management features.

## 🎯 Overview

This application provides a **comprehensive quantitative trading dashboard** featuring:

- **Multi-Source Data Pipeline**: Integration with Yahoo Finance, IEX Cloud, Alpha Vantage, Quandl, Finnhub, and Binance APIs
- **Advanced Data Processing**: Automated cleaning, enrichment, and quality assurance
- **Comprehensive Feature Engineering**: 150+ technical indicators, sentiment analysis, and regime detection
- **Advanced Technical Analysis**: 26+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Candlestick Pattern Recognition**: 9+ pattern types with strength scoring and performance analytics
- **Strategy Backtesting**: Test trading strategies with current year historical data and comprehensive performance metrics
- **Risk Management**: Comprehensive risk metrics, position sizing, and stop-loss management
- **Interactive Visualizations**: Professional-grade charts with Plotly for market analysis
- **Machine Learning Ready**: Deep learning model configurations for predictive analytics
- **Bias Reduction**: Balanced asset selection and stratified sampling for robust models

## 🏗️ Architecture

### Enhanced Data Pipeline
- **Multi-Source Integration**: Fetch data from 6+ financial APIs with automatic failover
- **Data Cleaning**: Outlier detection, missing value imputation, and data validation
- **Feature Engineering**: 150+ features including rolling statistics, volatility measures, and regime detection
- **Quality Assurance**: Automated bias detection, stratified sampling, and quality scoring
- **Parallel Processing**: Concurrent data fetching and processing for improved performance

### Streamlit Frontend
- **Modern Web Interface**: Responsive Streamlit dashboard with real-time data
- **Interactive Charts**: Candlestick charts, technical indicators, and market analysis
- **Risk Analytics**: Portfolio risk metrics and position management tools
- **Real-time Updates**: Live market data integration with caching

### Feature Engineering
- **Candlestick Patterns**: Advanced pattern recognition and signal generation
- **Technical Indicators**: Comprehensive trend and momentum analysis
- **Market Regime Detection**: Trend identification and volatility analysis
- **Earnings Analysis**: Earnings data integration and feature engineering

### Hyperparameter Optimization
- **Optuna Integration**: Bayesian optimization for model hyperparameters
- **Pluggable Architecture**: Easy optimization of any model component
- **Caching & Batching**: Efficient evaluation with persistent result storage
- **Multi-Model Support**: Optimizers for TrendAnalyzer, TrendSignalGenerator, EarningsFeatureEngineer

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Internet connection (for market data)

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/calebndavidson-del/unified-trading-bot.git
   cd unified-trading-bot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run dashboard.py
   ```

4. **Access the dashboard:**
   - Open your browser to `http://localhost:8501`
   - The dashboard will automatically load with sample data

## 📊 Enhanced Data Pipeline

### Multi-Source Data Integration

The trading bot now integrates with multiple financial data providers for comprehensive market coverage:

#### Supported Data Sources
- **Yahoo Finance** (Free): Stocks, ETFs, cryptocurrencies, forex
- **IEX Cloud** (API Key Required): Professional-grade stock data, news, sentiment
- **Alpha Vantage** (API Key Required): Comprehensive fundamental data, economic indicators
- **Quandl** (API Key Required): Alternative datasets, commodities, economic data
- **Finnhub** (API Key Required): Real-time data, news, social sentiment
- **Binance** (Public API): Cryptocurrency market data and order book

#### Data Sources Configuration

Set your API keys in `config.yaml` or environment variables:

```yaml
# API keys for enhanced data sources
alpha_vantage_key: "your_alpha_vantage_key"
iex_cloud_key: "your_iex_cloud_key"
iex_sandbox: true  # Use sandbox for testing
finnhub_key: "your_finnhub_key"
quandl_key: "your_quandl_key"
binance_api_key: "your_binance_key"  # Optional for public data
```

### Data Processing Pipeline

#### 1. Data Cleaning
- **Outlier Detection**: IQR, Z-score, and Isolation Forest methods
- **Missing Value Handling**: Interpolation, forward/backward fill
- **Data Validation**: OHLCV relationship validation, negative price removal
- **Time Series Alignment**: Automatic alignment across different data sources

#### 2. Feature Engineering (150+ Features)
- **Rolling Statistics**: SMA, EMA, standard deviation, min/max over multiple windows
- **Volatility Measures**: Realized volatility, GARCH-like volatility, Parkinson estimator
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R
- **Market Regime Detection**: Trend identification, volatility regimes, market phases
- **Sentiment Features**: News sentiment, social media sentiment, market sentiment indicators
- **Meta-data Tags**: Sector, industry, market cap category, asset type, time-based features

#### 3. Quality Assurance
- **Bias Reduction**: Balanced asset selection across sectors and market caps
- **Stratified Sampling**: Proper train/validation/test splits
- **Data Drift Detection**: Statistical tests for data consistency
- **Quality Scoring**: Comprehensive quality metrics and anomaly detection
- **Visualization**: Distribution analysis and correlation matrices

### Using the Data Pipeline

#### Basic Usage
```python
from features.data_pipeline import run_data_pipeline

# Run pipeline with default settings
symbols = ['AAPL', 'MSFT', 'BTC-USD']
results = run_data_pipeline(symbols, period="6mo")

# Results contain processed data for each symbol
for symbol, data in results.items():
    print(f"{symbol}: {data.shape[0]} rows, {data.shape[1]} features")
```

#### Advanced Configuration
```python
from features.data_pipeline import DataPipeline

config = {
    'alpha_vantage_key': 'your_key',
    'enable_multi_source': True,
    'parallel_processing': True,
    'balance_criteria': {
        'max_per_sector': 5,
        'market_cap_distribution': {
            'Large Cap': 0.4,
            'Mid Cap': 0.3,
            'Small Cap': 0.2,
            'Micro Cap': 0.1
        }
    }
}

pipeline = DataPipeline(config)
results = pipeline.process_pipeline(
    symbols=['AAPL', 'GOOGL', 'TSLA'],
    period="1y",
    enable_cleaning=True,
    enable_enrichment=True,
    enable_qa=True
)

# Save processed data
pipeline.save_pipeline_results(results, "data/processed")

# Get comprehensive summary
summary = pipeline.get_pipeline_summary()
```

#### Pipeline Testing
```bash
# Test the complete pipeline
python test_data_pipeline.py

# Test individual components
python -c "from features.data_sources import YahooFinanceAPI; api = YahooFinanceAPI(); print(api.fetch_market_data('AAPL').shape)"
```

### Data Quality Features

#### Bias Reduction
- **Sector Balance**: Limit maximum assets per sector
- **Geographic Diversity**: Balance across regions and exchanges
- **Market Cap Distribution**: Ensure representation across cap sizes
- **Data Quality Weighting**: Prioritize high-quality data sources

#### Quality Monitoring
- **Real-time Quality Scoring**: Automatic quality assessment
- **Anomaly Detection**: Statistical and ML-based anomaly detection
- **Data Drift Monitoring**: Track changes in data distributions
- **Missing Data Analysis**: Comprehensive missing data patterns

#### Visualization and Reporting
- **Distribution Analysis**: Feature distribution visualization
- **Correlation Matrices**: Feature correlation analysis
- **Quality Reports**: Comprehensive data quality documentation
- **Pipeline Logs**: Detailed processing logs and statistics

### Streamlit Community Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Deploy to Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository
   - Set the main file path to `dashboard.py`
   - Click "Deploy"

3. **Access your deployed app:**
   - Your app will be available at `https://your-app-name.streamlit.app`

## 📊 Dashboard Features

### Main Analytics Dashboard
- **📈 Real-time Market Data**: Live prices, volume, and market statistics
- **🕯️ Candlestick Charts**: Interactive price charts with technical overlays
- **📊 Technical Indicators**: RSI, MACD, Bollinger Bands, moving averages
- **🎯 Trading Signals**: Buy/sell signals based on pattern recognition

### Pattern Recognition
- **🕯️ Candlestick Patterns**: Doji, Hammer, Shooting Star, Engulfing, Harami
- **📈 Pattern Strength**: Reliability scoring and historical performance
- **🎯 Signal Generation**: Automated trading signals with confidence levels

### Risk Management
- **📉 Risk Metrics**: Sharpe ratio, maximum drawdown, VaR calculations
- **💰 Position Sizing**: Dynamic position sizing based on risk parameters
- **🛡️ Stop Loss**: Automated stop-loss level calculations

### Market Analysis
- **📊 Trend Analysis**: Market regime detection and trend strength
- **📈 Momentum Indicators**: RSI, Stochastic, Williams %R
- **💹 Volume Analysis**: OBV, VWAP, volume-based signals

## 🛠️ Development

### Project Structure
```
unified-trading-bot/
├── dashboard.py                    # Main Streamlit application
├── features/                       # Feature engineering modules
│   ├── data_sources/              # Multi-source data APIs
│   │   ├── yahoo_finance.py       # Yahoo Finance API
│   │   ├── iex_cloud.py          # IEX Cloud API
│   │   ├── alpha_vantage.py      # Alpha Vantage API
│   │   ├── quandl.py             # Quandl API
│   │   ├── finnhub.py            # Finnhub API
│   │   └── binance.py            # Binance API
│   ├── data_pipeline.py           # Comprehensive data pipeline
│   ├── candlestick.py             # Candlestick pattern detection
│   ├── earnings.py                # Earnings data features
│   └── market_trend.py            # Technical indicators
├── utils/                         # Utility modules
│   ├── data_cleaning.py           # Data cleaning and validation
│   ├── data_enrichment.py         # Feature engineering utilities
│   ├── data_quality.py            # Quality assurance and bias reduction
│   ├── visualization.py           # Chart and plot utilities
│   └── risk.py                   # Risk management functions
├── model_config.py                # Enhanced model configuration
├── config.yaml                    # Pipeline configuration
├── test_data_pipeline.py          # Pipeline testing suite
├── test_system.py                 # System integration tests
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

### Key Components

#### `dashboard.py` - Main Streamlit App
- Interactive user interface with sidebar controls
- Real-time data fetching and caching
- Multi-tab layout for different analysis views
- Responsive design for all devices

#### `features/candlestick.py` - Pattern Recognition
- 9+ candlestick pattern implementations
- Pattern strength and reliability scoring
- Historical performance analysis
- Trading signal generation

#### `features/market_trend.py` - Technical Analysis
- 26+ technical indicators
- Trend direction analysis
- Support/resistance level detection
- Market regime identification

#### `utils/risk.py` - Risk Management
- Comprehensive risk metrics calculation
- Position sizing algorithms
- Stop-loss management
- Portfolio-level risk analysis

### Configuration

The application uses `model_config.py` for centralized configuration:

- **Data Sources**: Symbol lists and market data settings
- **Technical Indicators**: Indicator parameters and calculations
- **Risk Parameters**: Risk thresholds and position sizing rules
- **Model Settings**: Deep learning model configurations

### Testing

Run the complete test suite to verify all components:

```bash
# Test original system components
python test_system.py

# Test enhanced data pipeline
python test_data_pipeline.py
```

The test suites will verify:
- ✅ Configuration loading and validation
- ✅ Multi-source data fetching (Yahoo Finance, Binance, etc.)
- ✅ Data cleaning and quality validation
- ✅ Feature engineering and enrichment
- ✅ Quality assurance and bias reduction
- ✅ Candlestick pattern detection
- ✅ Technical indicator calculations
- ✅ Risk metric calculations
- ✅ Complete pipeline integration
- ✅ Model configuration

## 📈 Supported Assets

### 🌐 Asset Universe Management
The trading bot now features a **comprehensive Asset Universe Management system** that allows users to fully customize their trading universe through an intuitive web interface:

#### Key Features:
- **🔍 Real-time Asset Search**: Search and discover assets by symbol or company name
- **📊 Smart Asset Organization**: Automatic categorization by asset type (stocks, ETFs, crypto, indexes)
- **✅ Built-in Validation**: Yahoo Finance API validation with detailed asset information
- **🎯 One-Click Management**: Easy addition and removal of assets from your universe

#### Preloaded Asset Lists:
- **🏢 Top 250 US Stocks** by market cap (mega-cap to small-cap)
- **📊 Top 50 ETFs** by volume (broad market, sector, bond, international, thematic)
- **🌍 Top 10 Global Indexes** (S&P 500, NASDAQ, international markets)
- **💎 Top 10 Cryptocurrencies** by market cap (Bitcoin, Ethereum, major alts, stablecoins)

#### How to Use:
1. Navigate to the **"🌐 Asset Universe"** tab in the dashboard
2. **Search** for specific assets or browse **preloaded lists**
3. **Add assets** individually or use **bulk operations** (Add All)
4. **Manage your universe** with real-time categorization and easy removal
5. **Universe automatically syncs** with all trading strategies and models

> 📖 **Detailed Documentation**: See [Asset Universe Management Guide](docs/ASSET_UNIVERSE.md) for complete usage instructions and best practices.

### Traditional Asset Support

#### Stocks
- **Tech Giants**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META
- **Market ETFs**: SPY, QQQ
- **Custom Symbols**: Add any Yahoo Finance supported ticker via Asset Universe

#### Cryptocurrencies
- **Major Coins**: BTC-USD, ETH-USD, SOL-USD, ADA-USD
- **Custom Crypto**: Add any Yahoo Finance crypto ticker via Asset Universe

## 🔧 Customization

### Adding New Data Sources
1. Create new API class in `features/data_sources/`
2. Implement `fetch_market_data()` method
3. Add to `DataPipeline` initialization
4. Update configuration in `model_config.py`

### Adding New Indicators
1. Implement indicator in `features/market_trend.py`
2. Add to configuration in `model_config.py`
3. Update enrichment pipeline in `utils/data_enrichment.py`
4. Update visualization in `utils/visualization.py`

### Adding New Patterns
1. Implement pattern detection in `features/candlestick.py`
2. Add pattern to extraction pipeline
3. Update signal generation logic

### Custom Feature Engineering
1. Add feature functions to `utils/data_enrichment.py`
2. Configure in `model_config.py` feature settings
3. Update pipeline processing in `features/data_pipeline.py`

### Custom Data Cleaning Rules
1. Add cleaning functions to `utils/data_cleaning.py`
2. Configure parameters in `model_config.py`
3. Update pipeline cleaning step

### Custom Risk Metrics
1. Add metric calculation to `utils/risk.py`
2. Update risk reporting functions
3. Add visualization to dashboard

### Pipeline Configuration
Customize the data pipeline behavior in `config.yaml`:

```yaml
data:
  # Enable/disable pipeline components
  enable_data_cleaning: true
  enable_data_enrichment: true
  enable_quality_assurance: true
  
  # Feature engineering settings
  rolling_windows: [5, 10, 20, 50]
  volatility_windows: [10, 20, 50]
  enable_regime_detection: true
  enable_sentiment_features: true
  
  # Data cleaning parameters
  outlier_detection_method: iqr  # iqr, zscore, isolation_forest
  missing_value_method: interpolate
  outlier_action: cap  # remove, cap, median
  
  # Bias reduction settings
  balance_criteria:
    max_per_sector: 5
    max_per_region: 10
```

## 🔍 Strategy Backtesting

The backtesting module allows users to test trading strategies against historical data from the current calendar year (January 1 to present date) with comprehensive performance analysis.

### 🎯 Supported Strategies

#### Technical Analysis Strategy
- **RSI Signals**: Oversold/overbought conditions with configurable thresholds
- **Moving Average Crossovers**: Short and long-term MA trend detection
- **Combined Signals**: Multi-indicator confirmation for stronger signals

#### Mean Reversion Strategy
- **Bollinger Bands**: Mean reversion based on price bands
- **Configurable Parameters**: Band period and standard deviation multipliers
- **Entry/Exit Rules**: Systematic band touch entry and exit signals

#### Momentum Strategy
- **MACD Signals**: Moving Average Convergence Divergence crossovers
- **Trend Following**: Capture trending market movements
- **Momentum Confirmation**: Histogram analysis for signal strength

#### Pattern Recognition Strategy
- **Candlestick Patterns**: 9+ pattern types including doji, hammer, engulfing
- **Pattern Strength**: Weighted signal generation based on pattern reliability
- **Multi-Pattern Analysis**: Combination of multiple patterns for robust signals

### 🔧 Backtesting Features

#### Comprehensive Analysis
- **Portfolio Simulation**: Full position tracking and P&L calculation
- **Risk Management**: Position sizing, stop-loss, and take-profit orders
- **Commission Modeling**: Realistic transaction cost inclusion
- **Current Year Data**: January 1 to present date historical data

#### Performance Metrics
- **Total Return**: Overall portfolio performance percentage
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Maximum Drawdown**: Worst peak-to-trough decline analysis
- **Win Rate**: Percentage of profitable trades
- **Volatility**: Portfolio volatility and risk measures
- **Value at Risk**: Downside risk exposure calculations

#### Interactive Visualizations
- **Equity Curve**: Portfolio value progression over time
- **Trade Analysis**: Trade distribution and cumulative P&L charts
- **Performance Radar**: Multi-metric performance overview
- **Monthly Heatmap**: Monthly return distribution visualization
- **Drawdown Analysis**: Portfolio decline and recovery periods

### 🚀 Quick Start

1. **Navigate to Backtesting Tab** in the dashboard
2. **Select Assets** from your asset universe (stocks, ETFs, crypto)
3. **Choose Strategy** and configure parameters
4. **Set Risk Parameters** (capital, position sizing, commission)
5. **Run Backtest** and analyze comprehensive results

```python
# Programmatic Usage
from features.backtesting import BacktestEngine
from model_config import TradingBotConfig

config = TradingBotConfig()
engine = BacktestEngine(config)

results = engine.run_backtest(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    strategy_name='Technical Analysis',
    confidence_threshold=0.75
)

print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
```

### 📊 Export Capabilities
- **Performance Report**: Comprehensive text-based analysis
- **Trade Log**: Detailed CSV export of all executed trades
- **Portfolio Data**: Historical portfolio values and metrics
- **Interactive Charts**: Save charts as images or interactive HTML

For detailed documentation, see [docs/BACKTESTING.md](docs/BACKTESTING.md).

## 🔬 Hyperparameter Optimization

The trading bot includes a comprehensive hyperparameter optimization framework using Optuna for Bayesian optimization. This allows you to automatically tune model parameters for optimal performance.

### 🎯 Supported Models

#### TrendAnalyzer Optimization
Optimizes technical analysis parameters:
- **Moving Average Windows**: Short, medium, and long-term periods
- **RSI Parameters**: Window size and calculation method
- **MACD Settings**: Fast, slow, and signal periods
- **Bollinger Bands**: Window size and standard deviation multiplier
- **Volatility Indicators**: ATR windows and calculation periods

#### TrendSignalGenerator Optimization
Optimizes trading signal parameters:
- **Signal Thresholds**: RSI overbought/oversold levels, Stochastic thresholds
- **Signal Combination**: Weights for momentum, trend, and mean reversion signals
- **Divergence Detection**: Window sizes for identifying signal divergences
- **Moving Average Crossovers**: Short and long-term MA windows

#### EarningsFeatureEngineer Optimization
Optimizes earnings analysis parameters:
- **Surprise Thresholds**: Beat/miss classification levels
- **Growth Analysis**: YoY and QoQ growth thresholds
- **Time Windows**: Pre/post earnings momentum periods
- **Signal Weights**: Combination weights for surprise, growth, and momentum

#### API Parameter Optimization (NEW!)
Optimizes data source API parameters for quality, efficiency, and cost:
- **Yahoo Finance**: Interval/period optimization, data validation strategies
- **IEX Cloud**: Multi-endpoint combinations, rate limiting, cost optimization
- **Alpha Vantage**: Rate limit compliance, function selection, technical indicators
- **Quandl**: Dataset selection, date ranges, data quality optimization
- **Finnhub**: Resolution optimization, alternative data integration
- **Binance**: Crypto symbol mapping, multi-timeframe analysis, weight management

### 🚀 Quick Start

#### Model Optimization
```python
from optimization import TrendAnalyzerOptimizer, OptimizationConfig
import yfinance as yf

# Download sample data
data = yf.Ticker("AAPL").history(period="1y")

# Configure optimization
config = OptimizationConfig(
    n_trials=100,
    direction='maximize',
    sampler='TPE'
)

# Create optimizer
optimizer = TrendAnalyzerOptimizer(config=config)

# Run optimization
result = optimizer.optimize(data)

# Get best parameters
print(f"Best score: {result['best_score']:.4f}")
print(f"Best parameters: {result['best_params']}")

# Use optimized model
best_model = result['best_model']
optimized_features = best_model.identify_trend_direction(data)
```

#### API Parameter Optimization (NEW!)
```python
from optimization import YahooFinanceAPIOptimizer, BinanceAPIOptimizer

# Optimize Yahoo Finance API parameters
yahoo_optimizer = YahooFinanceAPIOptimizer()
result = yahoo_optimizer.optimize_for_symbols(['AAPL', 'MSFT'])
print(f"Best Yahoo Finance score: {result['best_score']:.4f}")

# Optimize Binance API for cryptocurrency data
binance_optimizer = BinanceAPIOptimizer()
crypto_result = binance_optimizer.optimize_for_symbols(['BTC-USDT', 'ETH-USDT'])

# Customize optimization objectives
binance_optimizer.set_optimization_weights({
    'data_quality': 0.35,
    'efficiency': 0.35,      # Higher weight for crypto trading efficiency
    'cost_effectiveness': 0.20,
    'error_rate': 0.10
})
```

### 🔧 Advanced Features

#### Persistent Caching
Results are automatically cached to avoid redundant computations:
```python
from optimization import OptimizationCache

# View cache statistics
cache = OptimizationCache()
stats = cache.get_cache_stats()
print(f"Cache contains {stats['total_entries']} entries")

# Export/import cache
cache.export_cache("optimization_results.json")
cache.import_cache("shared_results.json")
```

#### Batch Processing
Optimize across multiple datasets for robustness:
```python
# Optimize across multiple stocks
datasets = [
    yf.Ticker("AAPL").history(period="1y"),
    yf.Ticker("MSFT").history(period="1y"),
    yf.Ticker("GOOGL").history(period="1y")
]

result = optimizer.optimize_batch(datasets)
print(f"Best overall score: {result['best_score']:.4f}")
```

#### Parameter Grid Search
Use batch processing for systematic parameter exploration:
```python
from optimization import BatchProcessor

processor = BatchProcessor()
param_grid = {
    'short_ma_window': [5, 10, 15],
    'long_ma_window': [20, 30, 40],
    'rsi_window': [14, 21]
}

combinations = processor.create_parameter_grid(param_grid)
results = processor.process_parameter_batch(combinations, model_factory, evaluate_func, data)
```

### 📊 Optimization Metrics

The framework optimizes models based on multiple criteria:

#### TrendAnalyzer Scoring
- **Feature Stability** (25%): Consistent indicator values
- **Trend Prediction** (35%): Correlation with future returns  
- **Information Content** (25%): Entropy and signal quality
- **Signal Quality** (15%): Signal-to-noise ratio

#### TrendSignalGenerator Scoring
- **Signal Accuracy** (30%): Correlation with future returns
- **Signal Frequency** (15%): Optimal signal density (5-15%)
- **Risk-Adjusted Returns** (25%): Sharpe ratio optimization
- **Signal Consistency** (15%): Temporal signal stability
- **Profit Potential** (15%): Total return vs. maximum drawdown

#### EarningsFeatureEngineer Scoring
- **Prediction Accuracy** (35%): Earnings surprise prediction quality
- **Timing Quality** (25%): Signal timing around earnings events
- **Feature Informativeness** (20%): Information content of features
- **Risk Management** (10%): Downside risk metrics
- **Signal Consistency** (10%): Feature stability over time

### 🛠️ Extending the Framework

Adding optimization for new models is straightforward:

```python
from optimization import BaseOptimizer
import optuna

class MyModelOptimizer(BaseOptimizer):
    def define_search_space(self, trial):
        return {
            'param1': trial.suggest_int('param1', 1, 100),
            'param2': trial.suggest_float('param2', 0.1, 1.0)
        }
    
    def create_model_instance(self, params):
        return MyModel(**params)
    
    def evaluate_model(self, model, data):
        # Implement your scoring logic
        return model.calculate_score(data)
```

### 📈 Example Results

Typical optimization improvements:
- **TrendAnalyzer**: 15-25% improvement in trend prediction accuracy
- **TrendSignalGenerator**: 20-30% improvement in signal Sharpe ratio
- **EarningsFeatureEngineer**: 10-20% improvement in earnings prediction

### 💡 Best Practices

1. **Use Validation Data**: Always optimize on training data and evaluate on held-out validation data
2. **Sufficient Trials**: Use 50-200 trials for meaningful optimization results  
3. **Multiple Datasets**: Use batch optimization for robust parameter selection
4. **Cache Management**: Leverage caching for faster iterative optimization
5. **Parameter Constraints**: Set reasonable bounds to avoid overfitting
6. **Regularization**: Framework includes automatic penalty for extreme parameters

For complete examples and advanced usage, see `optimization/examples.py`.

## 🚀 Deployment Options

### Streamlit Community Cloud (Recommended)
- **Free hosting** for public repositories
- **Automatic deployment** from GitHub
- **Built-in SSL** and custom domains
- **Easy scaling** and management

### Docker Deployment (Available on Request)
A Dockerfile is available for containerized deployment. Contact for access.

### Local Development Server
Perfect for development and testing:
```bash
streamlit run dashboard.py --server.port 8501
```

## 📊 Performance

- **Multi-Source Data**: Parallel fetching from 6+ data providers
- **Data Caching**: Intelligent caching with 5-minute TTL for market data
- **Efficient Processing**: Vectorized calculations for 150+ technical indicators
- **Parallel Pipeline**: Concurrent data processing and feature engineering
- **Quality Optimization**: Automated bias reduction and quality scoring
- **Responsive UI**: Optimized for real-time updates
- **Memory Management**: Efficient data handling for multiple assets and features
- **Scalable Architecture**: Modular design for easy extension and customization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the documentation
- Review the test suite for examples

---

**Built with ❤️ using Streamlit, Plotly, and Yahoo Finance**