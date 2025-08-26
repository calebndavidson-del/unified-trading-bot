# Unified Trading Bot - Quantitative Market Analysis Dashboard

A comprehensive Streamlit-based trading dashboard with advanced technical analysis, candlestick pattern recognition, and risk management features.

## 🎯 Overview

This application provides a **comprehensive quantitative trading dashboard** featuring:

- **Multi-Source Data Pipeline**: Integration with Yahoo Finance, IEX Cloud, Alpha Vantage, Quandl, Finnhub, and Binance APIs
- **Advanced Data Processing**: Automated cleaning, enrichment, and quality assurance
- **Comprehensive Feature Engineering**: 150+ technical indicators, sentiment analysis, and regime detection
- **Advanced Technical Analysis**: 26+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Candlestick Pattern Recognition**: 9+ pattern types with strength scoring and performance analytics
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

### Stocks
- **Tech Giants**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META
- **Market ETFs**: SPY, QQQ
- **Custom Symbols**: Add any Yahoo Finance supported ticker

### Cryptocurrencies
- **Major Coins**: BTC-USD, ETH-USD, SOL-USD, ADA-USD
- **Custom Crypto**: Add any Yahoo Finance crypto ticker

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