# Unified Trading Bot - Quantitative Market Analysis Dashboard

A comprehensive Streamlit-based trading dashboard with advanced technical analysis, candlestick pattern recognition, and risk management features.

## 🎯 Overview

This application provides a **comprehensive quantitative trading dashboard** featuring:

- **Multi-Asset Market Data**: Real-time data for stocks and cryptocurrencies via Yahoo Finance
- **Advanced Technical Analysis**: 26+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Candlestick Pattern Recognition**: 9+ pattern types with strength scoring and performance analytics
- **Risk Management**: Comprehensive risk metrics, position sizing, and stop-loss management
- **Interactive Visualizations**: Professional-grade charts with Plotly for market analysis
- **Machine Learning Ready**: Deep learning model configurations for predictive analytics

## 🏗️ Architecture

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
├── dashboard.py                # Main Streamlit application
├── features/                   # Feature engineering modules
│   ├── candlestick.py         # Candlestick pattern detection
│   ├── earnings.py            # Earnings data features
│   └── market_trend.py        # Technical indicators
├── utils/                     # Utility modules
│   ├── visualization.py       # Chart and plot utilities
│   └── risk.py               # Risk management functions
├── model_config.py            # Model configuration
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
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

Run the test suite to verify all components:

```bash
python test_system.py
```

This will test:
- ✅ Configuration loading
- ✅ Market data fetching
- ✅ Candlestick pattern detection
- ✅ Technical indicator calculations
- ✅ Risk metric calculations
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

### Adding New Indicators
1. Implement indicator in `features/market_trend.py`
2. Add to configuration in `model_config.py`
3. Update visualization in `utils/visualization.py`

### Adding New Patterns
1. Implement pattern detection in `features/candlestick.py`
2. Add pattern to extraction pipeline
3. Update signal generation logic

### Custom Risk Metrics
1. Add metric calculation to `utils/risk.py`
2. Update risk reporting functions
3. Add visualization to dashboard

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

- **Data Caching**: 5-minute TTL for market data
- **Efficient Processing**: Vectorized calculations for technical indicators
- **Responsive UI**: Optimized for real-time updates
- **Memory Management**: Efficient data handling for multiple assets

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