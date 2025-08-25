# Code Context Documentation

This document provides an up-to-date overview of the unified-trading-bot codebase for fast onboarding and bot automation context.

## 📝 Recent Commits Changelog

| Commit Hash | Author | Message | Date | Link |
|-------------|--------|---------|------|------|
| [9ad23bb](https://github.com/calebndavidson-del/unified-trading-bot/commit/9ad23bb400d48d58eac4a54caa3789c8a84b4e69) | copilot-swe-agent[bot] | Initial plan | 2025-08-10 | [View](https://github.com/calebndavidson-del/unified-trading-bot/commit/9ad23bb400d48d58eac4a54caa3789c8a84b4e69) |
| [2ead31e](https://github.com/calebndavidson-del/unified-trading-bot/commit/2ead31e6c9759a2b5310b0a1a2ec09f501b7f6cb) | calebndavidson-del | Update modal_app.py | 2025-08-10 | [View](https://github.com/calebndavidson-del/unified-trading-bot/commit/2ead31e6c9759a2b5310b0a1a2ec09f501b7f6cb) |

*Note: This repository currently has 2 commits. Table will expand as more commits are added.*

## 🏗️ Code Structure Overview

### 📊 Main Python Modules

### 📊 Main Python Modules

#### `dashboard.py` - Streamlit Trading Dashboard
**Classes:**
- `TradingDashboard` - Main dashboard class with market analysis and visualization

**Functions:**
- `fetch_market_data()` - Cached market data fetching via Yahoo Finance
- `create_enhanced_features()` - Comprehensive feature engineering pipeline
- `render_sidebar()` - Interactive sidebar controls and settings
- `render_main_metrics()` - Key performance metrics display
- `render_charts()` - Advanced candlestick and technical analysis charts
- `render_trading_signals()` - Trading signal analysis and recommendations

#### `features/candlestick.py` - Candlestick Pattern Recognition
**Classes:**
- `CandlestickPatternExtractor` - Advanced pattern detection and analysis

**Functions:**
- `extract_all_candlestick_features()` - Comprehensive pattern extraction pipeline
- `_detect_doji()` - Doji pattern detection with strength scoring
- `_detect_hammer()` - Hammer and hanging man pattern detection
- `_detect_shooting_star()` - Shooting star pattern recognition
- `_detect_engulfing()` - Bullish/bearish engulfing patterns
- `_detect_harami()` - Harami pattern detection
- `get_pattern_signals()` - Trading signal generation from patterns
- `calculate_pattern_performance()` - Historical pattern performance analysis

#### `features/market_trend.py` - Technical Analysis Engine
**Classes:**
- `TechnicalIndicators` - Comprehensive technical indicator calculations
- `TrendAnalyzer` - Market trend and pattern analysis
- `TrendSignalGenerator` - Trading signal generation

**Functions:**
- `create_comprehensive_trend_features()` - Complete technical analysis pipeline
- `calculate_momentum_indicators()` - RSI, Stochastic, Williams %R, momentum
- `calculate_volatility_indicators()` - ATR, Bollinger Bands, volatility analysis
- `detect_chart_patterns()` - Chart pattern recognition
- `calculate_support_resistance()` - Dynamic support/resistance levels

#### `features/earnings.py` - Earnings Analysis
**Classes:**
- `EarningsDataFetcher` - Earnings data collection and processing
- `EarningsFeatureEngineer` - Earnings-based feature engineering

**Functions:**
- `create_comprehensive_earnings_features()` - Earnings feature pipeline
- `calculate_earnings_growth()` - Earnings growth metrics
- `create_earnings_trading_signals()` - Earnings-based trading signals

#### `utils/visualization.py` - Advanced Chart Visualization
**Classes:**
- `CandlestickPlotter` - Professional candlestick chart generation
- `TrendVisualization` - Technical indicator and trend visualization

**Functions:**
- `create_dashboard_charts()` - Complete dashboard chart suite
- `plot_candlestick_with_indicators()` - Candlestick charts with technical overlays
- `plot_volume_analysis()` - Volume analysis and visualization
- `create_risk_charts()` - Risk metric visualization

#### `utils/risk.py` - Risk Management System
**Classes:**
- `RiskMetrics` - Comprehensive risk calculation engine
- `PositionSizing` - Dynamic position sizing algorithms
- `StopLossManager` - Advanced stop-loss management
- `PortfolioRiskManager` - Portfolio-level risk management

**Functions:**
- `calculate_comprehensive_risk_metrics()` - Complete risk analysis pipeline
- `calculate_var()` - Value at Risk calculations
- `calculate_sharpe_ratio()` - Risk-adjusted return metrics
- `optimize_position_size()` - Dynamic position sizing

#### `model_config.py` - Configuration Management
**Classes:**
- `TradingBotConfig` - Main configuration container
- `DataConfig` - Market data and symbol configuration
- `ModelConfig` - Deep learning model parameters
- `RiskConfig` - Risk management parameters

**Functions:**
- `load_config()` - Configuration loading and validation

## 🔗 Quick Links

### Code Search & Navigation
- **[Browse Code](https://github.com/calebndavidson-del/unified-trading-bot)** - Main repository
- **[Search Code](https://github.com/calebndavidson-del/unified-trading-bot/search?type=code)** - Search across all files
- **[View Issues](https://github.com/calebndavidson-del/unified-trading-bot/issues)** - Current issues and feature requests
- **[Pull Requests](https://github.com/calebndavidson-del/unified-trading-bot/pulls)** - Active pull requests

### Commit History & Analysis
- **[Commit History](https://github.com/calebndavidson-del/unified-trading-bot/commits/main)** - Full commit timeline
- **[Network Graph](https://github.com/calebndavidson-del/unified-trading-bot/network)** - Branch and merge visualization
- **[Contributors](https://github.com/calebndavidson-del/unified-trading-bot/graphs/contributors)** - Contributor statistics

### Project Documentation
- **[README.md](https://github.com/calebndavidson-del/unified-trading-bot/blob/main/README.md)** - Main project documentation
- **[Requirements](https://github.com/calebndavidson-del/unified-trading-bot/blob/main/requirements.txt)** - Python dependencies
- **[Config](https://github.com/calebndavidson-del/unified-trading-bot/blob/main/config.yaml)** - Bot configuration file

## 🤖 Architecture Summary

This is a **quantitative trading bot** with the following key components:

1. **Streamlit Dashboard** (`dashboard.py`) - Interactive web-based trading dashboard
2. **Feature Engineering** (`features/`) - Candlestick patterns, technical indicators, and market analysis
3. **Risk Management** (`utils/risk.py`) - Comprehensive risk metrics and position sizing
4. **Visualization** (`utils/visualization.py`) - Advanced chart plotting and market visualizations
5. **Model Configuration** (`model_config.py`) - Deep learning model configurations

The system supports:
- 📈 Technical analysis (RSI, MACD, Bollinger Bands, 26+ indicators)
- 🕯️ Candlestick pattern recognition (9+ patterns with strength scoring)
- 📊 Advanced risk management and position sizing
- 📱 Modern Streamlit web interface
- 🔄 Real-time market data via Yahoo Finance
- 🧠 Machine learning model configurations

---

**Last Updated:** 2025-08-10  
**Bot Context Version:** 1.0  
**Repository:** [calebndavidson-del/unified-trading-bot](https://github.com/calebndavidson-del/unified-trading-bot)