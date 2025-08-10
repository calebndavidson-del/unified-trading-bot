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

#### `quant_bot.py` - Core Trading Bot Engine
**Classes:**
- `Position` - Data class for trading position tracking
- `TradingParameters` - Dynamic trading parameters configuration  
- `PaperBroker` - Paper trading broker simulation
- `EnhancedBot` - Main enhanced trading bot with dynamic parameters

**Functions:**
- `read_symbol_list()` - Read trading symbols from CSV file
- `calculate_technical_indicators()` - Calculate RSI, Bollinger Bands, and moving averages
- `main()` - Bot entry point and main execution loop

#### `parameters.py` - Formal Parameter Schema
**Classes:**
- `MarketTimeframeParams` - Market and timeframe configuration
- `EntryParams` - Entry signal parameters (RSI, EMA, breakout)
- `ExitParams` - Exit signal parameters (profit targets, stop loss)
- `PositionSizingParams` - Position sizing and risk management
- `FrequencyControlParams` - Trade frequency control settings
- `ExecutionParams` - Execution quality parameters
- `BacktestConstraintParams` - Backtest performance constraints
- `AdditionalParams` - Additional configuration parameters
- `BacktestParameters` - Complete parameter schema consolidation

**Functions:**
- `get_parameter_ranges()` - Get parameter ranges for optimization
- `generate_random_parameters()` - Generate random parameters for testing
- `get_parameter_grid()` - Generate parameter grid for grid search
- `get_default_parameters()` - Get default parameter configuration

#### `dashboard.py` - Dash Web Dashboard
**Functions:**
- `generate_sample_data()` - Generate sample market data for demonstration
- `get_sample_market_data()` - Generate sample data for all configured symbols
- `create_price_chart()` - Create interactive candlestick price charts
- `create_summary_card()` - Create summary cards for market data display
- `create_overview_charts()` - Create overview performance comparison charts
- `render_tab_content()` - Callback function for tab content rendering

#### `backend/main.py` - FastAPI Backend Service
**Functions:**
- `fetch_symbol_data()` - Fetch time series data using yfinance
- `fetch_all_symbols_async()` - Asynchronously fetch data for multiple symbols
- `root()` - Health check endpoint
- `get_market_data()` - Main market data API endpoint
- `get_symbols()` - Available symbols metadata endpoint
- `get_symbol_data()` - Individual symbol data endpoint

#### `modal_app.py` - Modal Cloud Deployment
**Functions:**
- `get_fastapi_app()` - Get FastAPI application instance
- `fastapi_entrypoint()` - Modal cloud deployment entrypoint

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

1. **Core Engine** (`quant_bot.py`) - Paper trading with technical indicators
2. **Parameter Management** (`parameters.py`) - Comprehensive parameter schema for optimization
3. **Web Dashboard** (`dashboard.py`) - Real-time market data visualization
4. **API Backend** (`backend/main.py`) - FastAPI service for market data
5. **Cloud Deployment** (`modal_app.py`) - Modal cloud integration

The system supports:
- 📈 Technical analysis (RSI, Bollinger Bands, Moving Averages)
- 📊 Paper trading simulation
- 🎯 Dynamic parameter optimization
- 📱 Web-based dashboard interface
- ☁️ Cloud deployment via Modal
- 🔄 Real-time market data via yfinance

---

**Last Updated:** 2025-08-10  
**Bot Context Version:** 1.0  
**Repository:** [calebndavidson-del/unified-trading-bot](https://github.com/calebndavidson-del/unified-trading-bot)