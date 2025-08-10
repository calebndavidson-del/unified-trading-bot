# Unified Trading Bot - Current Market Dashboard

A clean, modern market dashboard displaying real-time data for major US indices and cryptocurrencies.

## 🎯 Overview

This application provides a **"Current Market"** dashboard that displays:

- **US Market Indices**: S&P 500 (SPY), NASDAQ (QQQ), Dow Jones (DIA), Russell 2000 (IWM)
- **Cryptocurrencies**: Bitcoin (BTC), Ethereum (ETH), Dogecoin (DOGE), XRP, Solana (SOL)
- **Interactive charts** with time series data
- **Real-time price updates** and percentage changes

## 🏗️ Architecture

### Backend (FastAPI)
- **FastAPI** REST API server on port 8000
- **Market data endpoint**: `/market-data` 
- **Yahoo Finance** integration for live data
- **Async data fetching** for optimal performance

### Frontend (React - Coming Soon)
- **React 18** single-page application
- **Chart.js** for interactive visualizations
- **Responsive design** for all devices
- **Real-time updates** every 5 minutes

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Internet connection (for market data)

### Run the Application

```bash
# Clone and navigate to the repository
cd unified-trading-bot

# Start the application (backend + frontend)
./start.sh
```

This will:
1. Install Python dependencies
2. Start the FastAPI backend server
3. Set up the React frontend (when ready)

### Manual Setup

#### Backend Only
```bash
# Install dependencies
pip install -r requirements.txt

# Start the backend server
cd backend
python main.py
```

Backend will be available at: http://localhost:8000

### Dash Dashboard
```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the Dash dashboard
python dashboard.py
```

Dashboard will be available at: http://localhost:8050

The dashboard includes:
- **📈 Market Overview**: Summary cards and performance comparison
- **📊 US Indices**: Interactive candlestick charts for SPY, QQQ, DIA, IWM
- **₿ Cryptocurrencies**: Charts for BTC, ETH, DOGE, XRP, SOL

*Note: Dashboard currently uses sample data for demonstration. For live data integration, connect to the FastAPI backend endpoints.*

### Streamlit Dashboard
```bash
# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the Streamlit dashboard
streamlit run streamlit_app.py
```

Dashboard will be available at: http://localhost:8501

The Streamlit dashboard includes:
- **📈 Market Overview**: Summary cards with real-time metrics and performance comparison chart
- **📊 Global Indices**: Interactive candlestick charts for SPY, QQQ, DIA, IWM
- **₿ Cryptocurrencies**: Charts for BTC, ETH, DOGE, XRP, SOL
- **🔄 Live Data Integration**: Connects to the FastAPI backend for real market data
- **⚠️ Graceful Error Handling**: Falls back to sample data if API is unavailable
- **📱 Responsive Design**: Works on desktop and mobile devices

*The Streamlit app automatically connects to the FastAPI backend when available, and gracefully falls back to sample data when the API is unavailable.*

#### Test the API
```bash
# Check API health
curl http://localhost:8000/

# Get market data for all symbols
curl http://localhost:8000/market-data

# Get data for specific categories
curl "http://localhost:8000/market-data?category=indices"
curl "http://localhost:8000/market-data?category=crypto"

# Get data for a specific symbol
curl http://localhost:8000/market-data/SPY
```

## 📊 API Endpoints

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/market-data` | GET | Get market data for all symbols |
| `/market-data/{symbol}` | GET | Get data for specific symbol |
| `/symbols` | GET | List available symbols |

### Query Parameters

- `category`: `"all"`, `"indices"`, `"crypto"` (default: `"all"`)
- `period`: `"1d"`, `"5d"`, `"1mo"`, `"3mo"`, etc. (default: `"5d"`)
- `interval`: `"1h"`, `"1d"`, etc. (default: `"1h"`)

### Example Response

```json
{
  "timestamp": "2024-01-10T15:30:00",
  "period": "5d",
  "interval": "1h",
  "category": "all",
  "data": {
    "SPY": {
      "symbol": "SPY",
      "latest_price": 475.32,
      "price_change": 2.15,
      "price_change_pct": 0.45,
      "is_positive": true,
      "data": [
        {
          "datetime": "2024-01-10T15:00:00",
          "timestamp": 1704906000000,
          "open": 475.10,
          "high": 475.85,
          "low": 474.90,
          "close": 475.32,
          "volume": 15642891
        }
      ]
    }
  }
}
```

## 🛠️ Development

### Project Structure

```
unified-trading-bot/
├── backend/
│   └── main.py              # FastAPI application
├── frontend/                # React application (in development)
│   ├── public/
│   └── src/
│       ├── components/      # React components
│       ├── services/        # API service
│       └── App.js          # Main app component
├── dashboard.py             # Dash dashboard application
├── parameters.py            # Formal parameter schema for backtesting
├── config.yaml             # Configuration file
├── quant_bot.py            # Bot learning logic
├── requirements.txt        # Python dependencies
├── start.sh               # Application startup script
└── README.md              # This file
```

### Key Features Implemented

✅ **FastAPI Backend**
- Market data fetching for 9 symbols
- RESTful API with async support
- Error handling and validation
- CORS enabled for frontend

✅ **Dash Dashboard**
- Interactive browser-based UI
- Tabbed interface for different markets
- Sample data with realistic price movements
- Candlestick charts with Plotly
- Market summary cards with price changes
- Structured for easy real data integration

✅ **Streamlit Dashboard**
- Modern, reactive web application
- Real-time data integration with FastAPI backend
- Graceful error handling and fallback to sample data
- Responsive design for all devices
- Interactive charts and summary cards
- Built-in caching for optimal performance

✅ **Clean Repository**
- Removed optimization files
- Removed parameter management
- Removed live/paper trading
- Kept only essential components

✅ **Real-time Data**
- Yahoo Finance integration
- Support for stocks and crypto
- Configurable time periods
- Price change calculations

### API Documentation

When the backend is running, visit http://localhost:8000/docs for interactive API documentation.

## 📝 Configuration

### Market Symbols

The application fetches data for these symbols:

**US Indices:**
- SPY - S&P 500 ETF
- QQQ - NASDAQ 100 ETF  
- DIA - Dow Jones ETF
- IWM - Russell 2000 ETF

**Cryptocurrencies:**
- BTC-USD - Bitcoin
- ETH-USD - Ethereum
- DOGE-USD - Dogecoin
- XRP-USD - XRP
- SOL-USD - Solana

### Environment Variables

- `REACT_APP_API_URL`: Frontend API URL (default: `http://localhost:8000`)

## 📊 Parameter Schema for Backtesting

### Formal Parameter Management

The repository now includes a comprehensive parameter schema (`parameters.py`) designed for algorithmic bot backtesting, optimization, and live parameter tracking. This formal schema provides:

**Key Features:**
- **Structured Parameters**: All parameters organized by category (market/timeframe, entry, exit, position sizing, trade frequency, execution, backtest constraints)
- **Optimization Ready**: Includes sensible default, minimum, and maximum values for grid search and random search optimization
- **Flexible Usage**: Can be used as configuration dictionary or dataclass
- **JSON Serialization**: Easy saving/loading of parameter sets
- **Grid Search Support**: Built-in parameter grid generation for optimization

**Parameter Categories:**

1. **Market & Timeframe**: `asset_class`, `candle_timeframe`, `session_start`, `session_end`
2. **Entry Signals**: `volatility_atr_min`, `ema_fast`, `ema_slow`, `rsi_period`, `rsi_overbought`, `rsi_oversold`, `breakout_lookback`
3. **Exit Signals**: `profit_target_mult`, `stop_loss_pct`, `trailing_stop`, `exit_on_signal`
4. **Position Sizing**: `risk_per_trade_pct`, `leverage`, `max_open_positions`
5. **Trade Frequency**: `max_trades_per_day`, `cooldown_minutes`
6. **Execution Quality**: `spread_max`, `volume_min`
7. **Backtest Constraints**: `max_drawdown_pct`, `min_sharpe_ratio`, `min_profit_factor`
8. **Walk-Forward Analysis**: `lookback` (for rolling optimization windows)

**Usage Examples:**

```python
from parameters import BacktestParameters, get_parameter_ranges, generate_random_parameters

# Get default parameters
params = BacktestParameters()
print(f"Default RSI period: {params.entry.rsi_period}")

# Create custom parameters
custom_params = BacktestParameters.from_dict({
    'ema_fast': 10,
    'ema_slow': 25,
    'risk_per_trade_pct': 1.5,
    'lookback': 180
})

# Generate random parameters for optimization
random_params = generate_random_parameters(seed=42)

# Get parameter ranges for optimization
ranges = get_parameter_ranges()
rsi_range = ranges['rsi_period']  # {'min': 5, 'max': 50, 'default': 14}

# Save/load parameters
params.save_to_json('my_strategy_params.json')
loaded_params = BacktestParameters.load_from_json('my_strategy_params.json')
```

**For Parameter Sweeps and Optimization:**

The schema is specifically designed to support:
- **Grid Search**: Use `get_parameter_grid()` for systematic parameter combinations
- **Random Search**: Use `generate_random_parameters()` for random sampling
- **Walk-Forward Analysis**: Use the `lookback` parameter for rolling optimization windows
- **Strategy Comparison**: Easy parameter set comparison and tracking
- **Live Trading**: Parameter validation and constraint checking

This formal parameter management system serves as the master parameter list for the bot, enabling consistent parameter handling across backtesting, optimization, and live trading scenarios.

## 🔧 Files Preserved

As per requirements, these files were kept unchanged:

- `.devcontainer/` - DevContainer configuration
- `.github/copilot-mcp-config.yml` - Copilot MCP configuration  
- `config.yaml` - API key management
- `quant_bot.py` - Bot learning logic (existing parameter logic preserved)

## 🎨 Frontend Preview

The React frontend will feature:

- **Dashboard Layout**: Clean, modern interface
- **Market Cards**: Summary cards for each symbol
- **Interactive Charts**: Line charts with zoom/pan
- **Real-time Updates**: Auto-refresh every 5 minutes
- **Responsive Design**: Works on desktop and mobile
- **Category Filters**: View indices, crypto, or all

## 🚧 Coming Soon

- [ ] Complete React frontend setup
- [ ] Chart.js integration
- [ ] Real-time WebSocket updates
- [ ] Mobile responsive design
- [ ] Dark mode support
- [ ] Additional technical indicators
- [ ] Live data integration for Dash dashboard

## 🔗 Links

- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Dash Dashboard**: http://localhost:8050
- **Streamlit Dashboard**: http://localhost:8501
- **Frontend**: http://localhost:3000 (when ready)

---

Built with ❤️ for clean, modern market data visualization.