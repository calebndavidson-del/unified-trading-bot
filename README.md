# Unified Trading Bot - Live Market Dashboard

A cloud-based market dashboard displaying real-time data for Apple Inc. (AAPL) stock, deployed on Modal.

## 🎯 Overview

This application provides a **live AAPL stock dashboard** that displays:

- **Live AAPL Stock Data**: Real-time price, volume, and percentage changes
- **Interactive Candlestick Charts**: 5-day price history with 1-hour intervals  
- **Volume Analysis**: Trading volume visualization
- **Real-time Updates**: Dashboard refreshes every 5 minutes with live data
- **Cloud Deployment**: Hosted on Modal for scalable, serverless operation

## 🏗️ Architecture

### Modal Deployment
- **Modal App**: Serverless deployment on Modal cloud platform
- **Dash Framework**: Interactive web dashboard with live data
- **Yahoo Finance Integration**: Real-time AAPL stock data via yfinance
- **Auto-refresh**: Live updates every 5 minutes

### Backend (FastAPI - Optional)
- **FastAPI** REST API server on port 8000
- **Market data endpoint**: `/market-data` 
- **Yahoo Finance** integration for live data
- **Async data fetching** for optimal performance

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Modal account (for cloud deployment)
- Internet connection (for market data)

### Modal Cloud Deployment

1. **Install Modal and dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup Modal authentication:**
```bash
python -m modal setup
```

3. **Deploy to Modal:**
```bash
modal deploy modal_app.py
```

The dashboard will be available at the URL provided by Modal after deployment.

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard locally
python modal_app.py
```

Dashboard will be available at: http://localhost:8050

## 📊 Dashboard Features

The dashboard includes:
- **📈 Live AAPL Overview**: Current price, price change, and percentage change cards
- **📊 Candlestick Chart**: Interactive 5-day AAPL price chart with 1-hour intervals
- **📈 Volume Chart**: AAPL trading volume visualization
- **🔄 Real-time Updates**: Automatic refresh every 5 minutes
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🔧 Modal Configuration

The `modal_app.py` file defines:
- **Modal App**: `unified-trading-bot-dashboard`
- **Image**: Debian slim with required Python packages
- **Web Endpoint**: Serves the Dash application
- **Dependencies**: modal, dash, plotly, pandas, yfinance, requests

## 📦 Dependencies

Main dependencies include:
- `modal>=0.64.0` - Serverless deployment platform
- `dash>=2.14.0` - Interactive web framework
- `yfinance>=0.2.18` - Live financial data
- `pandas>=2.0.0` - Data manipulation
- `plotly>=5.15.0` - Interactive charts
- `requests>=2.31.0` - HTTP requests

## 🎯 Live Data Source

The dashboard fetches live AAPL stock data including:
- **Time Period**: 5 days of historical data
- **Interval**: 1-hour price intervals
- **Data Points**: Open, High, Low, Close prices and Volume
- **Updates**: Real-time data refresh every 5 minutes
- **Source**: Yahoo Finance via yfinance library

## 🛠️ Development

### Project Structure
```
unified-trading-bot/
├── modal_app.py          # Main Modal application
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── backend/             # Optional FastAPI backend
│   └── main.py         # API server
└── config.yaml         # Configuration file
```

### Local Testing
```bash
# Test the Modal app locally
python modal_app.py

# Access dashboard at http://localhost:8050
```

### Backend Only (Optional)
```bash
# Start the backend server
cd backend
python main.py

# Backend available at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

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

## 🔧 Files Preserved

As per requirements, these files were kept unchanged:

- `.devcontainer/` - DevContainer configuration
- `.github/copilot-mcp-config.yml` - Copilot MCP configuration  
- `config.yaml` - API key management
- `quant_bot.py` - Bot learning logic

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