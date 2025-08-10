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

## 🔗 Links

- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000 (when ready)

---

Built with ❤️ for clean, modern market data visualization.