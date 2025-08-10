#!/usr/bin/env python3
"""
FastAPI backend for Unified Trading Bot
Provides market data for major US indices and cryptocurrencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified Trading Bot API",
    description="Market data API for current market dashboard",
    version="1.0.0"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Market symbols configuration
MARKET_SYMBOLS = {
    "indices": {
        "SPY": {"name": "S&P 500 ETF", "description": "SPDR S&P 500 ETF Trust"},
        "QQQ": {"name": "NASDAQ 100", "description": "Invesco QQQ Trust ETF"},
        "DIA": {"name": "Dow Jones", "description": "SPDR Dow Jones Industrial Average ETF"},
        "IWM": {"name": "Russell 2000", "description": "iShares Russell 2000 ETF"}
    },
    "crypto": {
        "BTC-USD": {"name": "Bitcoin", "description": "Bitcoin USD"},
        "ETH-USD": {"name": "Ethereum", "description": "Ethereum USD"},
        "DOGE-USD": {"name": "Dogecoin", "description": "Dogecoin USD"},
        "XRP-USD": {"name": "XRP", "description": "XRP USD"},
        "SOL-USD": {"name": "Solana", "description": "Solana USD"}
    }
}

def fetch_symbol_data(symbol: str, period: str = "5d", interval: str = "1h") -> Dict[str, Any]:
    """
    Fetch time series data for a symbol using yfinance
    
    Args:
        symbol: Stock/crypto symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        Dictionary with time series data and metadata
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            logger.warning(f"No data found for symbol: {symbol}")
            return None
        
        # Convert to JSON-friendly format
        data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "period": period,
            "interval": interval,
            "data": []
        }
        
        # Convert pandas DataFrame to list of dictionaries
        for index, row in hist.iterrows():
            data_point = {
                "datetime": index.isoformat(),
                "timestamp": int(index.timestamp() * 1000),  # JavaScript timestamp
                "open": float(row['Open']) if not pd.isna(row['Open']) else None,
                "high": float(row['High']) if not pd.isna(row['High']) else None,
                "low": float(row['Low']) if not pd.isna(row['Low']) else None,
                "close": float(row['Close']) if not pd.isna(row['Close']) else None,
                "volume": int(row['Volume']) if not pd.isna(row['Volume']) else None
            }
            data["data"].append(data_point)
        
        # Add latest price and change information
        if len(data["data"]) >= 2:
            latest = data["data"][-1]
            previous = data["data"][-2]
            
            if latest["close"] and previous["close"]:
                price_change = latest["close"] - previous["close"]
                price_change_pct = (price_change / previous["close"]) * 100
                
                data["latest_price"] = latest["close"]
                data["price_change"] = price_change
                data["price_change_pct"] = price_change_pct
                data["is_positive"] = price_change >= 0
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

async def fetch_all_symbols_async(symbols: List[str], period: str = "5d", interval: str = "1h") -> Dict[str, Any]:
    """
    Fetch data for multiple symbols asynchronously
    """
    loop = asyncio.get_event_loop()
    
    # Create tasks for each symbol
    tasks = []
    for symbol in symbols:
        task = loop.run_in_executor(None, fetch_symbol_data, symbol, period, interval)
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Organize results
    data = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.error(f"Error fetching {symbol}: {result}")
            data[symbol] = None
        else:
            data[symbol] = result
    
    return data

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Unified Trading Bot API",
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/market-data")
async def get_market_data(
    period: str = "5d",
    interval: str = "1h",
    category: str = "all"  # "all", "indices", "crypto"
):
    """
    Get market data for major US indices and popular cryptocurrencies
    
    Args:
        period: Time period (default: 5d)
        interval: Data interval (default: 1h)  
        category: Data category - "all", "indices", or "crypto" (default: all)
    
    Returns:
        JSON response with time series data for requested symbols
    """
    try:
        # Determine which symbols to fetch
        symbols_to_fetch = []
        
        if category in ["all", "indices"]:
            symbols_to_fetch.extend(list(MARKET_SYMBOLS["indices"].keys()))
        
        if category in ["all", "crypto"]:
            symbols_to_fetch.extend(list(MARKET_SYMBOLS["crypto"].keys()))
        
        if not symbols_to_fetch:
            raise HTTPException(status_code=400, detail="Invalid category. Use 'all', 'indices', or 'crypto'")
        
        logger.info(f"Fetching market data for {len(symbols_to_fetch)} symbols: {symbols_to_fetch}")
        
        # Fetch data for all symbols
        market_data = await fetch_all_symbols_async(symbols_to_fetch, period, interval)
        
        # Organize response
        response = {
            "timestamp": datetime.now().isoformat(),
            "period": period,
            "interval": interval,
            "category": category,
            "symbols": MARKET_SYMBOLS,
            "data": {}
        }
        
        # Add successful data
        successful_count = 0
        for symbol, data in market_data.items():
            if data is not None:
                response["data"][symbol] = data
                successful_count += 1
            else:
                response["data"][symbol] = {
                    "symbol": symbol,
                    "error": "Failed to fetch data",
                    "timestamp": datetime.now().isoformat()
                }
        
        response["summary"] = {
            "total_symbols": len(symbols_to_fetch),
            "successful": successful_count,
            "failed": len(symbols_to_fetch) - successful_count
        }
        
        logger.info(f"Market data fetch complete: {successful_count}/{len(symbols_to_fetch)} successful")
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Error in get_market_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/symbols")
async def get_symbols():
    """
    Get list of available symbols with metadata
    """
    return {
        "symbols": MARKET_SYMBOLS,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/market-data/{symbol}")
async def get_symbol_data(
    symbol: str,
    period: str = "5d", 
    interval: str = "1h"
):
    """
    Get market data for a specific symbol
    
    Args:
        symbol: Stock/crypto symbol (e.g., SPY, BTC-USD)
        period: Time period (default: 5d)
        interval: Data interval (default: 1h)
    """
    try:
        # Validate symbol
        all_symbols = list(MARKET_SYMBOLS["indices"].keys()) + list(MARKET_SYMBOLS["crypto"].keys())
        if symbol not in all_symbols:
            raise HTTPException(
                status_code=404, 
                detail=f"Symbol '{symbol}' not found. Available symbols: {all_symbols}"
            )
        
        logger.info(f"Fetching data for symbol: {symbol}")
        
        # Fetch data
        data = fetch_symbol_data(symbol, period, interval)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"No data available for symbol: {symbol}")
        
        return JSONResponse(content=data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")