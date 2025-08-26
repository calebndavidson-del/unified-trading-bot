#!/usr/bin/env python3
"""
Yahoo Finance Data Source
Enhanced Yahoo Finance integration with error handling and data validation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class YahooFinanceAPI:
    """Enhanced Yahoo Finance data fetcher with validation and error handling"""
    
    def __init__(self):
        self.source_name = "yahoo_finance"
        self.supported_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', 
                                  '1h', '1d', '5d', '1wk', '1mo', '3mo']
    
    def fetch_market_data(self, symbol: str, period: str = "1y", 
                         interval: str = "1d") -> pd.DataFrame:
        """Fetch market data with comprehensive error handling"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"Warning: No data returned for {symbol}")
                return pd.DataFrame()
                
            # Add metadata
            data['symbol'] = symbol
            data['source'] = self.source_name
            data['fetch_timestamp'] = datetime.now()
            
            # Validate data quality
            data = self._validate_ohlcv_data(data)
            
            return data
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_info(self, symbol: str) -> Dict:
        """Fetch company/asset information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant fields
            clean_info = {
                'symbol': symbol,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', None),
                'country': info.get('country', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'long_name': info.get('longName', symbol),
                'source': self.source_name
            }
            
            return clean_info
            
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")
            return {'symbol': symbol, 'source': self.source_name}
    
    def fetch_dividends(self, symbol: str) -> pd.DataFrame:
        """Fetch dividend history"""
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            
            if dividends.empty:
                return pd.DataFrame()
                
            div_df = dividends.to_frame('dividend')
            div_df['symbol'] = symbol
            div_df['source'] = self.source_name
            
            return div_df
            
        except Exception as e:
            print(f"Error fetching dividends for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_splits(self, symbol: str) -> pd.DataFrame:
        """Fetch stock split history"""
        try:
            ticker = yf.Ticker(symbol)
            splits = ticker.splits
            
            if splits.empty:
                return pd.DataFrame()
                
            split_df = splits.to_frame('split_ratio')
            split_df['symbol'] = symbol
            split_df['source'] = self.source_name
            
            return split_df
            
        except Exception as e:
            print(f"Error fetching splits for {symbol}: {e}")
            return pd.DataFrame()
    
    def _validate_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV data quality"""
        if data.empty:
            return data
            
        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in data.columns:
                data.loc[data[col] <= 0, col] = np.nan
        
        # Check for volume
        if 'Volume' in data.columns:
            data.loc[data['Volume'] < 0, 'Volume'] = 0
        
        # Validate high >= low, high >= open, high >= close
        if all(col in data.columns for col in price_cols):
            invalid_high = (data['High'] < data['Low']) | \
                          (data['High'] < data['Open']) | \
                          (data['High'] < data['Close'])
            
            # Mark invalid rows
            data.loc[invalid_high, price_cols] = np.nan
        
        return data
    
    def get_supported_symbols(self, asset_type: str = "all") -> List[str]:
        """Get list of commonly supported symbols by category"""
        symbols = {
            'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 
                      'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS'],
            'etfs': ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 
                    'TLT', 'GLD', 'SLV', 'XLF', 'XLK', 'XLE', 'XLV'],
            'crypto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'DOGE-USD', 
                      'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'ATOM-USD', 'LTC-USD'],
            'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X']
        }
        
        if asset_type == "all":
            return [symbol for category in symbols.values() for symbol in category]
        
        return symbols.get(asset_type, [])


if __name__ == "__main__":
    # Example usage
    api = YahooFinanceAPI()
    
    # Test data fetching
    data = api.fetch_market_data("AAPL", period="1mo")
    print(f"Fetched {len(data)} rows for AAPL")
    
    # Test info fetching
    info = api.fetch_info("AAPL")
    print(f"Company: {info.get('long_name')}, Sector: {info.get('sector')}")