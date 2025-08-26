#!/usr/bin/env python3
"""
IEX Cloud Data Source
Professional-grade market data from IEX Cloud with comprehensive features
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class IEXCloudAPI:
    """IEX Cloud data fetcher for stocks, options, and market data"""
    
    def __init__(self, api_key: Optional[str] = None, sandbox: bool = True):
        self.api_key = api_key
        self.sandbox = sandbox
        self.source_name = "iex_cloud"
        
        # Set base URL based on sandbox mode
        self.base_url = "https://sandbox.iexapis.com/stable" if sandbox else \
                       "https://cloud.iexapis.com/stable"
        
        # Rate limiting parameters
        self.max_symbols_per_request = 100
    
    def fetch_market_data(self, symbol: str, period: str = "1y", 
                         interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV market data"""
        if not self.api_key:
            print("Warning: IEX Cloud API key not provided, returning empty data")
            return pd.DataFrame()
        
        try:
            # Map periods to IEX range format
            range_map = {
                "1d": "1d", "5d": "5d", "1mo": "1m", "3mo": "3m", 
                "6mo": "6m", "1y": "1y", "2y": "2y", "5y": "5y"
            }
            
            iex_range = range_map.get(period, "1y")
            
            url = f"{self.base_url}/stock/{symbol}/chart/{iex_range}"
            params = {"token": self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Standardize column names
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume'
            })
            
            # Convert date to datetime index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Add metadata
            df['symbol'] = symbol
            df['source'] = self.source_name
            df['fetch_timestamp'] = datetime.now()
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume', 'symbol', 'source', 'fetch_timestamp']]
            
        except Exception as e:
            print(f"Error fetching IEX Cloud data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_company_info(self, symbol: str) -> Dict:
        """Fetch comprehensive company information"""
        if not self.api_key:
            return {'symbol': symbol, 'source': self.source_name}
        
        try:
            url = f"{self.base_url}/stock/{symbol}/company"
            params = {"token": self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract and standardize fields
            clean_info = {
                'symbol': symbol,
                'company_name': data.get('companyName', ''),
                'sector': data.get('sector', 'Unknown'),
                'industry': data.get('industry', 'Unknown'),
                'website': data.get('website', ''),
                'description': data.get('description', ''),
                'exchange': data.get('exchange', 'Unknown'),
                'ceo': data.get('CEO', ''),
                'employees': data.get('employees', None),
                'country': data.get('country', 'Unknown'),
                'source': self.source_name
            }
            
            return clean_info
            
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {e}")
            return {'symbol': symbol, 'source': self.source_name}
    
    def fetch_key_stats(self, symbol: str) -> Dict:
        """Fetch key financial statistics"""
        if not self.api_key:
            return {'symbol': symbol, 'source': self.source_name}
        
        try:
            url = f"{self.base_url}/stock/{symbol}/stats"
            params = {"token": self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract key metrics
            stats = {
                'symbol': symbol,
                'market_cap': data.get('marketcap', None),
                'pe_ratio': data.get('peRatio', None),
                'forward_pe': data.get('forwardPERatio', None),
                'peg_ratio': data.get('pegRatio', None),
                'price_to_book': data.get('priceToBook', None),
                'price_to_sales': data.get('priceToSales', None),
                'enterprise_value': data.get('enterpriseValue', None),
                'profit_margin': data.get('profitMargin', None),
                'operating_margin': data.get('operatingMargin', None),
                'return_on_assets': data.get('returnOnAssets', None),
                'return_on_equity': data.get('returnOnEquity', None),
                'revenue': data.get('revenue', None),
                'gross_profit': data.get('grossProfit', None),
                'cash': data.get('totalCash', None),
                'debt': data.get('totalDebt', None),
                'revenue_per_share': data.get('revenuePerShare', None),
                'revenue_per_employee': data.get('revenuePerEmployee', None),
                'debt_to_equity': data.get('debtToEquity', None),
                'current_ratio': data.get('currentRatio', None),
                'dividend_yield': data.get('dividendYield', None),
                'ex_dividend_date': data.get('exDividendDate', None),
                'latest_eps': data.get('latestEPS', None),
                'latest_eps_date': data.get('latestEPSDate', None),
                'shares_outstanding': data.get('sharesOutstanding', None),
                'float_shares': data.get('float', None),
                'avg_30_volume': data.get('avg30Volume', None),
                'avg_10_volume': data.get('avg10Volume', None),
                'week_52_high': data.get('week52high', None),
                'week_52_low': data.get('week52low', None),
                'week_52_change': data.get('week52change', None),
                'source': self.source_name
            }
            
            return stats
            
        except Exception as e:
            print(f"Error fetching key stats for {symbol}: {e}")
            return {'symbol': symbol, 'source': self.source_name}
    
    def fetch_news(self, symbol: str, last_n: int = 10) -> pd.DataFrame:
        """Fetch recent news for a symbol"""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            url = f"{self.base_url}/stock/{symbol}/news/last/{last_n}"
            params = {"token": self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            news_df = pd.DataFrame(data)
            
            # Parse datetime
            news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='ms')
            
            # Add metadata
            news_df['symbol'] = symbol
            news_df['source'] = self.source_name
            
            return news_df[['datetime', 'headline', 'summary', 'url', 'image', 
                           'lang', 'hasPaywall', 'symbol', 'source']]
            
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_social_sentiment(self, symbol: str) -> Dict:
        """Fetch social media sentiment data"""
        if not self.api_key:
            return {'symbol': symbol, 'source': self.source_name}
        
        try:
            url = f"{self.base_url}/stock/{symbol}/sentiment"
            params = {"token": self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            sentiment = {
                'symbol': symbol,
                'sentiment': data.get('sentiment', None),
                'total_scanned': data.get('totalScanned', None),
                'positive': data.get('positive', None),
                'negative': data.get('negative', None),
                'source': self.source_name
            }
            
            return sentiment
            
        except Exception as e:
            print(f"Error fetching sentiment for {symbol}: {e}")
            return {'symbol': symbol, 'source': self.source_name}
    
    def fetch_batch_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch real-time quotes for multiple symbols"""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            # Split into batches if needed
            all_quotes = []
            
            for i in range(0, len(symbols), self.max_symbols_per_request):
                batch_symbols = symbols[i:i + self.max_symbols_per_request]
                symbols_str = ",".join(batch_symbols)
                
                url = f"{self.base_url}/stock/market/batch"
                params = {
                    "symbols": symbols_str,
                    "types": "quote",
                    "token": self.api_key
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract quotes
                for symbol, info in data.items():
                    if 'quote' in info:
                        quote = info['quote']
                        quote['symbol'] = symbol
                        all_quotes.append(quote)
            
            if not all_quotes:
                return pd.DataFrame()
            
            # Convert to DataFrame
            quotes_df = pd.DataFrame(all_quotes)
            quotes_df['source'] = self.source_name
            quotes_df['fetch_timestamp'] = datetime.now()
            
            return quotes_df
            
        except Exception as e:
            print(f"Error fetching batch quotes: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage (requires API key)
    api = IEXCloudAPI(sandbox=True)
    
    print("IEX Cloud API initialized in sandbox mode")
    print("Set api_key parameter to use live data")