#!/usr/bin/env python3
"""
Finnhub Data Source
Real-time financial data, news, and alternative datasets from Finnhub
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class FinnhubAPI:
    """Finnhub data fetcher for stocks, forex, crypto, and alternative data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.source_name = "finnhub"
        self.base_url = "https://finnhub.io/api/v1"
        
        # Rate limiting
        self.requests_per_minute = 60  # Free tier
    
    def fetch_market_data(self, symbol: str, period: str = "1y", 
                         interval: str = "D") -> pd.DataFrame:
        """Fetch OHLCV market data"""
        if not self.api_key:
            print("Warning: Finnhub API key not provided, returning empty data")
            return pd.DataFrame()
        
        try:
            # Calculate date range
            end_date = datetime.now()
            
            if period == "1d":
                start_date = end_date - timedelta(days=1)
                resolution = "1"
            elif period == "1w":
                start_date = end_date - timedelta(weeks=1)
                resolution = "D"
            elif period == "1mo":
                start_date = end_date - timedelta(days=30)
                resolution = "D"
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
                resolution = "D"
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
                resolution = "D"
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
                resolution = "D"
            elif period == "2y":
                start_date = end_date - timedelta(days=730)
                resolution = "W"
            else:
                start_date = end_date - timedelta(days=365)
                resolution = "D"
            
            # Convert to Unix timestamps
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            url = f"{self.base_url}/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'from': start_timestamp,
                'to': end_timestamp,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors
            if data.get('s') != 'ok':
                print(f"Finnhub error for {symbol}: {data.get('s', 'Unknown error')}")
                return pd.DataFrame()
            
            # Extract OHLCV data
            timestamps = data.get('t', [])
            opens = data.get('o', [])
            highs = data.get('h', [])
            lows = data.get('l', [])
            closes = data.get('c', [])
            volumes = data.get('v', [])
            
            if not timestamps:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Volume': volumes
            })
            
            # Convert timestamps to datetime index
            df.index = pd.to_datetime(timestamps, unit='s')
            df.sort_index(inplace=True)
            
            # Add metadata
            df['symbol'] = symbol
            df['source'] = self.source_name
            df['fetch_timestamp'] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"Error fetching Finnhub data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_company_profile(self, symbol: str) -> Dict:
        """Fetch company profile and basic information"""
        if not self.api_key:
            return {'symbol': symbol, 'source': self.source_name}
        
        try:
            url = f"{self.base_url}/stock/profile2"
            params = {
                'symbol': symbol,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            profile = {
                'symbol': symbol,
                'name': data.get('name', ''),
                'country': data.get('country', ''),
                'currency': data.get('currency', ''),
                'exchange': data.get('exchange', ''),
                'ipo': data.get('ipo', ''),
                'market_capitalization': data.get('marketCapitalization', None),
                'shares_outstanding': data.get('shareOutstanding', None),
                'ticker': data.get('ticker', ''),
                'weburl': data.get('weburl', ''),
                'logo': data.get('logo', ''),
                'finnhub_industry': data.get('finnhubIndustry', ''),
                'source': self.source_name
            }
            
            return profile
            
        except Exception as e:
            print(f"Error fetching company profile for {symbol}: {e}")
            return {'symbol': symbol, 'source': self.source_name}
    
    def fetch_company_news(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """Fetch company-specific news"""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            url = f"{self.base_url}/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            news_df = pd.DataFrame(data)
            
            # Convert datetime
            news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='s')
            
            # Add metadata
            news_df['symbol'] = symbol
            news_df['source'] = self.source_name
            
            return news_df[['datetime', 'headline', 'summary', 'url', 'image', 
                           'category', 'source', 'symbol']]
            
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_market_news(self, category: str = "general", min_id: int = 0) -> pd.DataFrame:
        """Fetch general market news"""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            url = f"{self.base_url}/news"
            params = {
                'category': category,
                'minId': min_id,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            news_df = pd.DataFrame(data)
            
            # Convert datetime
            news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='s')
            
            # Add metadata
            news_df['category'] = category
            news_df['source'] = self.source_name
            
            return news_df
            
        except Exception as e:
            print(f"Error fetching market news: {e}")
            return pd.DataFrame()
    
    def fetch_sentiment(self, symbol: str) -> Dict:
        """Fetch social media sentiment data"""
        if not self.api_key:
            return {'symbol': symbol, 'source': self.source_name}
        
        try:
            url = f"{self.base_url}/stock/social-sentiment"
            params = {
                'symbol': symbol,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'data' not in data:
                return {'symbol': symbol, 'source': self.source_name}
            
            # Aggregate sentiment data
            sentiment_data = data['data']
            
            if sentiment_data:
                # Calculate aggregated metrics
                total_mention = sum(item.get('mention', 0) for item in sentiment_data)
                total_positive_mention = sum(item.get('positiveMention', 0) for item in sentiment_data)
                total_negative_mention = sum(item.get('negativeMention', 0) for item in sentiment_data)
                
                sentiment_score = 0
                if total_mention > 0:
                    sentiment_score = (total_positive_mention - total_negative_mention) / total_mention
                
                sentiment = {
                    'symbol': symbol,
                    'total_mention': total_mention,
                    'positive_mention': total_positive_mention,
                    'negative_mention': total_negative_mention,
                    'sentiment_score': sentiment_score,
                    'data_points': len(sentiment_data),
                    'source': self.source_name
                }
            else:
                sentiment = {'symbol': symbol, 'source': self.source_name}
            
            return sentiment
            
        except Exception as e:
            print(f"Error fetching sentiment for {symbol}: {e}")
            return {'symbol': symbol, 'source': self.source_name}
    
    def fetch_insider_trading(self, symbol: str) -> pd.DataFrame:
        """Fetch insider trading data"""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            url = f"{self.base_url}/stock/insider-transactions"
            params = {
                'symbol': symbol,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'data' not in data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            insider_df = pd.DataFrame(data['data'])
            
            if insider_df.empty:
                return pd.DataFrame()
            
            # Convert date columns
            if 'transactionDate' in insider_df.columns:
                insider_df['transactionDate'] = pd.to_datetime(insider_df['transactionDate'])
            
            # Add metadata
            insider_df['symbol'] = symbol
            insider_df['source'] = self.source_name
            
            return insider_df
            
        except Exception as e:
            print(f"Error fetching insider trading for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_earnings_calendar(self, days_ahead: int = 30) -> pd.DataFrame:
        """Fetch upcoming earnings calendar"""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            # Calculate date range
            start_date = datetime.now()
            end_date = start_date + timedelta(days=days_ahead)
            
            url = f"{self.base_url}/calendar/earnings"
            params = {
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'earningsCalendar' not in data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            earnings_df = pd.DataFrame(data['earningsCalendar'])
            
            if earnings_df.empty:
                return pd.DataFrame()
            
            # Convert date
            if 'date' in earnings_df.columns:
                earnings_df['date'] = pd.to_datetime(earnings_df['date'])
            
            # Add metadata
            earnings_df['source'] = self.source_name
            earnings_df['fetch_timestamp'] = datetime.now()
            
            return earnings_df
            
        except Exception as e:
            print(f"Error fetching earnings calendar: {e}")
            return pd.DataFrame()
    
    def fetch_economic_calendar(self) -> pd.DataFrame:
        """Fetch economic events calendar"""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            url = f"{self.base_url}/calendar/economic"
            params = {
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or 'economicCalendar' not in data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            econ_df = pd.DataFrame(data['economicCalendar'])
            
            if econ_df.empty:
                return pd.DataFrame()
            
            # Convert datetime
            if 'time' in econ_df.columns:
                econ_df['datetime'] = pd.to_datetime(econ_df['time'], unit='s')
            
            # Add metadata
            econ_df['source'] = self.source_name
            econ_df['fetch_timestamp'] = datetime.now()
            
            return econ_df
            
        except Exception as e:
            print(f"Error fetching economic calendar: {e}")
            return pd.DataFrame()
    
    def fetch_crypto_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch cryptocurrency data"""
        # Finnhub uses format like BINANCE:BTCUSDT
        if ':' not in symbol:
            # Convert common formats
            if symbol.upper().endswith('USD'):
                crypto_symbol = f"BINANCE:{symbol.upper().replace('-', '')}T"
            else:
                crypto_symbol = f"BINANCE:{symbol.upper()}USDT"
        else:
            crypto_symbol = symbol
        
        return self.fetch_market_data(crypto_symbol, period)


if __name__ == "__main__":
    # Example usage (requires API key)
    api = FinnhubAPI()
    
    print("Finnhub API initialized")
    print("Set api_key parameter to use live data")