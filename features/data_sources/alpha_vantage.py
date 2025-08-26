#!/usr/bin/env python3
"""
Alpha Vantage Data Source
Comprehensive market data, technical indicators, and fundamental data from Alpha Vantage
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class AlphaVantageAPI:
    """Alpha Vantage data fetcher for stocks, forex, crypto, and technical indicators"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.source_name = "alpha_vantage"
        self.base_url = "https://www.alphavantage.co/query"
        
        # Rate limiting - Alpha Vantage has strict limits
        self.requests_per_minute = 5
        self.requests_per_day = 500
    
    def fetch_market_data(self, symbol: str, period: str = "1y", 
                         interval: str = "1d", outputsize: str = "compact") -> pd.DataFrame:
        """Fetch OHLCV market data"""
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided, returning empty data")
            return pd.DataFrame()
        
        try:
            # Map interval to Alpha Vantage format
            function_map = {
                "1min": "TIME_SERIES_INTRADAY",
                "5min": "TIME_SERIES_INTRADAY", 
                "15min": "TIME_SERIES_INTRADAY",
                "30min": "TIME_SERIES_INTRADAY",
                "60min": "TIME_SERIES_INTRADAY",
                "1d": "TIME_SERIES_DAILY",
                "1wk": "TIME_SERIES_WEEKLY",
                "1mo": "TIME_SERIES_MONTHLY"
            }
            
            function = function_map.get(interval, "TIME_SERIES_DAILY")
            
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': outputsize,
                'datatype': 'json'
            }
            
            # Add interval parameter for intraday data
            if function == "TIME_SERIES_INTRADAY":
                params['interval'] = interval
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data:
                print(f"Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()
            
            if 'Information' in data:
                print(f"Alpha Vantage info: {data['Information']}")
                return pd.DataFrame()
            
            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if not time_series_key:
                print(f"No time series data found for {symbol}")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Standardize column names
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Add metadata
            df['symbol'] = symbol
            df['source'] = self.source_name
            df['fetch_timestamp'] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_company_overview(self, symbol: str) -> Dict:
        """Fetch comprehensive company fundamental data"""
        if not self.api_key:
            return {'symbol': symbol, 'source': self.source_name}
        
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Symbol' not in data:
                return {'symbol': symbol, 'source': self.source_name}
            
            # Extract and clean key metrics
            overview = {
                'symbol': symbol,
                'name': data.get('Name', ''),
                'description': data.get('Description', ''),
                'cik': data.get('CIK', ''),
                'exchange': data.get('Exchange', ''),
                'currency': data.get('Currency', ''),
                'country': data.get('Country', ''),
                'sector': data.get('Sector', ''),
                'industry': data.get('Industry', ''),
                'address': data.get('Address', ''),
                'fiscal_year_end': data.get('FiscalYearEnd', ''),
                'latest_quarter': data.get('LatestQuarter', ''),
                'market_cap': self._safe_float(data.get('MarketCapitalization')),
                'ebitda': self._safe_float(data.get('EBITDA')),
                'pe_ratio': self._safe_float(data.get('PERatio')),
                'peg_ratio': self._safe_float(data.get('PEGRatio')),
                'book_value': self._safe_float(data.get('BookValue')),
                'dividend_per_share': self._safe_float(data.get('DividendPerShare')),
                'dividend_yield': self._safe_float(data.get('DividendYield')),
                'eps': self._safe_float(data.get('EPS')),
                'revenue_per_share_ttm': self._safe_float(data.get('RevenuePerShareTTM')),
                'profit_margin': self._safe_float(data.get('ProfitMargin')),
                'operating_margin_ttm': self._safe_float(data.get('OperatingMarginTTM')),
                'return_on_assets_ttm': self._safe_float(data.get('ReturnOnAssetsTTM')),
                'return_on_equity_ttm': self._safe_float(data.get('ReturnOnEquityTTM')),
                'revenue_ttm': self._safe_float(data.get('RevenueTTM')),
                'gross_profit_ttm': self._safe_float(data.get('GrossProfitTTM')),
                'diluted_eps_ttm': self._safe_float(data.get('DilutedEPSTTM')),
                'quarterly_earnings_growth_yoy': self._safe_float(data.get('QuarterlyEarningsGrowthYOY')),
                'quarterly_revenue_growth_yoy': self._safe_float(data.get('QuarterlyRevenueGrowthYOY')),
                'analyst_target_price': self._safe_float(data.get('AnalystTargetPrice')),
                'trailing_pe': self._safe_float(data.get('TrailingPE')),
                'forward_pe': self._safe_float(data.get('ForwardPE')),
                'price_to_sales_ratio_ttm': self._safe_float(data.get('PriceToSalesRatioTTM')),
                'price_to_book_ratio': self._safe_float(data.get('PriceToBookRatio')),
                'ev_to_revenue': self._safe_float(data.get('EVToRevenue')),
                'ev_to_ebitda': self._safe_float(data.get('EVToEBITDA')),
                'beta': self._safe_float(data.get('Beta')),
                'week_52_high': self._safe_float(data.get('52WeekHigh')),
                'week_52_low': self._safe_float(data.get('52WeekLow')),
                'moving_average_50': self._safe_float(data.get('50DayMovingAverage')),
                'moving_average_200': self._safe_float(data.get('200DayMovingAverage')),
                'shares_outstanding': self._safe_float(data.get('SharesOutstanding')),
                'dividend_date': data.get('DividendDate', ''),
                'ex_dividend_date': data.get('ExDividendDate', ''),
                'source': self.source_name
            }
            
            return overview
            
        except Exception as e:
            print(f"Error fetching company overview for {symbol}: {e}")
            return {'symbol': symbol, 'source': self.source_name}
    
    def fetch_technical_indicator(self, symbol: str, indicator: str, 
                                 **kwargs) -> pd.DataFrame:
        """Fetch technical indicators"""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            # Map common indicators to Alpha Vantage functions
            indicator_map = {
                'sma': 'SMA',
                'ema': 'EMA',
                'rsi': 'RSI',
                'macd': 'MACD',
                'stoch': 'STOCH',
                'adx': 'ADX',
                'cci': 'CCI',
                'aroon': 'AROON',
                'bbands': 'BBANDS',
                'ad': 'AD',
                'obv': 'OBV'
            }
            
            function = indicator_map.get(indicator.lower())
            if not function:
                print(f"Indicator {indicator} not supported")
                return pd.DataFrame()
            
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'datatype': 'json'
            }
            
            # Add indicator-specific parameters
            params.update(kwargs)
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data:
                print(f"Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()
            
            # Extract technical indicator data
            tech_key = None
            for key in data.keys():
                if 'Technical Analysis' in key:
                    tech_key = key
                    break
            
            if not tech_key:
                return pd.DataFrame()
            
            tech_data = data[tech_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(tech_data, orient='index')
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Add metadata
            df['symbol'] = symbol
            df['indicator'] = indicator
            df['source'] = self.source_name
            
            return df
            
        except Exception as e:
            print(f"Error fetching {indicator} for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_forex_data(self, from_symbol: str, to_symbol: str) -> pd.DataFrame:
        """Fetch forex exchange rates"""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            params = {
                'function': 'FX_DAILY',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Time Series FX (Daily)' not in data:
                return pd.DataFrame()
            
            time_series = data['Time Series FX (Daily)']
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Standardize column names
            df.columns = ['Open', 'High', 'Low', 'Close']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Add metadata
            df['from_symbol'] = from_symbol
            df['to_symbol'] = to_symbol
            df['pair'] = f"{from_symbol}/{to_symbol}"
            df['source'] = self.source_name
            
            return df
            
        except Exception as e:
            print(f"Error fetching forex data for {from_symbol}/{to_symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_economic_indicators(self, indicator: str) -> pd.DataFrame:
        """Fetch economic indicators (GDP, inflation, etc.)"""
        if not self.api_key:
            return pd.DataFrame()
        
        try:
            # Map common economic indicators
            indicator_map = {
                'real_gdp': 'REAL_GDP',
                'real_gdp_per_capita': 'REAL_GDP_PER_CAPITA',
                'federal_funds_rate': 'FEDERAL_FUNDS_RATE',
                'cpi': 'CPI',
                'inflation': 'INFLATION',
                'retail_sales': 'RETAIL_SALES',
                'durables': 'DURABLES',
                'unemployment': 'UNEMPLOYMENT',
                'nonfarm_payroll': 'NONFARM_PAYROLL'
            }
            
            function = indicator_map.get(indicator.lower())
            if not function:
                print(f"Economic indicator {indicator} not supported")
                return pd.DataFrame()
            
            params = {
                'function': function,
                'apikey': self.api_key,
                'datatype': 'json'
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            
            # Convert date and value columns
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Set date as index
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Add metadata
            df['indicator'] = indicator
            df['source'] = self.source_name
            
            return df
            
        except Exception as e:
            print(f"Error fetching economic indicator {indicator}: {e}")
            return pd.DataFrame()
    
    def _safe_float(self, value: Union[str, float, None]) -> Optional[float]:
        """Safely convert value to float"""
        if value is None or value == 'None' or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None


if __name__ == "__main__":
    # Example usage (requires API key)
    api = AlphaVantageAPI()
    
    print("Alpha Vantage API initialized")
    print("Set api_key parameter to use live data")