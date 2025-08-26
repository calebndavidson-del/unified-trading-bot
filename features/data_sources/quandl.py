#!/usr/bin/env python3
"""
Quandl Data Source
Financial and economic data from Quandl (now part of Nasdaq Data Link)
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class QuandlAPI:
    """Quandl data fetcher for financial and economic datasets"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.source_name = "quandl"
        self.base_url = "https://www.quandl.com/api/v3"
    
    def fetch_dataset(self, database_code: str, dataset_code: str, 
                     start_date: Optional[str] = None, 
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch dataset from Quandl"""
        if not self.api_key:
            print("Warning: Quandl API key not provided, returning empty data")
            return pd.DataFrame()
        
        try:
            url = f"{self.base_url}/datasets/{database_code}/{dataset_code}/data.json"
            
            params = {
                'api_key': self.api_key,
                'order': 'asc'
            }
            
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'dataset_data' not in data:
                return pd.DataFrame()
            
            dataset_data = data['dataset_data']
            
            # Extract data and column names
            column_names = dataset_data.get('column_names', [])
            data_rows = dataset_data.get('data', [])
            
            if not data_rows or not column_names:
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=column_names)
            
            # Convert Date column to datetime and set as index
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            # Convert numeric columns
            for col in df.columns:
                if col != 'Date':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add metadata
            df['database_code'] = database_code
            df['dataset_code'] = dataset_code
            df['source'] = self.source_name
            df['fetch_timestamp'] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"Error fetching Quandl dataset {database_code}/{dataset_code}: {e}")
            return pd.DataFrame()
    
    def fetch_wiki_prices(self, symbol: str, start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch stock prices from WIKI database (deprecated but still useful for historical data)"""
        return self.fetch_dataset("WIKI", symbol, start_date, end_date)
    
    def fetch_fed_data(self, indicator: str, start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch Federal Reserve economic data (FRED)"""
        return self.fetch_dataset("FRED", indicator, start_date, end_date)
    
    def fetch_futures_data(self, contract: str, start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch futures contract data"""
        return self.fetch_dataset("CHRIS", contract, start_date, end_date)
    
    def fetch_commodity_data(self, commodity: str, start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch commodity prices"""
        # Common commodity codes
        commodity_map = {
            'gold': 'LBMA/GOLD',
            'silver': 'LBMA/SILVER',
            'oil_brent': 'EIA/PET_RBRTE_D',
            'oil_wti': 'EIA/PET_RWTC_D',
            'gas': 'EIA/NG_RNGWHHD_D',
            'copper': 'LME/PR_CU',
            'aluminum': 'LME/PR_AL',
            'zinc': 'LME/PR_ZN'
        }
        
        if commodity.lower() in commodity_map:
            code_parts = commodity_map[commodity.lower()].split('/')
            return self.fetch_dataset(code_parts[0], code_parts[1], start_date, end_date)
        else:
            print(f"Commodity {commodity} not found in predefined mappings")
            return pd.DataFrame()
    
    def fetch_economic_indicators(self, start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetch multiple key economic indicators"""
        indicators = {
            'gdp': 'GDP',
            'unemployment': 'UNRATE',
            'inflation': 'CPIAUCSL',
            'fed_funds_rate': 'DFF',
            'treasury_10y': 'DGS10',
            'treasury_2y': 'DGS2',
            'vix': 'VIXCLS',
            'dollar_index': 'DTWEXM'
        }
        
        data = {}
        for name, code in indicators.items():
            df = self.fetch_fed_data(code, start_date, end_date)
            if not df.empty:
                data[name] = df
        
        return data
    
    def fetch_sector_data(self, sector: str, start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch sector-specific data"""
        # Map sectors to common Quandl datasets
        sector_map = {
            'technology': 'NASDAQOMX/COMP',
            'finance': 'NASDAQOMX/BANK',
            'healthcare': 'NASDAQOMX/HLTH',
            'energy': 'NASDAQOMX/NRG',
            'utilities': 'NASDAQOMX/UTIL',
            'consumer': 'NASDAQOMX/COND',
            'industrials': 'NASDAQOMX/INDU',
            'materials': 'NASDAQOMX/MATR',
            'telecom': 'NASDAQOMX/TELE'
        }
        
        if sector.lower() in sector_map:
            code_parts = sector_map[sector.lower()].split('/')
            return self.fetch_dataset(code_parts[0], code_parts[1], start_date, end_date)
        else:
            print(f"Sector {sector} not found in predefined mappings")
            return pd.DataFrame()
    
    def fetch_currency_data(self, currency_pair: str, start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch currency exchange rates"""
        # Map common currency pairs
        currency_map = {
            'eurusd': 'ECB/EURUSD',
            'gbpusd': 'BOE/XUDLGBD',
            'usdjpy': 'BOJ/USD_JPY',
            'audusd': 'RBA/FXRAUD',
            'usdcad': 'BOC/FXUSDCAD',
            'usdchf': 'SNB/USDCHF'
        }
        
        pair_key = currency_pair.lower().replace('/', '').replace('_', '')
        
        if pair_key in currency_map:
            code_parts = currency_map[pair_key].split('/')
            return self.fetch_dataset(code_parts[0], code_parts[1], start_date, end_date)
        else:
            print(f"Currency pair {currency_pair} not found in predefined mappings")
            return pd.DataFrame()
    
    def search_datasets(self, query: str, per_page: int = 10) -> List[Dict]:
        """Search for datasets on Quandl"""
        if not self.api_key:
            return []
        
        try:
            url = f"{self.base_url}/datasets.json"
            params = {
                'api_key': self.api_key,
                'query': query,
                'per_page': per_page
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'datasets' not in data:
                return []
            
            # Extract relevant information
            results = []
            for dataset in data['datasets']:
                result = {
                    'id': dataset.get('id'),
                    'dataset_code': dataset.get('dataset_code'),
                    'database_code': dataset.get('database_code'),
                    'name': dataset.get('name'),
                    'description': dataset.get('description'),
                    'refreshed_at': dataset.get('refreshed_at'),
                    'newest_available_date': dataset.get('newest_available_date'),
                    'oldest_available_date': dataset.get('oldest_available_date'),
                    'frequency': dataset.get('frequency'),
                    'type': dataset.get('type')
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error searching Quandl datasets: {e}")
            return []
    
    def get_popular_datasets(self) -> Dict[str, List[str]]:
        """Get list of popular/useful datasets by category"""
        return {
            'stocks': [
                'WIKI/AAPL', 'WIKI/GOOGL', 'WIKI/MSFT', 'WIKI/AMZN',
                'WIKI/TSLA', 'WIKI/META', 'WIKI/NVDA'
            ],
            'indices': [
                'NASDAQOMX/COMP', 'YAHOO/INDEX_GSPC', 'YAHOO/INDEX_DJI',
                'YAHOO/INDEX_VIX', 'YAHOO/INDEX_RUT'
            ],
            'economic_indicators': [
                'FRED/GDP', 'FRED/UNRATE', 'FRED/CPIAUCSL', 'FRED/DFF',
                'FRED/DGS10', 'FRED/DGS2', 'FRED/VIXCLS'
            ],
            'commodities': [
                'LBMA/GOLD', 'LBMA/SILVER', 'EIA/PET_RBRTE_D',
                'EIA/PET_RWTC_D', 'EIA/NG_RNGWHHD_D'
            ],
            'currencies': [
                'ECB/EURUSD', 'BOE/XUDLGBD', 'BOJ/USD_JPY',
                'RBA/FXRAUD', 'BOC/FXUSDCAD'
            ],
            'crypto': [
                'BCHARTS/BITSTAMPUSD', 'BCHARTS/COINBASEUSD',
                'BITFINEX/BTCUSD', 'BITFINEX/ETHUSD'
            ]
        }


if __name__ == "__main__":
    # Example usage (requires API key)
    api = QuandlAPI()
    
    print("Quandl API initialized")
    print("Set api_key parameter to use live data")
    
    # Show popular datasets
    datasets = api.get_popular_datasets()
    print("\nPopular dataset categories:")
    for category, codes in datasets.items():
        print(f"  {category}: {len(codes)} datasets")