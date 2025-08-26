#!/usr/bin/env python3
"""
Binance Data Source
Cryptocurrency market data from Binance API
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class BinanceAPI:
    """Binance data fetcher for cryptocurrency market data"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.source_name = "binance"
        self.base_url = "https://api.binance.com/api/v3"
        
        # Rate limiting
        self.weight_limit = 1200  # Per minute
        self.requests_per_second = 10
    
    def fetch_market_data(self, symbol: str, period: str = "1y", 
                         interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV market data for cryptocurrency pairs"""
        try:
            # Clean symbol format (remove common separators)
            clean_symbol = symbol.upper().replace('-', '').replace('_', '').replace('/', '')
            
            # If symbol doesn't end with USDT, BTC, ETH, try adding USDT
            if not any(clean_symbol.endswith(base) for base in ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']):
                if clean_symbol.endswith('USD'):
                    clean_symbol = clean_symbol.replace('USD', 'USDT')
                else:
                    clean_symbol += 'USDT'
            
            # Map intervals to Binance format
            interval_map = {
                '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
                '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
            }
            
            binance_interval = interval_map.get(interval, '1d')
            
            # Calculate number of periods needed
            period_map = {
                '1d': 1, '7d': 7, '1mo': 30, '3mo': 90, 
                '6mo': 180, '1y': 365, '2y': 730
            }
            
            days = period_map.get(period, 365)
            
            # Calculate limit based on interval
            if binance_interval in ['1m', '3m', '5m']:
                limit = min(1000, days * 24 * 60 // int(binance_interval[:-1]))
            elif binance_interval in ['15m', '30m']:
                limit = min(1000, days * 24 * 60 // int(binance_interval[:-1]))
            elif binance_interval in ['1h', '2h', '4h', '6h', '8h', '12h']:
                limit = min(1000, days * 24 // int(binance_interval[:-1]))
            elif binance_interval == '1d':
                limit = min(1000, days)
            elif binance_interval == '1w':
                limit = min(1000, days // 7)
            else:
                limit = 500
            
            url = f"{self.base_url}/klines"
            params = {
                'symbol': clean_symbol,
                'interval': binance_interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                print(f"No data returned for {clean_symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            columns = [
                'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert price columns to float
            price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in price_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Keep only OHLCV columns
            df = df[price_cols]
            
            # Add metadata
            df['symbol'] = symbol
            df['binance_symbol'] = clean_symbol
            df['source'] = self.source_name
            df['fetch_timestamp'] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"Error fetching Binance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_symbol_info(self, symbol: str = None) -> Union[Dict, List[Dict]]:
        """Fetch symbol information or all symbols if symbol is None"""
        try:
            url = f"{self.base_url}/exchangeInfo"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if symbol:
                # Clean symbol format
                clean_symbol = symbol.upper().replace('-', '').replace('_', '').replace('/', '')
                if not any(clean_symbol.endswith(base) for base in ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']):
                    if clean_symbol.endswith('USD'):
                        clean_symbol = clean_symbol.replace('USD', 'USDT')
                    else:
                        clean_symbol += 'USDT'
                
                # Find specific symbol
                for symbol_info in data['symbols']:
                    if symbol_info['symbol'] == clean_symbol:
                        return {
                            'symbol': symbol,
                            'binance_symbol': clean_symbol,
                            'status': symbol_info.get('status'),
                            'base_asset': symbol_info.get('baseAsset'),
                            'quote_asset': symbol_info.get('quoteAsset'),
                            'price_precision': symbol_info.get('quotePrecision'),
                            'quantity_precision': symbol_info.get('baseAssetPrecision'),
                            'order_types': symbol_info.get('orderTypes', []),
                            'is_spot_trading_allowed': symbol_info.get('isSpotTradingAllowed'),
                            'is_margin_trading_allowed': symbol_info.get('isMarginTradingAllowed'),
                            'source': self.source_name
                        }
                
                return {'symbol': symbol, 'source': self.source_name, 'error': 'Symbol not found'}
            
            else:
                # Return all symbols
                symbols = []
                for symbol_info in data['symbols']:
                    if symbol_info.get('status') == 'TRADING':
                        symbols.append({
                            'symbol': symbol_info['symbol'],
                            'base_asset': symbol_info.get('baseAsset'),
                            'quote_asset': symbol_info.get('quoteAsset'),
                            'status': symbol_info.get('status'),
                            'source': self.source_name
                        })
                
                return symbols
            
        except Exception as e:
            print(f"Error fetching symbol info: {e}")
            return {'error': str(e), 'source': self.source_name}
    
    def fetch_ticker_24hr(self, symbol: str = None) -> Union[Dict, List[Dict]]:
        """Fetch 24hr ticker statistics"""
        try:
            url = f"{self.base_url}/ticker/24hr"
            
            if symbol:
                # Clean symbol format
                clean_symbol = symbol.upper().replace('-', '').replace('_', '').replace('/', '')
                if not any(clean_symbol.endswith(base) for base in ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']):
                    if clean_symbol.endswith('USD'):
                        clean_symbol = clean_symbol.replace('USD', 'USDT')
                    else:
                        clean_symbol += 'USDT'
                
                params = {'symbol': clean_symbol}
            else:
                params = {}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if symbol:
                # Single symbol response
                return {
                    'symbol': symbol,
                    'binance_symbol': clean_symbol,
                    'price_change': float(data.get('priceChange', 0)),
                    'price_change_percent': float(data.get('priceChangePercent', 0)),
                    'weighted_avg_price': float(data.get('weightedAvgPrice', 0)),
                    'prev_close_price': float(data.get('prevClosePrice', 0)),
                    'last_price': float(data.get('lastPrice', 0)),
                    'bid_price': float(data.get('bidPrice', 0)),
                    'ask_price': float(data.get('askPrice', 0)),
                    'open_price': float(data.get('openPrice', 0)),
                    'high_price': float(data.get('highPrice', 0)),
                    'low_price': float(data.get('lowPrice', 0)),
                    'volume': float(data.get('volume', 0)),
                    'quote_volume': float(data.get('quoteVolume', 0)),
                    'open_time': pd.to_datetime(data.get('openTime'), unit='ms'),
                    'close_time': pd.to_datetime(data.get('closeTime'), unit='ms'),
                    'count': int(data.get('count', 0)),
                    'source': self.source_name
                }
            else:
                # All symbols response
                tickers = []
                for ticker in data:
                    tickers.append({
                        'symbol': ticker['symbol'],
                        'price_change_percent': float(ticker.get('priceChangePercent', 0)),
                        'last_price': float(ticker.get('lastPrice', 0)),
                        'volume': float(ticker.get('volume', 0)),
                        'source': self.source_name
                    })
                
                return tickers
            
        except Exception as e:
            print(f"Error fetching 24hr ticker: {e}")
            return {'error': str(e), 'source': self.source_name}
    
    def fetch_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Fetch order book depth"""
        try:
            # Clean symbol format
            clean_symbol = symbol.upper().replace('-', '').replace('_', '').replace('/', '')
            if not any(clean_symbol.endswith(base) for base in ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']):
                if clean_symbol.endswith('USD'):
                    clean_symbol = clean_symbol.replace('USD', 'USDT')
                else:
                    clean_symbol += 'USDT'
            
            url = f"{self.base_url}/depth"
            params = {
                'symbol': clean_symbol,
                'limit': min(limit, 5000)  # Max limit is 5000
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrames
            bids_df = pd.DataFrame(data['bids'], columns=['price', 'quantity'])
            asks_df = pd.DataFrame(data['asks'], columns=['price', 'quantity'])
            
            # Convert to numeric
            bids_df['price'] = pd.to_numeric(bids_df['price'])
            bids_df['quantity'] = pd.to_numeric(bids_df['quantity'])
            asks_df['price'] = pd.to_numeric(asks_df['price'])
            asks_df['quantity'] = pd.to_numeric(asks_df['quantity'])
            
            return {
                'symbol': symbol,
                'binance_symbol': clean_symbol,
                'bids': bids_df,
                'asks': asks_df,
                'last_update_id': data.get('lastUpdateId'),
                'source': self.source_name,
                'fetch_timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return {'error': str(e), 'source': self.source_name}
    
    def fetch_recent_trades(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Fetch recent trades"""
        try:
            # Clean symbol format
            clean_symbol = symbol.upper().replace('-', '').replace('_', '').replace('/', '')
            if not any(clean_symbol.endswith(base) for base in ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']):
                if clean_symbol.endswith('USD'):
                    clean_symbol = clean_symbol.replace('USD', 'USDT')
                else:
                    clean_symbol += 'USDT'
            
            url = f"{self.base_url}/trades"
            params = {
                'symbol': clean_symbol,
                'limit': min(limit, 1000)  # Max limit is 1000
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            trades_df = pd.DataFrame(data)
            
            # Convert columns
            trades_df['price'] = pd.to_numeric(trades_df['price'])
            trades_df['qty'] = pd.to_numeric(trades_df['qty'])
            trades_df['time'] = pd.to_datetime(trades_df['time'], unit='ms')
            
            # Add metadata
            trades_df['symbol'] = symbol
            trades_df['binance_symbol'] = clean_symbol
            trades_df['source'] = self.source_name
            
            return trades_df[['time', 'price', 'qty', 'isBuyerMaker', 'symbol', 'source']]
            
        except Exception as e:
            print(f"Error fetching recent trades for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_popular_symbols(self) -> List[str]:
        """Get list of popular trading pairs"""
        return [
            'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 'SOL-USDT',
            'XRP-USDT', 'DOT-USDT', 'DOGE-USDT', 'AVAX-USDT', 'SHIB-USDT',
            'MATIC-USDT', 'LTC-USDT', 'UNI-USDT', 'LINK-USDT', 'ALGO-USDT',
            'BCH-USDT', 'XLM-USDT', 'VET-USDT', 'FIL-USDT', 'TRX-USDT'
        ]
    
    def get_top_volume_pairs(self, quote_asset: str = 'USDT', limit: int = 20) -> List[Dict]:
        """Get top volume trading pairs for a quote asset"""
        try:
            tickers = self.fetch_ticker_24hr()
            
            if isinstance(tickers, dict) and 'error' in tickers:
                return []
            
            # Filter by quote asset and sort by volume
            filtered_tickers = [
                ticker for ticker in tickers
                if ticker['symbol'].endswith(quote_asset)
            ]
            
            # Sort by volume (descending)
            sorted_tickers = sorted(
                filtered_tickers,
                key=lambda x: x['volume'],
                reverse=True
            )
            
            return sorted_tickers[:limit]
            
        except Exception as e:
            print(f"Error fetching top volume pairs: {e}")
            return []


if __name__ == "__main__":
    # Example usage
    api = BinanceAPI()
    
    print("Binance API initialized")
    
    # Test with sample data
    data = api.fetch_market_data("BTC-USD", period="1mo")
    if not data.empty:
        print(f"Fetched {len(data)} rows for BTC-USD")
    
    # Get popular symbols
    symbols = api.get_popular_symbols()
    print(f"Popular symbols: {symbols[:5]}")  # Show first 5