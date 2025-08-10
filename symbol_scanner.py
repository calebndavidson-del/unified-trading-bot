#!/usr/bin/env python3
"""
Intelligent Symbol Scanner for Trading Bot
Scans markets and scores symbols for trading opportunities
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import streamlit as st


@dataclass
class ScanResult:
    """Result of symbol scanning with scores and metadata"""
    symbol: str
    score: float
    sub_scores: Dict[str, float]
    price: float
    volume: float
    market_cap: Optional[float]
    reasoning: str
    data_quality: float


class MarketCategoryScanner:
    """Scanner for different market categories"""
    
    # Market category symbol lists - these are commonly tracked indices/ETFs that contain the symbols
    MARKET_CATEGORIES = {
        'SP500': [
            # Top S&P 500 stocks by market cap
            'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'BRK-B', 'LLY',
            'AVGO', 'V', 'JPM', 'UNH', 'XOM', 'COST', 'MA', 'HD', 'PG', 'NFLX',
            'JNJ', 'ABBV', 'BAC', 'CRM', 'CVX', 'KO', 'AMD', 'PEP', 'TMO', 'MRK',
            'WMT', 'ACN', 'LIN', 'CSCO', 'ABT', 'ADBE', 'DHR', 'VZ', 'TXN', 'ORCL',
            'NOW', 'MCD', 'PM', 'INTC', 'COP', 'DIS', 'CAT', 'GE', 'INTU', 'IBM'
        ],
        'NASDAQ100': [
            # Top NASDAQ 100 stocks
            'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'AVGO', 'COST',
            'NFLX', 'AMD', 'PEP', 'ADBE', 'CSCO', 'TXN', 'ORCL', 'INTC', 'CMCSA', 'QCOM',
            'INTU', 'AMAT', 'ISRG', 'BKNG', 'HON', 'AMGN', 'VRTX', 'ADP', 'PANW', 'GILD',
            'MU', 'ADI', 'LRCX', 'PYPL', 'REGN', 'KLAC', 'MRVL', 'SNPS', 'CRWD', 'FTNT',
            'ORLY', 'CSX', 'ABNB', 'NXPI', 'ROP', 'CDNS', 'WDAY', 'ADSK', 'PCAR', 'CPRT'
        ],
        'DOW': [
            # Dow Jones Industrial Average components
            'AAPL', 'MSFT', 'UNH', 'HD', 'PG', 'JNJ', 'JPM', 'V', 'CVX', 'MRK',
            'WMT', 'CRM', 'AMGN', 'MCD', 'DIS', 'HON', 'CAT', 'AXP', 'CSCO', 'IBM',
            'KO', 'INTC', 'GS', 'NKE', 'MMM', 'BA', 'TRV', 'SHW', 'VZ', 'WBA'
        ],
        'POPULAR_GROWTH': [
            # Popular growth stocks
            'NVDA', 'TSLA', 'AMD', 'PLTR', 'SNOW', 'CRWD', 'ZM', 'SHOP', 'SQ', 'ROKU',
            'UBER', 'LYFT', 'ABNB', 'COIN', 'RBLX', 'U', 'DKNG', 'ARKK', 'TQQQ', 'SOXL'
        ],
        'CRYPTO': [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LINK-USD'
        ]
    }
    
    @classmethod
    def get_symbols_for_scan(cls, categories: List[str], max_symbols: int = 100) -> List[str]:
        """Get unique symbols from selected market categories"""
        symbols = set()
        for category in categories:
            if category in cls.MARKET_CATEGORIES:
                symbols.update(cls.MARKET_CATEGORIES[category])
        
        # Limit to max_symbols and return as list
        return list(symbols)[:max_symbols]


class TechnicalScorer:
    """Calculate technical analysis scores for symbols"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators for a price dataframe"""
        if df.empty or len(df) < 50:
            return {}
        
        try:
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_ma = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            bb_upper = bb_ma + (bb_std * 2)
            bb_lower = bb_ma - (bb_std * 2)
            bb_position = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Moving averages
            ma_10 = df['Close'].rolling(window=10).mean()
            ma_20 = df['Close'].rolling(window=20).mean()
            ma_50 = df['Close'].rolling(window=50).mean()
            
            # Volume indicators
            vol_sma = df['Volume'].rolling(window=20).mean()
            vol_ratio = df['Volume'] / vol_sma
            
            # Price momentum
            price_change_5d = df['Close'].pct_change(5)
            price_change_20d = df['Close'].pct_change(20)
            
            # Volatility
            volatility = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            return {
                'rsi': rsi,
                'bb_position': bb_position,
                'ma_10': ma_10,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'vol_ratio': vol_ratio,
                'price_change_5d': price_change_5d,
                'price_change_20d': price_change_20d,
                'volatility': volatility
            }
        except Exception:
            return {}
    
    @staticmethod
    def score_symbol(symbol: str, data: pd.DataFrame, mode: str = 'balanced') -> Optional[ScanResult]:
        """Score a symbol based on technical analysis"""
        if data.empty or len(data) < 50:
            return None
        
        try:
            indicators = TechnicalScorer.calculate_indicators(data)
            if not indicators:
                return None
            
            current_price = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].tail(20).mean()
            
            # Sub-scores (0-100 scale)
            sub_scores = {}
            
            # 1. Momentum Score
            momentum_score = 50.0
            if 'price_change_5d' in indicators:
                pc_5d = indicators['price_change_5d'].iloc[-1]
                if not pd.isna(pc_5d):
                    momentum_score = max(0, min(100, 50 + (pc_5d * 500)))  # Scale momentum
            sub_scores['momentum'] = momentum_score
            
            # 2. Trend Score (MA alignment)
            trend_score = 50.0
            if all(k in indicators for k in ['ma_10', 'ma_20', 'ma_50']):
                ma_10_val = indicators['ma_10'].iloc[-1]
                ma_20_val = indicators['ma_20'].iloc[-1]
                ma_50_val = indicators['ma_50'].iloc[-1]
                
                if not any(pd.isna([ma_10_val, ma_20_val, ma_50_val])):
                    if ma_10_val > ma_20_val > ma_50_val:  # Bull trend
                        trend_score = 85
                    elif ma_10_val > ma_20_val:  # Partial bull
                        trend_score = 70
                    elif ma_10_val < ma_20_val < ma_50_val:  # Bear trend
                        trend_score = 15
                    else:  # Sideways
                        trend_score = 50
            sub_scores['trend'] = trend_score
            
            # 3. RSI Score (oversold/overbought opportunities)
            rsi_score = 50.0
            if 'rsi' in indicators:
                rsi_val = indicators['rsi'].iloc[-1]
                if not pd.isna(rsi_val):
                    if 30 <= rsi_val <= 70:  # Sweet spot
                        rsi_score = 80
                    elif rsi_val < 30:  # Oversold (good for buying)
                        rsi_score = 75
                    elif rsi_val > 70:  # Overbought (good for shorting)
                        rsi_score = 25
            sub_scores['rsi'] = rsi_score
            
            # 4. Volume Score
            volume_score = 50.0
            if current_volume > 0 and avg_volume > 0:
                vol_ratio = current_volume / avg_volume
                if vol_ratio > 1.5:  # High volume
                    volume_score = 80
                elif vol_ratio > 1.0:
                    volume_score = 65
                else:
                    volume_score = 40
            sub_scores['volume'] = volume_score
            
            # 5. Volatility Score (based on mode preference)
            volatility_score = 50.0
            if 'volatility' in indicators:
                vol_val = indicators['volatility'].iloc[-1]
                if not pd.isna(vol_val):
                    if mode == 'aggressive':
                        # High volatility preferred for aggressive mode
                        volatility_score = min(100, vol_val * 300)  # Scale volatility
                    elif mode == 'conservative':
                        # Low volatility preferred for conservative mode
                        volatility_score = max(0, 100 - (vol_val * 200))
                    else:  # balanced
                        # Moderate volatility preferred
                        if 0.15 <= vol_val <= 0.35:
                            volatility_score = 80
                        else:
                            volatility_score = 60
            sub_scores['volatility'] = volatility_score
            
            # Calculate overall score based on mode weights
            if mode == 'aggressive':
                weights = {'momentum': 0.3, 'trend': 0.2, 'rsi': 0.2, 'volume': 0.15, 'volatility': 0.15}
            elif mode == 'conservative':
                weights = {'momentum': 0.15, 'trend': 0.35, 'rsi': 0.25, 'volume': 0.15, 'volatility': 0.1}
            else:  # balanced
                weights = {'momentum': 0.25, 'trend': 0.25, 'rsi': 0.25, 'volume': 0.15, 'volatility': 0.1}
            
            overall_score = sum(sub_scores[key] * weights[key] for key in weights)
            
            # Generate reasoning
            reasoning_parts = []
            if sub_scores['momentum'] > 70:
                reasoning_parts.append("Strong momentum")
            if sub_scores['trend'] > 70:
                reasoning_parts.append("Bullish trend")
            if sub_scores['volume'] > 70:
                reasoning_parts.append("High volume")
            if sub_scores['volatility'] > 70 and mode == 'aggressive':
                reasoning_parts.append("High volatility (good for aggressive)")
            elif sub_scores['volatility'] > 70 and mode == 'conservative':
                reasoning_parts.append("Low volatility (good for conservative)")
            
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Mixed signals"
            
            # Data quality score
            data_quality = min(100, len(data) / 60 * 100)  # Prefer 60+ days of data
            
            return ScanResult(
                symbol=symbol,
                score=overall_score,
                sub_scores=sub_scores,
                price=current_price,
                volume=current_volume,
                market_cap=None,  # Could be added via yfinance info
                reasoning=reasoning,
                data_quality=data_quality
            )
            
        except Exception as e:
            print(f"Error scoring {symbol}: {e}")
            return None


class SymbolScanner:
    """Main symbol scanning engine"""
    
    def __init__(self, cache_duration_minutes: int = 15):
        self.cache_duration = cache_duration_minutes
        self.cache = {}
        
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self.cache:
            return False
        
        cache_time = self.cache[symbol].get('timestamp', 0)
        return (time.time() - cache_time) < (self.cache_duration * 60)
    
    def _get_symbol_data(self, symbol: str, period: str = "3mo") -> pd.DataFrame:
        """Get symbol data with caching"""
        if self._is_cache_valid(symbol):
            return self.cache[symbol]['data']
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            # Cache the data
            self.cache[symbol] = {
                'data': data,
                'timestamp': time.time()
            }
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            # Return mock data for demo purposes when network fails
            return self._get_mock_data(symbol)
    
    def _get_mock_data(self, symbol: str) -> pd.DataFrame:
        """Generate mock data for demo purposes when network is unavailable"""
        import random
        from datetime import datetime, timedelta
        
        # Create 60 days of mock data
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        
        # Base price varies by symbol
        base_price = hash(symbol) % 200 + 50  # Price between 50-250
        
        # Generate realistic price movement
        prices = []
        volume = []
        current_price = base_price
        
        for i in range(60):
            # Random walk with some trend
            change = random.uniform(-0.05, 0.05)  # ±5% daily change
            current_price *= (1 + change)
            prices.append(current_price)
            
            # Mock volume (varies by symbol and day)
            daily_volume = random.randint(500000, 5000000)
            volume.append(daily_volume)
        
        # Create OHLC data
        mock_data = pd.DataFrame({
            'Open': [p * random.uniform(0.99, 1.01) for p in prices],
            'High': [p * random.uniform(1.00, 1.03) for p in prices],
            'Low': [p * random.uniform(0.97, 1.00) for p in prices],
            'Close': prices,
            'Volume': volume
        }, index=dates)
        
        return mock_data
    
    def scan_symbols(self, symbols: List[str], mode: str = 'balanced', 
                     max_workers: int = 10, progress_callback=None) -> List[ScanResult]:
        """Scan multiple symbols in parallel"""
        results = []
        
        def scan_single_symbol(symbol: str) -> Optional[ScanResult]:
            try:
                data = self._get_symbol_data(symbol)
                return TechnicalScorer.score_symbol(symbol, data, mode)
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(scan_single_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_symbol)):
                if progress_callback:
                    progress_callback(i + 1, len(symbols))
                
                result = future.result()
                if result and result.score > 0:
                    results.append(result)
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def smart_scan(self, categories: List[str], mode: str = 'balanced', 
                   top_n: int = 20, filters: Dict = None) -> List[ScanResult]:
        """Perform a smart scan with filtering"""
        # Get symbols from categories
        symbols = MarketCategoryScanner.get_symbols_for_scan(categories)
        
        # Scan all symbols
        results = self.scan_symbols(symbols, mode)
        
        # Apply filters if provided
        if filters:
            filtered_results = []
            for result in results:
                # Price filter
                if 'min_price' in filters and result.price < filters['min_price']:
                    continue
                if 'max_price' in filters and result.price > filters['max_price']:
                    continue
                    
                # Volume filter
                if 'min_volume' in filters and result.volume < filters['min_volume']:
                    continue
                    
                # Score filter
                if 'min_score' in filters and result.score < filters['min_score']:
                    continue
                    
                filtered_results.append(result)
            results = filtered_results
        
        # Return top N results
        return results[:top_n]


# Caching functions for Streamlit
@st.cache_data(ttl=900)  # Cache for 15 minutes
def cached_symbol_scan(symbols_tuple: tuple, mode: str, cache_key: str) -> List[Dict]:
    """Cached version of symbol scanning for Streamlit"""
    scanner = SymbolScanner()
    symbols = list(symbols_tuple)  # Convert back from tuple (required for caching)
    results = scanner.scan_symbols(symbols, mode)
    
    # Convert ScanResult objects to dictionaries for caching
    return [
        {
            'symbol': r.symbol,
            'score': r.score,
            'sub_scores': r.sub_scores,
            'price': r.price,
            'volume': r.volume,
            'reasoning': r.reasoning,
            'data_quality': r.data_quality
        }
        for r in results
    ]


@st.cache_data(ttl=900)  # Cache for 15 minutes  
def cached_smart_scan(categories_tuple: tuple, mode: str, top_n: int, 
                      filters_key: str) -> List[Dict]:
    """Cached version of smart scanning for Streamlit"""
    scanner = SymbolScanner()
    categories = list(categories_tuple)
    
    # Parse filters from key (simple string encoding)
    filters = {}
    if filters_key and filters_key != "none":
        # This is a simple approach - in production you might use JSON encoding
        pass
    
    results = scanner.smart_scan(categories, mode, top_n, filters)
    
    # Convert to dictionaries for caching
    return [
        {
            'symbol': r.symbol,
            'score': r.score,
            'sub_scores': r.sub_scores,
            'price': r.price,
            'volume': r.volume,
            'reasoning': r.reasoning,
            'data_quality': r.data_quality
        }
        for r in results
    ]