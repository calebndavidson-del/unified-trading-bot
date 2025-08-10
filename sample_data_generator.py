#!/usr/bin/env python3
"""
Sample Data Generator for Offline Testing
Generates realistic financial data when network access is not available
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional


def generate_sample_market_data(symbol: str, days: int = 90, 
                               base_price: float = 150.0,
                               volatility: float = 0.02) -> pd.DataFrame:
    """
    Generate realistic financial market data for testing
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        days: Number of days of data to generate
        base_price: Starting price
        volatility: Daily volatility (standard deviation of returns)
    
    Returns:
        DataFrame with OHLCV data similar to yfinance format
    """
    
    # Set random seed based on symbol for consistent data
    np.random.seed(hash(symbol) % 1000000)
    
    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate returns with trending behavior and some autocorrelation
    trend = np.random.normal(0.001, 0.0005, days + 1)  # Slight upward trend
    noise = np.random.normal(0, volatility, days + 1)
    
    # Add some realistic patterns (momentum and mean reversion cycles)
    cycle_length = max(10, days // 6)  # Cycles every ~10 days
    cycle_factor = 0.3 * np.sin(2 * np.pi * np.arange(days + 1) / cycle_length)
    
    returns = trend + noise + cycle_factor * volatility
    
    # Generate prices
    prices = [base_price]
    for i in range(days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
    
    # Create OHLC data
    data = []
    for i, date in enumerate(dates):
        if i >= len(prices) - 1:
            break
            
        open_price = prices[i]
        close_price = prices[i + 1]
        
        # Generate realistic high/low based on volatility
        daily_range = abs(close_price - open_price) + np.random.exponential(open_price * 0.01)
        high_price = max(open_price, close_price) + np.random.uniform(0, daily_range * 0.5)
        low_price = min(open_price, close_price) - np.random.uniform(0, daily_range * 0.5)
        
        # Ensure logical price relationships
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume (higher volume on price changes)
        price_change = abs(close_price - open_price) / open_price
        base_volume = 1000000 + np.random.normal(500000, 200000)
        volume_multiplier = 1 + price_change * 3  # Higher volume on big moves
        volume = max(int(base_volume * volume_multiplier), 10000)
        
        data.append({
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates[:len(data)])
    return df


def generate_sample_data_for_symbols(symbols: list, days: int = 90) -> Dict[str, pd.DataFrame]:
    """Generate sample data for multiple symbols"""
    
    # Base prices for different symbols
    base_prices = {
        'AAPL': 175.0,
        'MSFT': 330.0,
        'NVDA': 450.0,
        'AMZN': 140.0,
        'GOOGL': 135.0,
        'TSLA': 250.0,
        'META': 320.0,
        'BTC-USD': 42000.0,
        'ETH-USD': 2500.0,
        'SPY': 440.0,
        'QQQ': 380.0
    }
    
    # Volatilities for different asset types
    volatilities = {
        'BTC-USD': 0.04,  # Crypto is more volatile
        'ETH-USD': 0.04,
        'TSLA': 0.035,    # TSLA is volatile
        'NVDA': 0.03,     # Tech stocks
        'META': 0.03,
        'AAPL': 0.025,    # Large caps
        'MSFT': 0.025,
        'GOOGL': 0.025,
        'AMZN': 0.025,
        'SPY': 0.015,     # ETFs are less volatile
        'QQQ': 0.02
    }
    
    data = {}
    for symbol in symbols:
        base_price = base_prices.get(symbol, 100.0)
        volatility = volatilities.get(symbol, 0.02)
        
        data[symbol] = generate_sample_market_data(
            symbol=symbol,
            days=days,
            base_price=base_price,
            volatility=volatility
        )
    
    return data


class MockYFinanceTicker:
    """Mock yfinance ticker for offline testing"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self._data_cache = {}
    
    def history(self, start=None, end=None, period=None) -> pd.DataFrame:
        """Mock history method that returns sample data"""
        
        # Calculate days based on parameters
        if start and end:
            days = (end - start).days
        elif period:
            # Convert period string to days
            period_days = {
                '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, 
                '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
            }
            days = period_days.get(period, 90)
        else:
            days = 90
        
        # Limit to reasonable range
        days = min(max(days, 1), 1000)
        
        # Generate or retrieve cached data
        cache_key = f"{self.symbol}_{days}"
        if cache_key not in self._data_cache:
            self._data_cache[cache_key] = generate_sample_market_data(
                symbol=self.symbol,
                days=days
            )
        
        return self._data_cache[cache_key]


def mock_yf_ticker(symbol: str) -> MockYFinanceTicker:
    """Create a mock yfinance ticker"""
    return MockYFinanceTicker(symbol)


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Sample Data Generator")
    print("=" * 40)
    
    # Test single symbol
    print("\n1. Testing single symbol data generation:")
    aapl_data = generate_sample_market_data('AAPL', days=30)
    print(f"Generated {len(aapl_data)} days of AAPL data")
    print(f"Price range: ${aapl_data['Low'].min():.2f} - ${aapl_data['High'].max():.2f}")
    print(f"Final price: ${aapl_data['Close'].iloc[-1]:.2f}")
    
    # Test multiple symbols
    print("\n2. Testing multiple symbols:")
    symbols = ['AAPL', 'MSFT', 'BTC-USD']
    multi_data = generate_sample_data_for_symbols(symbols, days=60)
    
    for symbol, data in multi_data.items():
        returns = data['Close'].pct_change().std()
        print(f"{symbol}: {len(data)} days, volatility: {returns:.3f}")
    
    # Test mock ticker
    print("\n3. Testing mock yfinance ticker:")
    mock_ticker = mock_yf_ticker('NVDA')
    hist_data = mock_ticker.history(period='1mo')
    print(f"Mock NVDA data: {len(hist_data)} days")
    print(f"Sample close prices: {hist_data['Close'].head(3).tolist()}")
    
    print("\n✅ Sample data generator working correctly!")