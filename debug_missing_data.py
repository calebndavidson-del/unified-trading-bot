#!/usr/bin/env python3
"""
Debug script for missing data handling
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features.backtesting import BacktestEngine, TechnicalAnalysisStrategy, MissingDataConfig, AssetTypeDetector
from model_config import TradingBotConfig
import pandas as pd

def debug_asset_detection():
    """Debug asset type detection"""
    print("🔍 Testing Asset Type Detection")
    print("=" * 35)
    
    test_symbols = ['AAPL', 'BTC-USD', 'ETH-USD', 'SPY', '^GSPC']
    
    for symbol in test_symbols:
        asset_type = AssetTypeDetector.detect_asset_type(symbol)
        print(f"  {symbol} -> {asset_type}")
    
    print()

def debug_date_filtering():
    """Debug date filtering logic"""
    print("🔍 Testing Date Filtering Logic")
    print("=" * 35)
    
    # Initialize configuration
    config = TradingBotConfig()
    missing_data_config = MissingDataConfig()
    
    # Initialize backtesting engine
    engine = BacktestEngine(config, missing_data_config)
    
    # Test date filtering for different asset types
    test_date = pd.Timestamp('2025-01-04', tz='UTC')  # Saturday
    
    for asset_type in ['stock', 'crypto', 'etf', 'index']:
        should_check = engine._should_check_date_for_asset(test_date, asset_type)
        print(f"  {test_date.strftime('%Y-%m-%d %A')} for {asset_type}: {should_check}")
    
    test_date = pd.Timestamp('2025-01-06', tz='UTC')  # Monday
    
    for asset_type in ['stock', 'crypto', 'etf', 'index']:
        should_check = engine._should_check_date_for_asset(test_date, asset_type)
        print(f"  {test_date.strftime('%Y-%m-%d %A')} for {asset_type}: {should_check}")
    
    print()

def debug_data_fetch():
    """Debug what data is actually fetched"""
    print("🔍 Debugging Data Fetch")
    print("=" * 25)
    
    # Initialize configuration
    config = TradingBotConfig()
    missing_data_config = MissingDataConfig()
    
    # Initialize backtesting engine
    engine = BacktestEngine(config, missing_data_config)
    
    # Fetch data
    symbols = ['AAPL', 'BTC-USD']
    data_dict = engine.fetch_current_year_data(symbols)
    
    for symbol, data in data_dict.items():
        asset_type = AssetTypeDetector.detect_asset_type(symbol)
        print(f"\n📊 {symbol} ({asset_type}):")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Total days: {len(data)}")
        print(f"  Sample dates: {data.index[:5].tolist()}")
        
        # Check weekends
        weekend_count = sum(1 for date in data.index if AssetTypeDetector.is_weekend(date))
        print(f"  Weekend days in data: {weekend_count}")
    
    print()

if __name__ == "__main__":
    debug_asset_detection()
    debug_date_filtering()
    debug_data_fetch()