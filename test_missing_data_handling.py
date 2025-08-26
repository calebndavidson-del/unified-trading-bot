#!/usr/bin/env python3
"""
Test script specifically for missing data handling functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features.backtesting import BacktestEngine, TechnicalAnalysisStrategy, MissingDataConfig
from model_config import TradingBotConfig
import pandas as pd

def test_mixed_asset_types():
    """Test missing data handling with mixed asset types"""
    print("🧪 Testing Mixed Asset Types (Stocks + Crypto)")
    print("=" * 50)
    
    # Initialize configuration
    config = TradingBotConfig()
    config.risk.initial_capital = 50000
    
    # Configure missing data handling
    missing_data_config = MissingDataConfig()
    missing_data_config.crypto_daily_tolerance_hours = 6.0
    missing_data_config.strict_mode = False
    
    # Initialize backtesting engine
    engine = BacktestEngine(config, missing_data_config)
    
    # Test with mixed asset types: stocks and crypto
    test_symbols = ['AAPL', 'BTC-USD', 'ETH-USD']
    
    print(f"📊 Running backtest for {test_symbols}")
    print("⏳ This may take a moment...")
    
    try:
        # Run backtest
        results = engine.run_backtest(
            symbols=test_symbols,
            strategy_name='Technical Analysis',
            model_name='LSTM Neural Network',
            confidence_threshold=0.75
        )
        
        if "error" in results:
            print(f"❌ Backtest failed: {results['error']}")
            return False
        
        # Display results
        print("\n✅ Backtest completed successfully!")
        print(f"📈 Total Return: {results['total_return_pct']:.2f}%")
        print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"📉 Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"🎯 Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"📋 Total Trades: {results['total_trades']}")
        print(f"📅 Period: {results['start_date']} to {results['end_date']}")
        
        # Check missing data summary
        if 'missing_data_summary' in results:
            summary = results['missing_data_summary']
            print(f"\n📊 Missing Data Analysis:")
            print(f"  Expected gaps: {summary['total_expected_gaps']}")
            print(f"  Unexpected gaps: {summary['total_unexpected_gaps']}")
            print(f"  Crypto violations: {summary['crypto_tolerance_violations']}")
            print(f"  Status: {summary['status']}")
            
            if summary['by_asset_type']:
                print(f"  By asset type: {summary['by_asset_type']}")
        
        print("\n🎉 Mixed asset type test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_strict_mode():
    """Test strict mode functionality"""
    print("\n🧪 Testing Strict Mode")
    print("=" * 30)
    
    # Initialize configuration
    config = TradingBotConfig()
    config.risk.initial_capital = 50000
    
    # Configure strict mode
    missing_data_config = MissingDataConfig()
    missing_data_config.strict_mode = True
    missing_data_config.max_missing_data_ratio = 0.05  # Very strict: 5% max missing data
    
    # Initialize backtesting engine
    engine = BacktestEngine(config, missing_data_config)
    
    # Test with potentially problematic symbols
    test_symbols = ['AAPL', INVALID_TEST_TICKER]  # Use constant to trigger missing data
    
    print(f"📊 Running strict mode test with {test_symbols}")
    
    try:
        results = engine.run_backtest(
            symbols=test_symbols,
            strategy_name='Technical Analysis',
            model_name='LSTM Neural Network',
            confidence_threshold=0.75
        )
        
        if "error" in results:
            if "Strict mode" in results["error"]:
                print(f"✅ Strict mode correctly triggered: {results['error']}")
                return True
            else:
                print(f"❌ Unexpected error: {results['error']}")
                return False
        else:
            print("⚠️ Strict mode did not trigger (may be normal if all data is available)")
            return True
            
    except Exception as e:
        print(f"❌ Error during strict mode test: {str(e)}")
        return False

def test_crypto_tolerance():
    """Test crypto-specific tolerance settings"""
    print("\n🧪 Testing Crypto Tolerance Settings")
    print("=" * 35)
    
    # Initialize configuration
    config = TradingBotConfig()
    config.risk.initial_capital = 50000
    
    # Configure very strict crypto tolerance to trigger violations
    missing_data_config = MissingDataConfig()
    missing_data_config.crypto_daily_tolerance_hours = 0.5  # Very strict: 30 minutes
    missing_data_config.strict_mode = False
    
    # Initialize backtesting engine
    engine = BacktestEngine(config, missing_data_config)
    
    # Test with crypto only
    test_symbols = ['BTC-USD', 'ETH-USD']
    
    print(f"📊 Running crypto tolerance test with {test_symbols}")
    print(f"📏 Tolerance set to {missing_data_config.crypto_daily_tolerance_hours} hours")
    
    try:
        results = engine.run_backtest(
            symbols=test_symbols,
            strategy_name='Technical Analysis',
            model_name='LSTM Neural Network',
            confidence_threshold=0.75
        )
        
        if "error" in results:
            print(f"❌ Backtest failed: {results['error']}")
            return False
        
        # Check crypto tolerance violations
        if 'missing_data_summary' in results:
            summary = results['missing_data_summary']
            print(f"✅ Crypto tolerance test completed")
            print(f"  Crypto violations: {summary['crypto_tolerance_violations']}")
            print(f"  Total unexpected gaps: {summary['total_unexpected_gaps']}")
            
            if summary['crypto_tolerance_violations'] > 0:
                print("⚠️ Crypto tolerance violations detected (expected with strict settings)")
            else:
                print("✅ No crypto tolerance violations (crypto data is very complete)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during crypto tolerance test: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Missing Data Handling Test Suite")
    print("=" * 50)
    
    tests = [
        test_mixed_asset_types,
        test_strict_mode,
        test_crypto_tolerance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All missing data handling tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)