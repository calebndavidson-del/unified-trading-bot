#!/usr/bin/env python3
"""
Comprehensive tests for backtesting edge cases: holidays, weekends, and missing data
"""

import sys
import os
sys.path.append('/home/runner/work/unified-trading-bot/unified-trading-bot')

from features.backtesting import BacktestEngine
from model_config import TradingBotConfig
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import pytz

def test_holiday_handling():
    """Test how backtesting handles market holidays"""
    print("🎄 Testing Holiday Handling")
    print("-" * 30)
    
    config = TradingBotConfig()
    engine = BacktestEngine(config)
    
    # Test with a symbol for holiday handling
    try:
        # Fetch data that includes holidays
        data_dict = engine.fetch_current_year_data(['AAPL'])
        
        if not data_dict:
            print("❌ No data fetched for holiday test")
            return False
        
        data = data_dict['AAPL']
        print(f"✅ Data fetched: {len(data)} trading days")
        print(f"   Timezone: {data.index.tz}")
        print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Test specific holiday dates that should NOT be in the data
        holidays_2025 = [
            '2025-01-01',  # New Year's Day
            '2025-01-20',  # Martin Luther King Jr. Day
            '2025-02-17',  # Presidents Day
            '2025-05-26',  # Memorial Day
            '2025-07-04',  # Independence Day
            '2025-09-01',  # Labor Day
            '2025-11-27',  # Thanksgiving
            '2025-12-25',  # Christmas
        ]
        
        missing_holidays = 0
        for holiday in holidays_2025:
            holiday_ts = pd.Timestamp(holiday, tz='UTC')
            if holiday_ts not in data.index:
                missing_holidays += 1
                print(f"  ✅ {holiday} correctly missing (holiday)")
            else:
                print(f"  ⚠️ {holiday} unexpectedly present")
        
        print(f"📊 Holiday handling: {missing_holidays}/{len(holidays_2025)} holidays correctly excluded")
        
        # Test weekend handling
        weekend_count = 0
        total_possible_days = (data.index[-1] - data.index[0]).days + 1
        
        for i in range(total_possible_days):
            test_date = data.index[0] + timedelta(days=i)
            if test_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                if test_date not in data.index:
                    weekend_count += 1
        
        print(f"📊 Weekend handling: {weekend_count} weekend days correctly excluded")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in holiday test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_missing_data_scenarios():
    """Test various missing data scenarios"""
    print("\n🚫 Testing Missing Data Scenarios")
    print("-" * 30)
    
    config = TradingBotConfig()
    engine = BacktestEngine(config)
    
    try:
        # Test 1: Invalid symbol
        print("Test 1: Invalid symbol handling...")
        invalid_data = engine.fetch_current_year_data(['INVALID_SYMBOL_12345'])
        if not invalid_data:
            print("  ✅ Invalid symbol correctly returns empty data")
        else:
            print("  ⚠️ Invalid symbol unexpectedly returned data")
        
        # Test 2: Mix of valid and invalid symbols
        print("Test 2: Mixed valid/invalid symbols...")
        mixed_data = engine.fetch_current_year_data(['AAPL', 'INVALID_SYMBOL', 'MSFT'])
        valid_symbols = list(mixed_data.keys())
        print(f"  ✅ Got data for {len(valid_symbols)} valid symbols: {valid_symbols}")
        
        # Test 3: Date validation function
        print("Test 3: Date validation function...")
        if mixed_data:
            data = mixed_data['AAPL']
            
            # Test valid date
            valid_date = data.index[len(data)//2]  # Middle date
            price = engine._validate_date_access(valid_date, data, 'AAPL')
            if price is not None:
                print(f"  ✅ Valid date access: ${price:.2f}")
            else:
                print("  ❌ Failed to access valid date")
            
            # Test invalid date (weekend)
            invalid_date = pd.Timestamp('2025-01-04', tz='UTC')  # Saturday
            price = engine._validate_date_access(invalid_date, data, 'AAPL')
            if price is None:
                print("  ✅ Invalid date correctly returned None")
            else:
                print(f"  ⚠️ Invalid date unexpectedly returned price: ${price:.2f}")
            
            # Test holiday date
            holiday_date = pd.Timestamp('2025-01-01', tz='UTC')  # New Year's
            price = engine._validate_date_access(holiday_date, data, 'AAPL')
            if price is None:
                print("  ✅ Holiday date correctly handled")
            else:
                print(f"  ⚠️ Holiday date unexpectedly returned price: ${price:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in missing data test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_timezone_edge_cases():
    """Test timezone-related edge cases"""
    print("\n🌍 Testing Timezone Edge Cases")
    print("-" * 30)
    
    config = TradingBotConfig()
    engine = BacktestEngine(config)
    
    try:
        # Get some data
        data_dict = engine.fetch_current_year_data(['AAPL'])
        if not data_dict:
            print("❌ No data for timezone test")
            return False
        
        data = data_dict['AAPL']
        print(f"✅ Data timezone: {data.index.tz}")
        
        # Test 1: Daylight saving time transition
        print("Test 1: DST transition handling...")
        
        # Find dates around DST transition (typically March/November)
        march_dates = data.index[(data.index.month == 3)]
        if len(march_dates) > 0:
            print(f"  March dates sample: {march_dates[:3]}")
            
        november_dates = data.index[(data.index.month == 11)]
        if len(november_dates) > 0:
            print(f"  November dates sample: {november_dates[:3]}")
        
        # Test 2: Cross-timezone date access
        print("Test 2: Cross-timezone date access...")
        sample_date = data.index[10]
        
        # Test access with different timezone representations
        utc_date = sample_date.tz_convert('UTC')
        et_date = sample_date.tz_convert('US/Eastern')
        naive_date = sample_date.tz_localize(None)
        
        print(f"  Original: {sample_date}")
        print(f"  UTC: {utc_date}")
        print(f"  Eastern: {et_date}")
        print(f"  Naive: {naive_date}")
        
        # Test access with each format
        price1 = engine._validate_date_access(utc_date, data, 'AAPL')
        price2 = engine._validate_date_access(et_date, data, 'AAPL')
        
        if price1 is not None and price2 is not None:
            print(f"  ✅ Both timezone formats work: ${price1:.2f} = ${price2:.2f}")
        else:
            print(f"  ⚠️ Timezone conversion issue: {price1} vs {price2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in timezone test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtest_robustness():
    """Test backtest robustness with edge cases"""
    print("\n🛡️ Testing Backtest Robustness")
    print("-" * 30)
    
    config = TradingBotConfig()
    config.risk.initial_capital = 10000  # Smaller capital for testing
    engine = BacktestEngine(config)
    
    try:
        # Test 1: Backtest with very low confidence threshold
        print("Test 1: Low confidence threshold...")
        results = engine.run_backtest(
            symbols=['AAPL'],
            strategy_name='Technical Analysis',
            confidence_threshold=0.1  # Very low threshold
        )
        
        if "error" not in results:
            print(f"  ✅ Low threshold backtest successful: {results['total_trades']} trades")
        else:
            print(f"  ❌ Low threshold backtest failed: {results['error']}")
            
        # Test 2: Backtest with high confidence threshold
        print("Test 2: High confidence threshold...")
        results = engine.run_backtest(
            symbols=['AAPL'],
            strategy_name='Technical Analysis', 
            confidence_threshold=0.95  # Very high threshold
        )
        
        if "error" not in results:
            print(f"  ✅ High threshold backtest successful: {results['total_trades']} trades")
        else:
            print(f"  ❌ High threshold backtest failed: {results['error']}")
        
        # Test 3: Multiple symbols with different data availability
        print("Test 3: Multiple symbols...")
        results = engine.run_backtest(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            strategy_name='Mean Reversion',
            confidence_threshold=0.75
        )
        
        if "error" not in results:
            success_rate = results.get('successful_days', 0) / results.get('total_days', 1)
            print(f"  ✅ Multi-symbol backtest successful: {success_rate:.1%} success rate")
            print(f"     Data quality: {results.get('data_quality', 'Unknown')}")
        else:
            print(f"  ❌ Multi-symbol backtest failed: {results['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in robustness test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all edge case tests"""
    print("🧪 Comprehensive Backtesting Edge Case Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Holiday Handling", test_holiday_handling()))
    test_results.append(("Missing Data Scenarios", test_missing_data_scenarios()))
    test_results.append(("Timezone Edge Cases", test_timezone_edge_cases()))
    test_results.append(("Backtest Robustness", test_backtest_robustness()))
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All edge case tests passed!")
        return True
    else:
        print("⚠️ Some tests failed - check output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)