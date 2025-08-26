#!/usr/bin/env python3
"""
Test script for backtesting functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features.backtesting import BacktestEngine, TechnicalAnalysisStrategy
from utils.backtesting_metrics import BacktestingMetrics
from model_config import TradingBotConfig
import pandas as pd

def test_backtesting():
    """Test the backtesting functionality"""
    print("🧪 Testing Backtesting Module")
    print("=" * 50)
    
    # Initialize configuration
    config = TradingBotConfig()
    config.risk.initial_capital = 50000  # Smaller capital for testing
    
    # Initialize backtesting engine
    engine = BacktestEngine(config)
    
    # Test with a small set of symbols
    test_symbols = ['AAPL', 'MSFT']
    
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
        
        # Test trade details
        trade_details = engine.get_trade_details()
        if not trade_details.empty:
            print(f"📊 Trade log generated with {len(trade_details)} trades")
        else:
            print("ℹ️ No completed trades in backtest period")
        
        # Test metrics calculation
        if 'portfolio_history' in results and not results['portfolio_history'].empty:
            advanced_metrics = BacktestingMetrics.calculate_advanced_metrics(results['portfolio_history'])
            print(f"📈 Advanced metrics calculated: {len(advanced_metrics)} metrics")
        
        print("\n🎉 All backtesting tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_backtesting()
    sys.exit(0 if success else 1)