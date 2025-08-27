#!/usr/bin/env python3
"""
Quick test to verify Unified Strategy in Backtesting Engine
"""

from features.backtesting import BacktestEngine
from model_config import TradingBotConfig

def test_unified_strategy_backtest():
    """Test Unified Strategy in backtesting engine"""
    print("🧪 Testing Unified Strategy in Backtesting Engine")
    print("=" * 55)
    
    # Create engine
    config = TradingBotConfig()
    engine = BacktestEngine(config)
    
    # List available strategies
    print(f"📋 Available strategies: {list(engine.strategies.keys())}")
    
    if 'Unified Strategy' not in engine.strategies:
        print("❌ Unified Strategy not found!")
        return False
    
    # Test unified strategy backtest
    print("\n🚀 Running Unified Strategy backtest...")
    try:
        results = engine.run_backtest(
            symbols=['AAPL'], 
            strategy_name='Unified Strategy',
            confidence_threshold=0.3
        )
        
        if 'error' in results:
            print(f"❌ Backtest failed: {results['error']}")
            return False
        
        print("✅ Unified Strategy backtest completed successfully!")
        print(f"📈 Total Return: {results.get('total_return_pct', 0):.2f}%")
        print(f"📊 Sharpe Ratio: {results.get('sharpe_ratio', 0):.4f}")
        print(f"📉 Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
        print(f"🎯 Total Trades: {results.get('total_trades', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return False

if __name__ == "__main__":
    success = test_unified_strategy_backtest()
    print(f"\n{'✅ Test PASSED' if success else '❌ Test FAILED'}")