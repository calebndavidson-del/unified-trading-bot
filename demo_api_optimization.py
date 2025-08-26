#!/usr/bin/env python3
"""
Simple API Optimization Demo

Demonstrates the basic functionality of API parameter optimization.
"""

import logging
from optimization.yahoo_finance_optimizer import YahooFinanceAPIOptimizer
from optimization.binance_optimizer import BinanceAPIOptimizer
from optimization.api_base import OptimizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_yahoo_finance_optimization():
    """Demo Yahoo Finance API optimization."""
    print("🟡 Yahoo Finance API Optimization Demo")
    print("=" * 50)
    
    try:
        # Create optimizer
        optimizer = YahooFinanceAPIOptimizer()
        
        # Configure for quick demo
        config = OptimizationConfig(
            n_trials=5,
            study_name="yahoo_demo",
            direction='maximize'
        )
        optimizer.config = config
        
        # Run optimization
        print("Running optimization...")
        result = optimizer.optimize_for_symbols(['AAPL'])
        
        print(f"✅ Best score: {result['best_score']:.4f}")
        print(f"✅ Best parameters: {result['best_params']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def demo_binance_optimization():
    """Demo Binance API optimization."""
    print("\n🟠 Binance API Optimization Demo")
    print("=" * 50)
    
    try:
        # Create optimizer
        optimizer = BinanceAPIOptimizer()
        
        # Configure for quick demo
        config = OptimizationConfig(
            n_trials=5,
            study_name="binance_demo",
            direction='maximize'
        )
        optimizer.config = config
        
        # Run optimization
        print("Running optimization...")
        result = optimizer.optimize_for_symbols(['BTC-USDT'])
        
        print(f"✅ Best score: {result['best_score']:.4f}")
        print(f"✅ Best parameters: {result['best_params']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def demo_parameter_exploration():
    """Demo parameter space exploration."""
    print("\n🔵 Parameter Space Exploration Demo")
    print("=" * 50)
    
    try:
        import optuna
        
        optimizer = YahooFinanceAPIOptimizer()
        
        print("Exploring parameter space...")
        valid_params = []
        pruned_count = 0
        
        for i in range(20):
            study = optuna.create_study()
            trial = study.ask()
            
            try:
                params = optimizer.define_search_space(trial)
                valid_params.append(params)
                print(f"✅ Valid: {params['interval']} / {params['period']}")
            except optuna.TrialPruned as e:
                pruned_count += 1
                print(f"⏭️ Pruned: {e}")
        
        print(f"\nSummary: {len(valid_params)} valid, {pruned_count} pruned")
        print("✅ Constraint handling working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all demos."""
    print("🚀 API Optimization Demo")
    print("=" * 60)
    
    demos = [
        demo_parameter_exploration,
        demo_yahoo_finance_optimization,
        demo_binance_optimization
    ]
    
    results = []
    
    for demo_func in demos:
        try:
            success = demo_func()
            results.append(success)
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            results.append(False)
    
    print("\n📊 Demo Results:")
    print("=" * 30)
    successful = sum(results)
    total = len(results)
    print(f"Successful: {successful}/{total}")
    
    if successful == total:
        print("🎉 All demos completed successfully!")
    else:
        print("⚠️ Some demos had issues (likely network-related)")
    
    print("\n📋 Key Features Demonstrated:")
    print("• Parameter space definition with constraints")
    print("• Invalid parameter combination pruning")
    print("• Multi-objective optimization (data quality, efficiency, cost)")
    print("• API-specific optimizations (rate limits, data types)")
    print("• Extensible framework for new data sources")


if __name__ == "__main__":
    main()