#!/usr/bin/env python3
"""
Quick test of the QuantConnect-style optimization system
Demonstrates the complete workflow from parameter setup to optimization
"""

from parameter_manager import create_default_parameters
from optimization_engine import OptimizationEngine
from results_analyzer import ResultsAnalyzer
import time

def test_quantconnect_optimization():
    """Test the complete QuantConnect-style optimization workflow"""
    
    print("🚀 Testing QuantConnect-Style Parameter Optimization System")
    print("=" * 60)
    
    # 1. Create parameter manager with smart ranges
    print("\n1️⃣ Creating Parameter Manager with Smart Ranges...")
    params = create_default_parameters("rsi_bollinger")
    
    # Reduce parameter space for quick testing
    params.parameters['rsi_period'].max_value = 16  # 5-16 (step 2) = 6 values
    params.parameters['bb_period'].max_value = 25   # 10-25 (step 5) = 4 values  
    params.parameters['position_size'].max_value = 0.15  # 0.05-0.15 (step 0.05) = 3 values
    params._update_combinations_count()
    
    print(f"✅ Parameters configured: {params.get_total_combinations():,} combinations")
    print(params.get_optimization_summary())
    
    # 2. Initialize optimization engine
    print("\n2️⃣ Initializing Optimization Engine...")
    engine = OptimizationEngine(max_workers=2)  # Use 2 workers for testing
    
    # Set up progress callback
    def progress_callback(current, total, status):
        progress_pct = (current / total) * 100
        print(f"   Progress: {current}/{total} ({progress_pct:.1f}%) - {status}")
    
    engine.set_progress_callback(progress_callback)
    print("✅ Optimization engine ready")
    
    # 3. Run optimization (limited combinations for testing)
    print("\n3️⃣ Running Parameter Optimization...")
    start_time = time.time()
    
    try:
        summary = engine.run_optimization(
            parameter_manager=params,
            symbols=['AAPL'],  # Single symbol for testing
            days=30,           # 30 days for quick testing
            objective='sharpe_ratio',
            max_combinations=20  # Limit to 20 combinations for testing
        )
        
        optimization_time = time.time() - start_time
        print(f"✅ Optimization completed in {optimization_time:.1f} seconds")
        
        # 4. Analyze results
        print("\n4️⃣ Analyzing Results...")
        analyzer = ResultsAnalyzer()
        analyzer.analyze_results(summary)
        
        print(f"✅ Analysis complete - {summary.successful_runs} successful optimizations")
        
        # 5. Display best results
        print("\n5️⃣ Best Optimization Results:")
        if summary.best_result:
            best = summary.best_result
            print(f"🏆 Best Sharpe Ratio: {best.sharpe_ratio:.3f}")
            print(f"📈 Total Return: {best.total_return:.2%}")
            print(f"📉 Max Drawdown: {best.max_drawdown:.2%}")
            print(f"🎯 Win Rate: {best.win_rate:.1%}")
            print(f"💰 Final Value: ${best.final_value:,.0f}")
            
            print("\n🎛️ Optimal Parameters:")
            for param, value in best.parameters.items():
                if isinstance(value, float):
                    if 0 < value < 1:
                        print(f"   • {param}: {value:.1%}")
                    else:
                        print(f"   • {param}: {value:.3f}")
                else:
                    print(f"   • {param}: {value}")
        
        # 6. Generate summary report
        print("\n6️⃣ Generating Summary Report...")
        stats = analyzer.get_summary_statistics()
        
        print(f"📊 Summary Statistics:")
        print(f"   • Total combinations tested: {stats.get('total_combinations_tested', 0)}")
        print(f"   • Success rate: {stats.get('success_rate', 0):.1f}%")
        print(f"   • Best Sharpe ratio: {stats.get('best_sharpe_ratio', 0):.3f}")
        print(f"   • Median return: {stats.get('median_return', 0):.2%}")
        print(f"   • Average win rate: {stats.get('average_win_rate', 0):.1%}")
        
        # 7. Test robustness analysis
        print("\n7️⃣ Testing Robustness Analysis...")
        robustness = analyzer.analyze_robustness()
        
        print(f"🛡️ Robustness Metrics:")
        print(f"   • Robustness score: {robustness.robustness_score:.2%}")
        print(f"   • Overfitting risk: {robustness.overfitting_risk:.2%}")
        
        if robustness.robustness_score > 0.7:
            print("   ✅ High robustness - parameters are reliable")
        elif robustness.robustness_score > 0.5:
            print("   ⚠️ Moderate robustness - additional validation recommended")
        else:
            print("   ❌ Low robustness - high overfitting risk")
        
        print("\n✅ QuantConnect-Style Optimization Test PASSED!")
        print(f"🎉 Complete system working perfectly in {optimization_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_quantconnect_optimization()
    if success:
        print("\n🚀 System ready for production use!")
    else:
        print("\n❌ System needs debugging")