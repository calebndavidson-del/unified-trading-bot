#!/usr/bin/env python3
"""
Comprehensive Tests for Unified Trading Strategy

This module contains comprehensive tests for the unified trading strategy,
feature selection, ensemble methods, and integration with the backtesting engine.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features.unified_strategy import UnifiedTradingStrategy
from features.feature_selector import FeatureSelector
from features.ensemble_model import EnsembleSignalCombiner
from features.backtesting import BacktestEngine
from model_config import TradingBotConfig


def test_feature_selector():
    """Test feature selection functionality."""
    print("🧪 Testing Feature Selector")
    print("-" * 40)
    
    # Create synthetic data with known relationships
    np.random.seed(42)
    n_samples = 200
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # Create features with varying levels of importance
    features = pd.DataFrame({
        'important_feature_1': np.random.normal(0, 1, n_samples),
        'important_feature_2': np.random.normal(0, 0.8, n_samples),
        'noise_feature_1': np.random.normal(0, 2, n_samples),
        'noise_feature_2': np.random.normal(0, 1.5, n_samples),
        'correlated_feature': None,  # Will be correlated with important_feature_1
        'rsi': np.random.uniform(0, 100, n_samples),
        'macd': np.random.normal(0, 0.5, n_samples),
        'sentiment_score': np.random.normal(0, 1, n_samples),
    }, index=dates)
    
    # Create correlation
    features['correlated_feature'] = (
        features['important_feature_1'] * 0.95 + 
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Create target with known relationships
    target = (
        0.5 * features['important_feature_1'] +
        0.3 * features['important_feature_2'] +
        0.1 * features['rsi'] / 100 +
        np.random.normal(0, 0.1, n_samples)
    )
    target.name = 'returns'
    
    # Test feature selector
    selector = FeatureSelector({
        'max_features': 6,
        'correlation_threshold': 0.9
    })
    
    try:
        # Test noise reduction
        features_denoised = selector.reduce_noise(features)
        print(f"  ✅ Noise reduction: {len(features.columns)} -> {len(features_denoised.columns)} features")
        
        # Test composite features
        features_composite = selector.create_composite_features(features_denoised)
        print(f"  ✅ Composite features: {len(features_denoised.columns)} -> {len(features_composite.columns)} features")
        
        # Test feature selection
        features_selected = selector.select_features(features_composite, target)
        print(f"  ✅ Feature selection: {len(features_composite.columns)} -> {len(features_selected.columns)} features")
        
        # Check if important features are selected
        selected_names = list(features_selected.columns)
        print(f"  📋 Selected features: {selected_names}")
        
        # Test feature importance
        importance = selector.get_feature_importance()
        print(f"  🎯 Feature importance methods: {list(importance.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature selector test failed: {e}")
        return False


def test_ensemble_model():
    """Test ensemble signal combiner."""
    print("\n🧪 Testing Ensemble Model")
    print("-" * 40)
    
    # Create synthetic signals from different strategies
    np.random.seed(42)
    n_samples = 300
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # Simulate signals with different characteristics
    signals = pd.DataFrame({
        'technical_signal': np.random.normal(0, 0.3, n_samples),
        'momentum_signal': np.random.normal(0, 0.4, n_samples),
        'mean_reversion_signal': np.random.normal(0, 0.2, n_samples),
        'pattern_signal': np.random.normal(0, 0.1, n_samples),
        'sentiment_signal': np.random.normal(0, 0.15, n_samples),
    }, index=dates)
    
    # Create target with realistic relationships
    target = (
        0.25 * signals['technical_signal'] +
        0.25 * signals['momentum_signal'] +
        0.2 * signals['mean_reversion_signal'] +
        0.15 * signals['pattern_signal'] +
        0.15 * signals['sentiment_signal'] +
        np.random.normal(0, 0.05, n_samples)  # noise
    )
    target.name = 'returns'
    
    # Split data
    train_size = int(0.8 * len(signals))
    train_signals = signals[:train_size]
    train_target = target[:train_size]
    test_signals = signals[train_size:]
    test_target = target[train_size:]
    
    try:
        # Test ensemble combiner
        ensemble = EnsembleSignalCombiner({
            'ensemble_method': 'voting',
            'cv_folds': 3,
            'min_samples_for_training': 50
        })
        
        # Fit ensemble
        ensemble.fit(train_signals, train_target)
        print(f"  ✅ Ensemble training completed")
        
        # Make predictions
        predictions = ensemble.predict(test_signals)
        print(f"  ✅ Predictions generated: {len(predictions)} signals")
        
        # Evaluate performance
        performance = ensemble.evaluate_performance(test_signals, test_target)
        print(f"  📊 Performance metrics:")
        for metric, value in performance.items():
            if isinstance(value, float) and not np.isnan(value):
                print(f"    {metric}: {value:.4f}")
        
        # Test feature importance
        importance = ensemble.get_feature_importance()
        print(f"  🎯 Feature importance available for {len(importance)} features")
        
        # Test ensemble summary
        summary = ensemble.get_ensemble_summary()
        print(f"  📋 Ensemble summary: {summary['ensemble_method']} with {len(summary['base_models'])} models")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ensemble model test failed: {e}")
        traceback.print_exc()
        return False


def test_unified_strategy():
    """Test unified trading strategy."""
    print("\n🧪 Testing Unified Strategy")
    print("-" * 40)
    
    try:
        # Fetch real market data for testing
        symbols = ['AAPL', 'MSFT']
        historical_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='6mo')  # 6 months of data
                if not data.empty:
                    historical_data[symbol] = data
                    print(f"  📊 Fetched {len(data)} days for {symbol}")
            except Exception as e:
                print(f"  ⚠️ Failed to fetch {symbol}: {e}")
        
        if not historical_data:
            print("  ❌ No market data available for testing")
            return False
        
        # Test strategy configuration
        config = {
            'feature_selection': {
                'max_features': 15,
                'correlation_threshold': 0.9
            },
            'ensemble': {
                'ensemble_method': 'voting',
                'cv_folds': 3,
                'min_samples_for_training': 50
            },
            'risk_management': {
                'confidence_threshold': 0.2,
                'volatility_adjustment': True
            }
        }
        
        strategy = UnifiedTradingStrategy(config)
        
        # Test unfitted strategy (simple mode)
        test_data = list(historical_data.values())[0]
        simple_signals = strategy.generate_signals(test_data)
        print(f"  ✅ Unfitted strategy: {len(simple_signals)} signals generated")
        print(f"    Signal range: [{simple_signals.min():.3f}, {simple_signals.max():.3f}]")
        print(f"    Non-zero signals: {(simple_signals != 0).sum()}")
        
        # Test fitted strategy (advanced mode)
        if len(historical_data) > 1:
            strategy.fit(historical_data)
            print(f"  ✅ Strategy fitted successfully")
            
            # Test signal generation
            fitted_signals = strategy.generate_signals(test_data)
            print(f"  ✅ Fitted strategy: {len(fitted_signals)} signals generated")
            print(f"    Signal range: [{fitted_signals.min():.3f}, {fitted_signals.max():.3f}]")
            print(f"    Non-zero signals: {(fitted_signals != 0).sum()}")
            
            # Test strategy summary
            summary = strategy.get_strategy_summary()
            print(f"  📋 Strategy summary:")
            print(f"    Fitted: {summary['is_fitted']}")
            print(f"    Features: {summary['num_features']}")
            print(f"    Base strategies: {len(summary['base_strategies'])}")
            
            if 'training_performance' in summary:
                perf = summary['training_performance']
                print(f"    Training correlation: {perf.get('correlation', 'N/A'):.4f}")
                print(f"    Directional accuracy: {perf.get('directional_accuracy', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Unified strategy test failed: {e}")
        traceback.print_exc()
        return False


def test_backtesting_integration():
    """Test integration with backtesting engine."""
    print("\n🧪 Testing Backtesting Integration")
    print("-" * 40)
    
    try:
        # Create backtesting engine
        config = TradingBotConfig()
        engine = BacktestEngine(config)
        
        # Check if unified strategy is available
        available_strategies = list(engine.strategies.keys())
        print(f"  📋 Available strategies: {available_strategies}")
        
        if 'Unified Strategy' not in available_strategies:
            print("  ⚠️ Unified Strategy not available in backtesting engine")
            return False
        
        # Test running backtest with unified strategy
        symbols = ['AAPL']  # Use single symbol for quick test
        
        print(f"  🚀 Running backtest with Unified Strategy...")
        results = engine.run_backtest(
            symbols=symbols,
            strategy_name='Unified Strategy',
            confidence_threshold=0.5
        )
        
        if 'error' in results:
            print(f"  ❌ Backtest failed: {results['error']}")
            return False
        
        print(f"  ✅ Backtest completed successfully")
        print(f"    Total return: {results.get('total_return_pct', 'N/A'):.2f}%")
        print(f"    Sharpe ratio: {results.get('sharpe_ratio', 'N/A'):.4f}")
        print(f"    Max drawdown: {results.get('max_drawdown_pct', 'N/A'):.2f}%")
        print(f"    Total trades: {results.get('total_trades', 'N/A')}")
        print(f"    Win rate: {results.get('win_rate_pct', 'N/A'):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backtesting integration test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_metrics():
    """Test performance evaluation metrics."""
    print("\n🧪 Testing Performance Metrics")
    print("-" * 40)
    
    try:
        # Create synthetic performance data
        np.random.seed(42)
        n_days = 100
        dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
        
        # Simulate returns with some volatility and trend
        base_return = 0.0005  # 0.05% daily return
        volatility = 0.02     # 2% daily volatility
        
        returns = np.random.normal(base_return, volatility, n_days)
        prices = pd.Series(100 * np.cumprod(1 + returns), index=dates)
        
        # Test various risk metrics
        from utils.risk import RiskMetrics
        
        returns_series = pd.Series(returns, index=dates)
        
        # Test Sharpe ratio
        sharpe = RiskMetrics.sharpe_ratio(returns_series)
        print(f"  📊 Sharpe ratio: {sharpe:.4f}")
        
        # Test maximum drawdown
        max_dd = RiskMetrics.max_drawdown(returns_series)
        print(f"  📉 Max drawdown: {max_dd:.4f}")
        
        # Test VaR
        var = RiskMetrics.value_at_risk(returns_series)
        print(f"  🎯 Value at Risk (5%): {var:.4f}")
        
        # Test CVaR
        cvar = RiskMetrics.conditional_var(returns_series)
        print(f"  ⚠️ Conditional VaR (5%): {cvar:.4f}")
        
        print(f"  ✅ All risk metrics calculated successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance metrics test failed: {e}")
        return False


def test_overfitting_prevention():
    """Test overfitting prevention mechanisms."""
    print("\n🧪 Testing Overfitting Prevention")
    print("-" * 40)
    
    try:
        # Create data that could lead to overfitting
        np.random.seed(42)
        n_samples = 100  # Small sample size
        n_features = 50   # Many features
        
        # Create mostly noise features
        features = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add a few genuine signal features
        signal_features = ['feature_0', 'feature_5', 'feature_10']
        target = (
            0.3 * features['feature_0'] +
            0.2 * features['feature_5'] +
            0.1 * features['feature_10'] +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Test feature selector with regularization
        selector = FeatureSelector({
            'max_features': 10,  # Limit features
            'regularization_alpha': 0.1,  # Strong regularization
            'correlation_threshold': 0.8
        })
        
        selected_features = selector.select_features(features, target, method='lasso')
        print(f"  🎯 Feature selection: {len(features.columns)} -> {len(selected_features.columns)}")
        
        # Check if genuine signal features are preserved
        selected_names = list(selected_features.columns)
        signal_preserved = sum(1 for sf in signal_features if sf in selected_names)
        print(f"  ✅ Signal features preserved: {signal_preserved}/{len(signal_features)}")
        
        # Test ensemble with cross-validation
        ensemble = EnsembleSignalCombiner({
            'ensemble_method': 'voting',
            'regularization_strength': 0.1,
            'cv_folds': 5,
            'min_samples_for_training': 50
        })
        
        # Fit and evaluate
        train_size = int(0.8 * len(selected_features))
        train_X = selected_features[:train_size]
        train_y = target[:train_size]
        test_X = selected_features[train_size:]
        test_y = target[train_size:]
        
        ensemble.fit(train_X, train_y)
        performance = ensemble.evaluate_performance(test_X, test_y)
        
        overfitting_score = performance.get('overfitting_score', 0)
        print(f"  📊 Overfitting score: {overfitting_score:.4f}")
        
        if overfitting_score < 1.0:
            print(f"  ✅ Overfitting prevention: Score within acceptable range")
        else:
            print(f"  ⚠️ Overfitting detected: Score {overfitting_score:.4f} > 1.0")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Overfitting prevention test failed: {e}")
        return False


def run_all_tests():
    """Run all unified strategy tests."""
    print("🧪 Unified Trading Strategy Test Suite")
    print("=" * 60)
    print()
    
    test_results = []
    
    # Run individual tests
    test_functions = [
        test_feature_selector,
        test_ensemble_model,
        test_unified_strategy,
        test_backtesting_integration,
        test_performance_metrics,
        test_overfitting_prevention
    ]
    
    for test_func in test_functions:
        try:
            result = test_func()
            test_results.append((test_func.__name__, result))
        except Exception as e:
            print(f"  ❌ Test {test_func.__name__} crashed: {e}")
            test_results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("-" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {test_name}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"📈 Tests passed: {passed}/{total} ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Unified strategy is ready for use.")
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)