#!/usr/bin/env python3
"""
Test suite for hyperparameter optimization framework.

This module provides tests for the optimization framework components.
"""

import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization import (
    BaseOptimizer,
    OptimizationConfig,
    OptimizationCache,
    BatchProcessor,
    TrendAnalyzerOptimizer,
    TrendSignalGeneratorOptimizer,
    EarningsFeatureEngineerOptimizer
)


def create_test_data(n_rows=100):
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible tests
    
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='D')
    
    # Generate realistic price movements
    returns = np.random.normal(0.001, 0.02, n_rows)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, n_rows)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_rows))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_rows))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_rows)
    }, index=dates)
    
    # Ensure High >= Low and both contain Open/Close
    data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
    data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
    
    return data


def test_optimization_config():
    """Test OptimizationConfig creation and defaults."""
    config = OptimizationConfig()
    
    assert config.n_trials == 100
    assert config.direction == 'maximize'
    assert config.sampler == 'TPE'
    assert config.n_jobs == 1
    
    # Test custom config
    custom_config = OptimizationConfig(
        n_trials=50,
        direction='minimize',
        sampler='Random'
    )
    
    assert custom_config.n_trials == 50
    assert custom_config.direction == 'minimize'
    assert custom_config.sampler == 'Random'


def test_optimization_cache():
    """Test OptimizationCache functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = OptimizationCache(temp_dir)
        
        # Test cache stats
        stats = cache.get_cache_stats()
        assert stats['total_entries'] == 0
        
        # Test put/get
        data = create_test_data(50)
        params = {'param1': 10, 'param2': 0.5}
        score = 0.75
        
        # Should return None initially
        result = cache.get(params, data, 'TestModel')
        assert result is None
        
        # Store result
        cache.put(params, score, data, 'TestModel')
        
        # Should return stored result
        result = cache.get(params, data, 'TestModel')
        assert result == score
        
        # Test cache stats after insertion
        stats = cache.get_cache_stats()
        assert stats['total_entries'] == 1
        assert 'TestModel' in stats['entries_by_model']


def test_batch_processor():
    """Test BatchProcessor functionality."""
    processor = BatchProcessor(n_workers=1)  # Single worker for testing
    
    # Test parameter grid creation
    param_space = {
        'param1': [1, 2, 3],
        'param2': [0.1, 0.2]
    }
    
    combinations = processor.create_parameter_grid(param_space)
    assert len(combinations) == 6  # 3 * 2 combinations
    
    # Check that all combinations are present
    expected_combinations = [
        {'param1': 1, 'param2': 0.1},
        {'param1': 1, 'param2': 0.2},
        {'param1': 2, 'param2': 0.1},
        {'param1': 2, 'param2': 0.2},
        {'param1': 3, 'param2': 0.1},
        {'param1': 3, 'param2': 0.2}
    ]
    
    for expected in expected_combinations:
        assert expected in combinations
    
    # Test random parameter sampling
    param_space_random = {
        'param1': (1, 10),  # Integer range
        'param2': (0.1, 1.0)  # Float range
    }
    
    random_combinations = processor.create_random_parameter_sample(
        param_space_random, n_samples=5, random_state=42
    )
    
    assert len(random_combinations) == 5
    for combo in random_combinations:
        assert 1 <= combo['param1'] <= 10
        assert 0.1 <= combo['param2'] <= 1.0


def test_trend_analyzer_optimizer():
    """Test TrendAnalyzerOptimizer basic functionality."""
    data = create_test_data(200)  # More data for meaningful optimization
    
    config = OptimizationConfig(n_trials=3, direction='maximize')  # Minimal trials for testing
    
    with tempfile.TemporaryDirectory() as temp_dir:
        optimizer = TrendAnalyzerOptimizer(config=config, cache_dir=temp_dir)
        
        # Test that we can create the optimizer
        assert optimizer.model_class.__name__ == 'OptimizedTrendAnalyzer'
        assert optimizer.config.n_trials == 3
        
        # Test optimization (this is the main test)
        result = optimizer.optimize(data)
        
        # Check result structure
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'best_model' in result
        assert 'n_trials' in result
        
        # Check that we got a valid score
        assert isinstance(result['best_score'], float)
        assert 0 <= result['best_score'] <= 1  # Scores should be normalized
        
        # Check that parameters are within expected ranges
        params = result['best_params']
        assert 5 <= params['short_ma_window'] <= 20
        assert 15 <= params['medium_ma_window'] <= 40
        assert 30 <= params['long_ma_window'] <= 100
        assert 10 <= params['rsi_window'] <= 25


def test_trend_signal_generator_optimizer():
    """Test TrendSignalGeneratorOptimizer basic functionality."""
    data = create_test_data(200)
    
    config = OptimizationConfig(n_trials=3, direction='maximize')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        optimizer = TrendSignalGeneratorOptimizer(config=config, cache_dir=temp_dir)
        
        # Test optimization
        result = optimizer.optimize(data)
        
        # Check result structure
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'best_model' in result
        
        # Check parameter ranges
        params = result['best_params']
        assert 20 <= params['rsi_oversold_threshold'] <= 35
        assert 65 <= params['rsi_overbought_threshold'] <= 80
        assert params['rsi_oversold_threshold'] < params['rsi_overbought_threshold']


def test_earnings_feature_engineer_optimizer():
    """Test EarningsFeatureEngineerOptimizer basic functionality."""
    data = create_test_data(200)
    
    # Add mock earnings data
    data['Actual'] = np.nan
    data['Estimate'] = np.nan
    
    # Add a few earnings points
    earnings_indices = [50, 100, 150]
    for idx in earnings_indices:
        if idx < len(data):
            data.iloc[idx, data.columns.get_loc('Actual')] = np.random.normal(1.0, 0.2)
            data.iloc[idx, data.columns.get_loc('Estimate')] = data.iloc[idx, data.columns.get_loc('Actual')] + np.random.normal(0, 0.1)
    
    config = OptimizationConfig(n_trials=3, direction='maximize')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        optimizer = EarningsFeatureEngineerOptimizer(config=config, cache_dir=temp_dir)
        
        # Test optimization
        result = optimizer.optimize(data)
        
        # Check result structure
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'best_model' in result
        
        # Check parameter ranges
        params = result['best_params']
        assert 5.0 <= params['strong_beat_threshold'] <= 20.0
        assert -20.0 <= params['strong_miss_threshold'] <= -5.0
        assert params['miss_threshold'] > params['strong_miss_threshold']
        assert params['beat_threshold'] < params['strong_beat_threshold']


def test_data_batching():
    """Test data batching functionality."""
    processor = BatchProcessor()
    data = create_test_data(100)
    
    # Test temporal batching
    temporal_batches = processor.create_data_batches(
        data, batch_method='temporal', n_batches=3, overlap_ratio=0.0
    )
    
    assert len(temporal_batches) == 3
    
    # Check that batches cover the data
    total_rows = sum(len(batch) for batch in temporal_batches)
    assert total_rows == len(data)
    
    # Test random batching
    random_batches = processor.create_data_batches(
        data, batch_method='random', n_batches=3, overlap_ratio=0.0
    )
    
    assert len(random_batches) == 3


def run_tests():
    """Run all tests."""
    print("Running hyperparameter optimization tests...")
    
    test_functions = [
        test_optimization_config,
        test_optimization_cache,
        test_batch_processor,
        test_trend_analyzer_optimizer,
        test_trend_signal_generator_optimizer,
        test_earnings_feature_engineer_optimizer,
        test_data_batching
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)