#!/usr/bin/env python3
"""
Example Usage of Hyperparameter Optimization Framework

This module demonstrates how to use the Optuna-based hyperparameter optimization
framework for trading bot models.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any
import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization import (
    OptimizationConfig,
    TrendAnalyzerOptimizer,
    TrendSignalGeneratorOptimizer,
    EarningsFeatureEngineerOptimizer,
    OptimizationCache,
    BatchProcessor
)


def download_sample_data(symbol: str = "AAPL", period: str = "2y") -> pd.DataFrame:
    """Download sample data for optimization."""
    logger.info(f"Downloading data for {symbol}")
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    
    # Ensure we have enough data
    if len(data) < 100:
        logger.warning(f"Limited data available for {symbol}: {len(data)} rows")
    
    return data


def example_trend_analyzer_optimization():
    """Example: Optimize TrendAnalyzer hyperparameters."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: TrendAnalyzer Optimization")
    logger.info("=" * 60)
    
    # Download sample data
    data = download_sample_data("AAPL", "1y")
    
    # Split data into train/validation
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    # Configure optimization
    config = OptimizationConfig(
        n_trials=50,  # Reduced for example
        direction='maximize',
        sampler='TPE',
        study_name='trend_analyzer_example'
    )
    
    # Create optimizer
    optimizer = TrendAnalyzerOptimizer(config=config)
    
    # Run optimization
    logger.info(f"Running optimization with {config.n_trials} trials...")
    result = optimizer.optimize(train_data, val_data)
    
    # Display results
    logger.info(f"Best score: {result['best_score']:.4f}")
    logger.info(f"Best parameters: {result['best_params']}")
    logger.info(f"Optimization time: {result['optimization_time']:.2f} seconds")
    
    # Create optimized model
    best_model = result['best_model']
    logger.info(f"Created optimized TrendAnalyzer with parameters: {best_model.hyperparams}")
    
    # Test the optimized model
    test_data = best_model.identify_trend_direction(val_data)
    test_data = best_model.calculate_momentum_indicators(test_data)
    logger.info(f"Generated features for validation data: {test_data.shape}")
    
    return result


def example_trend_signal_generator_optimization():
    """Example: Optimize TrendSignalGenerator hyperparameters."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: TrendSignalGenerator Optimization")
    logger.info("=" * 60)
    
    # Download sample data
    data = download_sample_data("MSFT", "1y")
    
    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    # Configure optimization
    config = OptimizationConfig(
        n_trials=30,
        direction='maximize',
        sampler='TPE',
        study_name='signal_generator_example'
    )
    
    # Create optimizer
    optimizer = TrendSignalGeneratorOptimizer(config=config)
    
    # Run optimization
    logger.info(f"Running signal optimization with {config.n_trials} trials...")
    result = optimizer.optimize(train_data, val_data)
    
    # Display results
    logger.info(f"Best score: {result['best_score']:.4f}")
    logger.info(f"Best parameters: {result['best_params']}")
    
    # Test signal generation
    best_model = result['best_model']
    
    # Note: TrendSignalGenerator needs technical indicators first
    from optimization.trend_analyzer_optimizer import OptimizedTrendAnalyzer
    analyzer = OptimizedTrendAnalyzer()
    
    # Add indicators to validation data
    enhanced_val = analyzer.identify_trend_direction(val_data)
    enhanced_val = analyzer.calculate_momentum_indicators(enhanced_val)
    enhanced_val = analyzer.calculate_volatility_indicators(enhanced_val)
    
    # Generate signals
    signals = best_model.generate_composite_signals(enhanced_val)
    
    # Count signals generated
    bullish_signals = signals['bullish_signal_strength'].sum()
    bearish_signals = signals['bearish_signal_strength'].sum()
    
    logger.info(f"Generated signals - Bullish: {bullish_signals:.1f}, Bearish: {bearish_signals:.1f}")
    
    return result


def example_earnings_feature_engineer_optimization():
    """Example: Optimize EarningsFeatureEngineer hyperparameters."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: EarningsFeatureEngineer Optimization")
    logger.info("=" * 60)
    
    # Download sample data
    data = download_sample_data("GOOGL", "1y")
    
    # Create mock earnings data for demonstration
    earnings_data = create_mock_earnings_data(data)
    
    # Split data
    split_idx = int(len(earnings_data) * 0.8)
    train_data = earnings_data.iloc[:split_idx]
    val_data = earnings_data.iloc[split_idx:]
    
    # Configure optimization
    config = OptimizationConfig(
        n_trials=25,
        direction='maximize',
        sampler='TPE',
        study_name='earnings_engineer_example'
    )
    
    # Create optimizer
    optimizer = EarningsFeatureEngineerOptimizer(config=config)
    
    # Run optimization
    logger.info(f"Running earnings optimization with {config.n_trials} trials...")
    result = optimizer.optimize(train_data, val_data)
    
    # Display results
    logger.info(f"Best score: {result['best_score']:.4f}")
    logger.info(f"Best parameters: {result['best_params']}")
    
    # Test earnings feature engineering
    best_model = result['best_model']
    
    # Extract earnings subset for feature generation
    earnings_subset = val_data[['Actual', 'Estimate']].dropna()
    
    if not earnings_subset.empty:
        surprise_features = best_model.calculate_earnings_surprise(earnings_subset)
        growth_features = best_model.calculate_earnings_growth(surprise_features)
        
        logger.info(f"Generated earnings features: {list(growth_features.columns)}")
    
    return result


def example_batch_optimization():
    """Example: Use batch processing for robust optimization."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Batch Optimization")
    logger.info("=" * 60)
    
    # Download data for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    datasets = []
    
    for symbol in symbols:
        try:
            data = download_sample_data(symbol, "1y")
            if len(data) > 100:
                datasets.append(data)
                logger.info(f"Added {symbol}: {len(data)} rows")
        except Exception as e:
            logger.warning(f"Failed to download {symbol}: {e}")
    
    if len(datasets) < 2:
        logger.warning("Not enough datasets for batch optimization")
        return None
    
    # Configure optimization
    config = OptimizationConfig(
        n_trials=60,  # Total trials across all batches
        direction='maximize',
        sampler='TPE',
        study_name='batch_optimization_example'
    )
    
    # Create optimizer
    optimizer = TrendAnalyzerOptimizer(config=config)
    
    # Run batch optimization
    logger.info(f"Running batch optimization across {len(datasets)} datasets...")
    result = optimizer.optimize_batch(datasets)
    
    # Display results
    logger.info(f"Best overall score: {result['best_score']:.4f}")
    logger.info(f"Best batch index: {result['best_batch_idx']}")
    logger.info(f"All batch scores: {[f'{s:.4f}' for s in result['all_scores']]}")
    logger.info(f"Best parameters: {result['best_params']}")
    
    return result


def example_caching_usage():
    """Example: Demonstrate caching functionality."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 5: Caching System Usage")
    logger.info("=" * 60)
    
    # Initialize cache
    cache = OptimizationCache(".example_cache")
    
    # Get cache statistics
    stats = cache.get_cache_stats()
    logger.info(f"Initial cache stats: {stats}")
    
    # Run a small optimization to populate cache
    data = download_sample_data("AAPL", "6m")
    
    config = OptimizationConfig(
        n_trials=10,
        direction='maximize',
        sampler='TPE'
    )
    
    optimizer = TrendAnalyzerOptimizer(config=config, cache_dir=".example_cache")
    
    # First run (will populate cache)
    logger.info("First optimization run (populating cache)...")
    result1 = optimizer.optimize(data)
    
    # Check cache after first run
    stats = cache.get_cache_stats()
    logger.info(f"Cache stats after first run: {stats}")
    
    # Second run (should use cache)
    logger.info("Second optimization run (using cache)...")
    result2 = optimizer.optimize(data)
    
    logger.info(f"First run time: {result1['optimization_time']:.2f}s")
    logger.info(f"Second run time: {result2['optimization_time']:.2f}s")
    
    # Get best cached entry
    best_entry = cache.get_best_entry('OptimizedTrendAnalyzer', 'maximize')
    if best_entry:
        logger.info(f"Best cached score: {best_entry.score:.4f}")
        logger.info(f"Best cached params: {best_entry.params}")
    
    return cache


def example_parameter_grid_search():
    """Example: Use batch processor for grid search."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 6: Parameter Grid Search")
    logger.info("=" * 60)
    
    # Download sample data
    data = download_sample_data("AAPL", "1y")
    
    # Define parameter grid
    param_space = {
        'short_ma_window': [5, 10, 15],
        'long_ma_window': [20, 30, 40],
        'rsi_window': [14, 21],
        'bb_std_dev': [2.0, 2.5]
    }
    
    # Create batch processor
    batch_processor = BatchProcessor(n_workers=2)
    
    # Generate parameter combinations
    param_combinations = batch_processor.create_parameter_grid(param_space)
    logger.info(f"Generated {len(param_combinations)} parameter combinations")
    
    # Define model factory and evaluation function
    from optimization.trend_analyzer_optimizer import OptimizedTrendAnalyzer
    
    def model_factory(params):
        return OptimizedTrendAnalyzer(**params)
    
    def evaluate_func(model, data):
        # Simple evaluation: calculate trend strength variance
        try:
            enhanced = model.identify_trend_direction(data)
            if 'trend_strength' in enhanced.columns:
                trend_data = enhanced['trend_strength'].dropna()
                if len(trend_data) > 10:
                    return 1.0 / (1.0 + trend_data.std())  # Lower variance is better
            return 0.5
        except:
            return 0.0
    
    # Process parameter batch
    logger.info("Processing parameter grid...")
    results = batch_processor.process_parameter_batch(
        param_combinations[:12],  # Limit to first 12 for example
        model_factory,
        evaluate_func,
        data
    )
    
    # Find best result
    successful_results = [r for r in results if r.success]
    if successful_results:
        best_result = max(successful_results, key=lambda r: r.score)
        logger.info(f"Best grid search score: {best_result.score:.4f}")
        logger.info(f"Best grid search params: {best_result.params}")
        logger.info(f"Success rate: {len(successful_results)}/{len(results)}")
    
    return results


def create_mock_earnings_data(price_data: pd.DataFrame) -> pd.DataFrame:
    """Create mock earnings data for demonstration."""
    data = price_data.copy()
    
    # Add mock earnings columns
    n_rows = len(data)
    n_earnings = max(1, n_rows // 60)  # Quarterly earnings
    
    # Initialize columns
    data['Actual'] = np.nan
    data['Estimate'] = np.nan
    
    # Add earnings data at regular intervals
    for i in range(n_earnings):
        idx = min(i * 60, n_rows - 1)
        actual_eps = np.random.normal(1.0, 0.3)
        estimate_eps = actual_eps + np.random.normal(0, 0.1)
        
        data.iloc[idx, data.columns.get_loc('Actual')] = actual_eps
        data.iloc[idx, data.columns.get_loc('Estimate')] = estimate_eps
    
    return data


def run_all_examples():
    """Run all optimization examples."""
    logger.info("🚀 Running Hyperparameter Optimization Examples")
    logger.info("=" * 80)
    
    results = {}
    
    try:
        # Example 1: TrendAnalyzer
        results['trend_analyzer'] = example_trend_analyzer_optimization()
    except Exception as e:
        logger.error(f"TrendAnalyzer example failed: {e}")
    
    try:
        # Example 2: TrendSignalGenerator
        results['signal_generator'] = example_trend_signal_generator_optimization()
    except Exception as e:
        logger.error(f"TrendSignalGenerator example failed: {e}")
    
    try:
        # Example 3: EarningsFeatureEngineer
        results['earnings_engineer'] = example_earnings_feature_engineer_optimization()
    except Exception as e:
        logger.error(f"EarningsFeatureEngineer example failed: {e}")
    
    try:
        # Example 4: Batch optimization
        results['batch_optimization'] = example_batch_optimization()
    except Exception as e:
        logger.error(f"Batch optimization example failed: {e}")
    
    try:
        # Example 5: Caching
        results['caching'] = example_caching_usage()
    except Exception as e:
        logger.error(f"Caching example failed: {e}")
    
    try:
        # Example 6: Grid search
        results['grid_search'] = example_parameter_grid_search()
    except Exception as e:
        logger.error(f"Grid search example failed: {e}")
    
    # Summary
    logger.info("=" * 80)
    logger.info("📊 EXAMPLES SUMMARY")
    logger.info("=" * 80)
    
    successful_examples = [name for name, result in results.items() if result is not None]
    logger.info(f"Successful examples: {len(successful_examples)}/{len(results)}")
    logger.info(f"Examples run: {', '.join(successful_examples)}")
    
    return results


if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()
    
    # Additional usage tips
    logger.info("\n" + "=" * 80)
    logger.info("💡 USAGE TIPS")
    logger.info("=" * 80)
    logger.info("1. Increase n_trials for better optimization results (50-200 recommended)")
    logger.info("2. Use validation data different from training data for unbiased evaluation")
    logger.info("3. Cache results are stored persistently and reused across runs")
    logger.info("4. Batch optimization provides more robust parameter selection")
    logger.info("5. Monitor optimization progress with study.best_value over trials")
    logger.info("6. Use different samplers (TPE, Random, CmaEs) for different search strategies")
    logger.info("7. Set timeouts for long-running optimizations")
    logger.info("8. Export/import cache for sharing optimization results")
    logger.info("=" * 80)