#!/usr/bin/env python3
"""
API Optimization Examples

Comprehensive examples showing how to use Optuna for optimizing data source API parameters.
"""

import logging
import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time

# API optimizers
from optimization.yahoo_finance_optimizer import YahooFinanceAPIOptimizer
from optimization.iex_cloud_optimizer import IEXCloudAPIOptimizer
from optimization.alpha_vantage_optimizer import AlphaVantageAPIOptimizer
from optimization.quandl_optimizer import QuandlAPIOptimizer
from optimization.finnhub_optimizer import FinnhubAPIOptimizer
from optimization.binance_optimizer import BinanceAPIOptimizer

from optimization.api_base import OptimizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_yahoo_finance_optimization():
    """Example: Optimize Yahoo Finance API parameters for data quality and efficiency."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Yahoo Finance API Optimization")
    logger.info("=" * 60)
    
    # Create optimizer
    optimizer = YahooFinanceAPIOptimizer()
    
    # Configure optimization
    config = OptimizationConfig(
        n_trials=20,
        study_name="yahoo_finance_example",
        direction='maximize',
        sampler='TPE',
        pruner='MedianPruner'
    )
    
    optimizer.config = config
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    logger.info(f"Optimizing Yahoo Finance parameters for symbols: {test_symbols}")
    
    # Run optimization
    result = optimizer.optimize_for_symbols(test_symbols)
    
    logger.info(f"Best score: {result['best_score']:.4f}")
    logger.info(f"Best parameters: {result['best_params']}")
    
    # Test best parameters
    if result['best_params']:
        test_result = optimizer.fetch_data_with_params(result['best_params'], ['AAPL'])
        logger.info(f"Test with AAPL - Success: {test_result.success}")
        if test_result.metrics:
            logger.info(f"Data quality: {test_result.metrics.get('data_quality_score', 0):.3f}")
            logger.info(f"Efficiency: {test_result.metrics.get('efficiency_score', 0):.3f}")
    
    return result


def example_binance_optimization():
    """Example: Optimize Binance API parameters for cryptocurrency data."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Binance API Optimization")
    logger.info("=" * 60)
    
    # Create optimizer
    optimizer = BinanceAPIOptimizer()
    
    # Configure optimization
    config = OptimizationConfig(
        n_trials=15,
        study_name="binance_crypto_example",
        direction='maximize',
        sampler='TPE',
        pruner='MedianPruner'
    )
    
    optimizer.config = config
    
    # Test crypto symbols
    crypto_symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT']
    
    logger.info(f"Optimizing Binance parameters for crypto symbols: {crypto_symbols}")
    
    # Run optimization
    result = optimizer.optimize_for_symbols(crypto_symbols)
    
    logger.info(f"Best score: {result['best_score']:.4f}")
    logger.info(f"Best parameters: {result['best_params']}")
    
    # Test best parameters
    if result['best_params']:
        test_result = optimizer.fetch_data_with_params(result['best_params'], ['BTC-USDT'])
        logger.info(f"Test with BTC-USDT - Success: {test_result.success}")
        if test_result.metrics:
            logger.info(f"Data quality: {test_result.metrics.get('data_quality_score', 0):.3f}")
            logger.info(f"Weight efficiency: {test_result.metrics.get('weight_efficiency', 0):.1f}")
    
    return result


def example_iex_cloud_optimization():
    """Example: Optimize IEX Cloud API for cost-effectiveness."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: IEX Cloud API Cost Optimization")
    logger.info("=" * 60)
    
    # Create optimizer (sandbox mode)
    optimizer = IEXCloudAPIOptimizer(sandbox=True)
    
    # Focus on cost optimization
    optimizer.set_optimization_weights({
        'data_quality': 0.25,
        'efficiency': 0.25,
        'cost_effectiveness': 0.40,  # Higher weight for cost
        'error_rate': 0.10
    })
    
    # Configure optimization
    config = OptimizationConfig(
        n_trials=10,
        study_name="iex_cloud_cost_example",
        direction='maximize',
        sampler='TPE'
    )
    
    optimizer.config = config
    
    # Test symbols (fewer for cost optimization)
    test_symbols = ['AAPL', 'MSFT']
    
    logger.info(f"Cost-optimizing IEX Cloud parameters for: {test_symbols}")
    logger.info("Note: Requires API key for actual optimization")
    
    # Run optimization
    result = optimizer.optimize_for_symbols(test_symbols)
    
    logger.info(f"Best cost-optimized score: {result['best_score']:.4f}")
    logger.info(f"Best parameters: {result['best_params']}")
    
    return result


def example_alpha_vantage_rate_limit_optimization():
    """Example: Optimize Alpha Vantage API for rate limit compliance."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Alpha Vantage Rate Limit Optimization")
    logger.info("=" * 60)
    
    # Create optimizer
    optimizer = AlphaVantageAPIOptimizer()
    
    # Configure optimization
    config = OptimizationConfig(
        n_trials=8,  # Small number due to rate limits
        study_name="alpha_vantage_rate_limit_example",
        direction='maximize',
        sampler='TPE'
    )
    
    optimizer.config = config
    
    # Test symbols (minimal for rate limit testing)
    test_symbols = ['AAPL']
    
    logger.info(f"Rate limit optimizing Alpha Vantage for: {test_symbols}")
    logger.info("Note: Requires API key and respects 5 calls/minute limit")
    
    # Run optimization
    result = optimizer.optimize_for_symbols(test_symbols)
    
    logger.info(f"Best rate-limit optimized score: {result['best_score']:.4f}")
    logger.info(f"Best parameters: {result['best_params']}")
    
    return result


def example_quandl_dataset_optimization():
    """Example: Optimize Quandl API for dataset selection and quality."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 5: Quandl Dataset Selection Optimization")
    logger.info("=" * 60)
    
    # Create optimizer
    optimizer = QuandlAPIOptimizer()
    
    # Configure optimization
    config = OptimizationConfig(
        n_trials=12,
        study_name="quandl_dataset_example",
        direction='maximize',
        sampler='TPE'
    )
    
    optimizer.config = config
    
    # Test datasets
    test_datasets = ['WIKI/AAPL', 'FRED/GDP', 'LBMA/GOLD']
    
    logger.info(f"Optimizing Quandl dataset parameters for: {test_datasets}")
    logger.info("Note: Requires API key for actual data fetching")
    
    # Run optimization
    result = optimizer.optimize_for_symbols(test_datasets)
    
    logger.info(f"Best dataset optimization score: {result['best_score']:.4f}")
    logger.info(f"Best parameters: {result['best_params']}")
    
    return result


def example_finnhub_resolution_optimization():
    """Example: Optimize Finnhub API for resolution and endpoint selection."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 6: Finnhub Resolution and Endpoint Optimization")
    logger.info("=" * 60)
    
    # Create optimizer
    optimizer = FinnhubAPIOptimizer()
    
    # Configure optimization
    config = OptimizationConfig(
        n_trials=15,
        study_name="finnhub_resolution_example",
        direction='maximize',
        sampler='TPE'
    )
    
    optimizer.config = config
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT']
    
    logger.info(f"Resolution optimizing Finnhub for: {test_symbols}")
    logger.info("Note: Requires API key and respects rate limits")
    
    # Run optimization
    result = optimizer.optimize_for_symbols(test_symbols)
    
    logger.info(f"Best resolution optimization score: {result['best_score']:.4f}")
    logger.info(f"Best parameters: {result['best_params']}")
    
    return result


def example_multi_api_comparison():
    """Example: Compare optimization results across multiple APIs."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 7: Multi-API Optimization Comparison")
    logger.info("=" * 60)
    
    # Common test symbol
    test_symbol = ['AAPL']
    
    # APIs to compare (ones that work without API keys)
    apis = {
        'yahoo_finance': YahooFinanceAPIOptimizer(),
        'binance': BinanceAPIOptimizer()  # For crypto equivalent
    }
    
    results = {}
    
    for api_name, optimizer in apis.items():
        logger.info(f"Optimizing {api_name}...")
        
        # Quick optimization
        config = OptimizationConfig(
            n_trials=10,
            study_name=f"{api_name}_comparison",
            direction='maximize'
        )
        optimizer.config = config
        
        try:
            if api_name == 'binance':
                # Use crypto symbol for Binance
                symbol = ['BTC-USDT']
            else:
                symbol = test_symbol
            
            result = optimizer.optimize_for_symbols(symbol)
            results[api_name] = result
            
            logger.info(f"{api_name} best score: {result['best_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error optimizing {api_name}: {e}")
            results[api_name] = None
    
    # Compare results
    logger.info("\nAPI Optimization Comparison:")
    for api_name, result in results.items():
        if result:
            logger.info(f"{api_name:15}: Score = {result['best_score']:.4f}")
    
    return results


def example_constraint_handling():
    """Example: Demonstrate constraint handling in API optimization."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 8: API Constraint Handling")
    logger.info("=" * 60)
    
    # Yahoo Finance with constraints
    optimizer = YahooFinanceAPIOptimizer()
    
    # Define a study with constraint handling
    def constrained_objective(trial):
        params = optimizer.define_search_space(trial)
        
        # Custom constraint: interval and period must be compatible
        interval = params['interval']
        period = params['period']
        
        # Yahoo Finance constraint validation
        if interval in ['1m', '2m', '5m'] and period not in ['1d', '5d']:
            raise optuna.TrialPruned(f"Invalid combination: {interval} with {period}")
        
        # Simulate fetching with constraint-validated parameters
        try:
            result = optimizer.fetch_data_with_params(params, ['AAPL'])
            return result.score
        except Exception as e:
            logger.debug(f"Trial failed: {e}")
            return 0.0
    
    # Run study with constraints
    study = optuna.create_study(direction='maximize', study_name="constraint_handling_example")
    study.optimize(constrained_objective, n_trials=15)
    
    logger.info(f"Best score with constraints: {study.best_value:.4f}")
    logger.info(f"Best constrained parameters: {study.best_params}")
    
    # Count pruned trials
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    logger.info(f"Pruned trials due to constraints: {len(pruned_trials)}/{len(study.trials)}")
    
    return study


def example_optimization_weights_tuning():
    """Example: Demonstrate optimization weight tuning for different objectives."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 9: Optimization Weight Tuning")
    logger.info("=" * 60)
    
    optimizer = YahooFinanceAPIOptimizer()
    test_symbols = ['AAPL']
    
    # Different optimization objectives
    weight_strategies = {
        'data_quality_focused': {
            'data_quality': 0.60,
            'efficiency': 0.20,
            'cost_effectiveness': 0.15,
            'error_rate': 0.05
        },
        'efficiency_focused': {
            'data_quality': 0.25,
            'efficiency': 0.50,
            'cost_effectiveness': 0.20,
            'error_rate': 0.05
        },
        'balanced': {
            'data_quality': 0.35,
            'efficiency': 0.30,
            'cost_effectiveness': 0.25,
            'error_rate': 0.10
        }
    }
    
    results = {}
    
    for strategy_name, weights in weight_strategies.items():
        logger.info(f"Testing {strategy_name} strategy...")
        
        # Set weights
        optimizer.set_optimization_weights(weights)
        
        # Quick optimization
        config = OptimizationConfig(
            n_trials=8,
            study_name=f"{strategy_name}_weights",
            direction='maximize'
        )
        optimizer.config = config
        
        result = optimizer.optimize_for_symbols(test_symbols)
        results[strategy_name] = result
        
        logger.info(f"{strategy_name} score: {result['best_score']:.4f}")
    
    # Compare strategies
    logger.info("\nWeight Strategy Comparison:")
    for strategy, result in results.items():
        logger.info(f"{strategy:20}: Score = {result['best_score']:.4f}")
    
    return results


def run_all_examples():
    """Run all API optimization examples."""
    logger.info("🚀 Running API Optimization Examples")
    logger.info("=" * 80)
    
    examples = [
        example_yahoo_finance_optimization,
        example_binance_optimization,
        example_iex_cloud_optimization,
        example_alpha_vantage_rate_limit_optimization,
        example_quandl_dataset_optimization,
        example_finnhub_resolution_optimization,
        example_multi_api_comparison,
        example_constraint_handling,
        example_optimization_weights_tuning
    ]
    
    results = {}
    
    for i, example_func in enumerate(examples, 1):
        try:
            logger.info(f"\n{'='*20} Running Example {i} {'='*20}")
            result = example_func()
            results[example_func.__name__] = result
        except Exception as e:
            logger.error(f"Example {i} failed: {e}")
            results[example_func.__name__] = None
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 EXAMPLES SUMMARY")
    logger.info("=" * 80)
    
    for example_name, result in results.items():
        if result:
            if isinstance(result, dict) and 'best_score' in result:
                logger.info(f"✅ {example_name}: Score = {result['best_score']:.4f}")
            else:
                logger.info(f"✅ {example_name}: Completed successfully")
        else:
            logger.info(f"❌ {example_name}: Failed")
    
    logger.info("=" * 80)
    logger.info("🎉 All examples completed!")
    
    return results


if __name__ == "__main__":
    # Run examples
    results = run_all_examples()
    
    print("\n📋 Key Takeaways:")
    print("1. Each API has unique constraints and optimization targets")
    print("2. Rate limiting strategies vary significantly between APIs")
    print("3. Data quality vs. efficiency tradeoffs depend on use case")
    print("4. Constraint handling prevents invalid parameter combinations")
    print("5. Weight tuning allows optimization for specific objectives")
    print("6. Optuna's pruning helps avoid expensive invalid trials")
    print("7. Caching improves optimization efficiency across runs")
    print("8. Multi-API comparison helps select the best data source")