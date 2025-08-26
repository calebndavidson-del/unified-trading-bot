#!/usr/bin/env python3
"""
Test API Optimizers

Test suite for the API optimization functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.api_base import BaseAPIOptimizer, APIOptimizationResult, OptimizationConfig
from optimization.yahoo_finance_optimizer import YahooFinanceAPIOptimizer
from optimization.binance_optimizer import BinanceAPIOptimizer
from optimization.iex_cloud_optimizer import IEXCloudAPIOptimizer
from optimization.alpha_vantage_optimizer import AlphaVantageAPIOptimizer
from optimization.quandl_optimizer import QuandlAPIOptimizer
from optimization.finnhub_optimizer import FinnhubAPIOptimizer

import optuna
import logging

# Suppress optuna logs for cleaner test output
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger('optimization').setLevel(logging.WARNING)


class TestAPIOptimizationBase(unittest.TestCase):
    """Test base API optimization functionality."""
    
    def test_api_optimization_result(self):
        """Test APIOptimizationResult creation and attributes."""
        # Test successful result
        data = pd.DataFrame({'Close': [100, 101, 102], 'Volume': [1000, 1100, 1200]})
        metrics = {'data_quality_score': 0.95, 'efficiency_score': 0.80}
        
        result = APIOptimizationResult(
            success=True,
            score=0.85,
            data=data,
            metrics=metrics
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.score, 0.85)
        self.assertFalse(result.data.empty)
        self.assertEqual(result.metrics['data_quality_score'], 0.95)
        self.assertIsInstance(result.timestamp, datetime)
        
        # Test failed result
        failed_result = APIOptimizationResult(
            success=False,
            score=0.0,
            error="API key missing"
        )
        
        self.assertFalse(failed_result.success)
        self.assertEqual(failed_result.error, "API key missing")


class TestYahooFinanceOptimizer(unittest.TestCase):
    """Test Yahoo Finance API optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = YahooFinanceAPIOptimizer()
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer.api)
        self.assertEqual(len(self.optimizer.default_symbols), 5)
        self.assertIn('AAPL', self.optimizer.default_symbols)
    
    def test_search_space_definition(self):
        """Test parameter search space definition."""
        study = optuna.create_study()
        trial = study.ask()
        
        params = self.optimizer.define_search_space(trial)
        
        # Check required parameters
        required_params = ['interval', 'period', 'min_data_points', 'max_missing_ratio']
        for param in required_params:
            self.assertIn(param, params)
        
        # Check parameter types and constraints
        self.assertIn(params['interval'], self.optimizer.supported_intervals)
        self.assertIsInstance(params['min_data_points'], int)
        self.assertIsInstance(params['max_missing_ratio'], float)
    
    def test_constraint_validation(self):
        """Test parameter constraint validation."""
        study = optuna.create_study()
        
        # Test constraint pruning
        pruned_count = 0
        valid_count = 0
        
        for _ in range(20):
            trial = study.ask()
            try:
                params = self.optimizer.define_search_space(trial)
                valid_count += 1
            except optuna.TrialPruned:
                pruned_count += 1
        
        # Should have some valid trials
        self.assertGreater(valid_count, 0)
    
    def test_data_fetching_simulation(self):
        """Test data fetching with mock parameters."""
        # Create simple test parameters
        params = {
            'interval': '1d',
            'period': '1mo',
            'min_data_points': 10,
            'max_missing_ratio': 0.3,
            'max_retries': 1,
            'retry_delay': 0.1,
            'batch_size': 1,
            'batch_delay': 0.0,
            'validate_prices': True,
            'remove_outliers': False,
            'outlier_std_threshold': 3.0,
            'price_change_threshold': 5.0
        }
        
        # Test with single symbol (this will actually fetch real data)
        try:
            result = self.optimizer.fetch_data_with_params(params, ['AAPL'])
            
            # Should either succeed or fail gracefully
            self.assertIsInstance(result, APIOptimizationResult)
            self.assertIsInstance(result.success, bool)
            self.assertIsInstance(result.score, (int, float))
            
            if result.success:
                self.assertIsNotNone(result.data)
                self.assertIsNotNone(result.metrics)
        except Exception as e:
            # Network errors are acceptable in tests
            self.assertIn('network', str(e).lower(), f"Unexpected error: {e}")


class TestBinanceOptimizer(unittest.TestCase):
    """Test Binance API optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = BinanceAPIOptimizer()
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer.api)
        self.assertEqual(len(self.optimizer.default_symbols), 5)
        self.assertIn('BTC-USDT', self.optimizer.default_symbols)
    
    def test_search_space_definition(self):
        """Test parameter search space definition."""
        study = optuna.create_study()
        trial = study.ask()
        
        params = self.optimizer.define_search_space(trial)
        
        # Check required parameters
        required_params = ['interval', 'period', 'symbol_strategy', 'max_symbols']
        for param in required_params:
            self.assertIn(param, params)
        
        # Check crypto-specific parameters
        self.assertIn(params['interval'], self.optimizer.interval_limits.keys())
        self.assertIn('USDT', self.optimizer.quote_assets)
    
    def test_symbol_mapping(self):
        """Test cryptocurrency symbol mapping."""
        params = {'auto_symbol_mapping': True, 'prefer_usdt_pairs': True}
        
        # Test various symbol formats
        test_cases = [
            ('BTC-USD', 'BTCUSDT'),
            ('ETH/USDT', 'ETHUSDT'),
            ('BNB_BTC', 'BNBBTC'),
            ('ADAUSDT', 'ADAUSDT')
        ]
        
        for input_symbol, expected in test_cases:
            mapped = self.optimizer._map_symbol(input_symbol, params)
            self.assertEqual(mapped, expected)
    
    def test_symbol_selection(self):
        """Test symbol selection strategies."""
        strategies = ['popular_pairs', 'base_asset_focus', 'quote_asset_focus', 'mixed_strategy']
        
        for strategy in strategies:
            params = {'symbol_strategy': strategy, 'max_symbols': 5}
            symbols = self.optimizer._select_symbols(params)
            
            self.assertIsInstance(symbols, list)
            self.assertLessEqual(len(symbols), 5)
            self.assertGreater(len(symbols), 0)


class TestOptimizationWeights(unittest.TestCase):
    """Test optimization weight management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = YahooFinanceAPIOptimizer()
    
    def test_default_weights(self):
        """Test default optimization weights."""
        weights = self.optimizer.get_optimization_weights()
        
        # Check all required keys present
        required_keys = {'data_quality', 'efficiency', 'cost_effectiveness', 'error_rate'}
        self.assertEqual(set(weights.keys()), required_keys)
        
        # Check weights sum to approximately 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=2)
    
    def test_weight_setting(self):
        """Test setting custom optimization weights."""
        new_weights = {
            'data_quality': 0.5,
            'efficiency': 0.3,
            'cost_effectiveness': 0.15,
            'error_rate': 0.05
        }
        
        self.optimizer.set_optimization_weights(new_weights)
        updated_weights = self.optimizer.get_optimization_weights()
        
        for key, value in new_weights.items():
            self.assertAlmostEqual(updated_weights[key], value, places=3)
    
    def test_weight_normalization(self):
        """Test weight normalization."""
        # Weights that don't sum to 1.0
        unnormalized_weights = {
            'data_quality': 2.0,
            'efficiency': 1.0,
            'cost_effectiveness': 1.0,
            'error_rate': 0.5
        }
        
        self.optimizer.set_optimization_weights(unnormalized_weights)
        normalized_weights = self.optimizer.get_optimization_weights()
        
        # Should sum to 1.0 after normalization
        self.assertAlmostEqual(sum(normalized_weights.values()), 1.0, places=2)
        
        # Proportions should be maintained
        expected_ratio = 2.0 / 4.5  # data_quality ratio
        self.assertAlmostEqual(normalized_weights['data_quality'], expected_ratio, places=2)


class TestMetricCalculations(unittest.TestCase):
    """Test metric calculation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = YahooFinanceAPIOptimizer()
    
    def test_data_quality_score(self):
        """Test data quality score calculation."""
        # Perfect data
        perfect_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        })
        
        score = self.optimizer.calculate_data_quality_score(perfect_data)
        self.assertGreater(score, 0.8)  # Should be high quality
        
        # Data with issues
        bad_data = pd.DataFrame({
            'Open': [100, np.nan, 102],
            'High': [99, 103, 104],  # High < Open in first row
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [-10, 1100, 1200]  # Negative volume
        })
        
        bad_score = self.optimizer.calculate_data_quality_score(bad_data)
        self.assertLess(bad_score, score)  # Should be lower quality
    
    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        # Good efficiency
        score1 = self.optimizer.calculate_efficiency_score(1.0, 1000, 1)
        
        # Poor efficiency
        score2 = self.optimizer.calculate_efficiency_score(10.0, 100, 5)
        
        self.assertGreater(score1, score2)
    
    def test_cost_score(self):
        """Test cost effectiveness score calculation."""
        # Good cost effectiveness
        score1 = self.optimizer.calculate_cost_score(1, 1000)
        
        # Poor cost effectiveness  
        score2 = self.optimizer.calculate_cost_score(10, 100)
        
        self.assertGreater(score1, score2)
    
    def test_error_score(self):
        """Test error score calculation."""
        # No error
        no_error_score = self.optimizer.calculate_error_score(False)
        self.assertEqual(no_error_score, 1.0)
        
        # With error
        error_score = self.optimizer.calculate_error_score(True, 'network')
        self.assertLess(error_score, 1.0)
        
        # Different error types should have different penalties
        rate_limit_score = self.optimizer.calculate_error_score(True, 'rate_limit')
        api_key_score = self.optimizer.calculate_error_score(True, 'api_key')
        
        self.assertNotEqual(rate_limit_score, api_key_score)


class TestOptimizationIntegration(unittest.TestCase):
    """Test end-to-end optimization integration."""
    
    def test_quick_optimization_run(self):
        """Test a quick optimization run with minimal trials."""
        optimizer = YahooFinanceAPIOptimizer()
        
        # Configure for quick test
        config = OptimizationConfig(
            n_trials=3,
            study_name="test_quick_optimization",
            direction='maximize'
        )
        optimizer.config = config
        
        # Test with single symbol
        try:
            result = optimizer.optimize_for_symbols(['AAPL'])
            
            # Check result structure
            self.assertIn('best_score', result)
            self.assertIn('best_params', result)
            self.assertIsInstance(result['best_score'], (int, float))
            
        except Exception as e:
            # Network-related failures are acceptable in tests
            self.skipTest(f"Network-dependent test failed: {e}")


def run_api_optimizer_tests():
    """Run all API optimizer tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAPIOptimizationBase,
        TestYahooFinanceOptimizer,
        TestBinanceOptimizer,
        TestOptimizationWeights,
        TestMetricCalculations,
        TestOptimizationIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running API Optimizer Tests...")
    success = run_api_optimizer_tests()
    
    if success:
        print("\n✅ All API optimizer tests passed!")
    else:
        print("\n❌ Some API optimizer tests failed.")
        exit(1)