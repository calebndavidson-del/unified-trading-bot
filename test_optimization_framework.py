#!/usr/bin/env python3
"""
Test script for the automated optimization backtesting framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime

from features.backtesting import AutomatedOptimizationBacktest, OptimizationConfig
from features.models import AutoModelSelector
from model_config import TradingBotConfig


class TestAutomatedOptimizationBacktest(unittest.TestCase):
    """Test cases for automated optimization backtesting"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = OptimizationConfig(
            n_trials=5,  # Small number for testing
            study_name="test_optimization",
            symbols=['AAPL', 'MSFT'],
            timeout=60  # 1 minute timeout for testing
        )
        self.optimizer = AutomatedOptimizationBacktest(self.config)
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertIsNotNone(self.optimizer)
        self.assertIsNotNone(self.optimizer.model_selector)
        self.assertEqual(self.optimizer.config.n_trials, 5)
        self.assertEqual(self.optimizer.config.symbols, ['AAPL', 'MSFT'])
    
    def test_model_selector_initialization(self):
        """Test model selector initialization"""
        model_selector = AutoModelSelector()
        
        # Test that available models and strategies are properly defined
        self.assertGreater(len(model_selector.available_models), 0)
        self.assertGreater(len(model_selector.available_strategies), 0)
        
        # Check specific models exist
        self.assertIn('lstm_neural_network', model_selector.available_models)
        self.assertIn('ensemble_ml', model_selector.available_models)
        
        # Check specific strategies exist
        self.assertIn('technical_analysis', model_selector.available_strategies)
        self.assertIn('unified_strategy', model_selector.available_strategies)
    
    @patch('features.backtesting.BacktestEngine')
    def test_objective_function_success(self, mock_backtest_engine):
        """Test objective function with successful backtest"""
        # Mock successful backtest results
        mock_results = {
            'sharpe_ratio': 1.5,
            'total_return': 0.25,
            'win_rate': 0.6,
            'max_drawdown': -0.15,
            'total_trades': 10,
            'profit_factor': 2.0
        }
        
        mock_engine_instance = Mock()
        mock_engine_instance.run_backtest.return_value = mock_results
        mock_backtest_engine.return_value = mock_engine_instance
        
        # Create a mock trial
        mock_trial = Mock()
        mock_trial.number = 1
        mock_trial.suggest_categorical = Mock()
        mock_trial.suggest_float = Mock(return_value=0.75)
        mock_trial.suggest_int = Mock(return_value=5)
        
        # Configure categorical suggestions
        def mock_categorical_side_effect(name, choices):
            if 'model_type' in name:
                return 'lstm_neural_network'
            elif 'strategy_type' in name:
                return 'technical_analysis'
            elif 'backtest_period' in name:
                return '1y'
            else:
                return choices[0] if choices else 'default'
        
        mock_trial.suggest_categorical.side_effect = mock_categorical_side_effect
        
        # Test objective function
        score = self.optimizer._objective_function(mock_trial, ['AAPL', 'MSFT'])
        
        # Should return Sharpe ratio (default objective)
        self.assertEqual(score, 1.5)
        
        # Verify backtest was called
        mock_engine_instance.run_backtest.assert_called_once()
    
    @patch('features.backtesting.BacktestEngine')
    def test_objective_function_failure(self, mock_backtest_engine):
        """Test objective function with failed backtest"""
        # Mock failed backtest results
        mock_results = {'error': 'Data fetch failed'}
        
        mock_engine_instance = Mock()
        mock_engine_instance.run_backtest.return_value = mock_results
        mock_backtest_engine.return_value = mock_engine_instance
        
        # Create a mock trial
        mock_trial = Mock()
        mock_trial.number = 1
        mock_trial.suggest_categorical = Mock(return_value='test_value')
        mock_trial.suggest_float = Mock(return_value=0.75)
        mock_trial.suggest_int = Mock(return_value=5)
        
        # Test objective function
        score = self.optimizer._objective_function(mock_trial, ['AAPL', 'MSFT'])
        
        # Should return -inf for failed backtest
        self.assertEqual(score, -np.inf)
    
    def test_calculate_objective_score(self):
        """Test objective score calculation"""
        # Test case 1: Good performance
        good_results = {
            'sharpe_ratio': 2.0,
            'win_rate': 0.7,
            'max_drawdown': -0.1,
            'total_trades': 15
        }
        
        mock_trial = Mock()
        score = self.optimizer._calculate_objective_score(good_results, mock_trial)
        self.assertEqual(score, 2.0)  # No penalties applied
        
        # Test case 2: Poor win rate
        poor_winrate_results = {
            'sharpe_ratio': 2.0,
            'win_rate': 0.3,  # Low win rate
            'max_drawdown': -0.1,
            'total_trades': 15
        }
        
        score = self.optimizer._calculate_objective_score(poor_winrate_results, mock_trial)
        self.assertEqual(score, 1.6)  # 20% penalty applied
        
        # Test case 3: High drawdown
        high_drawdown_results = {
            'sharpe_ratio': 2.0,
            'win_rate': 0.7,
            'max_drawdown': -0.25,  # High drawdown
            'total_trades': 15
        }
        
        score = self.optimizer._calculate_objective_score(high_drawdown_results, mock_trial)
        self.assertEqual(score, 1.4)  # 30% penalty applied
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        valid_config = OptimizationConfig(
            n_trials=10,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            objective_metric='sharpe_ratio'
        )
        optimizer = AutomatedOptimizationBacktest(valid_config)
        self.assertEqual(optimizer.config.n_trials, 10)
        self.assertEqual(len(optimizer.config.symbols), 3)
        
        # Test default configuration
        default_optimizer = AutomatedOptimizationBacktest()
        self.assertEqual(default_optimizer.config.n_trials, 100)
        self.assertEqual(default_optimizer.config.objective_metric, "sharpe_ratio")


class TestAutoModelSelector(unittest.TestCase):
    """Test cases for automatic model selector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.selector = AutoModelSelector()
    
    def test_model_configurations(self):
        """Test model configurations are properly defined"""
        models = self.selector.available_models
        
        # Check all models have required fields
        for model_name, model_config in models.items():
            self.assertIsNotNone(model_config.name)
            self.assertIsNotNone(model_config.model_class)
            self.assertIsNotNone(model_config.param_ranges)
            self.assertIsNotNone(model_config.description)
            
            # Check parameter ranges are valid
            for param, param_range in model_config.param_ranges.items():
                self.assertIsInstance(param_range, list)
                self.assertGreater(len(param_range), 0)
    
    def test_strategy_configurations(self):
        """Test strategy configurations are properly defined"""
        strategies = self.selector.available_strategies
        
        # Check all strategies have required fields
        for strategy_name, strategy_config in strategies.items():
            self.assertIsNotNone(strategy_config.name)
            self.assertIsNotNone(strategy_config.strategy_class)
            self.assertIsNotNone(strategy_config.param_ranges)
            self.assertIsNotNone(strategy_config.description)
            
            # Check parameter ranges are valid
            for param, param_range in strategy_config.param_ranges.items():
                self.assertIsInstance(param_range, list)
                self.assertEqual(len(param_range), 2)  # Should be min/max pair
    
    def test_suggest_configurations(self):
        """Test configuration suggestion methods"""
        mock_trial = Mock()
        
        # Mock trial methods
        mock_trial.suggest_categorical = Mock(return_value='lstm_neural_network')
        mock_trial.suggest_float = Mock(return_value=0.05)
        mock_trial.suggest_int = Mock(return_value=5)
        
        # Test model config suggestion
        model_config = self.selector.suggest_model_config(mock_trial)
        self.assertIn('model_name', model_config)
        
        # Reset mock for strategy test
        mock_trial.suggest_categorical = Mock(return_value='technical_analysis')
        
        # Test strategy config suggestion
        strategy_config = self.selector.suggest_strategy_config(mock_trial)
        self.assertIn('strategy_name', strategy_config)
        self.assertIn('strategy_class', strategy_config)
        
        # Test backtest config suggestion
        backtest_config = self.selector.suggest_backtest_config(mock_trial)
        self.assertIn('backtest_period', backtest_config)
        self.assertIn('lookback_window', backtest_config)


def run_integration_test():
    """Run a small integration test with real optimization"""
    print("🧪 Running Integration Test for Automated Optimization")
    print("=" * 60)
    
    try:
        # Create a minimal configuration for testing
        config = OptimizationConfig(
            n_trials=3,  # Very small for quick test
            study_name="integration_test",
            symbols=['AAPL'],  # Single symbol for speed
            timeout=120,  # 2 minute timeout
            objective_metric='sharpe_ratio'
        )
        
        print(f"📊 Running optimization with {config.n_trials} trials on {config.symbols}")
        
        # Create optimizer
        optimizer = AutomatedOptimizationBacktest(config)
        
        # Run optimization
        results = optimizer.optimize(symbols=config.symbols, n_trials=config.n_trials)
        
        # Check results
        if "error" in results:
            print(f"❌ Integration test failed: {results['error']}")
            return False
        
        print("✅ Integration test completed successfully!")
        print(f"📈 Best score: {results['optimization_stats']['best_score']:.4f}")
        print(f"📊 Successful trials: {results['optimization_stats']['successful_trials']}")
        print(f"⏱️ Total time: {results['optimization_stats']['optimization_time']:.1f}s")
        
        # Check leaderboard
        if results['leaderboard']:
            print(f"🏆 Top configuration: {results['leaderboard'][0]['model']} + {results['leaderboard'][0]['strategy']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run unit tests
    print("🧪 Running Unit Tests for Automated Optimization")
    print("=" * 60)
    
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 60)
    
    # Run integration test
    integration_success = run_integration_test()
    
    print("\n" + "=" * 60)
    if integration_success:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)