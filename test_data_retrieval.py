#!/usr/bin/env python3
"""
Test data retrieval and validation logic for optimization engine
Tests the retry logic and data quality validation without needing internet access
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from optimization_engine import OptimizationEngine


class TestDataRetrieval(unittest.TestCase):
    """Test data retrieval and validation logic"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = OptimizationEngine(max_workers=1, log_level=logging.WARNING)
    
    def test_data_quality_validation_valid_data(self):
        """Test data quality validation with valid data"""
        # Create valid test data
        dates = pd.date_range(start='2023-01-01', end='2023-03-01', freq='D')
        df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(105, 115, len(dates)),
            'Low': np.random.uniform(95, 105, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        # Test validation
        result = self.engine._validate_data_quality(df, 'TEST')
        self.assertTrue(result, "Valid data should pass validation")
    
    def test_data_quality_validation_empty_data(self):
        """Test data quality validation with empty data"""
        df = pd.DataFrame()
        result = self.engine._validate_data_quality(df, 'TEST')
        self.assertFalse(result, "Empty data should fail validation")
    
    def test_data_quality_validation_insufficient_days(self):
        """Test data quality validation with insufficient days"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')  # Only 10 days
        df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(105, 115, len(dates)),
            'Low': np.random.uniform(95, 105, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        result = self.engine._validate_data_quality(df, 'TEST')
        self.assertFalse(result, "Insufficient data should fail validation")
    
    def test_data_quality_validation_no_volume(self):
        """Test data quality validation with no volume data"""
        dates = pd.date_range(start='2023-01-01', end='2023-03-01', freq='D')
        df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(105, 115, len(dates)),
            'Low': np.random.uniform(95, 105, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.zeros(len(dates))  # No volume
        }, index=dates)
        
        result = self.engine._validate_data_quality(df, 'TEST')
        self.assertFalse(result, "Data with no volume should fail validation")
    
    def test_data_quality_validation_negative_prices(self):
        """Test data quality validation with negative prices"""
        dates = pd.date_range(start='2023-01-01', end='2023-03-01', freq='D')
        df = pd.DataFrame({
            'Open': np.random.uniform(-10, 10, len(dates)),  # Some negative prices
            'High': np.random.uniform(105, 115, len(dates)),
            'Low': np.random.uniform(95, 105, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        result = self.engine._validate_data_quality(df, 'TEST')
        self.assertFalse(result, "Data with negative prices should fail validation")
    
    @patch('yfinance.Ticker')
    def test_retry_logic_success_on_second_attempt(self, mock_ticker_class):
        """Test retry logic succeeds on second attempt"""
        # Set up mock
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # First call fails, second succeeds
        dates = pd.date_range(start='2023-01-01', end='2023-03-01', freq='D')
        valid_df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(105, 115, len(dates)),
            'Low': np.random.uniform(95, 105, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        mock_ticker.history.side_effect = [
            Exception("Network error"),  # First attempt fails
            valid_df  # Second attempt succeeds
        ]
        
        # Test retry logic
        result = self.engine._get_yahoo_data_with_retry('TEST', 60, max_retries=3)
        
        # Verify we got data
        self.assertIsNotNone(result, "Should get data on retry")
        self.assertEqual(len(result), len(valid_df), "Should return complete dataset")
        
        # Verify we called history twice (first failed, second succeeded)
        self.assertEqual(mock_ticker.history.call_count, 2)
    
    @patch('yfinance.Ticker')
    def test_retry_logic_all_attempts_fail(self, mock_ticker_class):
        """Test retry logic when all attempts fail"""
        # Set up mock
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # All calls fail
        mock_ticker.history.side_effect = Exception("Network error")
        
        # Test retry logic
        result = self.engine._get_yahoo_data_with_retry('TEST', 60, max_retries=2)
        
        # Verify we got None
        self.assertIsNone(result, "Should return None when all attempts fail")
        
        # Verify we tried the right number of times
        self.assertEqual(mock_ticker.history.call_count, 2)
    
    @patch('yfinance.Ticker')
    def test_retry_logic_empty_data_handling(self, mock_ticker_class):
        """Test retry logic handles empty data correctly"""
        # Set up mock
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        # Return empty DataFrame
        mock_ticker.history.return_value = pd.DataFrame()
        
        # Test retry logic
        result = self.engine._get_yahoo_data_with_retry('TEST', 60, max_retries=2)
        
        # Verify we got None for empty data
        self.assertIsNone(result, "Should return None for empty data")
    
    @patch('yfinance.Ticker')
    def test_backtest_with_valid_data(self, mock_ticker_class):
        """Test backtest succeeds with valid data"""
        # Set up mock with valid data
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        
        dates = pd.date_range(start='2023-01-01', end='2023-03-01', freq='D')
        valid_df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(105, 115, len(dates)),
            'Low': np.random.uniform(95, 105, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates)
        
        mock_ticker.history.return_value = valid_df
        
        # Test parameters
        params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std': 2.0,
            'position_size': 0.1,
            'stop_loss': 0.02,
            'starting_capital': 100000
        }
        
        # Run backtest
        result = self.engine._run_backtest('TEST', params, 60)
        
        # Verify we got a result
        self.assertIsNotNone(result, "Should get backtest result with valid data")
        self.assertIn('portfolio_series', result)
        self.assertIn('trades', result)
        self.assertIn('final_value', result)
    
    @patch('yfinance.Ticker')
    def test_backtest_with_network_failure(self, mock_ticker_class):
        """Test backtest fails gracefully with network issues"""
        # Set up mock to always fail
        mock_ticker = Mock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.side_effect = Exception("Could not resolve host")
        
        # Test parameters
        params = {
            'rsi_period': 14,
            'starting_capital': 100000
        }
        
        # Run backtest
        result = self.engine._run_backtest('TEST', params, 60)
        
        # Verify we got None (no fallback to mock data)
        self.assertIsNone(result, "Should return None when network fails")


def run_data_retrieval_tests():
    """Run data retrieval tests"""
    print("🧪 Testing Data Retrieval and Validation Logic")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDataRetrieval)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    if result.wasSuccessful():
        print("\n✅ All data retrieval tests passed!")
        print("🔧 Retry logic and data validation working correctly")
        return True
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed")
        print(f"💥 {len(result.errors)} error(s) occurred")
        return False


if __name__ == "__main__":
    success = run_data_retrieval_tests()
    if not success:
        exit(1)