#!/usr/bin/env python3
"""
Yahoo Finance API Optimizer

Optimizes Yahoo Finance API parameters for data quality, efficiency, and reliability.
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.data_sources.yahoo_finance import YahooFinanceAPI
from optimization.api_base import BaseAPIOptimizer, APIOptimizationResult, OptimizationConfig
import logging

logger = logging.getLogger(__name__)


class YahooFinanceAPIOptimizer(BaseAPIOptimizer):
    """
    Optimizer for Yahoo Finance API parameters.
    
    Optimizes:
    - Data fetching intervals and periods
    - Data validation thresholds
    - Batch processing parameters
    - Error handling strategies
    """
    
    def __init__(self, api_key: str = None, config: OptimizationConfig = None):
        """Initialize Yahoo Finance API optimizer."""
        api_instance = YahooFinanceAPI()
        super().__init__(api_instance, config)
        self.supported_intervals = api_instance.supported_intervals
        
        # Default test symbols for optimization
        self.default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Yahoo Finance specific constraints
        self.period_interval_constraints = {
            '1m': ['1d', '5d'],
            '2m': ['1d', '5d'], 
            '5m': ['1d', '5d'],
            '15m': ['1d', '5d'],
            '30m': ['1d', '5d'],
            '60m': ['1d', '5d'],
            '90m': ['1d', '5d'],
            '1h': ['1d', '5d'],
            '1d': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
            '5d': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
            '1wk': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
            '1mo': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
            '3mo': ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        }
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define Yahoo Finance API parameter search space."""
        
        # Data fetching parameters - use fixed categories to avoid dynamic value space issues
        interval = trial.suggest_categorical('interval', [
            '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        ])
        
        # Use all possible periods (Optuna will prune invalid combinations)
        period = trial.suggest_categorical('period', [
            '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        ])
        
        # Validate combination and prune if invalid
        valid_periods = self.period_interval_constraints.get(interval, ['1y'])
        if period not in valid_periods:
            logger.debug(f'Pruning invalid combination: {interval}/{period}')
            raise optuna.TrialPruned(f"Invalid period {period} for interval {interval}")
        
        # Data validation parameters
        params = {
            'interval': interval,
            'period': period,
            
            # Data validation thresholds
            'min_data_points': trial.suggest_int('min_data_points', 10, 1000),
            'max_missing_ratio': trial.suggest_float('max_missing_ratio', 0.0, 0.5),
            'price_change_threshold': trial.suggest_float('price_change_threshold', 0.5, 10.0),  # Max daily % change considered valid
            
            # Retry parameters
            'max_retries': trial.suggest_int('max_retries', 0, 3),
            'retry_delay': trial.suggest_float('retry_delay', 0.1, 2.0),
            
            # Batch processing
            'batch_size': trial.suggest_int('batch_size', 1, 10),
            'batch_delay': trial.suggest_float('batch_delay', 0.0, 1.0),
            
            # Data processing options
            'validate_prices': trial.suggest_categorical('validate_prices', [True, False]),
            'remove_outliers': trial.suggest_categorical('remove_outliers', [True, False]),
            'outlier_std_threshold': trial.suggest_float('outlier_std_threshold', 2.0, 5.0),
        }
        
        return params
    
    def fetch_data_with_params(self, params: Dict[str, Any], 
                              symbols: List[str] = None) -> APIOptimizationResult:
        """Fetch data using Yahoo Finance with specified parameters."""
        if symbols is None:
            symbols = self.default_symbols
        
        start_time = time.time()
        total_data_points = 0
        api_calls = 0
        errors = []
        all_data = []
        
        try:
            # Process symbols in batches
            batch_size = params.get('batch_size', 1)
            batch_delay = params.get('batch_delay', 0.0)
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                
                for symbol in batch_symbols:
                    success = False
                    retries = 0
                    max_retries = params.get('max_retries', 0)
                    retry_delay = params.get('retry_delay', 0.1)
                    
                    while not success and retries <= max_retries:
                        try:
                            # Fetch data
                            data = self.api.fetch_market_data(
                                symbol=symbol,
                                period=params['period'],
                                interval=params['interval']
                            )
                            
                            api_calls += 1
                            
                            if not data.empty:
                                # Apply data validation and processing
                                processed_data = self._process_data(data, params)
                                
                                if len(processed_data) >= params.get('min_data_points', 10):
                                    total_data_points += len(processed_data)
                                    all_data.append(processed_data)
                                    success = True
                                else:
                                    errors.append(f"{symbol}: Insufficient data points ({len(processed_data)})")
                            else:
                                errors.append(f"{symbol}: No data returned")
                                
                        except Exception as e:
                            retries += 1
                            if retries <= max_retries:
                                time.sleep(retry_delay)
                            else:
                                errors.append(f"{symbol}: {str(e)}")
                
                # Batch delay
                if i + batch_size < len(symbols) and batch_delay > 0:
                    time.sleep(batch_delay)
        
        except Exception as e:
            return APIOptimizationResult(
                success=False,
                score=0.0,
                error=str(e),
                metrics={'api_calls': api_calls, 'errors': errors}
            )
        
        fetch_time = time.time() - start_time
        
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
        else:
            combined_data = pd.DataFrame()
        
        # Calculate metrics
        data_quality_score = self.calculate_data_quality_score(combined_data)
        efficiency_score = self.calculate_efficiency_score(fetch_time, total_data_points, api_calls)
        cost_score = self.calculate_cost_score(api_calls, total_data_points)
        error_score = self.calculate_error_score(len(errors) > 0, 'fetch_error' if errors else None)
        
        # Calculate missing data ratio
        requested_symbols = len(symbols)
        successful_symbols = len(all_data)
        missing_ratio = 1.0 - (successful_symbols / requested_symbols) if requested_symbols > 0 else 1.0
        
        metrics = {
            'data_quality_score': data_quality_score,
            'efficiency_score': efficiency_score,
            'cost_score': cost_score,
            'error_score': error_score,
            'fetch_time': fetch_time,
            'total_data_points': total_data_points,
            'api_calls': api_calls,
            'successful_symbols': successful_symbols,
            'requested_symbols': requested_symbols,
            'missing_ratio': missing_ratio,
            'errors': errors[:5]  # Keep only first 5 errors for logging
        }
        
        # Apply missing data penalty
        if missing_ratio > params.get('max_missing_ratio', 0.3):
            data_quality_score *= (1.0 - missing_ratio)
        
        composite_score = self.calculate_composite_score(
            APIOptimizationResult(True, 0.0, combined_data, metrics)
        )
        
        return APIOptimizationResult(
            success=len(all_data) > 0,
            score=composite_score,
            data=combined_data,
            metrics=metrics
        )
    
    def _process_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Process fetched data according to parameters."""
        if data.empty:
            return data
        
        processed_data = data.copy()
        
        # Validate prices if enabled
        if params.get('validate_prices', True):
            processed_data = self._validate_ohlcv_data(processed_data, params)
        
        # Remove outliers if enabled
        if params.get('remove_outliers', False):
            processed_data = self._remove_outliers(processed_data, params)
        
        return processed_data
    
    def _validate_ohlcv_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Enhanced OHLCV data validation."""
        if data.empty:
            return data
        
        validated_data = data.copy()
        price_cols = ['Open', 'High', 'Low', 'Close']
        
        # Basic validation (from original API)
        for col in price_cols:
            if col in validated_data.columns:
                validated_data.loc[validated_data[col] <= 0, col] = np.nan
        
        if 'Volume' in validated_data.columns:
            validated_data.loc[validated_data['Volume'] < 0, 'Volume'] = 0
        
        # High >= Low, Open, Close validation
        if all(col in validated_data.columns for col in price_cols):
            invalid_high = ((validated_data['High'] < validated_data['Low']) | 
                           (validated_data['High'] < validated_data['Open']) | 
                           (validated_data['High'] < validated_data['Close']))
            validated_data.loc[invalid_high, price_cols] = np.nan
        
        # Price change validation
        price_change_threshold = params.get('price_change_threshold', 5.0)
        if 'Close' in validated_data.columns and len(validated_data) > 1:
            price_changes = validated_data['Close'].pct_change().abs()
            extreme_changes = price_changes > (price_change_threshold / 100.0)
            if extreme_changes.any():
                logger.warning(f"Found {extreme_changes.sum()} extreme price changes > {price_change_threshold}%")
                # Mark extreme changes as potentially invalid
                validated_data.loc[extreme_changes, 'extreme_change'] = True
        
        return validated_data
    
    def _remove_outliers(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Remove statistical outliers from data."""
        if data.empty:
            return data
        
        cleaned_data = data.copy()
        std_threshold = params.get('outlier_std_threshold', 3.0)
        
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                col_data = cleaned_data[col]
                mean_val = col_data.mean()
                std_val = col_data.std()
                
                if std_val > 0:  # Avoid division by zero
                    z_scores = np.abs((col_data - mean_val) / std_val)
                    outliers = z_scores > std_threshold
                    
                    if outliers.any():
                        logger.debug(f"Removing {outliers.sum()} outliers from {col}")
                        cleaned_data.loc[outliers, col] = np.nan
        
        return cleaned_data
    
    def optimize_for_asset_class(self, asset_class: str = 'stocks') -> Dict[str, Any]:
        """
        Optimize parameters for a specific asset class.
        
        Args:
            asset_class: 'stocks', 'etfs', 'crypto', or 'forex'
        """
        symbols = self.api.get_supported_symbols(asset_class)
        
        # Use a subset for optimization to keep it manageable
        if len(symbols) > 10:
            symbols = symbols[:10]
        
        return self.optimize_for_symbols(symbols)
    
    def run_parameter_study(self, study_name: str = "yahoo_finance_optimization",
                           n_trials: int = 100) -> optuna.Study:
        """
        Run a comprehensive parameter optimization study.
        
        Args:
            study_name: Name for the Optuna study
            n_trials: Number of trials to run
            
        Returns:
            Completed Optuna study
        """
        config = OptimizationConfig(
            n_trials=n_trials,
            study_name=study_name,
            direction='maximize',
            sampler='TPE',
            pruner='MedianPruner'
        )
        
        self.config = config
        
        # Use default symbols for the study
        result = self.optimize_for_symbols(self.default_symbols)
        
        return self.study


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = YahooFinanceAPIOptimizer()
    
    # Run optimization study
    print("Running Yahoo Finance API optimization...")
    study = optimizer.run_parameter_study(n_trials=20)
    
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Test with best parameters
    best_result = optimizer.fetch_data_with_params(
        study.best_params, 
        ['AAPL', 'MSFT']
    )
    
    print(f"Test result - Success: {best_result.success}")
    print(f"Test metrics: {best_result.metrics}")