#!/usr/bin/env python3
"""
Alpha Vantage API Optimizer

Optimizes Alpha Vantage API parameters for data quality, efficiency, and rate limit management.
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

from features.data_sources.alpha_vantage import AlphaVantageAPI
from optimization.api_base import BaseAPIOptimizer, APIOptimizationResult, OptimizationConfig
import logging

logger = logging.getLogger(__name__)


class AlphaVantageAPIOptimizer(BaseAPIOptimizer):
    """
    Optimizer for Alpha Vantage API parameters.
    
    Optimizes:
    - Function selection and output size parameters
    - Rate limiting strategies (strict 5 calls/minute limit)
    - Data endpoint combinations
    - Technical indicator parameters
    """
    
    def __init__(self, api_key: str = None, config: OptimizationConfig = None):
        """Initialize Alpha Vantage API optimizer."""
        api_instance = AlphaVantageAPI(api_key=api_key)
        super().__init__(api_instance, config)
        
        # Default test symbols for optimization
        self.default_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Fewer symbols due to rate limits
        
        # Alpha Vantage specific constraints
        self.time_series_functions = {
            "1min": "TIME_SERIES_INTRADAY",
            "5min": "TIME_SERIES_INTRADAY", 
            "15min": "TIME_SERIES_INTRADAY",
            "30min": "TIME_SERIES_INTRADAY",
            "60min": "TIME_SERIES_INTRADAY",
            "1d": "TIME_SERIES_DAILY",
            "1wk": "TIME_SERIES_WEEKLY",
            "1mo": "TIME_SERIES_MONTHLY"
        }
        
        self.technical_indicators = [
            'sma', 'ema', 'rsi', 'macd', 'stoch', 'adx', 'cci', 'bbands'
        ]
        
        # Set optimization weights (efficiency less important due to rate limits)
        self.set_optimization_weights({
            'data_quality': 0.45,
            'efficiency': 0.15,  # Lower weight due to unavoidable rate limits
            'cost_effectiveness': 0.30,
            'error_rate': 0.10
        })
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define Alpha Vantage API parameter search space."""
        
        # Data fetching parameters
        interval = trial.suggest_categorical('interval', list(self.time_series_functions.keys()))
        outputsize = trial.suggest_categorical('outputsize', ['compact', 'full'])
        
        # Endpoint selection
        fetch_market_data = trial.suggest_categorical('fetch_market_data', [True, False])
        fetch_company_overview = trial.suggest_categorical('fetch_company_overview', [True, False])
        fetch_technical_indicators = trial.suggest_categorical('fetch_technical_indicators', [True, False])
        
        # At least one endpoint must be selected
        if not any([fetch_market_data, fetch_company_overview, fetch_technical_indicators]):
            fetch_market_data = True
        
        # Technical indicator selection (if enabled)
        selected_indicators = []
        if fetch_technical_indicators:
            num_indicators = trial.suggest_int('num_indicators', 1, 3)
            for i in range(num_indicators):
                indicator = trial.suggest_categorical(f'indicator_{i}', self.technical_indicators)
                if indicator not in selected_indicators:
                    selected_indicators.append(indicator)
        
        params = {
            'interval': interval,
            'outputsize': outputsize,
            'fetch_market_data': fetch_market_data,
            'fetch_company_overview': fetch_company_overview,
            'fetch_technical_indicators': fetch_technical_indicators,
            'selected_indicators': selected_indicators,
            
            # Rate limiting parameters (critical for Alpha Vantage)
            'request_delay': trial.suggest_float('request_delay', 12.0, 20.0),  # Min 12 seconds for 5/minute limit
            'batch_delay': trial.suggest_float('batch_delay', 60.0, 120.0),   # Between symbol batches
            'error_backoff': trial.suggest_float('error_backoff', 30.0, 60.0), # On rate limit errors
            
            # Retry parameters
            'max_retries': trial.suggest_int('max_retries', 1, 3),
            'retry_delay': trial.suggest_float('retry_delay', 15.0, 30.0),
            
            # Data quality parameters
            'min_data_points': trial.suggest_int('min_data_points', 10, 200),
            'max_missing_ratio': trial.suggest_float('max_missing_ratio', 0.0, 0.3),
            
            # Technical indicator parameters
            'sma_period': trial.suggest_int('sma_period', 10, 50) if 'sma' in selected_indicators else 20,
            'ema_period': trial.suggest_int('ema_period', 10, 50) if 'ema' in selected_indicators else 20,
            'rsi_period': trial.suggest_int('rsi_period', 10, 30) if 'rsi' in selected_indicators else 14,
            
            # Data validation
            'validate_time_series': trial.suggest_categorical('validate_time_series', [True, False]),
            'remove_weekends': trial.suggest_categorical('remove_weekends', [True, False]),
        }
        
        return params
    
    def fetch_data_with_params(self, params: Dict[str, Any], 
                              symbols: List[str] = None) -> APIOptimizationResult:
        """Fetch data using Alpha Vantage with specified parameters."""
        if symbols is None:
            symbols = self.default_symbols
        
        if not self.api.api_key:
            return APIOptimizationResult(
                success=False,
                score=0.0,
                error="No API key provided",
                metrics={'api_calls': 0}
            )
        
        start_time = time.time()
        total_data_points = 0
        api_calls = 0
        errors = []
        all_data = []
        rate_limit_errors = 0
        
        try:
            request_delay = params.get('request_delay', 15.0)
            batch_delay = params.get('batch_delay', 60.0)
            
            for i, symbol in enumerate(symbols):
                symbol_data = {}
                symbol_success = False
                
                # Add batch delay between symbols (except first)
                if i > 0:
                    time.sleep(batch_delay)
                
                # Fetch market data
                if params.get('fetch_market_data', True):
                    result = self._fetch_with_retry(
                        lambda: self.api.fetch_market_data(
                            symbol, 
                            interval=params['interval'],
                            outputsize=params['outputsize']
                        ),
                        params, f"{symbol}_market_data"
                    )
                    if result['success']:
                        symbol_data['market_data'] = result['data']
                        total_data_points += len(result['data']) if not result['data'].empty else 0
                        symbol_success = True
                    api_calls += result['api_calls']
                    errors.extend(result['errors'])
                    rate_limit_errors += result.get('rate_limit_errors', 0)
                    
                    time.sleep(request_delay)
                
                # Fetch company overview
                if params.get('fetch_company_overview', False):
                    result = self._fetch_with_retry(
                        lambda: pd.DataFrame([self.api.fetch_company_overview(symbol)]),
                        params, f"{symbol}_overview"
                    )
                    if result['success']:
                        symbol_data['company_overview'] = result['data']
                        total_data_points += len(result['data']) if not result['data'].empty else 0
                    api_calls += result['api_calls']
                    errors.extend(result['errors'])
                    rate_limit_errors += result.get('rate_limit_errors', 0)
                    
                    time.sleep(request_delay)
                
                # Fetch technical indicators
                if params.get('fetch_technical_indicators', False):
                    for indicator in params.get('selected_indicators', []):
                        indicator_params = self._get_indicator_params(indicator, params)
                        
                        result = self._fetch_with_retry(
                            lambda: self.api.fetch_technical_indicator(
                                symbol, indicator, **indicator_params
                            ),
                            params, f"{symbol}_{indicator}"
                        )
                        if result['success']:
                            symbol_data[f'{indicator}_data'] = result['data']
                            total_data_points += len(result['data']) if not result['data'].empty else 0
                        api_calls += result['api_calls']
                        errors.extend(result['errors'])
                        rate_limit_errors += result.get('rate_limit_errors', 0)
                        
                        time.sleep(request_delay)
                
                # Store symbol data if any endpoint was successful
                if symbol_success and symbol_data:
                    # Combine market data for overall evaluation
                    if 'market_data' in symbol_data and not symbol_data['market_data'].empty:
                        processed_data = self._process_data(symbol_data['market_data'], params)
                        all_data.append(processed_data)
        
        except Exception as e:
            return APIOptimizationResult(
                success=False,
                score=0.0,
                error=str(e),
                metrics={'api_calls': api_calls, 'errors': errors, 'rate_limit_errors': rate_limit_errors}
            )
        
        fetch_time = time.time() - start_time
        
        # Combine all market data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
        else:
            combined_data = pd.DataFrame()
        
        # Calculate metrics
        data_quality_score = self.calculate_data_quality_score(combined_data)
        efficiency_score = self._calculate_alpha_vantage_efficiency_score(
            fetch_time, total_data_points, api_calls, rate_limit_errors
        )
        cost_score = self._calculate_alpha_vantage_cost_score(
            api_calls, total_data_points, rate_limit_errors
        )
        error_score = self.calculate_error_score(
            len(errors) > 0, 
            'rate_limit' if rate_limit_errors > 0 else 'api_error' if errors else None
        )
        
        # Calculate symbol coverage
        requested_symbols = len(symbols)
        successful_symbols = len(all_data)
        coverage_ratio = successful_symbols / requested_symbols if requested_symbols > 0 else 0.0
        
        metrics = {
            'data_quality_score': data_quality_score,
            'efficiency_score': efficiency_score,
            'cost_score': cost_score,
            'error_score': error_score,
            'fetch_time': fetch_time,
            'total_data_points': total_data_points,
            'api_calls': api_calls,
            'rate_limit_errors': rate_limit_errors,
            'successful_symbols': successful_symbols,
            'requested_symbols': requested_symbols,
            'coverage_ratio': coverage_ratio,
            'avg_data_per_call': total_data_points / api_calls if api_calls > 0 else 0,
            'calls_per_minute': api_calls / (fetch_time / 60.0) if fetch_time > 0 else 0,
            'errors': errors[:3]  # Keep only first 3 errors for logging
        }
        
        composite_score = self.calculate_composite_score(
            APIOptimizationResult(True, 0.0, combined_data, metrics)
        )
        
        return APIOptimizationResult(
            success=successful_symbols > 0,
            score=composite_score,
            data=combined_data,
            metrics=metrics
        )
    
    def _fetch_with_retry(self, fetch_func, params: Dict[str, Any], 
                         context: str) -> Dict[str, Any]:
        """Execute a fetch function with Alpha Vantage specific retry logic."""
        max_retries = params.get('max_retries', 2)
        retry_delay = params.get('retry_delay', 20.0)
        error_backoff = params.get('error_backoff', 45.0)
        
        api_calls = 0
        errors = []
        rate_limit_errors = 0
        
        for attempt in range(max_retries + 1):
            try:
                data = fetch_func()
                api_calls += 1
                
                # Validate response
                if isinstance(data, pd.DataFrame):
                    if data.empty:
                        raise ValueError("Empty response")
                    # Check for Alpha Vantage error messages
                    if 'Error Message' in str(data.columns) or 'Information' in str(data.columns):
                        raise ValueError("API error or rate limit response")
                elif isinstance(data, dict):
                    if 'Error Message' in data:
                        raise ValueError(f"API Error: {data['Error Message']}")
                    if 'Information' in data:
                        # This is usually a rate limit message
                        rate_limit_errors += 1
                        raise ValueError(f"Rate limit: {data['Information']}")
                
                return {
                    'success': True,
                    'data': data,
                    'api_calls': api_calls,
                    'errors': errors,
                    'rate_limit_errors': rate_limit_errors
                }
                
            except Exception as e:
                api_calls += 1
                error_msg = f"{context} attempt {attempt + 1}: {str(e)}"
                errors.append(error_msg)
                
                # Check if it's a rate limit error
                if 'rate limit' in str(e).lower() or 'information' in str(e).lower():
                    rate_limit_errors += 1
                
                if attempt < max_retries:
                    # Use longer delay for rate limit errors
                    sleep_time = error_backoff if rate_limit_errors > 0 else retry_delay
                    time.sleep(sleep_time)
                else:
                    break
        
        return {
            'success': False,
            'data': pd.DataFrame(),
            'api_calls': api_calls,
            'errors': errors,
            'rate_limit_errors': rate_limit_errors
        }
    
    def _get_indicator_params(self, indicator: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for technical indicators."""
        indicator_params = {}
        
        if indicator == 'sma':
            indicator_params['time_period'] = params.get('sma_period', 20)
        elif indicator == 'ema':
            indicator_params['time_period'] = params.get('ema_period', 20)
        elif indicator == 'rsi':
            indicator_params['time_period'] = params.get('rsi_period', 14)
        elif indicator == 'bbands':
            indicator_params['time_period'] = params.get('sma_period', 20)
            indicator_params['nbdevup'] = 2
            indicator_params['nbdevdn'] = 2
        
        # Add series type for most indicators
        if indicator in ['sma', 'ema', 'rsi']:
            indicator_params['series_type'] = 'close'
        
        return indicator_params
    
    def _process_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Process fetched data according to parameters."""
        if data.empty:
            return data
        
        processed_data = data.copy()
        
        # Remove weekends if enabled (for daily+ data)
        if params.get('remove_weekends', False) and params.get('interval') in ['1d', '1wk', '1mo']:
            if hasattr(processed_data.index, 'weekday'):
                # Remove Saturday (5) and Sunday (6)
                processed_data = processed_data[processed_data.index.weekday < 5]
        
        # Validate time series if enabled
        if params.get('validate_time_series', True):
            processed_data = self._validate_time_series_data(processed_data)
        
        return processed_data
    
    def _validate_time_series_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate time series data quality."""
        if data.empty:
            return data
        
        validated_data = data.copy()
        
        # Basic OHLCV validation
        price_cols = ['Open', 'High', 'Low', 'Close']
        existing_price_cols = [col for col in price_cols if col in validated_data.columns]
        
        if existing_price_cols:
            # Remove negative prices
            for col in existing_price_cols:
                validated_data.loc[validated_data[col] <= 0, col] = np.nan
            
            # High >= Low validation
            if 'High' in validated_data.columns and 'Low' in validated_data.columns:
                invalid_high_low = validated_data['High'] < validated_data['Low']
                validated_data.loc[invalid_high_low, existing_price_cols] = np.nan
        
        # Volume validation
        if 'Volume' in validated_data.columns:
            validated_data.loc[validated_data['Volume'] < 0, 'Volume'] = 0
        
        return validated_data
    
    def _calculate_alpha_vantage_efficiency_score(self, fetch_time: float, 
                                                 data_points: int, api_calls: int,
                                                 rate_limit_errors: int) -> float:
        """Calculate Alpha Vantage specific efficiency score."""
        if fetch_time <= 0 or api_calls <= 0:
            return 0.0
        
        # Data points per API call (higher is better)
        data_per_call = data_points / api_calls
        
        # Rate limit compliance (fewer rate limit errors is better)
        rate_limit_penalty = rate_limit_errors / api_calls if api_calls > 0 else 0
        
        # Time efficiency (considering mandatory delays)
        expected_min_time = api_calls * 12.0  # Minimum 12 seconds per call
        time_efficiency = expected_min_time / fetch_time if fetch_time > expected_min_time else 1.0
        
        # Combine metrics
        data_efficiency = min(1.0, data_per_call / 200.0)  # 200 data points per call = 1.0
        compliance_score = max(0.0, 1.0 - rate_limit_penalty * 2.0)  # Penalty for rate limit errors
        
        return float(data_efficiency * 0.5 + compliance_score * 0.3 + time_efficiency * 0.2)
    
    def _calculate_alpha_vantage_cost_score(self, api_calls: int, data_points: int,
                                          rate_limit_errors: int) -> float:
        """Calculate Alpha Vantage specific cost effectiveness score."""
        if api_calls <= 0:
            return 0.0
        
        # Alpha Vantage free tier: 25 calls per day, 5 per minute
        # Premium tiers have higher limits but cost money
        
        # Data value per API call
        data_per_call = data_points / api_calls
        
        # Penalty for rate limit errors (wasted calls)
        error_penalty = rate_limit_errors / api_calls if api_calls > 0 else 0
        
        # Daily quota efficiency (assuming free tier)
        daily_quota_usage = api_calls / 25.0  # Free tier daily limit
        quota_efficiency = max(0.0, 1.0 - daily_quota_usage) if daily_quota_usage <= 1.0 else 0.0
        
        # Combine metrics
        data_value = min(1.0, data_per_call / 300.0)  # 300 data points per call = 1.0
        error_score = max(0.0, 1.0 - error_penalty * 3.0)  # Heavy penalty for errors
        
        return float(data_value * 0.4 + error_score * 0.4 + quota_efficiency * 0.2)
    
    def run_rate_limit_optimization(self, study_name: str = "alpha_vantage_rate_limit_optimization",
                                   n_trials: int = 30) -> optuna.Study:
        """
        Run optimization focused on rate limit compliance and efficiency.
        
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
        
        # Use minimal symbols for rate limit optimization
        rate_limit_symbols = self.default_symbols[:2]
        result = self.optimize_for_symbols(rate_limit_symbols)
        
        return self.study


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = AlphaVantageAPIOptimizer()
    
    print("Running Alpha Vantage API optimization...")
    print("Note: This requires an Alpha Vantage API key to work properly")
    print("Free tier has strict rate limits (5 calls/minute, 25/day)")
    
    # Run rate limit optimization study
    study = optimizer.run_rate_limit_optimization(n_trials=5)  # Small number due to rate limits
    
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Test with best parameters
    if study.best_params:
        best_result = optimizer.fetch_data_with_params(
            study.best_params, 
            ['AAPL']  # Test with single symbol
        )
        
        print(f"Test result - Success: {best_result.success}")
        if best_result.metrics:
            print(f"API calls: {best_result.metrics.get('api_calls', 0)}")
            print(f"Rate limit errors: {best_result.metrics.get('rate_limit_errors', 0)}")
            print(f"Data points: {best_result.metrics.get('total_data_points', 0)}")
            print(f"Efficiency score: {best_result.metrics.get('efficiency_score', 0):.3f}")