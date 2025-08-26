#!/usr/bin/env python3
"""
Finnhub API Optimizer

Optimizes Finnhub API parameters for data resolution, endpoints, and rate limiting.
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.data_sources.finnhub import FinnhubAPI
from optimization.api_base import BaseAPIOptimizer, APIOptimizationResult, OptimizationConfig
import logging

logger = logging.getLogger(__name__)


class FinnhubAPIOptimizer(BaseAPIOptimizer):
    """
    Optimizer for Finnhub API parameters.
    
    Optimizes:
    - Data resolution and time periods
    - Multi-endpoint data combinations
    - Rate limiting and request strategies
    - Alternative data usage (news, sentiment, insider trading)
    """
    
    def __init__(self, api_key: str = None, config: OptimizationConfig = None):
        """Initialize Finnhub API optimizer."""
        api_instance = FinnhubAPI(api_key=api_key)
        super().__init__(api_instance, config)
        
        # Default test symbols for optimization
        self.default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Finnhub resolution mappings
        self.resolution_periods = {
            '1': ['1d', '1w'],           # 1 minute
            '5': ['1d', '1w', '1mo'],    # 5 minutes
            '15': ['1d', '1w', '1mo'],   # 15 minutes
            '30': ['1d', '1w', '1mo'],   # 30 minutes
            '60': ['1w', '1mo', '3mo'],  # 1 hour
            'D': ['1mo', '3mo', '6mo', '1y', '2y'],  # Daily
            'W': ['6mo', '1y', '2y', '5y'],          # Weekly
            'M': ['2y', '5y']                        # Monthly
        }
        
        # Data endpoints available
        self.data_endpoints = [
            'market_data', 'company_profile', 'company_news', 
            'market_news', 'sentiment', 'insider_trading', 
            'earnings_calendar'
        ]
        
        # Set optimization weights for Finnhub
        self.set_optimization_weights({
            'data_quality': 0.40,
            'efficiency': 0.30,
            'cost_effectiveness': 0.20,
            'error_rate': 0.10
        })
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define Finnhub API parameter search space."""
        
        # Resolution and period selection
        resolution = trial.suggest_categorical('resolution', list(self.resolution_periods.keys()))
        valid_periods = self.resolution_periods[resolution]
        period = trial.suggest_categorical('period', valid_periods)
        
        # Endpoint selection
        fetch_market_data = trial.suggest_categorical('fetch_market_data', [True, False])
        fetch_company_profile = trial.suggest_categorical('fetch_company_profile', [True, False])
        fetch_company_news = trial.suggest_categorical('fetch_company_news', [True, False])
        fetch_market_news = trial.suggest_categorical('fetch_market_news', [True, False])
        fetch_sentiment = trial.suggest_categorical('fetch_sentiment', [True, False])
        fetch_insider_trading = trial.suggest_categorical('fetch_insider_trading', [True, False])
        
        # At least market data or company profile should be fetched
        if not any([fetch_market_data, fetch_company_profile]):
            fetch_market_data = True
        
        params = {
            'resolution': resolution,
            'period': period,
            'fetch_market_data': fetch_market_data,
            'fetch_company_profile': fetch_company_profile,
            'fetch_company_news': fetch_company_news,
            'fetch_market_news': fetch_market_news,
            'fetch_sentiment': fetch_sentiment,
            'fetch_insider_trading': fetch_insider_trading,
            
            # Rate limiting parameters (Finnhub: 60 calls/minute free tier)
            'request_delay': trial.suggest_float('request_delay', 1.0, 3.0),
            'batch_delay': trial.suggest_float('batch_delay', 5.0, 15.0),
            'symbol_delay': trial.suggest_float('symbol_delay', 0.5, 2.0),
            
            # Retry parameters
            'max_retries': trial.suggest_int('max_retries', 1, 3),
            'retry_delay': trial.suggest_float('retry_delay', 2.0, 10.0),
            'backoff_multiplier': trial.suggest_float('backoff_multiplier', 1.5, 3.0),
            
            # News parameters
            'news_days_back': trial.suggest_int('news_days_back', 7, 60) if fetch_company_news else 30,
            'market_news_category': trial.suggest_categorical('market_news_category', 
                                                             ['general', 'forex', 'crypto', 'merger']) if fetch_market_news else 'general',
            
            # Data quality parameters
            'min_data_points': trial.suggest_int('min_data_points', 10, 500),
            'max_missing_ratio': trial.suggest_float('max_missing_ratio', 0.0, 0.4),
            'require_volume_data': trial.suggest_categorical('require_volume_data', [True, False]),
            
            # Processing parameters
            'validate_ohlc': trial.suggest_categorical('validate_ohlc', [True, False]),
            'remove_outliers': trial.suggest_categorical('remove_outliers', [True, False]),
            'outlier_threshold': trial.suggest_float('outlier_threshold', 2.0, 5.0),
            
            # Constraint validation
            'check_weekends': trial.suggest_categorical('check_weekends', [True, False]),
            'trading_hours_only': trial.suggest_categorical('trading_hours_only', [True, False]) if resolution in ['1', '5', '15', '30', '60'] else False,
        }
        
        return params
    
    def fetch_data_with_params(self, params: Dict[str, Any], 
                              symbols: List[str] = None) -> APIOptimizationResult:
        """Fetch data using Finnhub with specified parameters."""
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
        symbol_results = {}
        rate_limit_hits = 0
        
        try:
            request_delay = params.get('request_delay', 1.5)
            symbol_delay = params.get('symbol_delay', 1.0)
            batch_delay = params.get('batch_delay', 10.0)
            
            for i, symbol in enumerate(symbols):
                symbol_data = {}
                symbol_success = False
                
                # Inter-symbol delay
                if i > 0:
                    time.sleep(symbol_delay)
                
                # Fetch market data
                if params.get('fetch_market_data', True):
                    result = self._fetch_with_retry(
                        lambda: self.api.fetch_market_data(
                            symbol, 
                            period=params['period'], 
                            interval=params['resolution']
                        ),
                        params, f"{symbol}_market_data"
                    )
                    if result['success']:
                        symbol_data['market_data'] = result['data']
                        total_data_points += len(result['data']) if not result['data'].empty else 0
                        symbol_success = True
                    api_calls += result['api_calls']
                    errors.extend(result['errors'])
                    rate_limit_hits += result.get('rate_limit_hits', 0)
                    
                    time.sleep(request_delay)
                
                # Fetch company profile
                if params.get('fetch_company_profile', False):
                    result = self._fetch_with_retry(
                        lambda: pd.DataFrame([self.api.fetch_company_profile(symbol)]),
                        params, f"{symbol}_profile"
                    )
                    if result['success']:
                        symbol_data['company_profile'] = result['data']
                        total_data_points += len(result['data']) if not result['data'].empty else 0
                    api_calls += result['api_calls']
                    errors.extend(result['errors'])
                    rate_limit_hits += result.get('rate_limit_hits', 0)
                    
                    time.sleep(request_delay)
                
                # Fetch company news
                if params.get('fetch_company_news', False):
                    result = self._fetch_with_retry(
                        lambda: self.api.fetch_company_news(
                            symbol, 
                            days_back=params.get('news_days_back', 30)
                        ),
                        params, f"{symbol}_news"
                    )
                    if result['success']:
                        symbol_data['company_news'] = result['data']
                        total_data_points += len(result['data']) if not result['data'].empty else 0
                    api_calls += result['api_calls']
                    errors.extend(result['errors'])
                    rate_limit_hits += result.get('rate_limit_hits', 0)
                    
                    time.sleep(request_delay)
                
                # Fetch sentiment
                if params.get('fetch_sentiment', False):
                    result = self._fetch_with_retry(
                        lambda: pd.DataFrame([self.api.fetch_sentiment(symbol)]),
                        params, f"{symbol}_sentiment"
                    )
                    if result['success']:
                        symbol_data['sentiment'] = result['data']
                        total_data_points += len(result['data']) if not result['data'].empty else 0
                    api_calls += result['api_calls']
                    errors.extend(result['errors'])
                    rate_limit_hits += result.get('rate_limit_hits', 0)
                    
                    time.sleep(request_delay)
                
                # Fetch insider trading
                if params.get('fetch_insider_trading', False):
                    result = self._fetch_with_retry(
                        lambda: self.api.fetch_insider_trading(symbol),
                        params, f"{symbol}_insider"
                    )
                    if result['success']:
                        symbol_data['insider_trading'] = result['data']
                        total_data_points += len(result['data']) if not result['data'].empty else 0
                    api_calls += result['api_calls']
                    errors.extend(result['errors'])
                    rate_limit_hits += result.get('rate_limit_hits', 0)
                    
                    time.sleep(request_delay)
                
                # Store symbol data if successful
                if symbol_success and symbol_data:
                    symbol_results[symbol] = symbol_data
                    # Process market data for combination
                    if 'market_data' in symbol_data and not symbol_data['market_data'].empty:
                        processed_data = self._process_data(symbol_data['market_data'], params)
                        all_data.append(processed_data)
                
                # Batch delay after every few symbols
                if (i + 1) % 3 == 0 and i + 1 < len(symbols):
                    time.sleep(batch_delay)
            
            # Fetch market news (once per optimization, not per symbol)
            if params.get('fetch_market_news', False):
                result = self._fetch_with_retry(
                    lambda: self.api.fetch_market_news(
                        category=params.get('market_news_category', 'general')
                    ),
                    params, "market_news"
                )
                if result['success']:
                    total_data_points += len(result['data']) if not result['data'].empty else 0
                api_calls += result['api_calls']
                errors.extend(result['errors'])
                rate_limit_hits += result.get('rate_limit_hits', 0)
        
        except Exception as e:
            return APIOptimizationResult(
                success=False,
                score=0.0,
                error=str(e),
                metrics={'api_calls': api_calls, 'errors': errors, 'rate_limit_hits': rate_limit_hits}
            )
        
        fetch_time = time.time() - start_time
        
        # Combine all market data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
        else:
            combined_data = pd.DataFrame()
        
        # Calculate metrics
        data_quality_score = self._calculate_finnhub_data_quality_score(symbol_results, params)
        efficiency_score = self._calculate_finnhub_efficiency_score(
            fetch_time, total_data_points, api_calls, rate_limit_hits
        )
        cost_score = self._calculate_finnhub_cost_score(
            api_calls, len(symbol_results), rate_limit_hits
        )
        error_score = self.calculate_error_score(
            len(errors) > 0,
            'rate_limit' if rate_limit_hits > 0 else 'api_error' if errors else None
        )
        
        # Calculate coverage and endpoint usage
        requested_symbols = len(symbols)
        successful_symbols = len(symbol_results)
        coverage_ratio = successful_symbols / requested_symbols if requested_symbols > 0 else 0.0
        
        endpoint_count = sum([
            params.get('fetch_market_data', False),
            params.get('fetch_company_profile', False),
            params.get('fetch_company_news', False),
            params.get('fetch_market_news', False),
            params.get('fetch_sentiment', False),
            params.get('fetch_insider_trading', False)
        ])
        
        metrics = {
            'data_quality_score': data_quality_score,
            'efficiency_score': efficiency_score,
            'cost_score': cost_score,
            'error_score': error_score,
            'fetch_time': fetch_time,
            'total_data_points': total_data_points,
            'api_calls': api_calls,
            'rate_limit_hits': rate_limit_hits,
            'successful_symbols': successful_symbols,
            'requested_symbols': requested_symbols,
            'coverage_ratio': coverage_ratio,
            'endpoint_count': endpoint_count,
            'calls_per_minute': api_calls / (fetch_time / 60.0) if fetch_time > 0 else 0,
            'avg_data_per_call': total_data_points / api_calls if api_calls > 0 else 0,
            'errors': errors[:5]  # Keep only first 5 errors for logging
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
        """Execute a fetch function with Finnhub specific retry logic."""
        max_retries = params.get('max_retries', 2)
        retry_delay = params.get('retry_delay', 5.0)
        backoff_multiplier = params.get('backoff_multiplier', 2.0)
        
        api_calls = 0
        errors = []
        rate_limit_hits = 0
        
        for attempt in range(max_retries + 1):
            try:
                data = fetch_func()
                api_calls += 1
                
                # Validate response
                if isinstance(data, pd.DataFrame) and data.empty:
                    raise ValueError("Empty response")
                elif isinstance(data, dict) and not data:
                    raise ValueError("Empty response dict")
                
                return {
                    'success': True,
                    'data': data,
                    'api_calls': api_calls,
                    'errors': errors,
                    'rate_limit_hits': rate_limit_hits
                }
                
            except Exception as e:
                api_calls += 1
                error_msg = f"{context} attempt {attempt + 1}: {str(e)}"
                errors.append(error_msg)
                
                # Check for rate limit indicators
                error_str = str(e).lower()
                if 'rate limit' in error_str or '429' in error_str or 'too many' in error_str:
                    rate_limit_hits += 1
                
                if attempt < max_retries:
                    sleep_time = retry_delay * (backoff_multiplier ** attempt)
                    # Extra delay for rate limit errors
                    if rate_limit_hits > 0:
                        sleep_time *= 2
                    time.sleep(sleep_time)
                else:
                    break
        
        return {
            'success': False,
            'data': pd.DataFrame(),
            'api_calls': api_calls,
            'errors': errors,
            'rate_limit_hits': rate_limit_hits
        }
    
    def _process_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Process fetched data according to parameters."""
        if data.empty:
            return data
        
        processed_data = data.copy()
        
        # OHLC validation
        if params.get('validate_ohlc', True):
            processed_data = self._validate_ohlc_data(processed_data, params)
        
        # Remove outliers
        if params.get('remove_outliers', False):
            processed_data = self._remove_outliers(processed_data, params)
        
        # Weekend filtering
        if params.get('check_weekends', True):
            processed_data = self._filter_weekends(processed_data)
        
        return processed_data
    
    def _validate_ohlc_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Validate OHLC data integrity."""
        if data.empty:
            return data
        
        validated_data = data.copy()
        price_cols = ['Open', 'High', 'Low', 'Close']
        
        # Check for required columns
        existing_cols = [col for col in price_cols if col in validated_data.columns]
        
        if len(existing_cols) >= 2:
            # Remove negative prices
            for col in existing_cols:
                validated_data.loc[validated_data[col] <= 0, col] = np.nan
            
            # Validate High >= Low
            if 'High' in validated_data.columns and 'Low' in validated_data.columns:
                invalid_high_low = validated_data['High'] < validated_data['Low']
                validated_data.loc[invalid_high_low, existing_cols] = np.nan
            
            # Validate volume if required
            if params.get('require_volume_data', False) and 'Volume' in validated_data.columns:
                validated_data = validated_data[validated_data['Volume'] > 0]
        
        return validated_data
    
    def _remove_outliers(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Remove statistical outliers from price data."""
        if data.empty:
            return data
        
        cleaned_data = data.copy()
        threshold = params.get('outlier_threshold', 3.0)
        
        price_cols = ['Open', 'High', 'Low', 'Close']
        
        for col in price_cols:
            if col in cleaned_data.columns:
                col_data = cleaned_data[col].dropna()
                if len(col_data) > 10:
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    
                    if std_val > 0:
                        z_scores = np.abs((col_data - mean_val) / std_val)
                        outliers = z_scores > threshold
                        
                        if outliers.any():
                            logger.debug(f"Removing {outliers.sum()} outliers from {col}")
                            cleaned_data.loc[outliers, col] = np.nan
        
        return cleaned_data
    
    def _filter_weekends(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter out weekend data for stock markets."""
        if data.empty or not hasattr(data.index, 'weekday'):
            return data
        
        # Remove Saturday (5) and Sunday (6)
        return data[data.index.weekday < 5]
    
    def _calculate_finnhub_data_quality_score(self, symbol_results: Dict[str, Dict], 
                                            params: Dict[str, Any]) -> float:
        """Calculate Finnhub-specific data quality score."""
        if not symbol_results:
            return 0.0
        
        total_score = 0.0
        count = 0
        
        for symbol, data_dict in symbol_results.items():
            if 'market_data' in data_dict and not data_dict['market_data'].empty:
                data = data_dict['market_data']
                
                # Basic completeness
                total_cells = len(data) * len(data.columns)
                missing_cells = data.isnull().sum().sum()
                completeness = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
                
                # Data consistency for OHLC
                consistency = 1.0
                if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    # Check OHLC relationships
                    valid_high = (data['High'] >= data['Low']).mean()
                    consistency = float(valid_high)
                
                # Volume data availability
                volume_score = 1.0
                if params.get('require_volume_data', False):
                    if 'Volume' in data.columns:
                        volume_score = (data['Volume'] > 0).mean()
                    else:
                        volume_score = 0.0
                
                symbol_score = completeness * 0.5 + consistency * 0.3 + volume_score * 0.2
                total_score += symbol_score
                count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def _calculate_finnhub_efficiency_score(self, fetch_time: float, data_points: int,
                                          api_calls: int, rate_limit_hits: int) -> float:
        """Calculate Finnhub-specific efficiency score."""
        if fetch_time <= 0 or api_calls <= 0:
            return 0.0
        
        # Data points per second
        data_per_second = data_points / fetch_time
        
        # API calls per minute (free tier limit: 60/minute)
        calls_per_minute = api_calls / (fetch_time / 60.0) if fetch_time > 0 else 0
        rate_limit_compliance = min(1.0, (60.0 - calls_per_minute) / 60.0) if calls_per_minute <= 60 else 0.0
        
        # Rate limit penalty
        rate_limit_penalty = rate_limit_hits / api_calls if api_calls > 0 else 0
        
        # Combine metrics
        throughput_score = min(1.0, data_per_second / 100.0)  # 100 data points/sec = 1.0
        compliance_score = max(0.0, rate_limit_compliance - rate_limit_penalty)
        
        return float(throughput_score * 0.4 + compliance_score * 0.6)
    
    def _calculate_finnhub_cost_score(self, api_calls: int, successful_symbols: int,
                                    rate_limit_hits: int) -> float:
        """Calculate Finnhub-specific cost effectiveness score."""
        if api_calls <= 0:
            return 0.0
        
        # Success rate
        success_rate = successful_symbols / api_calls if api_calls > 0 else 0
        
        # Rate limit efficiency (fewer hits is better)
        rate_limit_efficiency = max(0.0, 1.0 - (rate_limit_hits / api_calls * 2.0))
        
        # API quota efficiency (free tier: 60 calls/minute)
        quota_efficiency = min(1.0, successful_symbols / 60.0) if successful_symbols <= 60 else 0.5
        
        return float(success_rate * 0.4 + rate_limit_efficiency * 0.4 + quota_efficiency * 0.2)
    
    def run_resolution_optimization(self, study_name: str = "finnhub_resolution_optimization",
                                  n_trials: int = 50) -> optuna.Study:
        """
        Run optimization focused on resolution and time period selection.
        
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
        
        # Use fewer symbols for resolution optimization
        resolution_symbols = self.default_symbols[:3]
        result = self.optimize_for_symbols(resolution_symbols)
        
        return self.study


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = FinnhubAPIOptimizer()
    
    print("Running Finnhub API optimization...")
    print("Note: This requires a Finnhub API key to work properly")
    print("Free tier has rate limits (60 calls/minute)")
    
    # Run resolution optimization study
    study = optimizer.run_resolution_optimization(n_trials=20)
    
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Test with best parameters
    if study.best_params:
        best_result = optimizer.fetch_data_with_params(
            study.best_params, 
            ['AAPL', 'MSFT']
        )
        
        print(f"Test result - Success: {best_result.success}")
        if best_result.metrics:
            print(f"API calls: {best_result.metrics.get('api_calls', 0)}")
            print(f"Rate limit hits: {best_result.metrics.get('rate_limit_hits', 0)}")
            print(f"Data points: {best_result.metrics.get('total_data_points', 0)}")
            print(f"Coverage ratio: {best_result.metrics.get('coverage_ratio', 0):.3f}")