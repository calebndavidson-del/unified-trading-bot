#!/usr/bin/env python3
"""
IEX Cloud API Optimizer

Optimizes IEX Cloud API parameters for data quality, efficiency, and cost-effectiveness.
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

from features.data_sources.iex_cloud import IEXCloudAPI
from optimization.api_base import BaseAPIOptimizer, APIOptimizationResult, OptimizationConfig
import logging

logger = logging.getLogger(__name__)


class IEXCloudAPIOptimizer(BaseAPIOptimizer):
    """
    Optimizer for IEX Cloud API parameters.
    
    Optimizes:
    - Data fetching strategies and batch sizes
    - Rate limiting and request optimization
    - Data endpoint selection and combinations
    - Error handling and retry strategies
    """
    
    def __init__(self, api_key: str = None, sandbox: bool = True, 
                 config: OptimizationConfig = None):
        """Initialize IEX Cloud API optimizer."""
        api_instance = IEXCloudAPI(api_key=api_key, sandbox=sandbox)
        super().__init__(api_instance, config)
        
        # Default test symbols for optimization
        self.default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # IEX Cloud specific constraints
        self.period_mappings = {
            "1d": "1d", "5d": "5d", "1mo": "1m", "3mo": "3m", 
            "6mo": "6m", "1y": "1y", "2y": "2y", "5y": "5y"
        }
        
        # Data endpoints to optimize
        self.data_endpoints = [
            'market_data',
            'company_info', 
            'key_stats',
            'news',
            'social_sentiment'
        ]
        
        # Set optimization weights for IEX Cloud (cost is more important due to billing)
        self.set_optimization_weights({
            'data_quality': 0.35,
            'efficiency': 0.25,
            'cost_effectiveness': 0.30,  # Higher weight for cost
            'error_rate': 0.10
        })
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define IEX Cloud API parameter search space."""
        
        # Data fetching parameters
        period = trial.suggest_categorical('period', list(self.period_mappings.keys()))
        
        # Endpoint selection (which data to fetch)
        fetch_market_data = trial.suggest_categorical('fetch_market_data', [True, False])
        fetch_company_info = trial.suggest_categorical('fetch_company_info', [True, False])
        fetch_key_stats = trial.suggest_categorical('fetch_key_stats', [True, False])
        fetch_news = trial.suggest_categorical('fetch_news', [True, False])
        fetch_sentiment = trial.suggest_categorical('fetch_sentiment', [True, False])
        
        # At least one endpoint must be selected
        if not any([fetch_market_data, fetch_company_info, fetch_key_stats, fetch_news, fetch_sentiment]):
            fetch_market_data = True  # Force at least market data
        
        # Batch processing parameters
        params = {
            'period': period,
            'fetch_market_data': fetch_market_data,
            'fetch_company_info': fetch_company_info,
            'fetch_key_stats': fetch_key_stats,
            'fetch_news': fetch_news,
            'fetch_sentiment': fetch_sentiment,
            
            # Rate limiting and batching
            'batch_size': trial.suggest_int('batch_size', 1, 20),  # IEX allows up to 100 symbols per request
            'request_delay': trial.suggest_float('request_delay', 0.0, 2.0),
            'batch_delay': trial.suggest_float('batch_delay', 0.0, 1.0),
            
            # Retry parameters
            'max_retries': trial.suggest_int('max_retries', 0, 3),
            'retry_delay': trial.suggest_float('retry_delay', 0.1, 2.0),
            'retry_backoff': trial.suggest_float('retry_backoff', 1.0, 3.0),
            
            # Data quality parameters
            'min_data_points': trial.suggest_int('min_data_points', 5, 100),
            'max_missing_ratio': trial.suggest_float('max_missing_ratio', 0.0, 0.5),
            
            # News parameters (if news is fetched)
            'news_count': trial.suggest_int('news_count', 5, 50) if fetch_news else 10,
            
            # Data validation
            'validate_responses': trial.suggest_categorical('validate_responses', [True, False]),
            'combine_requests': trial.suggest_categorical('combine_requests', [True, False]),
        }
        
        return params
    
    def fetch_data_with_params(self, params: Dict[str, Any], 
                              symbols: List[str] = None) -> APIOptimizationResult:
        """Fetch data using IEX Cloud with specified parameters."""
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
        endpoint_data = {}
        
        try:
            # Process symbols in batches
            batch_size = params.get('batch_size', 5)
            batch_delay = params.get('batch_delay', 0.0)
            request_delay = params.get('request_delay', 0.1)
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                
                for symbol in batch_symbols:
                    symbol_data = {}
                    symbol_success = False
                    
                    # Fetch market data
                    if params.get('fetch_market_data', True):
                        result = self._fetch_with_retry(
                            lambda: self.api.fetch_market_data(symbol, period=params['period']),
                            params, f"{symbol}_market_data"
                        )
                        if result['success']:
                            symbol_data['market_data'] = result['data']
                            total_data_points += len(result['data']) if not result['data'].empty else 0
                            symbol_success = True
                        api_calls += result['api_calls']
                        errors.extend(result['errors'])
                        
                        time.sleep(request_delay)
                    
                    # Fetch company info
                    if params.get('fetch_company_info', False):
                        result = self._fetch_with_retry(
                            lambda: pd.DataFrame([self.api.fetch_company_info(symbol)]),
                            params, f"{symbol}_company_info"
                        )
                        if result['success']:
                            symbol_data['company_info'] = result['data']
                            total_data_points += len(result['data']) if not result['data'].empty else 0
                        api_calls += result['api_calls']
                        errors.extend(result['errors'])
                        
                        time.sleep(request_delay)
                    
                    # Fetch key stats
                    if params.get('fetch_key_stats', False):
                        result = self._fetch_with_retry(
                            lambda: pd.DataFrame([self.api.fetch_key_stats(symbol)]),
                            params, f"{symbol}_key_stats"
                        )
                        if result['success']:
                            symbol_data['key_stats'] = result['data']
                            total_data_points += len(result['data']) if not result['data'].empty else 0
                        api_calls += result['api_calls']
                        errors.extend(result['errors'])
                        
                        time.sleep(request_delay)
                    
                    # Fetch news
                    if params.get('fetch_news', False):
                        result = self._fetch_with_retry(
                            lambda: self.api.fetch_news(symbol, last_n=params.get('news_count', 10)),
                            params, f"{symbol}_news"
                        )
                        if result['success']:
                            symbol_data['news'] = result['data']
                            total_data_points += len(result['data']) if not result['data'].empty else 0
                        api_calls += result['api_calls']
                        errors.extend(result['errors'])
                        
                        time.sleep(request_delay)
                    
                    # Fetch social sentiment
                    if params.get('fetch_sentiment', False):
                        result = self._fetch_with_retry(
                            lambda: pd.DataFrame([self.api.fetch_social_sentiment(symbol)]),
                            params, f"{symbol}_sentiment"
                        )
                        if result['success']:
                            symbol_data['sentiment'] = result['data']
                            total_data_points += len(result['data']) if not result['data'].empty else 0
                        api_calls += result['api_calls']
                        errors.extend(result['errors'])
                        
                        time.sleep(request_delay)
                    
                    # Store symbol data if any endpoint was successful
                    if symbol_success and symbol_data:
                        endpoint_data[symbol] = symbol_data
                        # Combine all data for the symbol
                        if 'market_data' in symbol_data and not symbol_data['market_data'].empty:
                            all_data.append(symbol_data['market_data'])
                
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
        
        # Combine all market data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
        else:
            combined_data = pd.DataFrame()
        
        # Calculate metrics
        data_quality_score = self.calculate_data_quality_score(combined_data)
        efficiency_score = self.calculate_efficiency_score(fetch_time, total_data_points, api_calls)
        
        # IEX Cloud specific cost calculation (based on API calls)
        cost_score = self._calculate_iex_cost_score(api_calls, total_data_points, len(endpoint_data))
        
        error_score = self.calculate_error_score(len(errors) > 0, 'api_error' if errors else None)
        
        # Calculate coverage metrics
        requested_symbols = len(symbols)
        successful_symbols = len(endpoint_data)
        coverage_ratio = successful_symbols / requested_symbols if requested_symbols > 0 else 0.0
        
        # Calculate endpoint efficiency
        endpoint_usage = sum([
            params.get('fetch_market_data', False),
            params.get('fetch_company_info', False),
            params.get('fetch_key_stats', False),
            params.get('fetch_news', False),
            params.get('fetch_sentiment', False)
        ])
        
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
            'coverage_ratio': coverage_ratio,
            'endpoint_usage': endpoint_usage,
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
        """Execute a fetch function with retry logic."""
        max_retries = params.get('max_retries', 1)
        retry_delay = params.get('retry_delay', 0.5)
        retry_backoff = params.get('retry_backoff', 1.5)
        
        api_calls = 0
        errors = []
        
        for attempt in range(max_retries + 1):
            try:
                data = fetch_func()
                api_calls += 1
                
                # Validate response if enabled
                if params.get('validate_responses', True):
                    if isinstance(data, pd.DataFrame) and data.empty:
                        raise ValueError("Empty response")
                    elif isinstance(data, dict) and not data:
                        raise ValueError("Empty response dict")
                
                return {
                    'success': True,
                    'data': data,
                    'api_calls': api_calls,
                    'errors': errors
                }
                
            except Exception as e:
                api_calls += 1
                error_msg = f"{context} attempt {attempt + 1}: {str(e)}"
                errors.append(error_msg)
                
                if attempt < max_retries:
                    sleep_time = retry_delay * (retry_backoff ** attempt)
                    time.sleep(sleep_time)
                else:
                    break
        
        return {
            'success': False,
            'data': pd.DataFrame(),
            'api_calls': api_calls,
            'errors': errors
        }
    
    def _calculate_iex_cost_score(self, api_calls: int, data_points: int, 
                                 successful_symbols: int) -> float:
        """Calculate IEX Cloud specific cost effectiveness score."""
        if api_calls <= 0:
            return 0.0
        
        # IEX Cloud charges per API call, so efficiency is data per call
        data_per_call = data_points / api_calls
        symbols_per_call = successful_symbols / api_calls
        
        # Normalize scores
        data_efficiency = min(1.0, data_per_call / 100.0)  # 100 data points per call = 1.0
        symbol_efficiency = min(1.0, symbols_per_call / 1.0)  # 1 symbol per call = 1.0
        
        # Combine efficiencies
        return float(data_efficiency * 0.6 + symbol_efficiency * 0.4)
    
    def optimize_for_data_combination(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Optimize for the best combination of data endpoints.
        
        Args:
            symbols: Symbols to optimize for
        """
        if symbols is None:
            symbols = self.default_symbols[:3]  # Use fewer symbols for endpoint optimization
        
        return self.optimize_for_symbols(symbols)
    
    def run_cost_optimization_study(self, study_name: str = "iex_cloud_cost_optimization",
                                   n_trials: int = 50) -> optuna.Study:
        """
        Run optimization focused on cost-effectiveness.
        
        Args:
            study_name: Name for the Optuna study
            n_trials: Number of trials to run
            
        Returns:
            Completed Optuna study
        """
        # Adjust weights for cost optimization
        original_weights = self.get_optimization_weights()
        self.set_optimization_weights({
            'data_quality': 0.25,
            'efficiency': 0.20,
            'cost_effectiveness': 0.45,  # Higher focus on cost
            'error_rate': 0.10
        })
        
        try:
            config = OptimizationConfig(
                n_trials=n_trials,
                study_name=study_name,
                direction='maximize',
                sampler='TPE',
                pruner='MedianPruner'
            )
            
            self.config = config
            
            # Use smaller symbol set for cost optimization
            cost_symbols = self.default_symbols[:3]
            result = self.optimize_for_symbols(cost_symbols)
            
            return self.study
            
        finally:
            # Restore original weights
            self.set_optimization_weights(original_weights)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer (using sandbox mode)
    optimizer = IEXCloudAPIOptimizer(sandbox=True)
    
    print("Running IEX Cloud API optimization...")
    print("Note: This requires an IEX Cloud API key to work properly")
    
    # Run cost optimization study
    study = optimizer.run_cost_optimization_study(n_trials=10)
    
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
            print(f"Data points: {best_result.metrics.get('total_data_points', 0)}")
            print(f"Cost score: {best_result.metrics.get('cost_score', 0):.3f}")