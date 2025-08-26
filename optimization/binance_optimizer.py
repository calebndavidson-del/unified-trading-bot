#!/usr/bin/env python3
"""
Binance API Optimizer

Optimizes Binance API parameters for cryptocurrency data fetching, intervals, and efficiency.
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

from features.data_sources.binance import BinanceAPI
from optimization.api_base import BaseAPIOptimizer, APIOptimizationResult, OptimizationConfig
import logging

logger = logging.getLogger(__name__)


class BinanceAPIOptimizer(BaseAPIOptimizer):
    """
    Optimizer for Binance API parameters.
    
    Optimizes:
    - Cryptocurrency symbol mapping and selection
    - Interval and period combinations for crypto data
    - Rate limiting and weight management
    - Multi-timeframe data strategies
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, 
                 config: OptimizationConfig = None):
        """Initialize Binance API optimizer."""
        api_instance = BinanceAPI(api_key=api_key, api_secret=api_secret)
        super().__init__(api_instance, config)
        
        # Default test symbols for optimization (crypto pairs)
        self.default_symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 'SOL-USDT']
        
        # Binance interval mappings and constraints
        self.interval_limits = {
            '1m': {'max_days': 30, 'max_limit': 1000},
            '3m': {'max_days': 30, 'max_limit': 1000},
            '5m': {'max_days': 60, 'max_limit': 1000},
            '15m': {'max_days': 180, 'max_limit': 1000},
            '30m': {'max_days': 365, 'max_limit': 1000},
            '1h': {'max_days': 730, 'max_limit': 1000},
            '2h': {'max_days': 730, 'max_limit': 1000},
            '4h': {'max_days': 730, 'max_limit': 1000},
            '6h': {'max_days': 730, 'max_limit': 1000},
            '8h': {'max_days': 730, 'max_limit': 1000},
            '12h': {'max_days': 730, 'max_limit': 1000},
            '1d': {'max_days': 1000, 'max_limit': 1000},
            '3d': {'max_days': 3000, 'max_limit': 1000},
            '1w': {'max_days': 3000, 'max_limit': 1000},
            '1M': {'max_days': 3000, 'max_limit': 1000}
        }
        
        # Base and quote asset categories
        self.base_assets = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'MATIC', 'AVAX', 'DOGE', 'LTC']
        self.quote_assets = ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']
        
        # Set optimization weights for Binance (efficiency important due to high frequency)
        self.set_optimization_weights({
            'data_quality': 0.35,
            'efficiency': 0.35,  # Higher weight for crypto trading efficiency
            'cost_effectiveness': 0.20,
            'error_rate': 0.10
        })
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define Binance API parameter search space."""
        
        # Interval and period selection - use fixed lists to avoid dynamic value space
        interval = trial.suggest_categorical('interval', [
            '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
        ])
        
        # Use all possible periods and validate with constraints
        period = trial.suggest_categorical('period', [
            '1d', '7d', '1mo', '3mo', '6mo', '1y', '2y'
        ])
        
        # Validate period based on interval constraints
        max_days = self.interval_limits[interval]['max_days']
        period_days_map = {
            '1d': 1, '7d': 7, '1mo': 30, '3mo': 90, 
            '6mo': 180, '1y': 365, '2y': 730
        }
        
        if period_days_map.get(period, 365) > max_days:
            raise optuna.TrialPruned(f"Period {period} not supported for interval {interval}")
        
        # Symbol selection strategy
        symbol_strategy = trial.suggest_categorical('symbol_strategy', [
            'popular_pairs', 'base_asset_focus', 'quote_asset_focus', 'mixed_strategy'
        ])
        
        # Multi-timeframe analysis
        fetch_multiple_timeframes = trial.suggest_categorical('fetch_multiple_timeframes', [True, False])
        
        params = {
            'interval': interval,
            'period': period,
            'symbol_strategy': symbol_strategy,
            'fetch_multiple_timeframes': fetch_multiple_timeframes,
            
            # Symbol mapping and validation
            'auto_symbol_mapping': trial.suggest_categorical('auto_symbol_mapping', [True, False]),
            'prefer_usdt_pairs': trial.suggest_categorical('prefer_usdt_pairs', [True, False]),
            'validate_symbol_exists': trial.suggest_categorical('validate_symbol_exists', [True, False]),
            
            # Rate limiting (Binance has weight-based limits)
            'request_delay': trial.suggest_float('request_delay', 0.1, 1.0),
            'batch_delay': trial.suggest_float('batch_delay', 1.0, 5.0),
            'weight_limit_buffer': trial.suggest_int('weight_limit_buffer', 100, 500),  # Safety buffer for weight limits
            
            # Data fetching parameters
            'max_symbols': trial.suggest_int('max_symbols', 5, 20),
            'limit_per_request': trial.suggest_int('limit_per_request', 100, 1000),
            
            # Additional data endpoints
            'fetch_ticker_24hr': trial.suggest_categorical('fetch_ticker_24hr', [True, False]),
            'fetch_order_book': trial.suggest_categorical('fetch_order_book', [True, False]),
            'order_book_limit': trial.suggest_int('order_book_limit', 50, 500) if trial.suggest_categorical('temp_fetch_order_book', [True, False]) else 100,
            
            # Retry parameters
            'max_retries': trial.suggest_int('max_retries', 1, 3),
            'retry_delay': trial.suggest_float('retry_delay', 1.0, 5.0),
            'exponential_backoff': trial.suggest_categorical('exponential_backoff', [True, False]),
            
            # Data quality parameters
            'min_data_points': trial.suggest_int('min_data_points', 50, 1000),
            'max_missing_ratio': trial.suggest_float('max_missing_ratio', 0.0, 0.2),
            'require_volume': trial.suggest_categorical('require_volume', [True, False]),
            'min_volume_threshold': trial.suggest_float('min_volume_threshold', 0.0, 1000.0),
            
            # Data processing
            'remove_zero_volume': trial.suggest_categorical('remove_zero_volume', [True, False]),
            'validate_price_continuity': trial.suggest_categorical('validate_price_continuity', [True, False]),
            'price_change_threshold': trial.suggest_float('price_change_threshold', 0.1, 2.0),  # Max % change between consecutive periods
            
            # Timeframe parameters (if multiple timeframes)
            'secondary_intervals': [['1h'], ['4h'], ['1d'], ['1h', '4h'], ['4h', '1d']][trial.suggest_int('secondary_intervals_idx', 0, 4)] if fetch_multiple_timeframes else [],
        }
        
        return params
    
    def fetch_data_with_params(self, params: Dict[str, Any], 
                              symbols: List[str] = None) -> APIOptimizationResult:
        """Fetch data using Binance with specified parameters."""
        if symbols is None:
            symbols = self._select_symbols(params)
        
        start_time = time.time()
        total_data_points = 0
        api_calls = 0
        weight_used = 0
        errors = []
        all_data = []
        symbol_results = {}
        
        try:
            request_delay = params.get('request_delay', 0.2)
            batch_delay = params.get('batch_delay', 2.0)
            max_symbols = params.get('max_symbols', 10)
            
            # Limit symbols to max_symbols
            limited_symbols = symbols[:max_symbols]
            
            for i, symbol in enumerate(limited_symbols):
                symbol_data = {}
                symbol_success = False
                
                # Map symbol to Binance format
                binance_symbol = self._map_symbol(symbol, params)
                
                # Validate symbol exists if enabled
                if params.get('validate_symbol_exists', False):
                    symbol_info = self.api.fetch_symbol_info(binance_symbol)
                    if isinstance(symbol_info, dict) and 'error' in symbol_info:
                        errors.append(f"Symbol {symbol} not found: {symbol_info.get('error')}")
                        continue
                
                # Fetch market data (primary timeframe)
                result = self._fetch_market_data_with_retry(
                    binance_symbol, params, f"{symbol}_primary"
                )
                if result['success']:
                    symbol_data['primary_data'] = result['data']
                    total_data_points += len(result['data']) if not result['data'].empty else 0
                    symbol_success = True
                    all_data.append(result['data'])
                
                api_calls += result['api_calls']
                weight_used += result.get('weight_used', 1)
                errors.extend(result['errors'])
                
                time.sleep(request_delay)
                
                # Fetch multiple timeframes if enabled
                if params.get('fetch_multiple_timeframes', False) and symbol_success:
                    secondary_intervals = params.get('secondary_intervals', [])
                    for sec_interval in secondary_intervals:
                        sec_params = params.copy()
                        sec_params['interval'] = sec_interval
                        
                        result = self._fetch_market_data_with_retry(
                            binance_symbol, sec_params, f"{symbol}_{sec_interval}"
                        )
                        if result['success']:
                            symbol_data[f'{sec_interval}_data'] = result['data']
                            total_data_points += len(result['data']) if not result['data'].empty else 0
                        
                        api_calls += result['api_calls']
                        weight_used += result.get('weight_used', 1)
                        errors.extend(result['errors'])
                        
                        time.sleep(request_delay)
                
                # Fetch 24hr ticker if enabled
                if params.get('fetch_ticker_24hr', False):
                    result = self._fetch_with_retry(
                        lambda: pd.DataFrame([self.api.fetch_ticker_24hr(binance_symbol)]),
                        params, f"{symbol}_ticker"
                    )
                    if result['success']:
                        symbol_data['ticker_24hr'] = result['data']
                        total_data_points += len(result['data']) if not result['data'].empty else 0
                    
                    api_calls += result['api_calls']
                    weight_used += result.get('weight_used', 1)
                    errors.extend(result['errors'])
                    
                    time.sleep(request_delay)
                
                # Fetch order book if enabled
                if params.get('fetch_order_book', False):
                    result = self._fetch_with_retry(
                        lambda: self._process_order_book(
                            self.api.fetch_order_book(binance_symbol, limit=params.get('order_book_limit', 100))
                        ),
                        params, f"{symbol}_orderbook"
                    )
                    if result['success']:
                        symbol_data['order_book'] = result['data']
                        total_data_points += len(result['data']) if not result['data'].empty else 0
                    
                    api_calls += result['api_calls']
                    weight_used += result.get('weight_used', 1)
                    errors.extend(result['errors'])
                    
                    time.sleep(request_delay)
                
                # Store symbol data if successful
                if symbol_success and symbol_data:
                    symbol_results[symbol] = symbol_data
                
                # Batch delay and weight management
                if (i + 1) % 5 == 0 and i + 1 < len(limited_symbols):
                    time.sleep(batch_delay)
                
                # Check weight limits
                weight_limit = self.api.weight_limit - params.get('weight_limit_buffer', 200)
                if weight_used > weight_limit:
                    logger.warning(f"Approaching weight limit. Used: {weight_used}/{self.api.weight_limit}")
                    time.sleep(60)  # Wait for weight reset
                    weight_used = 0
        
        except Exception as e:
            return APIOptimizationResult(
                success=False,
                score=0.0,
                error=str(e),
                metrics={'api_calls': api_calls, 'weight_used': weight_used, 'errors': errors}
            )
        
        fetch_time = time.time() - start_time
        
        # Combine all market data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
        else:
            combined_data = pd.DataFrame()
        
        # Process combined data
        if not combined_data.empty:
            processed_data = self._process_crypto_data(combined_data, params)
            combined_data = processed_data
        
        # Calculate metrics
        data_quality_score = self._calculate_binance_data_quality_score(symbol_results, params)
        efficiency_score = self._calculate_binance_efficiency_score(
            fetch_time, total_data_points, api_calls, weight_used
        )
        cost_score = self._calculate_binance_cost_score(
            api_calls, weight_used, len(symbol_results)
        )
        error_score = self.calculate_error_score(len(errors) > 0, 'api_error' if errors else None)
        
        # Calculate coverage metrics
        requested_symbols = len(limited_symbols)
        successful_symbols = len(symbol_results)
        coverage_ratio = successful_symbols / requested_symbols if requested_symbols > 0 else 0.0
        
        metrics = {
            'data_quality_score': data_quality_score,
            'efficiency_score': efficiency_score,
            'cost_score': cost_score,
            'error_score': error_score,
            'fetch_time': fetch_time,
            'total_data_points': total_data_points,
            'api_calls': api_calls,
            'weight_used': weight_used,
            'successful_symbols': successful_symbols,
            'requested_symbols': requested_symbols,
            'coverage_ratio': coverage_ratio,
            'data_per_second': total_data_points / fetch_time if fetch_time > 0 else 0,
            'weight_efficiency': total_data_points / weight_used if weight_used > 0 else 0,
            'errors': errors[:5]
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
    
    def _select_symbols(self, params: Dict[str, Any]) -> List[str]:
        """Select cryptocurrency symbols based on strategy."""
        strategy = params.get('symbol_strategy', 'popular_pairs')
        max_symbols = params.get('max_symbols', 10)
        
        if strategy == 'base_asset_focus':
            # Focus on major cryptocurrencies with USDT pairs
            base_focus = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL'][:max_symbols//2]
            symbols = [f"{base}-USDT" for base in base_focus]
        elif strategy == 'quote_asset_focus':
            # Mix of quote assets
            symbols = ['BTC-USDT', 'ETH-BTC', 'BNB-ETH', 'ADA-BTC', 'SOL-USDT'][:max_symbols]
        elif strategy == 'mixed_strategy':
            # Mix of different market caps and pairs
            symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 'DOT-USDT', 
                      'MATIC-USDT', 'AVAX-USDT', 'DOGE-USDT'][:max_symbols]
        else:  # popular_pairs
            symbols = self.api.get_popular_symbols()[:max_symbols]
        
        return symbols
    
    def _map_symbol(self, symbol: str, params: Dict[str, Any]) -> str:
        """Map input symbol to Binance format."""
        if not params.get('auto_symbol_mapping', True):
            return symbol
        
        # Clean and format symbol
        clean_symbol = symbol.upper().replace('-', '').replace('_', '').replace('/', '')
        
        # If symbol doesn't end with a quote asset, add USDT
        quote_assets = ['USDT', 'BTC', 'ETH', 'BNB', 'BUSD']
        if not any(clean_symbol.endswith(quote) for quote in quote_assets):
            if clean_symbol.endswith('USD'):
                clean_symbol = clean_symbol.replace('USD', 'USDT')
            elif params.get('prefer_usdt_pairs', True):
                clean_symbol += 'USDT'
        
        return clean_symbol
    
    def _fetch_market_data_with_retry(self, symbol: str, params: Dict[str, Any], 
                                     context: str) -> Dict[str, Any]:
        """Fetch market data with Binance-specific retry logic."""
        max_retries = params.get('max_retries', 2)
        retry_delay = params.get('retry_delay', 2.0)
        exponential_backoff = params.get('exponential_backoff', True)
        
        api_calls = 0
        weight_used = 0
        errors = []
        
        for attempt in range(max_retries + 1):
            try:
                data = self.api.fetch_market_data(
                    symbol, 
                    period=params['period'], 
                    interval=params['interval']
                )
                api_calls += 1
                weight_used += 1  # Approximate weight for klines endpoint
                
                # Validate response
                if data.empty:
                    raise ValueError("Empty market data response")
                
                # Apply minimum data points filter
                if len(data) < params.get('min_data_points', 50):
                    raise ValueError(f"Insufficient data points: {len(data)}")
                
                return {
                    'success': True,
                    'data': data,
                    'api_calls': api_calls,
                    'weight_used': weight_used,
                    'errors': errors
                }
                
            except Exception as e:
                api_calls += 1
                weight_used += 1
                error_msg = f"{context} attempt {attempt + 1}: {str(e)}"
                errors.append(error_msg)
                
                if attempt < max_retries:
                    sleep_time = retry_delay
                    if exponential_backoff:
                        sleep_time *= (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    break
        
        return {
            'success': False,
            'data': pd.DataFrame(),
            'api_calls': api_calls,
            'weight_used': weight_used,
            'errors': errors
        }
    
    def _fetch_with_retry(self, fetch_func, params: Dict[str, Any], 
                         context: str) -> Dict[str, Any]:
        """Generic fetch function with retry logic."""
        max_retries = params.get('max_retries', 2)
        retry_delay = params.get('retry_delay', 2.0)
        
        api_calls = 0
        weight_used = 0
        errors = []
        
        for attempt in range(max_retries + 1):
            try:
                data = fetch_func()
                api_calls += 1
                weight_used += 1
                
                return {
                    'success': True,
                    'data': data,
                    'api_calls': api_calls,
                    'weight_used': weight_used,
                    'errors': errors
                }
                
            except Exception as e:
                api_calls += 1
                weight_used += 1
                error_msg = f"{context} attempt {attempt + 1}: {str(e)}"
                errors.append(error_msg)
                
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    break
        
        return {
            'success': False,
            'data': pd.DataFrame(),
            'api_calls': api_calls,
            'weight_used': weight_used,
            'errors': errors
        }
    
    def _process_order_book(self, order_book_data: Dict) -> pd.DataFrame:
        """Process order book data into DataFrame format."""
        if not order_book_data or 'error' in order_book_data:
            return pd.DataFrame()
        
        # Create summary statistics from order book
        bids = order_book_data.get('bids', pd.DataFrame())
        asks = order_book_data.get('asks', pd.DataFrame())
        
        if bids.empty or asks.empty:
            return pd.DataFrame()
        
        summary_data = {
            'symbol': [order_book_data.get('symbol', '')],
            'best_bid': [bids['price'].iloc[0] if not bids.empty else 0],
            'best_ask': [asks['price'].iloc[0] if not asks.empty else 0],
            'bid_ask_spread': [asks['price'].iloc[0] - bids['price'].iloc[0] if not bids.empty and not asks.empty else 0],
            'total_bid_volume': [bids['quantity'].sum() if not bids.empty else 0],
            'total_ask_volume': [asks['quantity'].sum() if not asks.empty else 0],
            'depth_levels': [min(len(bids), len(asks))]
        }
        
        return pd.DataFrame(summary_data)
    
    def _process_crypto_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Process cryptocurrency data according to parameters."""
        if data.empty:
            return data
        
        processed_data = data.copy()
        
        # Remove zero volume if enabled
        if params.get('remove_zero_volume', True) and 'Volume' in processed_data.columns:
            processed_data = processed_data[processed_data['Volume'] > 0]
        
        # Apply minimum volume threshold
        if params.get('require_volume', False) and 'Volume' in processed_data.columns:
            min_volume = params.get('min_volume_threshold', 0.0)
            processed_data = processed_data[processed_data['Volume'] >= min_volume]
        
        # Validate price continuity
        if params.get('validate_price_continuity', True):
            processed_data = self._validate_price_continuity(processed_data, params)
        
        return processed_data
    
    def _validate_price_continuity(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Validate price continuity and remove extreme gaps."""
        if data.empty or 'Close' not in data.columns:
            return data
        
        validated_data = data.copy()
        threshold = params.get('price_change_threshold', 1.0) / 100.0  # Convert to decimal
        
        # Calculate price changes
        price_changes = validated_data['Close'].pct_change().abs()
        
        # Mark extreme changes
        extreme_changes = price_changes > threshold
        
        if extreme_changes.any():
            logger.debug(f"Found {extreme_changes.sum()} extreme price changes")
            # Option: remove extreme changes or mark them
            # For now, we'll keep them but log them
            validated_data.loc[extreme_changes, 'extreme_change'] = True
        
        return validated_data
    
    def _calculate_binance_data_quality_score(self, symbol_results: Dict[str, Dict], 
                                            params: Dict[str, Any]) -> float:
        """Calculate Binance-specific data quality score."""
        if not symbol_results:
            return 0.0
        
        total_score = 0.0
        count = 0
        
        for symbol, data_dict in symbol_results.items():
            if 'primary_data' in data_dict and not data_dict['primary_data'].empty:
                data = data_dict['primary_data']
                
                # Completeness score
                completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
                
                # Volume consistency (important for crypto)
                volume_score = 1.0
                if 'Volume' in data.columns:
                    zero_volume_ratio = (data['Volume'] == 0).mean()
                    volume_score = 1.0 - zero_volume_ratio
                
                # Price consistency
                price_score = 1.0
                if 'Close' in data.columns and len(data) > 1:
                    price_changes = data['Close'].pct_change().abs()
                    extreme_changes = (price_changes > 0.5).mean()  # >50% changes
                    price_score = 1.0 - extreme_changes
                
                # Multi-timeframe bonus
                timeframe_bonus = 1.0
                if len(data_dict) > 1:  # Multiple timeframes
                    timeframe_bonus = 1.1
                
                symbol_score = (completeness * 0.4 + volume_score * 0.3 + price_score * 0.3) * timeframe_bonus
                total_score += min(1.0, symbol_score)  # Cap at 1.0
                count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def _calculate_binance_efficiency_score(self, fetch_time: float, data_points: int,
                                          api_calls: int, weight_used: int) -> float:
        """Calculate Binance-specific efficiency score."""
        if fetch_time <= 0 or api_calls <= 0:
            return 0.0
        
        # Data points per second
        data_per_second = data_points / fetch_time
        
        # Weight efficiency (Binance uses weight-based rate limiting)
        weight_efficiency = data_points / weight_used if weight_used > 0 else 0
        
        # API call efficiency
        data_per_call = data_points / api_calls
        
        # Normalize scores
        throughput_score = min(1.0, data_per_second / 1000.0)  # 1000 data points/sec = 1.0
        weight_score = min(1.0, weight_efficiency / 500.0)     # 500 data points/weight = 1.0
        call_score = min(1.0, data_per_call / 1000.0)         # 1000 data points/call = 1.0
        
        return float(throughput_score * 0.3 + weight_score * 0.4 + call_score * 0.3)
    
    def _calculate_binance_cost_score(self, api_calls: int, weight_used: int,
                                    successful_symbols: int) -> float:
        """Calculate Binance-specific cost effectiveness score."""
        if api_calls <= 0 or weight_used <= 0:
            return 0.0
        
        # Success rate
        success_per_call = successful_symbols / api_calls
        
        # Weight efficiency (important for Binance rate limits)
        symbols_per_weight = successful_symbols / weight_used
        
        # Overall efficiency
        efficiency_score = min(1.0, success_per_call)
        weight_efficiency_score = min(1.0, symbols_per_weight / 0.5)  # 0.5 symbols/weight = 1.0
        
        return float(efficiency_score * 0.6 + weight_efficiency_score * 0.4)
    
    def run_crypto_optimization_study(self, study_name: str = "binance_crypto_optimization",
                                     n_trials: int = 60) -> optuna.Study:
        """
        Run optimization study for cryptocurrency trading parameters.
        
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
        
        # Use crypto-specific symbols
        crypto_symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT']
        result = self.optimize_for_symbols(crypto_symbols)
        
        return self.study


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = BinanceAPIOptimizer()
    
    print("Running Binance API optimization...")
    print("This works with public endpoints (no API key required)")
    
    # Run crypto optimization study
    study = optimizer.run_crypto_optimization_study(n_trials=25)
    
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Test with best parameters
    if study.best_params:
        best_result = optimizer.fetch_data_with_params(
            study.best_params, 
            ['BTC-USDT', 'ETH-USDT']
        )
        
        print(f"Test result - Success: {best_result.success}")
        if best_result.metrics:
            print(f"API calls: {best_result.metrics.get('api_calls', 0)}")
            print(f"Weight used: {best_result.metrics.get('weight_used', 0)}")
            print(f"Data points: {best_result.metrics.get('total_data_points', 0)}")
            print(f"Weight efficiency: {best_result.metrics.get('weight_efficiency', 0):.1f} data/weight")