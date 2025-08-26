#!/usr/bin/env python3
"""
Base API Optimizer Framework

Provides the foundation for optimizing data source API parameters using Optuna
with API-specific constraints and objectives.
"""

import optuna
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import logging
import time
from datetime import datetime, timedelta

from .base import BaseOptimizer, OptimizationConfig

logger = logging.getLogger(__name__)


class APIOptimizationResult:
    """Result of API parameter optimization"""
    
    def __init__(self, success: bool, score: float, data: pd.DataFrame = None, 
                 metrics: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.score = score
        self.data = data
        self.metrics = metrics or {}
        self.error = error
        self.timestamp = datetime.now()


class BaseAPIOptimizer(BaseOptimizer):
    """
    Base class for optimizing data source API parameters.
    
    This class extends BaseOptimizer to handle API-specific optimization including:
    1. Data quality metrics (completeness, accuracy, validation)
    2. Efficiency metrics (fetch speed, API calls, data per call)
    3. Cost optimization (API usage vs. data value)
    4. Constraint handling (rate limits, valid combinations)
    """
    
    def __init__(self, 
                 api_instance: Any,
                 config: OptimizationConfig = None,
                 cache_dir: str = ".api_optimization_cache"):
        """
        Initialize API optimizer.
        
        Args:
            api_instance: Instance of the API class to optimize
            config: Optimization configuration
            cache_dir: Directory for caching results
        """
        super().__init__(api_instance.__class__, config, cache_dir)
        self.api = api_instance
        self.optimization_weights = {
            'data_quality': 0.4,
            'efficiency': 0.3,
            'cost_effectiveness': 0.2,
            'error_rate': 0.1
        }
    
    def create_model_instance(self, params: Dict[str, Any]) -> Any:
        """
        Create API instance with optimized parameters.
        For APIs, this typically means configuring fetch parameters.
        """
        return self.api
    
    def evaluate_model(self, model: Any, data: pd.DataFrame) -> float:
        """
        Evaluate API performance with given parameters.
        This method should be implemented by subclasses but we provide a default.
        """
        # Default evaluation based on data quality
        if data.empty:
            return 0.0
        
        # Basic data quality metrics
        completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        
        return float(completeness)
    
    @abstractmethod
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the search space for API parameters.
        Should include constraints specific to the API.
        """
        pass
    
    @abstractmethod
    def fetch_data_with_params(self, params: Dict[str, Any], 
                              symbols: List[str] = None) -> APIOptimizationResult:
        """
        Fetch data using the specified parameters and return optimization result.
        
        Args:
            params: Parameters to use for data fetching
            symbols: List of symbols to fetch (if applicable)
            
        Returns:
            APIOptimizationResult with success status, score, and metrics
        """
        pass
    
    def calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score (0-1)."""
        if data.empty:
            return 0.0
        
        # Completeness (no missing values)
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        completeness = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
        
        # Validity (for OHLCV data, check basic constraints)
        validity = 1.0
        if 'Open' in data.columns and 'High' in data.columns and 'Low' in data.columns and 'Close' in data.columns:
            # High should be >= Open, Low, Close
            invalid_high = ((data['High'] < data['Open']) | 
                           (data['High'] < data['Low']) | 
                           (data['High'] < data['Close'])).sum()
            validity = 1.0 - (invalid_high / len(data)) if len(data) > 0 else 0.0
        
        # Volume should be non-negative
        if 'Volume' in data.columns:
            negative_volume = (data['Volume'] < 0).sum()
            volume_validity = 1.0 - (negative_volume / len(data)) if len(data) > 0 else 0.0
            validity = min(validity, volume_validity)
        
        return float(completeness * 0.7 + validity * 0.3)
    
    def calculate_efficiency_score(self, fetch_time: float, data_size: int, 
                                 api_calls: int = 1) -> float:
        """Calculate efficiency score based on fetch time and data retrieved."""
        if fetch_time <= 0 or data_size <= 0:
            return 0.0
        
        # Data points per second
        data_per_second = data_size / fetch_time
        
        # Data points per API call
        data_per_call = data_size / api_calls
        
        # Normalize to 0-1 scale (these are heuristic thresholds)
        time_score = min(1.0, data_per_second / 1000.0)  # 1000 data points/sec = 1.0
        call_score = min(1.0, data_per_call / 500.0)     # 500 data points/call = 1.0
        
        return float(time_score * 0.6 + call_score * 0.4)
    
    def calculate_cost_score(self, api_calls: int, data_value: float) -> float:
        """Calculate cost effectiveness score."""
        if api_calls <= 0:
            return 0.0
        
        # Higher data value per API call is better
        value_per_call = data_value / api_calls
        
        # Normalize (heuristic threshold)
        return float(min(1.0, value_per_call / 100.0))  # 100 value units/call = 1.0
    
    def calculate_error_score(self, has_error: bool, error_type: str = None) -> float:
        """Calculate error score (1.0 = no error, 0.0 = error)."""
        if has_error:
            # Different penalty for different error types
            error_penalties = {
                'rate_limit': 0.3,  # Rate limit errors are somewhat expected
                'invalid_symbol': 0.1,  # Invalid symbols should be heavily penalized
                'network': 0.5,  # Network errors are temporary
                'api_key': 0.0,  # API key errors are critical
                'unknown': 0.2
            }
            return error_penalties.get(error_type, 0.2)
        return 1.0
    
    def calculate_composite_score(self, result: APIOptimizationResult) -> float:
        """Calculate composite optimization score."""
        metrics = result.metrics
        
        data_quality = metrics.get('data_quality_score', 0.0)
        efficiency = metrics.get('efficiency_score', 0.0)
        cost_effectiveness = metrics.get('cost_score', 0.0)
        error_score = metrics.get('error_score', 1.0)
        
        # Weighted combination
        composite = (
            data_quality * self.optimization_weights['data_quality'] +
            efficiency * self.optimization_weights['efficiency'] +
            cost_effectiveness * self.optimization_weights['cost_effectiveness'] +
            error_score * self.optimization_weights['error_rate']
        )
        
        return float(composite)
    
    def optimize_for_symbols(self, symbols: List[str], 
                           validation_symbols: List[str] = None) -> Dict[str, Any]:
        """
        Optimize API parameters for a specific set of symbols.
        
        Args:
            symbols: Symbols to optimize for
            validation_symbols: Symbols to validate on (if different)
            
        Returns:
            Optimization results
        """
        def objective_wrapper(trial, data):
            # Ignore the data parameter for API optimization
            return self._api_objective(trial, symbols)
        
        # Create temporary objective function
        original_objective = self._objective
        self._objective = objective_wrapper
        
        try:
            # Run optimization
            dummy_data = pd.DataFrame({'dummy': [1, 2, 3]})  # Not used in API optimization
            result = self.optimize(dummy_data)
            
            # Validate on different symbols if provided
            if validation_symbols and result['best_params']:
                validation_result = self.fetch_data_with_params(
                    result['best_params'], validation_symbols
                )
                result['validation_score'] = validation_result.score
                result['validation_metrics'] = validation_result.metrics
            
            return result
            
        finally:
            # Restore original objective
            self._objective = original_objective
    
    def _api_objective(self, trial: optuna.Trial, symbols: List[str]) -> float:
        """API-specific objective function."""
        params = self.define_search_space(trial)
        
        # Generate cache key for API parameters
        import hashlib
        import json
        params_str = json.dumps(params, sort_keys=True)
        symbols_str = json.dumps(sorted(symbols))
        cache_key = hashlib.md5(f"{params_str}_{symbols_str}".encode()).hexdigest()
        
        if cache_key in self.cache:
            logger.debug(f"Cache hit for trial {trial.number}")
            return self.cache[cache_key]
        
        try:
            # Fetch data with parameters
            result = self.fetch_data_with_params(params, symbols)
            
            if not result.success:
                # Return poor score for failed fetches
                score = 0.0
            else:
                score = self.calculate_composite_score(result)
            
            # Cache result
            self.cache[cache_key] = score
            
            # Log progress
            logger.info(f"Trial {trial.number}: Score={score:.4f}, Params={params}")
            if result.metrics:
                logger.debug(f"Metrics: {result.metrics}")
            
            return score
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            return 0.0  # Return worst possible score
    
    def get_optimization_weights(self) -> Dict[str, float]:
        """Get current optimization weights."""
        return self.optimization_weights.copy()
    
    def set_optimization_weights(self, weights: Dict[str, float]):
        """
        Set optimization weights.
        
        Args:
            weights: Dictionary with keys: data_quality, efficiency, 
                    cost_effectiveness, error_rate
        """
        # Validate weights
        required_keys = {'data_quality', 'efficiency', 'cost_effectiveness', 'error_rate'}
        if not all(key in weights for key in required_keys):
            raise ValueError(f"Missing required weight keys: {required_keys}")
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            normalized_weights = {key: value / total for key, value in weights.items()}
            self.optimization_weights.update(normalized_weights)
        else:
            raise ValueError("Weights must sum to a positive value")