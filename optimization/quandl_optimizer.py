#!/usr/bin/env python3
"""
Quandl API Optimizer

Optimizes Quandl API parameters for dataset selection, date ranges, and data quality.
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

from features.data_sources.quandl import QuandlAPI
from optimization.api_base import BaseAPIOptimizer, APIOptimizationResult, OptimizationConfig
import logging

logger = logging.getLogger(__name__)


class QuandlAPIOptimizer(BaseAPIOptimizer):
    """
    Optimizer for Quandl API parameters.
    
    Optimizes:
    - Dataset selection and database choices
    - Date range optimization for historical data
    - Multi-dataset combination strategies
    - Data completeness and quality metrics
    """
    
    def __init__(self, api_key: str = None, config: OptimizationConfig = None):
        """Initialize Quandl API optimizer."""
        api_instance = QuandlAPI(api_key=api_key)
        super().__init__(api_instance, config)
        
        # Popular datasets for testing
        self.popular_datasets = {
            'stocks': ['WIKI/AAPL', 'WIKI/MSFT', 'WIKI/GOOGL'],
            'economic': ['FRED/GDP', 'FRED/UNRATE', 'FRED/CPIAUCSL'],
            'commodities': ['LBMA/GOLD', 'LBMA/SILVER'],
            'indices': ['YAHOO/INDEX_GSPC', 'YAHOO/INDEX_VIX']
        }
        
        # Default test datasets
        self.default_datasets = [
            'WIKI/AAPL', 'WIKI/MSFT', 'FRED/GDP', 'LBMA/GOLD'
        ]
        
        # Set optimization weights for Quandl (data quality very important)
        self.set_optimization_weights({
            'data_quality': 0.50,  # Higher weight for data quality
            'efficiency': 0.25,
            'cost_effectiveness': 0.15,
            'error_rate': 0.10
        })
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define Quandl API parameter search space."""
        
        # Dataset selection strategy
        dataset_strategy = trial.suggest_categorical('dataset_strategy', [
            'single_category', 'mixed_categories', 'specific_datasets'
        ])
        
        # Date range parameters
        end_date = datetime.now()
        
        # Historical data range
        years_back = trial.suggest_int('years_back', 1, 10)
        start_date = end_date - timedelta(days=years_back * 365)
        
        # Date range strategy
        date_strategy = trial.suggest_categorical('date_strategy', [
            'full_range', 'recent_only', 'yearly_samples', 'quarterly_samples'
        ])
        
        params = {
            'dataset_strategy': dataset_strategy,
            'years_back': years_back,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'date_strategy': date_strategy,
            
            # Dataset selection (based on strategy)
            'max_datasets': trial.suggest_int('max_datasets', 2, 8),
            'include_stocks': trial.suggest_categorical('include_stocks', [True, False]),
            'include_economic': trial.suggest_categorical('include_economic', [True, False]),
            'include_commodities': trial.suggest_categorical('include_commodities', [True, False]),
            'include_indices': trial.suggest_categorical('include_indices', [True, False]),
            
            # Data processing parameters
            'min_data_points': trial.suggest_int('min_data_points', 50, 1000),
            'max_missing_ratio': trial.suggest_float('max_missing_ratio', 0.0, 0.4),
            'require_recent_data': trial.suggest_categorical('require_recent_data', [True, False]),
            'recent_data_days': trial.suggest_int('recent_data_days', 30, 365),
            
            # Request parameters
            'request_delay': trial.suggest_float('request_delay', 0.1, 2.0),
            'batch_size': trial.suggest_int('batch_size', 1, 5),
            'max_retries': trial.suggest_int('max_retries', 0, 3),
            'retry_delay': trial.suggest_float('retry_delay', 1.0, 5.0),
            
            # Data quality filters
            'remove_outliers': trial.suggest_categorical('remove_outliers', [True, False]),
            'outlier_std_threshold': trial.suggest_float('outlier_std_threshold', 2.0, 5.0),
            'interpolate_missing': trial.suggest_categorical('interpolate_missing', [True, False]),
            'interpolation_method': trial.suggest_categorical('interpolation_method', 
                                                             ['linear', 'forward', 'backward']),
            
            # Validation parameters
            'validate_data_types': trial.suggest_categorical('validate_data_types', [True, False]),
            'check_data_frequency': trial.suggest_categorical('check_data_frequency', [True, False]),
        }
        
        # Ensure at least one data category is selected
        if not any([params['include_stocks'], params['include_economic'], 
                   params['include_commodities'], params['include_indices']]):
            params['include_stocks'] = True
        
        return params
    
    def fetch_data_with_params(self, params: Dict[str, Any], 
                              datasets: List[str] = None) -> APIOptimizationResult:
        """Fetch data using Quandl with specified parameters."""
        if datasets is None:
            datasets = self._select_datasets(params)
        
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
        dataset_results = {}
        
        try:
            request_delay = params.get('request_delay', 0.5)
            batch_size = params.get('batch_size', 3)
            
            # Process datasets in batches
            for i in range(0, len(datasets), batch_size):
                batch_datasets = datasets[i:i + batch_size]
                
                for dataset in batch_datasets:
                    # Parse dataset code
                    if '/' in dataset:
                        database_code, dataset_code = dataset.split('/', 1)
                    else:
                        continue  # Skip invalid dataset format
                    
                    # Determine date range based on strategy
                    start_date, end_date = self._get_date_range(params, dataset)
                    
                    result = self._fetch_dataset_with_retry(
                        database_code, dataset_code, start_date, end_date, params
                    )
                    
                    api_calls += result['api_calls']
                    errors.extend(result['errors'])
                    
                    if result['success'] and not result['data'].empty:
                        processed_data = self._process_data(result['data'], params)
                        
                        if len(processed_data) >= params.get('min_data_points', 50):
                            dataset_results[dataset] = processed_data
                            total_data_points += len(processed_data)
                            all_data.append(processed_data)
                        else:
                            errors.append(f"{dataset}: Insufficient data points ({len(processed_data)})")
                    
                    time.sleep(request_delay)
                
                # Batch delay
                if i + batch_size < len(datasets):
                    time.sleep(1.0)
        
        except Exception as e:
            return APIOptimizationResult(
                success=False,
                score=0.0,
                error=str(e),
                metrics={'api_calls': api_calls, 'errors': errors}
            )
        
        fetch_time = time.time() - start_time
        
        # Combine all data for analysis
        if all_data:
            # For Quandl, we'll create a summary DataFrame
            combined_data = self._combine_datasets(dataset_results)
        else:
            combined_data = pd.DataFrame()
        
        # Calculate metrics
        data_quality_score = self._calculate_quandl_data_quality_score(dataset_results, params)
        efficiency_score = self.calculate_efficiency_score(fetch_time, total_data_points, api_calls)
        cost_score = self._calculate_quandl_cost_score(api_calls, len(dataset_results), errors)
        error_score = self.calculate_error_score(len(errors) > 0, 'fetch_error' if errors else None)
        
        # Calculate coverage metrics
        requested_datasets = len(datasets)
        successful_datasets = len(dataset_results)
        coverage_ratio = successful_datasets / requested_datasets if requested_datasets > 0 else 0.0
        
        # Data recency score
        recency_score = self._calculate_data_recency_score(dataset_results, params)
        
        metrics = {
            'data_quality_score': data_quality_score,
            'efficiency_score': efficiency_score,
            'cost_score': cost_score,
            'error_score': error_score,
            'recency_score': recency_score,
            'fetch_time': fetch_time,
            'total_data_points': total_data_points,
            'api_calls': api_calls,
            'successful_datasets': successful_datasets,
            'requested_datasets': requested_datasets,
            'coverage_ratio': coverage_ratio,
            'avg_data_per_dataset': total_data_points / successful_datasets if successful_datasets > 0 else 0,
            'errors': errors[:5]  # Keep only first 5 errors for logging
        }
        
        composite_score = self.calculate_composite_score(
            APIOptimizationResult(True, 0.0, combined_data, metrics)
        )
        
        return APIOptimizationResult(
            success=successful_datasets > 0,
            score=composite_score,
            data=combined_data,
            metrics=metrics
        )
    
    def _select_datasets(self, params: Dict[str, Any]) -> List[str]:
        """Select datasets based on parameters."""
        strategy = params.get('dataset_strategy', 'mixed_categories')
        max_datasets = params.get('max_datasets', 5)
        
        selected_datasets = []
        
        if strategy == 'specific_datasets':
            selected_datasets = self.default_datasets[:max_datasets]
        else:
            # Build dataset list based on categories
            if params.get('include_stocks', True):
                selected_datasets.extend(self.popular_datasets['stocks'][:2])
            if params.get('include_economic', True):
                selected_datasets.extend(self.popular_datasets['economic'][:2])
            if params.get('include_commodities', False):
                selected_datasets.extend(self.popular_datasets['commodities'][:1])
            if params.get('include_indices', False):
                selected_datasets.extend(self.popular_datasets['indices'][:1])
            
            # Limit to max_datasets
            selected_datasets = selected_datasets[:max_datasets]
        
        return selected_datasets if selected_datasets else self.default_datasets[:3]
    
    def _get_date_range(self, params: Dict[str, Any], dataset: str) -> tuple:
        """Get optimized date range for dataset."""
        strategy = params.get('date_strategy', 'full_range')
        end_date = params.get('end_date')
        start_date = params.get('start_date')
        
        if strategy == 'recent_only':
            # Only get recent data
            recent_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            return recent_start, end_date
        elif strategy == 'yearly_samples':
            # Sample one year of data
            sample_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            return sample_start, end_date
        elif strategy == 'quarterly_samples':
            # Sample quarterly data
            quarterly_start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            return quarterly_start, end_date
        else:
            # Full range
            return start_date, end_date
    
    def _fetch_dataset_with_retry(self, database_code: str, dataset_code: str,
                                 start_date: str, end_date: str, 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch a single dataset with retry logic."""
        max_retries = params.get('max_retries', 2)
        retry_delay = params.get('retry_delay', 2.0)
        
        api_calls = 0
        errors = []
        
        for attempt in range(max_retries + 1):
            try:
                data = self.api.fetch_dataset(database_code, dataset_code, start_date, end_date)
                api_calls += 1
                
                # Validate response
                if data.empty:
                    raise ValueError("Empty dataset response")
                
                return {
                    'success': True,
                    'data': data,
                    'api_calls': api_calls,
                    'errors': errors
                }
                
            except Exception as e:
                api_calls += 1
                error_msg = f"{database_code}/{dataset_code} attempt {attempt + 1}: {str(e)}"
                errors.append(error_msg)
                
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    break
        
        return {
            'success': False,
            'data': pd.DataFrame(),
            'api_calls': api_calls,
            'errors': errors
        }
    
    def _process_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Process fetched dataset according to parameters."""
        if data.empty:
            return data
        
        processed_data = data.copy()
        
        # Remove outliers if enabled
        if params.get('remove_outliers', False):
            processed_data = self._remove_outliers(processed_data, params)
        
        # Interpolate missing values if enabled
        if params.get('interpolate_missing', False):
            method = params.get('interpolation_method', 'linear')
            processed_data = processed_data.interpolate(method=method)
        
        # Validate data types if enabled
        if params.get('validate_data_types', True):
            processed_data = self._validate_data_types(processed_data)
        
        return processed_data
    
    def _remove_outliers(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Remove statistical outliers from data."""
        if data.empty:
            return data
        
        cleaned_data = data.copy()
        std_threshold = params.get('outlier_std_threshold', 3.0)
        
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['database_code', 'dataset_code', 'source']:
                col_data = cleaned_data[col]
                if len(col_data.dropna()) > 10:  # Need sufficient data for outlier detection
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    
                    if std_val > 0:
                        z_scores = np.abs((col_data - mean_val) / std_val)
                        outliers = z_scores > std_threshold
                        
                        if outliers.any():
                            logger.debug(f"Removing {outliers.sum()} outliers from {col}")
                            cleaned_data.loc[outliers, col] = np.nan
        
        return cleaned_data
    
    def _validate_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data types."""
        if data.empty:
            return data
        
        validated_data = data.copy()
        
        # Convert numeric columns
        for col in validated_data.columns:
            if col not in ['database_code', 'dataset_code', 'source', 'fetch_timestamp']:
                try:
                    validated_data[col] = pd.to_numeric(validated_data[col], errors='coerce')
                except:
                    pass  # Keep as is if conversion fails
        
        return validated_data
    
    def _combine_datasets(self, dataset_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple datasets into a summary DataFrame."""
        if not dataset_results:
            return pd.DataFrame()
        
        # Create a summary with key statistics for each dataset
        summary_data = []
        
        for dataset_name, data in dataset_results.items():
            if not data.empty:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if col not in ['database_code', 'dataset_code', 'source']:
                        col_data = data[col].dropna()
                        if len(col_data) > 0:
                            summary_data.append({
                                'dataset': dataset_name,
                                'column': col,
                                'count': len(col_data),
                                'mean': col_data.mean(),
                                'std': col_data.std(),
                                'min': col_data.min(),
                                'max': col_data.max(),
                                'first_date': data.index.min() if hasattr(data.index, 'min') else None,
                                'last_date': data.index.max() if hasattr(data.index, 'max') else None
                            })
        
        return pd.DataFrame(summary_data)
    
    def _calculate_quandl_data_quality_score(self, dataset_results: Dict[str, pd.DataFrame], 
                                           params: Dict[str, Any]) -> float:
        """Calculate Quandl-specific data quality score."""
        if not dataset_results:
            return 0.0
        
        total_score = 0.0
        
        for dataset_name, data in dataset_results.items():
            if data.empty:
                continue
            
            # Completeness score
            total_cells = len(data) * len(data.columns)
            missing_cells = data.isnull().sum().sum()
            completeness = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
            
            # Recency score (for datasets that should have recent data)
            recency = 1.0
            if params.get('require_recent_data', False) and hasattr(data.index, 'max'):
                try:
                    last_date = pd.to_datetime(data.index.max())
                    days_old = (datetime.now() - last_date).days
                    max_days = params.get('recent_data_days', 90)
                    recency = max(0.0, 1.0 - (days_old / max_days))
                except:
                    recency = 0.5  # Neutral score if can't determine recency
            
            # Data consistency score
            consistency = 1.0
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['database_code', 'dataset_code', 'source']:
                    col_data = data[col].dropna()
                    if len(col_data) > 1:
                        # Check for extreme variations
                        cv = col_data.std() / col_data.mean() if col_data.mean() != 0 else 0
                        if cv > 10:  # Coefficient of variation > 10 indicates high volatility
                            consistency *= 0.9
            
            dataset_score = completeness * 0.5 + recency * 0.3 + consistency * 0.2
            total_score += dataset_score
        
        return total_score / len(dataset_results)
    
    def _calculate_quandl_cost_score(self, api_calls: int, successful_datasets: int, 
                                   errors: List[str]) -> float:
        """Calculate Quandl-specific cost effectiveness score."""
        if api_calls <= 0:
            return 0.0
        
        # Datasets per API call
        datasets_per_call = successful_datasets / api_calls
        
        # Error penalty
        error_penalty = len(errors) / api_calls if api_calls > 0 else 0
        
        # Efficiency score
        efficiency = min(1.0, datasets_per_call)  # 1 dataset per call = 1.0
        error_score = max(0.0, 1.0 - error_penalty * 2.0)
        
        return float(efficiency * 0.7 + error_score * 0.3)
    
    def _calculate_data_recency_score(self, dataset_results: Dict[str, pd.DataFrame], 
                                    params: Dict[str, Any]) -> float:
        """Calculate average data recency score."""
        if not dataset_results:
            return 0.0
        
        total_recency = 0.0
        count = 0
        
        for dataset_name, data in dataset_results.items():
            if data.empty or not hasattr(data.index, 'max'):
                continue
            
            try:
                last_date = pd.to_datetime(data.index.max())
                days_old = (datetime.now() - last_date).days
                
                # Recent data is better (exponential decay)
                recency = np.exp(-days_old / 365.0)  # Decay over 1 year
                total_recency += recency
                count += 1
            except:
                continue
        
        return total_recency / count if count > 0 else 0.0
    
    def optimize_for_data_category(self, category: str = 'mixed') -> Dict[str, Any]:
        """
        Optimize parameters for a specific data category.
        
        Args:
            category: 'stocks', 'economic', 'commodities', 'indices', or 'mixed'
        """
        if category in self.popular_datasets:
            datasets = self.popular_datasets[category]
        else:
            datasets = None  # Use parameter-based selection
        
        return self.optimize_for_symbols(datasets)  # Use symbols parameter for datasets
    
    def run_dataset_optimization_study(self, study_name: str = "quandl_dataset_optimization",
                                      n_trials: int = 40) -> optuna.Study:
        """
        Run optimization study for dataset selection and processing.
        
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
        
        # Use default datasets for the study
        result = self.optimize_for_symbols(self.default_datasets)
        
        return self.study


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create optimizer
    optimizer = QuandlAPIOptimizer()
    
    print("Running Quandl API optimization...")
    print("Note: This requires a Quandl API key to work properly")
    
    # Run dataset optimization study
    study = optimizer.run_dataset_optimization_study(n_trials=15)
    
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # Test with best parameters
    if study.best_params:
        best_result = optimizer.fetch_data_with_params(
            study.best_params, 
            ['WIKI/AAPL', 'FRED/GDP']
        )
        
        print(f"Test result - Success: {best_result.success}")
        if best_result.metrics:
            print(f"API calls: {best_result.metrics.get('api_calls', 0)}")
            print(f"Successful datasets: {best_result.metrics.get('successful_datasets', 0)}")
            print(f"Data quality score: {best_result.metrics.get('data_quality_score', 0):.3f}")
            print(f"Recency score: {best_result.metrics.get('recency_score', 0):.3f}")