#!/usr/bin/env python3
"""
Batch Processing for Hyperparameter Optimization

Provides efficient batching capabilities for processing multiple parameter
combinations and datasets simultaneously.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result from batch processing"""
    params: Dict[str, Any]
    score: float
    model: Any
    processing_time: float
    success: bool
    error: Optional[str] = None


class BatchProcessor:
    """
    Batch processor for efficient evaluation of multiple parameter combinations.
    
    Features:
    - Parallel processing of parameter combinations
    - Efficient data batching
    - Result aggregation and statistics
    - Error handling and recovery
    """
    
    def __init__(self, 
                 n_workers: int = None,
                 use_processes: bool = False,
                 timeout_per_job: float = 300.0):
        """
        Initialize batch processor.
        
        Args:
            n_workers: Number of parallel workers (default: CPU count)
            use_processes: Whether to use processes instead of threads
            timeout_per_job: Timeout in seconds for each job
        """
        self.n_workers = n_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.timeout_per_job = timeout_per_job
        
        logger.info(f"Initialized BatchProcessor with {self.n_workers} workers "
                   f"({'processes' if use_processes else 'threads'})")
    
    def process_parameter_batch(self,
                               param_combinations: List[Dict[str, Any]],
                               model_factory: Callable[[Dict[str, Any]], Any],
                               evaluate_func: Callable[[Any, pd.DataFrame], float],
                               data: pd.DataFrame) -> List[BatchResult]:
        """
        Process a batch of parameter combinations in parallel.
        
        Args:
            param_combinations: List of parameter dictionaries to evaluate
            model_factory: Function that creates model instance from parameters
            evaluate_func: Function that evaluates model on data
            data: Data to evaluate models on
            
        Returns:
            List of BatchResult objects
        """
        logger.info(f"Processing batch of {len(param_combinations)} parameter combinations")
        
        # Create worker function
        def worker_func(params: Dict[str, Any]) -> BatchResult:
            start_time = time.time()
            
            try:
                # Create model
                model = model_factory(params)
                
                # Evaluate model
                score = evaluate_func(model, data)
                
                processing_time = time.time() - start_time
                
                return BatchResult(
                    params=params,
                    score=score,
                    model=model,
                    processing_time=processing_time,
                    success=True
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Error processing params {params}: {str(e)}")
                
                return BatchResult(
                    params=params,
                    score=np.nan,
                    model=None,
                    processing_time=processing_time,
                    success=False,
                    error=str(e)
                )
        
        # Execute in parallel
        if self.use_processes:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        results = []
        with executor_class(max_workers=self.n_workers) as executor:
            # Submit all jobs
            futures = [executor.submit(worker_func, params) 
                      for params in param_combinations]
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=self.timeout_per_job)
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{len(futures)} evaluations")
                        
                except Exception as e:
                    logger.error(f"Job {i} failed with timeout or error: {str(e)}")
                    results.append(BatchResult(
                        params=param_combinations[i],
                        score=np.nan,
                        model=None,
                        processing_time=self.timeout_per_job,
                        success=False,
                        error=str(e)
                    ))
        
        logger.info(f"Batch processing completed. "
                   f"Success rate: {sum(r.success for r in results)}/{len(results)}")
        
        return results
    
    def process_data_batches(self,
                            data_batches: List[pd.DataFrame],
                            params: Dict[str, Any],
                            model_factory: Callable[[Dict[str, Any]], Any],
                            evaluate_func: Callable[[Any, pd.DataFrame], float]) -> List[BatchResult]:
        """
        Evaluate a single parameter combination on multiple data batches.
        
        Args:
            data_batches: List of data batches to evaluate on
            params: Parameter dictionary
            model_factory: Function that creates model instance
            evaluate_func: Function that evaluates model on data
            
        Returns:
            List of BatchResult objects (one per data batch)
        """
        logger.info(f"Evaluating parameters on {len(data_batches)} data batches")
        
        def worker_func(data_batch: pd.DataFrame) -> BatchResult:
            start_time = time.time()
            
            try:
                # Create model
                model = model_factory(params)
                
                # Evaluate on this data batch
                score = evaluate_func(model, data_batch)
                
                processing_time = time.time() - start_time
                
                return BatchResult(
                    params=params,
                    score=score,
                    model=model,
                    processing_time=processing_time,
                    success=True
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Error evaluating on batch: {str(e)}")
                
                return BatchResult(
                    params=params,
                    score=np.nan,
                    model=None,
                    processing_time=processing_time,
                    success=False,
                    error=str(e)
                )
        
        # Execute in parallel
        if self.use_processes:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        results = []
        with executor_class(max_workers=self.n_workers) as executor:
            futures = [executor.submit(worker_func, batch) for batch in data_batches]
            
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=self.timeout_per_job)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Data batch {i} evaluation failed: {str(e)}")
                    results.append(BatchResult(
                        params=params,
                        score=np.nan,
                        model=None,
                        processing_time=self.timeout_per_job,
                        success=False,
                        error=str(e)
                    ))
        
        return results
    
    def aggregate_batch_results(self, 
                               results: List[BatchResult],
                               aggregation_method: str = 'mean') -> Dict[str, Any]:
        """
        Aggregate results from batch processing.
        
        Args:
            results: List of BatchResult objects
            aggregation_method: How to aggregate scores ('mean', 'median', 'min', 'max')
            
        Returns:
            Dictionary with aggregated results
        """
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            logger.warning("No successful results to aggregate")
            return {
                'aggregated_score': np.nan,
                'n_successful': 0,
                'n_total': len(results),
                'success_rate': 0.0,
                'processing_times': [],
                'errors': [r.error for r in results if not r.success]
            }
        
        scores = [r.score for r in successful_results]
        processing_times = [r.processing_time for r in successful_results]
        
        # Aggregate scores
        if aggregation_method == 'mean':
            aggregated_score = np.mean(scores)
        elif aggregation_method == 'median':
            aggregated_score = np.median(scores)
        elif aggregation_method == 'min':
            aggregated_score = np.min(scores)
        elif aggregation_method == 'max':
            aggregated_score = np.max(scores)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        return {
            'aggregated_score': aggregated_score,
            'score_std': np.std(scores),
            'score_min': np.min(scores),
            'score_max': np.max(scores),
            'scores': scores,
            'n_successful': len(successful_results),
            'n_total': len(results),
            'success_rate': len(successful_results) / len(results),
            'avg_processing_time': np.mean(processing_times),
            'total_processing_time': sum(processing_times),
            'processing_times': processing_times,
            'errors': [r.error for r in results if not r.success]
        }
    
    def create_parameter_grid(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create a grid of parameter combinations from parameter space.
        
        Args:
            param_space: Dictionary where keys are parameter names and values are lists of values
            
        Returns:
            List of parameter dictionaries representing all combinations
        """
        import itertools
        
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_combinations.append(param_dict)
        
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        return param_combinations
    
    def create_random_parameter_sample(self,
                                     param_space: Dict[str, Any],
                                     n_samples: int,
                                     random_state: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create random sample of parameter combinations.
        
        Args:
            param_space: Dictionary where keys are parameter names and values are:
                        - Lists for categorical parameters
                        - Tuples (min, max) for continuous parameters
            n_samples: Number of random samples to generate
            random_state: Random seed for reproducibility
            
        Returns:
            List of parameter dictionaries
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        param_combinations = []
        
        for _ in range(n_samples):
            params = {}
            
            for param_name, param_config in param_space.items():
                if isinstance(param_config, list):
                    # Categorical parameter
                    params[param_name] = np.random.choice(param_config)
                elif isinstance(param_config, tuple) and len(param_config) == 2:
                    # Continuous parameter (min, max)
                    min_val, max_val = param_config
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer parameter
                        params[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        # Float parameter
                        params[param_name] = np.random.uniform(min_val, max_val)
                else:
                    raise ValueError(f"Invalid parameter configuration for {param_name}: {param_config}")
            
            param_combinations.append(params)
        
        logger.info(f"Generated {n_samples} random parameter combinations")
        return param_combinations
    
    def create_data_batches(self,
                           data: pd.DataFrame,
                           batch_method: str = 'temporal',
                           n_batches: int = 5,
                           overlap_ratio: float = 0.0) -> List[pd.DataFrame]:
        """
        Create data batches for robust evaluation.
        
        Args:
            data: Input DataFrame
            batch_method: Method for creating batches ('temporal', 'random', 'rolling')
            n_batches: Number of batches to create
            overlap_ratio: Ratio of overlap between consecutive batches (0-1)
            
        Returns:
            List of DataFrame batches
        """
        if batch_method == 'temporal':
            return self._create_temporal_batches(data, n_batches, overlap_ratio)
        elif batch_method == 'random':
            return self._create_random_batches(data, n_batches, overlap_ratio)
        elif batch_method == 'rolling':
            return self._create_rolling_batches(data, n_batches, overlap_ratio)
        else:
            raise ValueError(f"Unknown batch method: {batch_method}")
    
    def _create_temporal_batches(self,
                                data: pd.DataFrame,
                                n_batches: int,
                                overlap_ratio: float) -> List[pd.DataFrame]:
        """Create temporal batches (sequential time periods)."""
        batches = []
        total_length = len(data)
        
        if overlap_ratio == 0:
            # Non-overlapping batches
            batch_size = total_length // n_batches
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size if i < n_batches - 1 else total_length
                batches.append(data.iloc[start_idx:end_idx])
        else:
            # Overlapping batches
            step_size = int(total_length * (1 - overlap_ratio) / (n_batches - 1))
            batch_size = int(total_length / n_batches * (1 + overlap_ratio))
            
            for i in range(n_batches):
                start_idx = i * step_size
                end_idx = min(start_idx + batch_size, total_length)
                
                if end_idx > start_idx:
                    batches.append(data.iloc[start_idx:end_idx])
        
        logger.info(f"Created {len(batches)} temporal batches")
        return batches
    
    def _create_random_batches(self,
                              data: pd.DataFrame,
                              n_batches: int,
                              overlap_ratio: float) -> List[pd.DataFrame]:
        """Create random batches (randomly sampled data points)."""
        batches = []
        total_length = len(data)
        batch_size = int(total_length / n_batches * (1 + overlap_ratio))
        
        for i in range(n_batches):
            # Randomly sample indices
            sample_indices = np.random.choice(
                total_length, 
                size=min(batch_size, total_length), 
                replace=False
            )
            sample_indices = sorted(sample_indices)
            
            batches.append(data.iloc[sample_indices])
        
        logger.info(f"Created {len(batches)} random batches")
        return batches
    
    def _create_rolling_batches(self,
                               data: pd.DataFrame,
                               n_batches: int,
                               overlap_ratio: float) -> List[pd.DataFrame]:
        """Create rolling window batches."""
        batches = []
        total_length = len(data)
        window_size = int(total_length * 0.8)  # Use 80% of data per batch
        step_size = (total_length - window_size) // (n_batches - 1) if n_batches > 1 else 0
        
        for i in range(n_batches):
            start_idx = i * step_size
            end_idx = min(start_idx + window_size, total_length)
            
            if end_idx > start_idx:
                batches.append(data.iloc[start_idx:end_idx])
        
        logger.info(f"Created {len(batches)} rolling batches")
        return batches