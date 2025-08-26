#!/usr/bin/env python3
"""
Base Hyperparameter Optimization Framework

Provides the foundation for optimizing hyperparameters using Optuna
with caching and batching support.
"""

import optuna
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
import hashlib
import json
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    n_jobs: int = 1
    study_name: Optional[str] = None
    storage: Optional[str] = None  # SQLite database path
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    sampler: str = 'TPE'  # 'TPE', 'Random', 'CmaEs'
    pruner: Optional[str] = 'MedianPruner'  # 'MedianPruner', 'SuccessiveHalvingPruner'


class BaseOptimizer(ABC):
    """
    Base class for hyperparameter optimization of trading models.
    
    This class provides a framework for:
    1. Defining parameter search spaces
    2. Implementing objective functions
    3. Caching results for efficiency
    4. Batch processing capabilities
    """
    
    def __init__(self, 
                 model_class: type,
                 config: OptimizationConfig = None,
                 cache_dir: str = ".optimization_cache"):
        """
        Initialize the optimizer.
        
        Args:
            model_class: The class to optimize (e.g., TrendAnalyzer)
            config: Optimization configuration
            cache_dir: Directory for caching results
        """
        self.model_class = model_class
        self.config = config or OptimizationConfig()
        self.cache_dir = cache_dir
        self.cache = {}  # In-memory cache
        self.study = None
        
        # Initialize cache directory
        import os
        os.makedirs(cache_dir, exist_ok=True)
    
    @abstractmethod
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters sampled from the search space
        """
        pass
    
    @abstractmethod
    def create_model_instance(self, params: Dict[str, Any]) -> Any:
        """
        Create an instance of the model with given parameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            Model instance configured with the parameters
        """
        pass
    
    @abstractmethod
    def evaluate_model(self, model: Any, data: pd.DataFrame) -> float:
        """
        Evaluate the model and return a score.
        
        Args:
            model: Model instance to evaluate
            data: Data to evaluate on
            
        Returns:
            Score (higher is better for maximize, lower for minimize)
        """
        pass
    
    def _generate_cache_key(self, params: Dict[str, Any], data_hash: str) -> str:
        """Generate a unique cache key for parameters and data."""
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{params_str}_{data_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _hash_data(self, data: pd.DataFrame) -> str:
        """Generate a hash for the input data."""
        return hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()
    
    def _objective(self, trial: optuna.Trial, data: pd.DataFrame) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            data: Training data
            
        Returns:
            Score to optimize
        """
        # Get hyperparameters from search space
        params = self.define_search_space(trial)
        
        # Check cache first
        data_hash = self._hash_data(data)
        cache_key = self._generate_cache_key(params, data_hash)
        
        if cache_key in self.cache:
            logger.debug(f"Cache hit for trial {trial.number}")
            return self.cache[cache_key]
        
        try:
            # Create model instance
            model = self.create_model_instance(params)
            
            # Evaluate model
            score = self.evaluate_model(model, data)
            
            # Cache result
            self.cache[cache_key] = score
            
            # Log progress
            logger.info(f"Trial {trial.number}: Score={score:.4f}, Params={params}")
            
            return score
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            # Return worst possible score
            return -np.inf if self.config.direction == 'maximize' else np.inf
    
    def optimize(self, 
                 data: pd.DataFrame,
                 validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            data: Training data
            validation_data: Optional validation data for evaluation
            
        Returns:
            Dictionary containing optimization results
        """
        start_time = time.time()
        
        # Use validation data if provided, otherwise use training data
        eval_data = validation_data if validation_data is not None else data
        
        # Create sampler
        if self.config.sampler == 'TPE':
            sampler = optuna.samplers.TPESampler()
        elif self.config.sampler == 'Random':
            sampler = optuna.samplers.RandomSampler()
        elif self.config.sampler == 'CmaEs':
            sampler = optuna.samplers.CmaEsSampler()
        else:
            sampler = optuna.samplers.TPESampler()
        
        # Create pruner
        pruner = None
        if self.config.pruner == 'MedianPruner':
            pruner = optuna.pruners.MedianPruner()
        elif self.config.pruner == 'SuccessiveHalvingPruner':
            pruner = optuna.pruners.SuccessiveHalvingPruner()
        
        # Create study
        study_name = self.config.study_name or f"{self.model_class.__name__}_optimization"
        
        self.study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=self.config.storage,
            load_if_exists=True
        )
        
        # Run optimization
        logger.info(f"Starting optimization for {self.model_class.__name__}")
        logger.info(f"Configuration: {self.config}")
        
        self.study.optimize(
            lambda trial: self._objective(trial, eval_data),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        # Extract results
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        optimization_time = time.time() - start_time
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': self.study,
            'optimization_time': optimization_time,
            'n_trials': len(self.study.trials),
            'best_model': self.create_model_instance(best_params)
        }
    
    def optimize_batch(self, 
                      data_batches: List[pd.DataFrame],
                      validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run optimization on multiple data batches for robustness.
        
        Args:
            data_batches: List of training data batches
            validation_data: Optional validation data
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Starting batch optimization with {len(data_batches)} batches")
        
        batch_results = []
        
        for i, batch_data in enumerate(data_batches):
            logger.info(f"Optimizing batch {i+1}/{len(data_batches)}")
            
            # Create a new study for this batch
            batch_config = OptimizationConfig(
                n_trials=self.config.n_trials // len(data_batches),
                timeout=self.config.timeout,
                n_jobs=self.config.n_jobs,
                study_name=f"{self.config.study_name}_batch_{i}" if self.config.study_name else None,
                direction=self.config.direction,
                sampler=self.config.sampler,
                pruner=self.config.pruner
            )
            
            # Temporarily replace config
            original_config = self.config
            self.config = batch_config
            
            try:
                batch_result = self.optimize(batch_data, validation_data)
                batch_results.append(batch_result)
            finally:
                # Restore original config
                self.config = original_config
        
        # Aggregate results across batches
        all_params = [result['best_params'] for result in batch_results]
        all_scores = [result['best_score'] for result in batch_results]
        
        # Find best overall result
        if self.config.direction == 'maximize':
            best_idx = np.argmax(all_scores)
        else:
            best_idx = np.argmin(all_scores)
        
        best_batch_result = batch_results[best_idx]
        
        logger.info(f"Batch optimization completed")
        logger.info(f"Best batch: {best_idx + 1}, Score: {all_scores[best_idx]:.4f}")
        
        return {
            'best_params': best_batch_result['best_params'],
            'best_score': best_batch_result['best_score'],
            'best_model': best_batch_result['best_model'],
            'batch_results': batch_results,
            'all_scores': all_scores,
            'best_batch_idx': best_idx
        }
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as a DataFrame."""
        if self.study is None:
            raise ValueError("No optimization study found. Run optimize() first.")
        
        trials_df = self.study.trials_dataframe()
        return trials_df
    
    def plot_optimization_history(self):
        """Plot optimization history (requires plotly)."""
        if self.study is None:
            raise ValueError("No optimization study found. Run optimize() first.")
        
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Get trials data
            trials_df = self.get_optimization_history()
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Optimization History', 'Parameter Importance'),
                vertical_spacing=0.1
            )
            
            # Plot objective value over trials
            fig.add_trace(
                go.Scatter(
                    x=trials_df.index,
                    y=trials_df['value'],
                    mode='lines+markers',
                    name='Objective Value',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Plot best value so far
            best_values = []
            best_so_far = -np.inf if self.config.direction == 'maximize' else np.inf
            
            for value in trials_df['value']:
                if self.config.direction == 'maximize':
                    best_so_far = max(best_so_far, value)
                else:
                    best_so_far = min(best_so_far, value)
                best_values.append(best_so_far)
            
            fig.add_trace(
                go.Scatter(
                    x=trials_df.index,
                    y=best_values,
                    mode='lines',
                    name='Best Value So Far',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"Hyperparameter Optimization - {self.model_class.__name__}",
                height=600,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Trial", row=1, col=1)
            fig.update_yaxes(title_text="Objective Value", row=1, col=1)
            
            return fig
            
        except ImportError:
            logger.warning("Plotly not available. Cannot generate plots.")
            return None