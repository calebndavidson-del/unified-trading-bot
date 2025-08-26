#!/usr/bin/env python3
"""
Hyperparameter Optimization Framework for Trading Bot Models

This module provides a comprehensive framework for optimizing hyperparameters
of trading models using Optuna's Bayesian optimization.
"""

from .base import BaseOptimizer, OptimizationConfig
from .cache import OptimizationCache
from .batch import BatchProcessor
from .trend_analyzer_optimizer import TrendAnalyzerOptimizer
from .trend_signal_generator_optimizer import TrendSignalGeneratorOptimizer
from .earnings_feature_engineer_optimizer import EarningsFeatureEngineerOptimizer

__all__ = [
    'BaseOptimizer',
    'OptimizationConfig',
    'OptimizationCache',
    'BatchProcessor',
    'TrendAnalyzerOptimizer',
    'TrendSignalGeneratorOptimizer',
    'EarningsFeatureEngineerOptimizer'
]

__version__ = '1.0.0'