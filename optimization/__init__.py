#!/usr/bin/env python3
"""
Hyperparameter Optimization Framework for Trading Bot Models

This module provides a comprehensive framework for optimizing hyperparameters
of trading models and data source APIs using Optuna's Bayesian optimization.
"""

# Core optimization framework
from .base import BaseOptimizer, OptimizationConfig
from .cache import OptimizationCache
from .batch import BatchProcessor

# Model optimizers
from .trend_analyzer_optimizer import TrendAnalyzerOptimizer
from .trend_signal_generator_optimizer import TrendSignalGeneratorOptimizer
from .earnings_feature_engineer_optimizer import EarningsFeatureEngineerOptimizer

# API optimizers
from .api_base import BaseAPIOptimizer, APIOptimizationResult
from .yahoo_finance_optimizer import YahooFinanceAPIOptimizer
from .iex_cloud_optimizer import IEXCloudAPIOptimizer
from .alpha_vantage_optimizer import AlphaVantageAPIOptimizer
from .quandl_optimizer import QuandlAPIOptimizer
from .finnhub_optimizer import FinnhubAPIOptimizer
from .binance_optimizer import BinanceAPIOptimizer

__all__ = [
    # Core framework
    'BaseOptimizer',
    'OptimizationConfig',
    'OptimizationCache',
    'BatchProcessor',
    
    # Model optimizers
    'TrendAnalyzerOptimizer',
    'TrendSignalGeneratorOptimizer',
    'EarningsFeatureEngineerOptimizer',
    
    # API optimization framework
    'BaseAPIOptimizer',
    'APIOptimizationResult',
    
    # API optimizers
    'YahooFinanceAPIOptimizer',
    'IEXCloudAPIOptimizer',
    'AlphaVantageAPIOptimizer',
    'QuandlAPIOptimizer',
    'FinnhubAPIOptimizer',
    'BinanceAPIOptimizer'
]

__version__ = '2.0.0'  # Updated for API optimization support