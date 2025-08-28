#!/usr/bin/env python3
"""
Automated Model and Strategy Selection for Optimization Framework
"""

from typing import Dict, List, Type, Any, Optional
from dataclasses import dataclass
import optuna
from features.backtesting import (
    TradingStrategy, TechnicalAnalysisStrategy, MeanReversionStrategy, 
    MomentumStrategy, PatternRecognitionStrategy
)
from features.unified_strategy import UnifiedTradingStrategy
from features.ensemble_model import EnsembleSignalCombiner


@dataclass
class ModelConfig:
    """Configuration for a model type"""
    name: str
    model_class: Type
    param_ranges: Dict[str, Any]
    description: str


@dataclass
class StrategyConfig:
    """Configuration for a strategy type"""
    name: str
    strategy_class: Type[TradingStrategy]
    param_ranges: Dict[str, Any]
    description: str


class AutoModelSelector:
    """Automatically selects and configures models for optimization"""
    
    def __init__(self):
        self.available_models = self._define_available_models()
        self.available_strategies = self._define_available_strategies()
    
    def _define_available_models(self) -> Dict[str, ModelConfig]:
        """Define available model types with their parameter ranges"""
        return {
            'lstm_neural_network': ModelConfig(
                name="LSTM Neural Network",
                model_class=EnsembleSignalCombiner,  # Unified uses ensemble
                param_ranges={
                    'ensemble_method': ['voting', 'stacking', 'blending'],
                    'regularization_strength': [0.001, 0.1],
                    'cv_folds': [3, 10],
                    'overfitting_threshold': [0.1, 0.3],
                },
                description="LSTM-based neural network with ensemble capabilities"
            ),
            'ensemble_ml': ModelConfig(
                name="Ensemble ML",
                model_class=EnsembleSignalCombiner,
                param_ranges={
                    'ensemble_method': ['voting', 'stacking', 'blending'],
                    'regularization_strength': [0.01, 0.05],
                    'cv_folds': [5, 8],
                    'overfitting_threshold': [0.15, 0.25],
                },
                description="Ensemble of multiple ML models"
            ),
            'random_forest': ModelConfig(
                name="Random Forest",
                model_class=EnsembleSignalCombiner,
                param_ranges={
                    'ensemble_method': ['voting'],
                    'regularization_strength': [0.005, 0.02],
                    'cv_folds': [3, 5],
                    'overfitting_threshold': [0.2, 0.4],
                },
                description="Random Forest-based ensemble"
            )
        }
    
    def _define_available_strategies(self) -> Dict[str, StrategyConfig]:
        """Define available strategy types with their parameter ranges"""
        return {
            'technical_analysis': StrategyConfig(
                name="Technical Analysis",
                strategy_class=TechnicalAnalysisStrategy,
                param_ranges={
                    'confidence_threshold': [0.6, 0.85],
                    'position_size_pct': [0.05, 0.25],
                    'stop_loss_pct': [0.02, 0.08],
                    'take_profit_pct': [0.05, 0.15],
                },
                description="Technical indicators based strategy"
            ),
            'mean_reversion': StrategyConfig(
                name="Mean Reversion",
                strategy_class=MeanReversionStrategy,
                param_ranges={
                    'confidence_threshold': [0.65, 0.9],
                    'position_size_pct': [0.1, 0.3],
                    'stop_loss_pct': [0.03, 0.1],
                    'take_profit_pct': [0.04, 0.12],
                },
                description="Mean reversion based strategy"
            ),
            'momentum': StrategyConfig(
                name="Momentum",
                strategy_class=MomentumStrategy,
                param_ranges={
                    'confidence_threshold': [0.7, 0.85],
                    'position_size_pct': [0.08, 0.2],
                    'stop_loss_pct': [0.025, 0.075],
                    'take_profit_pct': [0.06, 0.18],
                },
                description="Momentum based strategy"
            ),
            'pattern_recognition': StrategyConfig(
                name="Pattern Recognition",
                strategy_class=PatternRecognitionStrategy,
                param_ranges={
                    'confidence_threshold': [0.65, 0.8],
                    'position_size_pct': [0.1, 0.25],
                    'stop_loss_pct': [0.02, 0.06],
                    'take_profit_pct': [0.05, 0.14],
                },
                description="Candlestick pattern recognition strategy"
            ),
            'unified_strategy': StrategyConfig(
                name="Unified Strategy",
                strategy_class=UnifiedTradingStrategy,
                param_ranges={
                    'confidence_threshold': [0.6, 0.85],
                    'position_size_pct': [0.1, 0.3],
                    'stop_loss_pct': [0.02, 0.08],
                    'take_profit_pct': [0.05, 0.15],
                },
                description="Unified strategy combining all data sources"
            )
        }
    
    def suggest_model_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest a model configuration for optimization trial"""
        # Select model type
        model_key = trial.suggest_categorical('model_type', list(self.available_models.keys()))
        model_config = self.available_models[model_key]
        
        # Generate parameters for this model
        config = {'model_name': model_config.name}
        
        for param, param_range in model_config.param_ranges.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                if isinstance(param_range[0], (int, float)):
                    if isinstance(param_range[0], int):
                        config[param] = trial.suggest_int(f'model_{param}', param_range[0], param_range[1])
                    else:
                        config[param] = trial.suggest_float(f'model_{param}', param_range[0], param_range[1])
                else:
                    config[param] = trial.suggest_categorical(f'model_{param}', param_range)
            else:
                config[param] = trial.suggest_categorical(f'model_{param}', param_range)
        
        return config
    
    def suggest_strategy_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest a strategy configuration for optimization trial"""
        # Select strategy type
        strategy_key = trial.suggest_categorical('strategy_type', list(self.available_strategies.keys()))
        strategy_config = self.available_strategies[strategy_key]
        
        # Generate parameters for this strategy
        config = {
            'strategy_name': strategy_config.name,
            'strategy_class': strategy_config.strategy_class
        }
        
        for param, param_range in strategy_config.param_ranges.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                if isinstance(param_range[0], (int, float)):
                    if isinstance(param_range[0], int):
                        config[param] = trial.suggest_int(f'strategy_{param}', param_range[0], param_range[1])
                    else:
                        config[param] = trial.suggest_float(f'strategy_{param}', param_range[0], param_range[1])
                else:
                    config[param] = trial.suggest_categorical(f'strategy_{param}', param_range)
            else:
                config[param] = trial.suggest_categorical(f'strategy_{param}', param_range)
        
        return config
    
    def suggest_backtest_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest general backtesting configuration"""
        return {
            'backtest_period': trial.suggest_categorical('backtest_period', ['6mo', '1y', '2y']),
            'lookback_window': trial.suggest_int('lookback_window', 30, 252),
            'rebalance_frequency': trial.suggest_categorical('rebalance_frequency', ['daily', 'weekly', 'monthly']),
            'max_positions': trial.suggest_int('max_positions', 3, 15),
            'risk_free_rate': trial.suggest_float('risk_free_rate', 0.02, 0.05),
        }
    
    def get_model_info(self, model_name: str) -> Optional[ModelConfig]:
        """Get information about a specific model"""
        for config in self.available_models.values():
            if config.name == model_name:
                return config
        return None
    
    def get_strategy_info(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get information about a specific strategy"""
        for config in self.available_strategies.values():
            if config.name == strategy_name:
                return config
        return None