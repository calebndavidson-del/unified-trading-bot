#!/usr/bin/env python3
"""
Hyperparameter Optimizer for TrendAnalyzer

Optimizes parameters for the TrendAnalyzer class including window sizes,
thresholds, and technical indicator parameters.
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path to import trading bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.market_trend import TrendAnalyzer, TechnicalIndicators
from optimization.base import BaseOptimizer, OptimizationConfig
from optimization.cache import OptimizationCache
import logging

logger = logging.getLogger(__name__)


class OptimizedTrendAnalyzer(TrendAnalyzer):
    """
    TrendAnalyzer with optimizable hyperparameters.
    
    This wrapper allows the TrendAnalyzer to accept hyperparameters
    and modify its behavior accordingly.
    """
    
    def __init__(self, **hyperparams):
        """
        Initialize with hyperparameters.
        
        Args:
            **hyperparams: Hyperparameters for optimization
        """
        super().__init__()
        self.hyperparams = hyperparams
        
        # Extract key parameters with defaults
        self.short_ma_window = hyperparams.get('short_ma_window', 10)
        self.medium_ma_window = hyperparams.get('medium_ma_window', 20)
        self.long_ma_window = hyperparams.get('long_ma_window', 50)
        self.rsi_window = hyperparams.get('rsi_window', 14)
        self.macd_fast = hyperparams.get('macd_fast', 12)
        self.macd_slow = hyperparams.get('macd_slow', 26)
        self.macd_signal = hyperparams.get('macd_signal', 9)
        self.bb_window = hyperparams.get('bb_window', 20)
        self.bb_std_dev = hyperparams.get('bb_std_dev', 2.0)
        self.atr_window = hyperparams.get('atr_window', 14)
        self.stoch_k_period = hyperparams.get('stoch_k_period', 14)
        self.stoch_d_period = hyperparams.get('stoch_d_period', 3)
        self.volatility_window = hyperparams.get('volatility_window', 20)
        self.support_resistance_window = hyperparams.get('support_resistance_window', 20)
    
    def identify_trend_direction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Override with parameterized windows."""
        result_df = data.copy()
        
        # Calculate moving averages with optimized windows
        result_df['sma_short'] = self.indicators.sma(data['Close'], self.short_ma_window)
        result_df['sma_medium'] = self.indicators.sma(data['Close'], self.medium_ma_window)
        result_df['sma_long'] = self.indicators.sma(data['Close'], self.long_ma_window)
        
        # Trend direction based on MA relationships
        result_df['trend_short'] = np.where(
            result_df['sma_short'] > result_df['sma_medium'], 1, -1
        )
        result_df['trend_medium'] = np.where(
            result_df['sma_medium'] > result_df['sma_long'], 1, -1
        )
        result_df['trend_long'] = np.where(
            result_df['sma_medium'] > result_df['sma_long'], 1, -1
        )
        
        # Overall trend strength
        result_df['trend_strength'] = (
            result_df['trend_short'] + result_df['trend_medium'] + result_df['trend_long']
        ) / 3
        
        return result_df
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Override with parameterized indicators."""
        result_df = data.copy()
        
        # RSI with optimized window
        result_df['rsi'] = self.indicators.rsi(data['Close'], self.rsi_window)
        
        # MACD with optimized parameters
        macd_data = self.indicators.macd(
            data['Close'], 
            fast=self.macd_fast, 
            slow=self.macd_slow, 
            signal=self.macd_signal
        )
        result_df['macd'] = macd_data['macd']
        result_df['macd_signal'] = macd_data['signal']
        result_df['macd_histogram'] = macd_data['histogram']
        
        # Stochastic with optimized parameters
        if all(col in data.columns for col in ['High', 'Low']):
            stoch_data = self.indicators.stochastic(
                data['High'], data['Low'], data['Close'],
                k_period=self.stoch_k_period,
                d_period=self.stoch_d_period
            )
            result_df['stoch_k'] = stoch_data['k']
            result_df['stoch_d'] = stoch_data['d']
        
        return result_df
    
    def calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Override with parameterized volatility indicators."""
        result_df = data.copy()
        
        # Bollinger Bands with optimized parameters
        bb_data = self.indicators.bollinger_bands(
            data['Close'], 
            window=self.bb_window, 
            std_dev=self.bb_std_dev
        )
        result_df['bb_upper'] = bb_data['upper']
        result_df['bb_middle'] = bb_data['middle']
        result_df['bb_lower'] = bb_data['lower']
        result_df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        result_df['bb_position'] = (data['Close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # ATR with optimized window
        if all(col in data.columns for col in ['High', 'Low']):
            result_df['atr'] = self.indicators.atr(
                data['High'], data['Low'], data['Close'], 
                window=self.atr_window
            )
        
        # Historical volatility with optimized window
        returns = data['Close'].pct_change()
        result_df['volatility'] = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
        
        return result_df
    
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = None) -> pd.DataFrame:
        """Override with parameterized support/resistance window."""
        window = window or self.support_resistance_window
        return super().calculate_support_resistance(data, window)


class TrendAnalyzerOptimizer(BaseOptimizer):
    """
    Hyperparameter optimizer for TrendAnalyzer.
    
    Optimizes parameters such as:
    - Moving average windows
    - Technical indicator parameters
    - Volatility calculation windows
    - Support/resistance detection windows
    """
    
    def __init__(self, 
                 config: OptimizationConfig = None,
                 cache_dir: str = ".optimization_cache"):
        """Initialize the TrendAnalyzer optimizer."""
        super().__init__(
            model_class=OptimizedTrendAnalyzer,
            config=config,
            cache_dir=cache_dir
        )
        
        # Initialize cache
        self.persistent_cache = OptimizationCache(cache_dir)
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for TrendAnalyzer.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {
            # Moving average windows
            'short_ma_window': trial.suggest_int('short_ma_window', 5, 20),
            'medium_ma_window': trial.suggest_int('medium_ma_window', 15, 40),
            'long_ma_window': trial.suggest_int('long_ma_window', 30, 100),
            
            # RSI parameters
            'rsi_window': trial.suggest_int('rsi_window', 10, 25),
            
            # MACD parameters
            'macd_fast': trial.suggest_int('macd_fast', 8, 18),
            'macd_slow': trial.suggest_int('macd_slow', 20, 35),
            'macd_signal': trial.suggest_int('macd_signal', 6, 15),
            
            # Bollinger Bands parameters
            'bb_window': trial.suggest_int('bb_window', 15, 30),
            'bb_std_dev': trial.suggest_float('bb_std_dev', 1.5, 3.0),
            
            # ATR window
            'atr_window': trial.suggest_int('atr_window', 10, 25),
            
            # Stochastic parameters
            'stoch_k_period': trial.suggest_int('stoch_k_period', 10, 20),
            'stoch_d_period': trial.suggest_int('stoch_d_period', 2, 5),
            
            # Volatility window
            'volatility_window': trial.suggest_int('volatility_window', 15, 30),
            
            # Support/resistance window
            'support_resistance_window': trial.suggest_int('support_resistance_window', 15, 30),
        }
        
        # Ensure logical ordering of MA windows and MACD fast < slow
        if not (params['short_ma_window'] < params['medium_ma_window'] < params['long_ma_window']):
            raise optuna.TrialPruned(f"Invalid MA window ordering: short={params['short_ma_window']}, medium={params['medium_ma_window']}, long={params['long_ma_window']}")
        
        if not (params['macd_fast'] < params['macd_slow']):
            raise optuna.TrialPruned(f"Invalid MACD ordering: fast={params['macd_fast']}, slow={params['macd_slow']}")
        
        return params
    
    def create_model_instance(self, params: Dict[str, Any]) -> OptimizedTrendAnalyzer:
        """
        Create TrendAnalyzer instance with given parameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            OptimizedTrendAnalyzer instance
        """
        return OptimizedTrendAnalyzer(**params)
    
    def evaluate_model(self, model: OptimizedTrendAnalyzer, data: pd.DataFrame) -> float:
        """
        Evaluate TrendAnalyzer model and return a composite score.
        
        The evaluation focuses on:
        1. Feature stability (low variance in signals)
        2. Trend prediction accuracy
        3. Information content of indicators
        
        Args:
            model: OptimizedTrendAnalyzer instance
            data: Data to evaluate on
            
        Returns:
            Composite score (higher is better)
        """
        try:
            # Generate features
            enhanced_data = self._generate_features(model, data)
            
            # Calculate multiple scoring metrics
            scores = {}
            
            # 1. Feature stability score
            scores['stability'] = self._calculate_stability_score(enhanced_data)
            
            # 2. Trend prediction score
            scores['trend_prediction'] = self._calculate_trend_prediction_score(enhanced_data)
            
            # 3. Information content score
            scores['information_content'] = self._calculate_information_score(enhanced_data)
            
            # 4. Signal quality score
            scores['signal_quality'] = self._calculate_signal_quality_score(enhanced_data)
            
            # Combine scores with weights
            weights = {
                'stability': 0.25,
                'trend_prediction': 0.35,
                'information_content': 0.25,
                'signal_quality': 0.15
            }
            
            composite_score = sum(weights[key] * scores[key] for key in scores)
            
            # Add regularization penalty for extreme parameters
            penalty = self._calculate_parameter_penalty(model.hyperparams)
            composite_score -= penalty
            
            logger.debug(f"Evaluation scores: {scores}, Penalty: {penalty:.4f}, Final: {composite_score:.4f}")
            
            return composite_score
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            return -1.0  # Return poor score for failed evaluations
    
    def _generate_features(self, model: OptimizedTrendAnalyzer, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features using the model."""
        enhanced_data = data.copy()
        
        # Apply all TrendAnalyzer methods
        enhanced_data = model.identify_trend_direction(enhanced_data)
        enhanced_data = model.calculate_momentum_indicators(enhanced_data)
        enhanced_data = model.calculate_volatility_indicators(enhanced_data)
        enhanced_data = model.calculate_support_resistance(enhanced_data)
        
        return enhanced_data
    
    def _calculate_stability_score(self, data: pd.DataFrame) -> float:
        """Calculate feature stability score (lower variance is better)."""
        try:
            stability_features = ['trend_strength', 'rsi', 'bb_position']
            stability_scores = []
            
            for feature in stability_features:
                if feature in data.columns:
                    # Calculate normalized variance (coefficient of variation)
                    feature_data = data[feature].dropna()
                    if len(feature_data) > 10 and feature_data.std() > 0:
                        cv = feature_data.std() / (abs(feature_data.mean()) + 1e-8)
                        stability_score = 1.0 / (1.0 + cv)  # Higher score for lower variance
                        stability_scores.append(stability_score)
            
            return np.mean(stability_scores) if stability_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_trend_prediction_score(self, data: pd.DataFrame) -> float:
        """Calculate how well trend indicators predict future price movements."""
        try:
            if 'trend_strength' not in data.columns:
                return 0.5
            
            # Calculate future returns
            returns = data['Close'].pct_change().shift(-1)  # Next period return
            trend_strength = data['trend_strength']
            
            # Remove NaN values
            valid_mask = ~(returns.isna() | trend_strength.isna())
            returns_clean = returns[valid_mask]
            trend_clean = trend_strength[valid_mask]
            
            if len(returns_clean) < 20:
                return 0.5
            
            # Calculate correlation between trend strength and future returns
            correlation = np.corrcoef(trend_clean, returns_clean)[0, 1]
            
            # Convert correlation to score (0-1 range)
            score = (abs(correlation) + 1) / 2  # Higher absolute correlation is better
            
            return score if not np.isnan(score) else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_information_score(self, data: pd.DataFrame) -> float:
        """Calculate information content of generated indicators."""
        try:
            info_features = ['rsi', 'macd', 'bb_position', 'atr']
            info_scores = []
            
            for feature in info_features:
                if feature in data.columns:
                    feature_data = data[feature].dropna()
                    if len(feature_data) > 10:
                        # Calculate entropy as a measure of information content
                        # Discretize the feature into bins
                        bins = min(10, len(feature_data) // 5)
                        hist, _ = np.histogram(feature_data, bins=bins)
                        
                        # Calculate normalized entropy
                        probs = hist / hist.sum()
                        probs = probs[probs > 0]  # Remove zero probabilities
                        entropy = -np.sum(probs * np.log2(probs))
                        max_entropy = np.log2(len(probs))
                        
                        if max_entropy > 0:
                            normalized_entropy = entropy / max_entropy
                            info_scores.append(normalized_entropy)
            
            return np.mean(info_scores) if info_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_signal_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate quality of trading signals."""
        try:
            signal_features = ['trend_strength']
            signal_scores = []
            
            for feature in signal_features:
                if feature in data.columns:
                    feature_data = data[feature].dropna()
                    if len(feature_data) > 10:
                        # Calculate signal-to-noise ratio
                        signal_range = feature_data.max() - feature_data.min()
                        signal_std = feature_data.std()
                        
                        if signal_std > 0:
                            snr = signal_range / signal_std
                            score = min(1.0, snr / 10.0)  # Normalize to 0-1
                            signal_scores.append(score)
            
            return np.mean(signal_scores) if signal_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_parameter_penalty(self, params: Dict[str, Any]) -> float:
        """Calculate penalty for extreme parameter values."""
        try:
            penalty = 0.0
            
            # Penalize extreme window sizes
            if params.get('long_ma_window', 50) > 80:
                penalty += 0.1
            
            if params.get('bb_std_dev', 2.0) > 2.8:
                penalty += 0.05
            
            # Penalize very small windows that might cause overfitting
            if params.get('short_ma_window', 10) < 7:
                penalty += 0.05
            
            return penalty
            
        except Exception:
            return 0.0
    
    def _objective_with_cache(self, trial: optuna.Trial, data: pd.DataFrame) -> float:
        """Objective function with persistent caching."""
        params = self.define_search_space(trial)
        
        # Check persistent cache
        cached_score = self.persistent_cache.get(
            params, data, self.model_class.__name__
        )
        
        if cached_score is not None:
            logger.debug(f"Cache hit for trial {trial.number}")
            return cached_score
        
        # Evaluate model
        try:
            model = self.create_model_instance(params)
            score = self.evaluate_model(model, data)
            
            # Store in persistent cache
            self.persistent_cache.put(
                params, score, data, self.model_class.__name__
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            return -1.0
    
    def optimize(self, 
                 data: pd.DataFrame,
                 validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Override to use persistent caching."""
        # Temporarily replace objective function
        original_objective = self._objective
        self._objective = self._objective_with_cache
        
        try:
            result = super().optimize(data, validation_data)
            return result
        finally:
            # Restore original objective
            self._objective = original_objective