#!/usr/bin/env python3
"""
Hyperparameter Optimizer for TrendSignalGenerator

Optimizes parameters for the TrendSignalGenerator class including signal thresholds,
combination weights, and signal detection parameters.
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path to import trading bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.market_trend import TrendSignalGenerator, TrendAnalyzer
from optimization.base import BaseOptimizer, OptimizationConfig
from optimization.cache import OptimizationCache
import logging

logger = logging.getLogger(__name__)


class OptimizedTrendSignalGenerator(TrendSignalGenerator):
    """
    TrendSignalGenerator with optimizable hyperparameters.
    
    This wrapper allows the TrendSignalGenerator to accept hyperparameters
    and modify signal generation behavior accordingly.
    """
    
    def __init__(self, **hyperparams):
        """
        Initialize with hyperparameters.
        
        Args:
            **hyperparams: Hyperparameters for optimization
        """
        super().__init__()
        self.hyperparams = hyperparams
        
        # Signal threshold parameters
        self.rsi_oversold_threshold = hyperparams.get('rsi_oversold_threshold', 30)
        self.rsi_overbought_threshold = hyperparams.get('rsi_overbought_threshold', 70)
        self.stoch_oversold_threshold = hyperparams.get('stoch_oversold_threshold', 20)
        self.stoch_overbought_threshold = hyperparams.get('stoch_overbought_threshold', 80)
        self.bb_oversold_threshold = hyperparams.get('bb_oversold_threshold', 0.1)
        self.bb_overbought_threshold = hyperparams.get('bb_overbought_threshold', 0.9)
        self.bb_mean_reversion_low = hyperparams.get('bb_mean_reversion_low', 0.2)
        self.bb_mean_reversion_high = hyperparams.get('bb_mean_reversion_high', 0.8)
        self.trend_strength_threshold = hyperparams.get('trend_strength_threshold', 0.6)
        
        # Moving average parameters for trend following
        self.ma_short_window = hyperparams.get('ma_short_window', 10)
        self.ma_long_window = hyperparams.get('ma_long_window', 30)
        
        # Signal combination weights
        self.momentum_weight = hyperparams.get('momentum_weight', 0.4)
        self.trend_weight = hyperparams.get('trend_weight', 0.4)
        self.mean_reversion_weight = hyperparams.get('mean_reversion_weight', 0.2)
        
        # Divergence detection parameters
        self.divergence_window = hyperparams.get('divergence_window', 14)
    
    def generate_momentum_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Override with parameterized thresholds."""
        result_df = data.copy()
        
        # RSI signals with optimized thresholds
        if 'rsi' in result_df.columns:
            result_df['rsi_oversold'] = (result_df['rsi'] < self.rsi_oversold_threshold).astype(int)
            result_df['rsi_overbought'] = (result_df['rsi'] > self.rsi_overbought_threshold).astype(int)
            result_df['rsi_bullish_divergence'] = self._detect_rsi_divergence(result_df, 'bullish', self.divergence_window)
            result_df['rsi_bearish_divergence'] = self._detect_rsi_divergence(result_df, 'bearish', self.divergence_window)
        
        # MACD signals (unchanged from base class)
        if all(col in result_df.columns for col in ['macd', 'macd_signal']):
            result_df['macd_bullish_cross'] = (
                (result_df['macd'] > result_df['macd_signal']) & 
                (result_df['macd'].shift(1) <= result_df['macd_signal'].shift(1))
            ).astype(int)
            
            result_df['macd_bearish_cross'] = (
                (result_df['macd'] < result_df['macd_signal']) & 
                (result_df['macd'].shift(1) >= result_df['macd_signal'].shift(1))
            ).astype(int)
        
        # Stochastic signals with optimized thresholds
        if all(col in result_df.columns for col in ['stoch_k', 'stoch_d']):
            result_df['stoch_oversold'] = (
                (result_df['stoch_k'] < self.stoch_oversold_threshold) & 
                (result_df['stoch_d'] < self.stoch_oversold_threshold)
            ).astype(int)
            result_df['stoch_overbought'] = (
                (result_df['stoch_k'] > self.stoch_overbought_threshold) & 
                (result_df['stoch_d'] > self.stoch_overbought_threshold)
            ).astype(int)
        
        return result_df
    
    def generate_trend_following_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Override with parameterized trend following."""
        from features.market_trend import TechnicalIndicators
        
        result_df = data.copy()
        
        # Moving average crossovers with optimized windows
        sma_short = TechnicalIndicators.sma(data['Close'], self.ma_short_window)
        sma_long = TechnicalIndicators.sma(data['Close'], self.ma_long_window)
        
        result_df['ma_bullish_cross'] = (
            (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
        ).astype(int)
        
        result_df['ma_bearish_cross'] = (
            (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))
        ).astype(int)
        
        # Breakout signals (unchanged)
        if 'resistance_breakout' in result_df.columns:
            result_df['breakout_signal'] = result_df['resistance_breakout']
        
        if 'support_breakdown' in result_df.columns:
            result_df['breakdown_signal'] = result_df['support_breakdown']
        
        # Trend strength signals with optimized threshold
        if 'trend_strength' in result_df.columns:
            result_df['strong_uptrend'] = (result_df['trend_strength'] > self.trend_strength_threshold).astype(int)
            result_df['strong_downtrend'] = (result_df['trend_strength'] < -self.trend_strength_threshold).astype(int)
        
        return result_df
    
    def generate_mean_reversion_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Override with parameterized mean reversion thresholds."""
        result_df = data.copy()
        
        # Bollinger Band signals with optimized thresholds
        if all(col in result_df.columns for col in ['bb_upper', 'bb_lower', 'bb_position']):
            result_df['bb_oversold'] = (result_df['bb_position'] < self.bb_oversold_threshold).astype(int)
            result_df['bb_overbought'] = (result_df['bb_position'] > self.bb_overbought_threshold).astype(int)
            
            # Mean reversion with optimized thresholds
            result_df['bb_mean_reversion_long'] = (
                (result_df['bb_position'] < self.bb_mean_reversion_low) & 
                (result_df['bb_position'].shift(1) < result_df['bb_position'])
            ).astype(int)
            
            result_df['bb_mean_reversion_short'] = (
                (result_df['bb_position'] > self.bb_mean_reversion_high) & 
                (result_df['bb_position'].shift(1) > result_df['bb_position'])
            ).astype(int)
        
        return result_df
    
    def generate_composite_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Override with weighted signal combination."""
        result_df = data.copy()
        
        # Generate all signal types first
        result_df = self.generate_momentum_signals(result_df)
        result_df = self.generate_trend_following_signals(result_df)
        result_df = self.generate_mean_reversion_signals(result_df)
        
        # Collect signal components with weights
        bullish_signals = []
        bearish_signals = []
        
        # Momentum signals
        momentum_bullish = []
        momentum_bearish = []
        
        if 'rsi_oversold' in result_df.columns:
            momentum_bullish.append(result_df['rsi_oversold'])
        if 'macd_bullish_cross' in result_df.columns:
            momentum_bullish.append(result_df['macd_bullish_cross'])
        if 'stoch_oversold' in result_df.columns:
            momentum_bullish.append(result_df['stoch_oversold'])
        
        if 'rsi_overbought' in result_df.columns:
            momentum_bearish.append(result_df['rsi_overbought'])
        if 'macd_bearish_cross' in result_df.columns:
            momentum_bearish.append(result_df['macd_bearish_cross'])
        if 'stoch_overbought' in result_df.columns:
            momentum_bearish.append(result_df['stoch_overbought'])
        
        # Trend following signals
        trend_bullish = []
        trend_bearish = []
        
        if 'ma_bullish_cross' in result_df.columns:
            trend_bullish.append(result_df['ma_bullish_cross'])
        if 'strong_uptrend' in result_df.columns:
            trend_bullish.append(result_df['strong_uptrend'])
        
        if 'ma_bearish_cross' in result_df.columns:
            trend_bearish.append(result_df['ma_bearish_cross'])
        if 'strong_downtrend' in result_df.columns:
            trend_bearish.append(result_df['strong_downtrend'])
        
        # Mean reversion signals
        reversion_bullish = []
        reversion_bearish = []
        
        if 'bb_oversold' in result_df.columns:
            reversion_bullish.append(result_df['bb_oversold'])
        if 'bb_mean_reversion_long' in result_df.columns:
            reversion_bullish.append(result_df['bb_mean_reversion_long'])
        
        if 'bb_overbought' in result_df.columns:
            reversion_bearish.append(result_df['bb_overbought'])
        if 'bb_mean_reversion_short' in result_df.columns:
            reversion_bearish.append(result_df['bb_mean_reversion_short'])
        
        # Calculate weighted signal strengths
        momentum_bullish_strength = np.mean(momentum_bullish, axis=0) if momentum_bullish else 0
        momentum_bearish_strength = np.mean(momentum_bearish, axis=0) if momentum_bearish else 0
        
        trend_bullish_strength = np.mean(trend_bullish, axis=0) if trend_bullish else 0
        trend_bearish_strength = np.mean(trend_bearish, axis=0) if trend_bearish else 0
        
        reversion_bullish_strength = np.mean(reversion_bullish, axis=0) if reversion_bullish else 0
        reversion_bearish_strength = np.mean(reversion_bearish, axis=0) if reversion_bearish else 0
        
        # Combine with weights
        result_df['bullish_signal_strength'] = (
            self.momentum_weight * momentum_bullish_strength +
            self.trend_weight * trend_bullish_strength +
            self.mean_reversion_weight * reversion_bullish_strength
        )
        
        result_df['bearish_signal_strength'] = (
            self.momentum_weight * momentum_bearish_strength +
            self.trend_weight * trend_bearish_strength +
            self.mean_reversion_weight * reversion_bearish_strength
        )
        
        # Final composite signal
        result_df['composite_signal'] = (
            result_df['bullish_signal_strength'] - result_df['bearish_signal_strength']
        )
        
        # Signal categories
        result_df['signal_category'] = pd.cut(
            result_df['composite_signal'],
            bins=[-1.1, -0.3, -0.1, 0.1, 0.3, 1.1],
            labels=['strong_sell', 'sell', 'hold', 'buy', 'strong_buy']
        )
        
        return result_df


class TrendSignalGeneratorOptimizer(BaseOptimizer):
    """
    Hyperparameter optimizer for TrendSignalGenerator.
    
    Optimizes parameters such as:
    - Signal detection thresholds (RSI, Stochastic, Bollinger Bands)
    - Moving average windows for trend following
    - Signal combination weights
    - Divergence detection parameters
    """
    
    def __init__(self, 
                 config: OptimizationConfig = None,
                 cache_dir: str = ".optimization_cache"):
        """Initialize the TrendSignalGenerator optimizer."""
        super().__init__(
            model_class=OptimizedTrendSignalGenerator,
            config=config,
            cache_dir=cache_dir
        )
        
        # Initialize cache
        self.persistent_cache = OptimizationCache(cache_dir)
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for TrendSignalGenerator.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {
            # RSI thresholds
            'rsi_oversold_threshold': trial.suggest_int('rsi_oversold_threshold', 20, 35),
            'rsi_overbought_threshold': trial.suggest_int('rsi_overbought_threshold', 65, 80),
            
            # Stochastic thresholds
            'stoch_oversold_threshold': trial.suggest_int('stoch_oversold_threshold', 15, 25),
            'stoch_overbought_threshold': trial.suggest_int('stoch_overbought_threshold', 75, 85),
            
            # Bollinger Band thresholds
            'bb_oversold_threshold': trial.suggest_float('bb_oversold_threshold', 0.05, 0.15),
            'bb_overbought_threshold': trial.suggest_float('bb_overbought_threshold', 0.85, 0.95),
            'bb_mean_reversion_low': trial.suggest_float('bb_mean_reversion_low', 0.15, 0.25),
            'bb_mean_reversion_high': trial.suggest_float('bb_mean_reversion_high', 0.75, 0.85),
            
            # Trend strength threshold
            'trend_strength_threshold': trial.suggest_float('trend_strength_threshold', 0.4, 0.8),
            
            # Moving average windows
            'ma_short_window': trial.suggest_int('ma_short_window', 5, 15),
            'ma_long_window': trial.suggest_int('ma_long_window', 20, 40),
            
            # Signal combination weights (must sum to 1)
            'momentum_weight': trial.suggest_float('momentum_weight', 0.2, 0.6),
            'trend_weight': trial.suggest_float('trend_weight', 0.2, 0.6),
            
            # Divergence detection window
            'divergence_window': trial.suggest_int('divergence_window', 10, 20),
        }
        
        # Ensure logical constraints
        if params['rsi_oversold_threshold'] >= params['rsi_overbought_threshold']:
            params['rsi_overbought_threshold'] = params['rsi_oversold_threshold'] + 20
        
        if params['stoch_oversold_threshold'] >= params['stoch_overbought_threshold']:
            params['stoch_overbought_threshold'] = params['stoch_oversold_threshold'] + 20
        
        if params['bb_oversold_threshold'] >= params['bb_overbought_threshold']:
            params['bb_overbought_threshold'] = params['bb_oversold_threshold'] + 0.5
        
        if params['ma_short_window'] >= params['ma_long_window']:
            params['ma_long_window'] = params['ma_short_window'] + 10
        
        # Normalize weights to sum to 1
        weight_sum = params['momentum_weight'] + params['trend_weight']
        params['mean_reversion_weight'] = 1.0 - weight_sum
        
        # Ensure all weights are positive
        if params['mean_reversion_weight'] < 0:
            # Re-normalize
            total = params['momentum_weight'] + params['trend_weight']
            params['momentum_weight'] = params['momentum_weight'] / total * 0.8
            params['trend_weight'] = params['trend_weight'] / total * 0.8
            params['mean_reversion_weight'] = 0.2
        
        return params
    
    def create_model_instance(self, params: Dict[str, Any]) -> OptimizedTrendSignalGenerator:
        """
        Create TrendSignalGenerator instance with given parameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            OptimizedTrendSignalGenerator instance
        """
        return OptimizedTrendSignalGenerator(**params)
    
    def evaluate_model(self, model: OptimizedTrendSignalGenerator, data: pd.DataFrame) -> float:
        """
        Evaluate TrendSignalGenerator model and return a composite score.
        
        The evaluation focuses on:
        1. Signal accuracy (correlation with future returns)
        2. Signal frequency (not too sparse or too frequent)
        3. Risk-adjusted returns (Sharpe ratio)
        4. Signal consistency
        
        Args:
            model: OptimizedTrendSignalGenerator instance
            data: Data to evaluate on
            
        Returns:
            Composite score (higher is better)
        """
        try:
            # Generate signals and features
            enhanced_data = self._generate_signals(model, data)
            
            # Calculate multiple scoring metrics
            scores = {}
            
            # 1. Signal accuracy score
            scores['accuracy'] = self._calculate_signal_accuracy_score(enhanced_data)
            
            # 2. Signal frequency score
            scores['frequency'] = self._calculate_signal_frequency_score(enhanced_data)
            
            # 3. Risk-adjusted returns score
            scores['risk_adjusted'] = self._calculate_risk_adjusted_score(enhanced_data)
            
            # 4. Signal consistency score
            scores['consistency'] = self._calculate_signal_consistency_score(enhanced_data)
            
            # 5. Profit potential score
            scores['profit'] = self._calculate_profit_potential_score(enhanced_data)
            
            # Combine scores with weights
            weights = {
                'accuracy': 0.3,
                'frequency': 0.15,
                'risk_adjusted': 0.25,
                'consistency': 0.15,
                'profit': 0.15
            }
            
            composite_score = sum(weights[key] * scores[key] for key in scores)
            
            # Add regularization penalty for extreme parameters
            penalty = self._calculate_parameter_penalty(model.hyperparams)
            composite_score -= penalty
            
            logger.debug(f"Signal evaluation scores: {scores}, Penalty: {penalty:.4f}, Final: {composite_score:.4f}")
            
            return composite_score
            
        except Exception as e:
            logger.error(f"Error in signal evaluation: {str(e)}")
            return -1.0  # Return poor score for failed evaluations
    
    def _generate_signals(self, model: OptimizedTrendSignalGenerator, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using the model."""
        # First need to add technical indicators for signal generation
        from optimization.trend_analyzer_optimizer import OptimizedTrendAnalyzer
        
        # Create a trend analyzer with default params to get indicators
        analyzer = OptimizedTrendAnalyzer()
        enhanced_data = data.copy()
        
        # Add necessary indicators
        enhanced_data = analyzer.identify_trend_direction(enhanced_data)
        enhanced_data = analyzer.calculate_momentum_indicators(enhanced_data)
        enhanced_data = analyzer.calculate_volatility_indicators(enhanced_data)
        enhanced_data = analyzer.calculate_support_resistance(enhanced_data)
        
        # Generate signals
        enhanced_data = model.generate_composite_signals(enhanced_data)
        
        return enhanced_data
    
    def _calculate_signal_accuracy_score(self, data: pd.DataFrame) -> float:
        """Calculate signal accuracy based on correlation with future returns."""
        try:
            if 'composite_signal' not in data.columns:
                return 0.5
            
            # Calculate future returns (1-3 periods ahead)
            returns_1 = data['Close'].pct_change().shift(-1)
            returns_3 = data['Close'].pct_change(3).shift(-3)
            
            signals = data['composite_signal']
            
            # Remove NaN values
            valid_mask = ~(returns_1.isna() | returns_3.isna() | signals.isna())
            
            if valid_mask.sum() < 20:
                return 0.5
            
            returns_1_clean = returns_1[valid_mask]
            returns_3_clean = returns_3[valid_mask]
            signals_clean = signals[valid_mask]
            
            # Calculate correlations
            corr_1 = np.corrcoef(signals_clean, returns_1_clean)[0, 1]
            corr_3 = np.corrcoef(signals_clean, returns_3_clean)[0, 1]
            
            # Average correlation (absolute value)
            avg_corr = (abs(corr_1) + abs(corr_3)) / 2
            
            # Convert to score
            score = min(1.0, avg_corr * 2)  # Scale correlation to 0-1 range
            
            return score if not np.isnan(score) else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_signal_frequency_score(self, data: pd.DataFrame) -> float:
        """Calculate signal frequency score (optimal signal density)."""
        try:
            signal_columns = ['bullish_signal_strength', 'bearish_signal_strength']
            frequency_scores = []
            
            for col in signal_columns:
                if col in data.columns:
                    # Count significant signals (> threshold)
                    significant_signals = (data[col] > 0.3).sum()
                    total_periods = len(data)
                    
                    signal_frequency = significant_signals / total_periods
                    
                    # Optimal frequency is around 5-15% (not too sparse, not too frequent)
                    optimal_min, optimal_max = 0.05, 0.15
                    
                    if signal_frequency < optimal_min:
                        score = signal_frequency / optimal_min
                    elif signal_frequency > optimal_max:
                        score = 1.0 - (signal_frequency - optimal_max) / (1.0 - optimal_max)
                    else:
                        score = 1.0
                    
                    frequency_scores.append(score)
            
            return np.mean(frequency_scores) if frequency_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_risk_adjusted_score(self, data: pd.DataFrame) -> float:
        """Calculate risk-adjusted returns based on signals."""
        try:
            if 'composite_signal' not in data.columns:
                return 0.5
            
            # Calculate returns
            returns = data['Close'].pct_change().fillna(0)
            signals = data['composite_signal'].fillna(0)
            
            # Strategy returns (simplified: position = signal)
            strategy_returns = signals.shift(1) * returns
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) < 20:
                return 0.5
            
            # Calculate Sharpe ratio
            mean_return = strategy_returns.mean()
            std_return = strategy_returns.std()
            
            if std_return > 0:
                sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
                
                # Convert Sharpe ratio to score (0-1 range)
                score = min(1.0, max(0.0, (sharpe_ratio + 1) / 3))  # Normalize around 0
            else:
                score = 0.5
            
            return score
            
        except Exception:
            return 0.5
    
    def _calculate_signal_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate signal consistency over time."""
        try:
            if 'composite_signal' not in data.columns:
                return 0.5
            
            signals = data['composite_signal'].dropna()
            
            if len(signals) < 20:
                return 0.5
            
            # Calculate rolling correlation with lagged signals
            rolling_corr = []
            window = 20
            
            for i in range(window, len(signals)):
                current_window = signals.iloc[i-window:i]
                prev_window = signals.iloc[i-window-5:i-5]
                
                if len(current_window) == len(prev_window) and len(current_window) > 5:
                    corr = np.corrcoef(current_window, prev_window)[0, 1]
                    if not np.isnan(corr):
                        rolling_corr.append(abs(corr))
            
            if rolling_corr:
                consistency_score = np.mean(rolling_corr)
                return consistency_score
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_profit_potential_score(self, data: pd.DataFrame) -> float:
        """Calculate profit potential of generated signals."""
        try:
            if 'composite_signal' not in data.columns:
                return 0.5
            
            returns = data['Close'].pct_change().fillna(0)
            signals = data['composite_signal'].fillna(0)
            
            # Calculate cumulative strategy returns
            strategy_returns = signals.shift(1) * returns
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            if len(cumulative_returns) < 20:
                return 0.5
            
            # Total return
            total_return = cumulative_returns.iloc[-1] - 1
            
            # Maximum drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdown = cumulative_returns / rolling_max - 1
            max_drawdown = drawdown.min()
            
            # Profit score (return vs drawdown)
            if max_drawdown < -0.5:  # Too much drawdown
                score = 0.1
            else:
                score = min(1.0, max(0.0, total_return * 2))  # Scale total return
                
                # Penalize high drawdown
                if max_drawdown < -0.1:
                    score *= (1 + max_drawdown * 2)  # Reduce score for drawdown
            
            return score
            
        except Exception:
            return 0.5
    
    def _calculate_parameter_penalty(self, params: Dict[str, Any]) -> float:
        """Calculate penalty for extreme parameter values."""
        try:
            penalty = 0.0
            
            # Penalize extreme thresholds
            if params.get('rsi_oversold_threshold', 30) < 25:
                penalty += 0.05
            if params.get('rsi_overbought_threshold', 70) > 75:
                penalty += 0.05
            
            # Penalize unbalanced weights
            weights = [
                params.get('momentum_weight', 0.4),
                params.get('trend_weight', 0.4),
                params.get('mean_reversion_weight', 0.2)
            ]
            
            # Penalize if any weight is too dominant
            if max(weights) > 0.7:
                penalty += 0.1
            
            return penalty
            
        except Exception:
            return 0.0