#!/usr/bin/env python3
"""
Hyperparameter Optimizer for EarningsFeatureEngineer

Optimizes parameters for the EarningsFeatureEngineer class including thresholds,
lookback periods, and feature engineering parameters.
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path to import trading bot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.earnings import EarningsFeatureEngineer, EarningsDataFetcher
from optimization.base import BaseOptimizer, OptimizationConfig
from optimization.cache import OptimizationCache
import logging

logger = logging.getLogger(__name__)


class OptimizedEarningsFeatureEngineer(EarningsFeatureEngineer):
    """
    EarningsFeatureEngineer with optimizable hyperparameters.
    
    This wrapper allows the EarningsFeatureEngineer to accept hyperparameters
    and modify feature engineering behavior accordingly.
    """
    
    def __init__(self, **hyperparams):
        """
        Initialize with hyperparameters.
        
        Args:
            **hyperparams: Hyperparameters for optimization
        """
        super().__init__()
        self.hyperparams = hyperparams
        
        # Earnings surprise thresholds
        self.strong_beat_threshold = hyperparams.get('strong_beat_threshold', 10.0)
        self.strong_miss_threshold = hyperparams.get('strong_miss_threshold', -10.0)
        self.beat_threshold = hyperparams.get('beat_threshold', 5.0)
        self.miss_threshold = hyperparams.get('miss_threshold', -5.0)
        
        # Growth analysis parameters
        self.growth_acceleration_threshold = hyperparams.get('growth_acceleration_threshold', 0.05)
        self.high_growth_threshold = hyperparams.get('high_growth_threshold', 0.2)
        self.declining_growth_threshold = hyperparams.get('declining_growth_threshold', -0.1)
        
        # Momentum calculation parameters
        self.pre_earnings_lookback = hyperparams.get('pre_earnings_lookback', 20)
        self.post_earnings_window = hyperparams.get('post_earnings_window', 5)
        
        # Revenue growth thresholds
        self.revenue_growth_threshold = hyperparams.get('revenue_growth_threshold', 0.1)
        self.revenue_acceleration_threshold = hyperparams.get('revenue_acceleration_threshold', 0.02)
        
        # Signal strength weights
        self.surprise_weight = hyperparams.get('surprise_weight', 0.4)
        self.growth_weight = hyperparams.get('growth_weight', 0.3)
        self.momentum_weight = hyperparams.get('momentum_weight', 0.3)
        
        # Time windows for feature calculation
        self.rolling_avg_window = hyperparams.get('rolling_avg_window', 4)
        self.volatility_window = hyperparams.get('volatility_window', 8)
    
    def calculate_earnings_surprise(self, earnings_df: pd.DataFrame) -> pd.DataFrame:
        """Override with parameterized thresholds."""
        result_df = earnings_df.copy()
        
        if 'Actual' in result_df.columns and 'Estimate' in result_df.columns:
            # EPS surprise percentage
            result_df['eps_surprise_pct'] = (
                (result_df['Actual'] - result_df['Estimate']) / 
                abs(result_df['Estimate'] + 1e-8) * 100
            )
            
            # Absolute surprise
            result_df['eps_surprise_abs'] = result_df['Actual'] - result_df['Estimate']
            
            # Beat/miss classification with optimized thresholds
            result_df['earnings_beat'] = (result_df['eps_surprise_pct'] > self.beat_threshold).astype(int)
            result_df['earnings_miss'] = (result_df['eps_surprise_pct'] < self.miss_threshold).astype(int)
            result_df['strong_beat'] = (result_df['eps_surprise_pct'] > self.strong_beat_threshold).astype(int)
            result_df['strong_miss'] = (result_df['eps_surprise_pct'] < self.strong_miss_threshold).astype(int)
            
            # Surprise magnitude categories with optimized bins
            result_df['surprise_magnitude'] = pd.cut(
                result_df['eps_surprise_pct'],
                bins=[-np.inf, self.strong_miss_threshold, self.miss_threshold, 
                      self.beat_threshold, self.strong_beat_threshold, np.inf],
                labels=['large_miss', 'small_miss', 'inline', 'small_beat', 'large_beat']
            )
        
        return result_df
    
    def calculate_earnings_growth(self, earnings_df: pd.DataFrame) -> pd.DataFrame:
        """Override with parameterized growth analysis."""
        result_df = earnings_df.copy()
        
        if 'Actual' in result_df.columns:
            # Sort by date
            if 'date' in result_df.columns:
                result_df = result_df.sort_values('date')
            
            # Quarter-over-quarter growth
            result_df['eps_qoq_growth'] = result_df['Actual'].pct_change()
            
            # Year-over-year growth (assuming quarterly data)
            result_df['eps_yoy_growth'] = result_df['Actual'].pct_change(periods=4)
            
            # Growth acceleration with optimized calculation
            result_df['eps_growth_acceleration'] = result_df['eps_qoq_growth'].diff()
            
            # Earnings trend with optimized window
            result_df['eps_trend_score'] = (
                result_df['eps_qoq_growth'].rolling(window=self.rolling_avg_window).mean()
            )
            
            # Consistency score with optimized window
            result_df['eps_consistency'] = (
                result_df['eps_qoq_growth'].rolling(window=self.volatility_window).std()
            )
            
            # Growth classification with optimized thresholds
            result_df['high_growth'] = (result_df['eps_yoy_growth'] > self.high_growth_threshold).astype(int)
            result_df['declining_growth'] = (result_df['eps_yoy_growth'] < self.declining_growth_threshold).astype(int)
            result_df['accelerating_growth'] = (result_df['eps_growth_acceleration'] > self.growth_acceleration_threshold).astype(int)
        
        return result_df
    
    def calculate_pre_earnings_momentum(self, price_df: pd.DataFrame,
                                      earnings_dates: list,
                                      lookback_days: int = None) -> pd.DataFrame:
        """Override with parameterized lookback period."""
        lookback_days = lookback_days or self.pre_earnings_lookback
        return super().calculate_pre_earnings_momentum(price_df, earnings_dates, lookback_days)
    
    def create_earnings_event_features(self, price_df: pd.DataFrame,
                                     earnings_dates: list,
                                     event_window: int = None) -> pd.DataFrame:
        """Create features around earnings announcement dates with optimized window."""
        event_window = event_window or self.post_earnings_window
        result_df = price_df.copy()
        result_df.index = pd.to_datetime(result_df.index)
        
        # Initialize event features
        result_df['pre_earnings_period'] = 0
        result_df['post_earnings_period'] = 0
        result_df['earnings_announcement'] = 0
        result_df['days_to_earnings'] = np.nan
        result_df['days_since_earnings'] = np.nan
        
        for earnings_date in earnings_dates:
            earnings_date = pd.to_datetime(earnings_date)
            
            # Mark earnings announcement date
            announcement_mask = result_df.index == earnings_date
            result_df.loc[announcement_mask, 'earnings_announcement'] = 1
            
            # Mark pre-earnings period
            pre_start = earnings_date - pd.Timedelta(days=event_window)
            pre_mask = (result_df.index >= pre_start) & (result_df.index < earnings_date)
            result_df.loc[pre_mask, 'pre_earnings_period'] = 1
            
            # Mark post-earnings period
            post_end = earnings_date + pd.Timedelta(days=event_window)
            post_mask = (result_df.index > earnings_date) & (result_df.index <= post_end)
            result_df.loc[post_mask, 'post_earnings_period'] = 1
            
            # Calculate days to/since earnings
            for idx in result_df.index:
                days_diff = (earnings_date - idx).days
                if days_diff > 0:  # Before earnings
                    if pd.isna(result_df.loc[idx, 'days_to_earnings']) or days_diff < result_df.loc[idx, 'days_to_earnings']:
                        result_df.loc[idx, 'days_to_earnings'] = days_diff
                elif days_diff < 0:  # After earnings
                    if pd.isna(result_df.loc[idx, 'days_since_earnings']) or abs(days_diff) < result_df.loc[idx, 'days_since_earnings']:
                        result_df.loc[idx, 'days_since_earnings'] = abs(days_diff)
        
        return result_df
    
    def create_earnings_trading_signals(self, price_df: pd.DataFrame,
                                      earnings_df: pd.DataFrame,
                                      estimates_df: pd.DataFrame) -> pd.DataFrame:
        """Override with weighted signal combination."""
        result_df = price_df.copy()
        
        # Initialize signal columns
        result_df['earnings_bullish_signal'] = 0
        result_df['earnings_bearish_signal'] = 0
        result_df['earnings_signal_strength'] = 0
        
        # Merge earnings data
        if not earnings_df.empty:
            # Calculate earnings features with optimized parameters
            earnings_features = self.calculate_earnings_surprise(earnings_df)
            earnings_features = self.calculate_earnings_growth(earnings_features)
            
            earnings_signals = self._generate_earnings_signals(earnings_features)
            
            # Map signals to price data (simplified approach)
            for idx, row in earnings_signals.iterrows():
                if 'date' in row and not pd.isna(row['date']):
                    signal_date = pd.to_datetime(row['date'])
                    
                    # Find closest price date
                    price_dates = pd.to_datetime(result_df.index)
                    closest_idx = np.argmin(abs(price_dates - signal_date))
                    
                    if closest_idx < len(result_df):
                        # Apply weighted signal combination
                        surprise_signal = row.get('bullish_signal', 0) * self.surprise_weight
                        growth_signal = row.get('high_growth', 0) * self.growth_weight
                        
                        # Calculate momentum signal if price data is available
                        momentum_signal = 0
                        if 'Close' in result_df.columns:
                            pre_earnings_data = result_df.iloc[max(0, closest_idx-self.pre_earnings_lookback):closest_idx]
                            if len(pre_earnings_data) > 5:
                                momentum = pre_earnings_data['Close'].pct_change().mean()
                                momentum_signal = (momentum > 0) * self.momentum_weight
                        
                        combined_signal = surprise_signal + growth_signal + momentum_signal
                        
                        result_df.iloc[closest_idx, result_df.columns.get_loc('earnings_bullish_signal')] = int(combined_signal > 0.5)
                        result_df.iloc[closest_idx, result_df.columns.get_loc('earnings_signal_strength')] = combined_signal
        
        return result_df
    
    def _generate_earnings_signals(self, earnings_df: pd.DataFrame) -> pd.DataFrame:
        """Override with parameterized signal generation."""
        signals_df = earnings_df.copy()
        
        # Initialize signals
        signals_df['bullish_signal'] = 0
        signals_df['bearish_signal'] = 0
        signals_df['signal_strength'] = 0
        
        # EPS surprise signals with optimized thresholds
        if 'eps_surprise_pct' in signals_df.columns:
            strong_beat_mask = signals_df['eps_surprise_pct'] > self.strong_beat_threshold
            signals_df.loc[strong_beat_mask, 'bullish_signal'] = 1
            signals_df.loc[strong_beat_mask, 'signal_strength'] = 0.8
            
            strong_miss_mask = signals_df['eps_surprise_pct'] < self.strong_miss_threshold
            signals_df.loc[strong_miss_mask, 'bearish_signal'] = 1
            signals_df.loc[strong_miss_mask, 'signal_strength'] = 0.8
        
        # Growth signals with optimized thresholds
        if 'eps_yoy_growth' in signals_df.columns:
            high_growth_mask = signals_df['eps_yoy_growth'] > self.high_growth_threshold
            signals_df.loc[high_growth_mask, 'bullish_signal'] = 1
            signals_df.loc[high_growth_mask, 'signal_strength'] += 0.3
            
            declining_mask = signals_df['eps_yoy_growth'] < self.declining_growth_threshold
            signals_df.loc[declining_mask, 'bearish_signal'] = 1
            signals_df.loc[declining_mask, 'signal_strength'] += 0.3
        
        # Growth acceleration signals
        if 'eps_growth_acceleration' in signals_df.columns:
            accel_mask = signals_df['eps_growth_acceleration'] > self.growth_acceleration_threshold
            signals_df.loc[accel_mask, 'bullish_signal'] = 1
            signals_df.loc[accel_mask, 'signal_strength'] += 0.2
        
        # Cap signal strength
        signals_df['signal_strength'] = signals_df['signal_strength'].clip(0, 1)
        
        return signals_df


class EarningsFeatureEngineerOptimizer(BaseOptimizer):
    """
    Hyperparameter optimizer for EarningsFeatureEngineer.
    
    Optimizes parameters such as:
    - Earnings surprise thresholds
    - Growth analysis thresholds
    - Time windows for momentum calculation
    - Signal combination weights
    """
    
    def __init__(self, 
                 config: OptimizationConfig = None,
                 cache_dir: str = ".optimization_cache"):
        """Initialize the EarningsFeatureEngineer optimizer."""
        super().__init__(
            model_class=OptimizedEarningsFeatureEngineer,
            config=config,
            cache_dir=cache_dir
        )
        
        # Initialize cache
        self.persistent_cache = OptimizationCache(cache_dir)
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for EarningsFeatureEngineer.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        params = {
            # Earnings surprise thresholds
            'strong_beat_threshold': trial.suggest_float('strong_beat_threshold', 5.0, 20.0),
            'strong_miss_threshold': trial.suggest_float('strong_miss_threshold', -20.0, -5.0),
            'beat_threshold': trial.suggest_float('beat_threshold', 2.0, 8.0),
            'miss_threshold': trial.suggest_float('miss_threshold', -8.0, -2.0),
            
            # Growth thresholds
            'growth_acceleration_threshold': trial.suggest_float('growth_acceleration_threshold', 0.01, 0.1),
            'high_growth_threshold': trial.suggest_float('high_growth_threshold', 0.1, 0.4),
            'declining_growth_threshold': trial.suggest_float('declining_growth_threshold', -0.2, -0.05),
            
            # Time windows
            'pre_earnings_lookback': trial.suggest_int('pre_earnings_lookback', 10, 30),
            'post_earnings_window': trial.suggest_int('post_earnings_window', 3, 10),
            'rolling_avg_window': trial.suggest_int('rolling_avg_window', 2, 8),
            'volatility_window': trial.suggest_int('volatility_window', 4, 12),
            
            # Revenue thresholds
            'revenue_growth_threshold': trial.suggest_float('revenue_growth_threshold', 0.05, 0.2),
            'revenue_acceleration_threshold': trial.suggest_float('revenue_acceleration_threshold', 0.01, 0.05),
            
            # Signal weights (must sum to 1)
            'surprise_weight': trial.suggest_float('surprise_weight', 0.2, 0.6),
            'growth_weight': trial.suggest_float('growth_weight', 0.2, 0.5),
        }
        
        # Ensure logical constraints
        if params['beat_threshold'] >= params['strong_beat_threshold']:
            # Clamp to allowed range [5.0, 20.0]
            params['strong_beat_threshold'] = min(max(params['beat_threshold'] + 5.0, 5.0), 20.0)
        
        if params['miss_threshold'] <= params['strong_miss_threshold']:
            # Clamp to allowed range [-20.0, -5.0]
            params['strong_miss_threshold'] = min(max(params['miss_threshold'] - 5.0, -20.0), -5.0)
        
        # Normalize weights
        weight_sum = params['surprise_weight'] + params['growth_weight']
        if weight_sum > 1.0:
            params['surprise_weight'] /= weight_sum
            params['growth_weight'] /= weight_sum
            params['momentum_weight'] = 0.0
        else:
            params['momentum_weight'] = 1.0 - weight_sum
        
        return params
    
    def create_model_instance(self, params: Dict[str, Any]) -> OptimizedEarningsFeatureEngineer:
        """
        Create EarningsFeatureEngineer instance with given parameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            OptimizedEarningsFeatureEngineer instance
        """
        return OptimizedEarningsFeatureEngineer(**params)
    
    def evaluate_model(self, model: OptimizedEarningsFeatureEngineer, data: pd.DataFrame) -> float:
        """
        Evaluate EarningsFeatureEngineer model and return a composite score.
        
        The evaluation focuses on:
        1. Earnings prediction accuracy
        2. Signal timing quality
        3. Feature informativeness
        4. Risk management effectiveness
        
        Args:
            model: OptimizedEarningsFeatureEngineer instance
            data: Data to evaluate on (should include earnings data)
            
        Returns:
            Composite score (higher is better)
        """
        try:
            # Generate features and signals
            enhanced_data = self._generate_earnings_features(model, data)
            
            # Calculate multiple scoring metrics
            scores = {}
            
            # 1. Earnings prediction accuracy
            scores['prediction_accuracy'] = self._calculate_prediction_accuracy_score(enhanced_data)
            
            # 2. Signal timing quality
            scores['timing_quality'] = self._calculate_timing_quality_score(enhanced_data)
            
            # 3. Feature informativeness
            scores['informativeness'] = self._calculate_informativeness_score(enhanced_data)
            
            # 4. Risk management
            scores['risk_management'] = self._calculate_risk_management_score(enhanced_data)
            
            # 5. Signal consistency
            scores['consistency'] = self._calculate_earnings_consistency_score(enhanced_data)
            
            # Combine scores with weights
            weights = {
                'prediction_accuracy': 0.35,
                'timing_quality': 0.25,
                'informativeness': 0.2,
                'risk_management': 0.1,
                'consistency': 0.1
            }
            
            composite_score = sum(weights[key] * scores[key] for key in scores)
            
            # Add regularization penalty
            penalty = self._calculate_parameter_penalty(model.hyperparams)
            composite_score -= penalty
            
            logger.debug(f"Earnings evaluation scores: {scores}, Penalty: {penalty:.4f}, Final: {composite_score:.4f}")
            
            return composite_score
            
        except Exception as e:
            logger.error(f"Error in earnings evaluation: {str(e)}")
            return -1.0
    
    def _generate_earnings_features(self, model: OptimizedEarningsFeatureEngineer, data: pd.DataFrame) -> pd.DataFrame:
        """Generate earnings features using the model."""
        enhanced_data = data.copy()
        
        # Create mock earnings data for testing if not provided
        if 'Actual' not in data.columns:
            enhanced_data = self._create_mock_earnings_data(data)
        
        # Apply earnings feature engineering
        earnings_data = enhanced_data[['Actual', 'Estimate']].dropna() if 'Estimate' in enhanced_data.columns else enhanced_data[['Actual']].dropna()
        
        if not earnings_data.empty:
            earnings_features = model.calculate_earnings_surprise(earnings_data)
            earnings_features = model.calculate_earnings_growth(earnings_features)
            
            # Merge back to main data
            for col in earnings_features.columns:
                if col not in enhanced_data.columns:
                    enhanced_data[col] = np.nan
                    enhanced_data.loc[earnings_features.index, col] = earnings_features[col]
        
        return enhanced_data
    
    def _create_mock_earnings_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create mock earnings data for testing purposes."""
        mock_data = data.copy()
        
        # Generate mock EPS data
        n_quarters = len(data) // 60  # Assume quarterly earnings
        
        if n_quarters > 0:
            eps_actual = np.random.normal(1.0, 0.2, n_quarters)
            eps_estimate = eps_actual + np.random.normal(0, 0.1, n_quarters)
            
            # Add earnings columns
            mock_data['Actual'] = np.nan
            mock_data['Estimate'] = np.nan
            
            # Place earnings data at regular intervals
            for i in range(n_quarters):
                idx = min(i * 60, len(mock_data) - 1)
                mock_data.iloc[idx, mock_data.columns.get_loc('Actual')] = eps_actual[i]
                mock_data.iloc[idx, mock_data.columns.get_loc('Estimate')] = eps_estimate[i]
        
        return mock_data
    
    def _calculate_prediction_accuracy_score(self, data: pd.DataFrame) -> float:
        """Calculate earnings prediction accuracy."""
        try:
            if 'eps_surprise_pct' not in data.columns or 'Close' not in data.columns:
                return 0.5
            
            # Calculate post-earnings returns
            returns = data['Close'].pct_change(5).shift(-5)  # 5-day forward return
            surprises = data['eps_surprise_pct']
            
            # Remove NaN values
            valid_mask = ~(returns.isna() | surprises.isna())
            
            if valid_mask.sum() < 5:
                return 0.5
            
            returns_clean = returns[valid_mask]
            surprises_clean = surprises[valid_mask]
            
            # Calculate correlation between surprises and future returns
            if len(returns_clean) > 1:
                correlation = np.corrcoef(surprises_clean, returns_clean)[0, 1]
                score = (abs(correlation) + 1) / 2 if not np.isnan(correlation) else 0.5
            else:
                score = 0.5
            
            return score
            
        except Exception:
            return 0.5
    
    def _calculate_timing_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate signal timing quality."""
        try:
            signal_cols = ['earnings_bullish_signal', 'earnings_bearish_signal']
            timing_scores = []
            
            for col in signal_cols:
                if col in data.columns and 'Close' in data.columns:
                    signals = data[col]
                    returns = data['Close'].pct_change()
                    
                    # Calculate returns around signal dates
                    signal_dates = data.index[signals > 0]
                    
                    if len(signal_dates) > 2:
                        signal_returns = []
                        for date in signal_dates:
                            date_idx = data.index.get_loc(date)
                            
                            # Look at returns in the following 5 days
                            if date_idx + 5 < len(data):
                                period_return = returns.iloc[date_idx:date_idx+5].mean()
                                if not np.isnan(period_return):
                                    signal_returns.append(period_return)
                        
                        if signal_returns:
                            # Positive mean return indicates good timing
                            mean_return = np.mean(signal_returns)
                            score = max(0, min(1, (mean_return + 0.02) * 25))  # Normalize
                            timing_scores.append(score)
            
            return np.mean(timing_scores) if timing_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_informativeness_score(self, data: pd.DataFrame) -> float:
        """Calculate informativeness of earnings features."""
        try:
            info_features = ['eps_surprise_pct', 'eps_yoy_growth', 'eps_growth_acceleration']
            info_scores = []
            
            for feature in info_features:
                if feature in data.columns:
                    feature_data = data[feature].dropna()
                    
                    if len(feature_data) > 5:
                        # Calculate information content (variance normalized by range)
                        data_range = feature_data.max() - feature_data.min()
                        data_std = feature_data.std()
                        
                        if data_range > 0:
                            info_score = min(1.0, data_std / (data_range / 4))  # Normalize
                            info_scores.append(info_score)
            
            return np.mean(info_scores) if info_scores else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_risk_management_score(self, data: pd.DataFrame) -> float:
        """Calculate risk management effectiveness."""
        try:
            if 'earnings_signal_strength' not in data.columns or 'Close' not in data.columns:
                return 0.5
            
            signals = data['earnings_signal_strength']
            returns = data['Close'].pct_change()
            
            # Calculate strategy returns
            strategy_returns = signals.shift(1) * returns
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) < 10:
                return 0.5
            
            # Calculate downside risk metrics
            negative_returns = strategy_returns[strategy_returns < 0]
            
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                total_std = strategy_returns.std()
                
                # Lower downside risk relative to total risk is better
                if total_std > 0:
                    risk_score = 1.0 - (downside_std / total_std)
                else:
                    risk_score = 0.5
            else:
                risk_score = 1.0  # No negative returns
            
            return max(0.0, min(1.0, risk_score))
            
        except Exception:
            return 0.5
    
    def _calculate_earnings_consistency_score(self, data: pd.DataFrame) -> float:
        """Calculate consistency of earnings signals."""
        try:
            if 'earnings_signal_strength' not in data.columns:
                return 0.5
            
            signals = data['earnings_signal_strength'].dropna()
            
            if len(signals) < 10:
                return 0.5
            
            # Calculate signal stability (low variance in signal strength)
            signal_std = signals.std()
            signal_mean = abs(signals.mean())
            
            if signal_mean > 0:
                cv = signal_std / signal_mean
                consistency_score = 1.0 / (1.0 + cv)  # Lower coefficient of variation is better
            else:
                consistency_score = 0.5
            
            return consistency_score
            
        except Exception:
            return 0.5
    
    def _calculate_parameter_penalty(self, params: Dict[str, Any]) -> float:
        """Calculate penalty for extreme parameter values."""
        try:
            penalty = 0.0
            
            # Penalize extreme thresholds
            if params.get('strong_beat_threshold', 10) > 18:
                penalty += 0.05
            if params.get('strong_miss_threshold', -10) < -18:
                penalty += 0.05
            
            # Penalize very long lookback periods
            if params.get('pre_earnings_lookback', 20) > 25:
                penalty += 0.05
            
            # Penalize unbalanced weights
            weights = [
                params.get('surprise_weight', 0.4),
                params.get('growth_weight', 0.3),
                params.get('momentum_weight', 0.3)
            ]
            
            if max(weights) > 0.8:
                penalty += 0.1
            
            return penalty
            
        except Exception:
            return 0.0