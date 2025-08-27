#!/usr/bin/env python3
"""
Unified Trading Strategy

This module implements a comprehensive unified trading strategy that combines
all available strategies and data sources using ensemble methods and advanced
feature selection to create robust trading signals while avoiding overfitting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import existing strategies and components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.backtesting import (
    TradingStrategy, TechnicalAnalysisStrategy, MeanReversionStrategy,
    MomentumStrategy, PatternRecognitionStrategy
)
from features.market_trend import create_comprehensive_trend_features
from features.candlestick import extract_all_candlestick_features
from features.earnings import EarningsFeatureEngineer
from features.feature_selector import FeatureSelector
from features.ensemble_model import EnsembleSignalCombiner
from utils.data_enrichment import DataEnricher
from utils.risk import RiskMetrics


class UnifiedTradingStrategy(TradingStrategy):
    """
    Unified trading strategy that combines all available strategies and data sources
    using ensemble methods and advanced feature selection.
    
    This strategy integrates:
    - Technical Analysis (RSI, MA, MACD, etc.)
    - Mean Reversion (Bollinger Bands)
    - Momentum (MACD, Price momentum)
    - Pattern Recognition (Candlestick patterns)
    - Earnings Analysis
    - Market Sentiment
    - Risk Management
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize unified trading strategy.
        
        Args:
            config: Configuration dictionary for the strategy
        """
        super().__init__("Unified Strategy", config)
        
        # Configuration
        self.config = config or {}
        self.feature_config = self.config.get('feature_selection', {})
        self.ensemble_config = self.config.get('ensemble', {})
        self.risk_config = self.config.get('risk_management', {})
        
        # Components
        self.feature_selector = FeatureSelector(self.feature_config)
        self.ensemble_combiner = EnsembleSignalCombiner(self.ensemble_config)
        self.data_enricher = DataEnricher()
        self.earnings_engineer = EarningsFeatureEngineer()
        
        # Base strategies
        self.base_strategies = {
            'technical_analysis': TechnicalAnalysisStrategy(
                self.config.get('technical_analysis', {})
            ),
            'mean_reversion': MeanReversionStrategy(
                self.config.get('mean_reversion', {})
            ),
            'momentum': MomentumStrategy(
                self.config.get('momentum', {})
            ),
            'pattern_recognition': PatternRecognitionStrategy(
                self.config.get('pattern_recognition', {})
            )
        }
        
        # State tracking
        self.is_fitted = False
        self.feature_names = []
        self.signal_history = []
        self.performance_metrics = {}
        
        # Risk management parameters
        self.max_signal_strength = self.risk_config.get('max_signal_strength', 1.0)
        self.confidence_threshold = self.risk_config.get('confidence_threshold', 0.3)
        self.signal_decay_rate = self.risk_config.get('signal_decay_rate', 0.95)
        self.volatility_adjustment = self.risk_config.get('volatility_adjustment', True)
    
    def fit(self, 
            historical_data: Dict[str, pd.DataFrame],
            earnings_data: Dict[str, pd.DataFrame] = None,
            sentiment_data: Dict[str, pd.DataFrame] = None) -> 'UnifiedTradingStrategy':
        """
        Fit the unified strategy on historical data.
        
        Args:
            historical_data: Dictionary of price data by symbol
            earnings_data: Optional earnings data by symbol
            sentiment_data: Optional sentiment data by symbol
            
        Returns:
            Fitted strategy instance
        """
        print("🚀 Training Unified Trading Strategy...")
        print("=" * 50)
        
        if not historical_data:
            raise ValueError("Historical data is required for training")
        
        # Combine and prepare all data
        combined_features = {}
        combined_targets = {}
        
        for symbol, price_data in historical_data.items():
            print(f"📊 Processing {symbol}...")
            
            # Generate comprehensive features
            features = self._generate_comprehensive_features(
                symbol, price_data, earnings_data, sentiment_data
            )
            
            # Create target (future returns)
            target = self._create_target_variable(price_data)
            
            # Align features and target
            aligned_features, aligned_target = self._align_data(features, target)
            
            if len(aligned_features) > 100:  # Minimum samples for training
                combined_features[symbol] = aligned_features
                combined_targets[symbol] = aligned_target
                print(f"  ✅ {symbol}: {len(aligned_features)} samples, {len(aligned_features.columns)} features")
            else:
                print(f"  ⚠️ {symbol}: Insufficient data ({len(aligned_features)} samples)")
        
        if not combined_features:
            raise ValueError("No symbols have sufficient data for training")
        
        # Combine features across all symbols
        print("\n🔧 Combining features across symbols...")
        all_features = pd.concat(combined_features.values(), ignore_index=True)
        all_targets = pd.concat(combined_targets.values(), ignore_index=True)
        
        print(f"📈 Total training samples: {len(all_features)}")
        print(f"🎯 Total features before selection: {len(all_features.columns)}")
        
        # Apply feature selection and noise reduction
        print("\n🎯 Applying feature selection...")
        all_features_denoised = self.feature_selector.reduce_noise(all_features)
        all_features_composite = self.feature_selector.create_composite_features(all_features_denoised)
        all_features_selected = self.feature_selector.select_features(
            all_features_composite, all_targets
        )
        
        print(f"✨ Features after selection: {len(all_features_selected.columns)}")
        self.feature_names = list(all_features_selected.columns)
        
        # Generate base strategy signals
        print("\n📡 Generating base strategy signals...")
        strategy_signals = self._generate_base_strategy_signals(combined_features)
        
        # Combine strategy signals with selected features
        print("\n🔗 Combining strategy signals with features...")
        if len(strategy_signals) > 0:
            final_features = self._combine_features_and_signals(
                all_features_selected, strategy_signals, all_targets.index
            )
        else:
            final_features = all_features_selected
        
        # Train ensemble model
        print("\n🎪 Training ensemble model...")
        print(f"  Final features shape: {final_features.shape}")
        print(f"  Target shape: {all_targets.shape}")
        
        # Adjust minimum samples if needed
        original_min_samples = self.ensemble_combiner.min_samples_for_training
        if len(final_features) < original_min_samples:
            self.ensemble_combiner.min_samples_for_training = max(50, len(final_features))
            print(f"  Adjusted min samples from {original_min_samples} to {self.ensemble_combiner.min_samples_for_training}")
        
        self.ensemble_combiner.fit(final_features, all_targets)
        
        # Evaluate training performance
        print("\n📊 Evaluating training performance...")
        self.performance_metrics = self.ensemble_combiner.evaluate_performance(
            final_features, all_targets
        )
        
        for metric, value in self.performance_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
        
        self.is_fitted = True
        print("\n✅ Unified strategy training completed!")
        
        return self
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate unified trading signals for given data.
        
        Args:
            data: OHLCV price data
            
        Returns:
            Series of trading signals (-1 to 1)
        """
        if not self.is_fitted:
            # If not fitted, use a simple combination of base strategies
            return self._generate_unfitted_signals(data)
        
        try:
            # Generate comprehensive features
            features = self._generate_comprehensive_features('', data)
            
            # Apply the same feature processing as training
            features_denoised = self.feature_selector.reduce_noise(features)
            features_composite = self.feature_selector.create_composite_features(features_denoised)
            
            # Select only the features used in training
            available_features = [f for f in self.feature_names if f in features_composite.columns]
            features_selected = features_composite[available_features]
            
            # Fill missing features with zeros
            for feature in self.feature_names:
                if feature not in features_selected.columns:
                    features_selected[feature] = 0.0
            
            # Reorder columns to match training
            features_selected = features_selected[self.feature_names]
            
            # Generate base strategy signals
            base_signals = {}
            for name, strategy in self.base_strategies.items():
                try:
                    signals = strategy.generate_signals(data)
                    base_signals[f'{name}_signal'] = signals
                except Exception as e:
                    print(f"Warning: Failed to generate {name} signals: {e}")
                    base_signals[f'{name}_signal'] = pd.Series(0.0, index=data.index)
            
            # Combine with features
            strategy_signals_df = pd.DataFrame(base_signals)
            
            # Align all data
            common_index = features_selected.index.intersection(strategy_signals_df.index)
            if len(common_index) == 0:
                return pd.Series(0.0, index=data.index)
            
            features_aligned = features_selected.loc[common_index]
            signals_aligned = strategy_signals_df.loc[common_index]
            
            # Combine features and signals
            final_features = pd.concat([features_aligned, signals_aligned], axis=1)
            
            # Generate ensemble predictions
            ensemble_signals = self.ensemble_combiner.predict(final_features)
            
            # Apply risk management and signal processing
            processed_signals = self._process_signals(ensemble_signals, data)
            
            # Return signals aligned with original data index
            return processed_signals.reindex(data.index, fill_value=0.0)
            
        except Exception as e:
            print(f"Warning: Error generating unified signals: {e}")
            return self._generate_unfitted_signals(data)
    
    def _generate_comprehensive_features(self, 
                                       symbol: str,
                                       price_data: pd.DataFrame,
                                       earnings_data: Dict[str, pd.DataFrame] = None,
                                       sentiment_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Generate comprehensive features from all available data sources."""
        
        # Start with price data
        features = price_data.copy()
        
        # Add technical analysis features
        try:
            features = create_comprehensive_trend_features(features)
        except Exception as e:
            print(f"Warning: Failed to create trend features for {symbol}: {e}")
        
        # Add candlestick pattern features
        try:
            candlestick_features = extract_all_candlestick_features(features)
            # Merge candlestick features
            for col in candlestick_features.columns:
                if col not in features.columns:
                    features[col] = candlestick_features[col]
        except Exception as e:
            print(f"Warning: Failed to create candlestick features for {symbol}: {e}")
        
        # Add enriched features
        try:
            features = self.data_enricher.add_rolling_features(features)
            features = self.data_enricher.add_volatility_features(features)
            features = self.data_enricher.add_technical_features(features)
            features = self.data_enricher.add_regime_detection(features)
        except Exception as e:
            print(f"Warning: Failed to enrich features for {symbol}: {e}")
        
        # Add earnings features if available
        if earnings_data and symbol in earnings_data:
            try:
                earnings_features = self.earnings_engineer.create_earnings_trading_signals(
                    features, earnings_data[symbol], pd.DataFrame()
                )
                # Merge earnings features
                for col in earnings_features.columns:
                    if col not in features.columns and 'earnings' in col:
                        features[col] = earnings_features[col]
            except Exception as e:
                print(f"Warning: Failed to create earnings features for {symbol}: {e}")
        
        # Add sentiment features if available
        if sentiment_data and symbol in sentiment_data:
            try:
                features = self.data_enricher.add_sentiment_features(
                    features, sentiment_data[symbol]
                )
            except Exception as e:
                print(f"Warning: Failed to add sentiment features for {symbol}: {e}")
        
        # Remove non-numeric columns and handle missing values
        numeric_features = features.select_dtypes(include=[np.number])
        numeric_features = numeric_features.ffill().fillna(0)
        
        return numeric_features
    
    def _create_target_variable(self, price_data: pd.DataFrame, 
                              horizon: int = 1) -> pd.Series:
        """Create target variable (future returns)."""
        if 'Close' not in price_data.columns:
            raise ValueError("Price data must contain 'Close' column")
        
        # Calculate future returns
        future_returns = price_data['Close'].pct_change(horizon).shift(-horizon)
        
        # Cap extreme returns to reduce noise
        return future_returns.clip(-0.1, 0.1)  # Cap at ±10%
    
    def _align_data(self, features: pd.DataFrame, 
                   target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and target data."""
        # Find common index
        common_index = features.index.intersection(target.index)
        
        # Align data
        features_aligned = features.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Remove rows with missing target values
        valid_mask = ~target_aligned.isna()
        features_clean = features_aligned[valid_mask]
        target_clean = target_aligned[valid_mask]
        
        return features_clean, target_clean
    
    def _generate_base_strategy_signals(self, 
                                      combined_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate signals from all base strategies."""
        all_signals = []
        
        for symbol, features in combined_features.items():
            # Reconstruct price data for strategies
            price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            price_data = features[price_cols] if all(col in features.columns for col in price_cols) else features
            
            symbol_signals = {}
            
            for name, strategy in self.base_strategies.items():
                try:
                    signals = strategy.generate_signals(price_data)
                    # Take only the signals that align with the features
                    aligned_signals = signals.reindex(features.index, fill_value=0.0)
                    symbol_signals[f'{name}_signal'] = aligned_signals
                except Exception as e:
                    print(f"Warning: Failed to generate {name} signals for {symbol}: {e}")
                    symbol_signals[f'{name}_signal'] = pd.Series(0.0, index=features.index)
            
            # Convert to DataFrame
            signals_df = pd.DataFrame(symbol_signals)
            all_signals.append(signals_df)
        
        # Combine all signals
        if all_signals:
            return pd.concat(all_signals, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _combine_features_and_signals(self, 
                                    features: pd.DataFrame,
                                    signals: pd.DataFrame,
                                    target_index: pd.Index) -> pd.DataFrame:
        """Combine selected features with strategy signals."""
        
        if signals.empty:
            return features
        
        # Ensure both have the same number of rows as target
        min_length = min(len(features), len(signals), len(target_index))
        
        features_trimmed = features.head(min_length)
        signals_trimmed = signals.head(min_length)
        target_index_trimmed = target_index[:min_length]
        
        # Reset indices to ensure alignment
        features_trimmed.index = range(min_length)
        signals_trimmed.index = range(min_length)
        
        # Combine features and signals
        combined = pd.concat([features_trimmed, signals_trimmed], axis=1)
        
        # Remove any remaining NaN values
        combined = combined.fillna(0)
        
        return combined
    
    def _generate_unfitted_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate simple signals when strategy is not fitted."""
        signals = pd.Series(0.0, index=data.index)
        
        # Simple combination of base strategies
        strategy_signals = []
        for name, strategy in self.base_strategies.items():
            try:
                strategy_signal = strategy.generate_signals(data)
                strategy_signals.append(strategy_signal)
            except Exception as e:
                print(f"Warning: Failed to generate {name} signals: {e}")
                continue
        
        if strategy_signals:
            # Simple average of all strategy signals
            combined_signals = pd.concat(strategy_signals, axis=1)
            signals = combined_signals.mean(axis=1).fillna(0)
            
            # Apply bounds
            signals = signals.clip(-1.0, 1.0)
        
        return signals
    
    def _process_signals(self, raw_signals: pd.Series, data: pd.DataFrame) -> pd.Series:
        """Apply risk management and signal processing."""
        processed_signals = raw_signals.copy()
        
        # Apply confidence threshold
        processed_signals = processed_signals.where(
            abs(processed_signals) >= self.confidence_threshold, 0.0
        )
        
        # Apply volatility adjustment if enabled
        if self.volatility_adjustment and 'Close' in data.columns:
            try:
                # Calculate rolling volatility
                returns = data['Close'].pct_change()
                volatility = returns.rolling(window=20).std()
                
                # Align volatility with signals
                vol_aligned = volatility.reindex(processed_signals.index)
                
                # Reduce signal strength during high volatility periods
                vol_threshold = vol_aligned.quantile(0.8)  # 80th percentile
                high_vol_mask = vol_aligned > vol_threshold
                processed_signals.loc[high_vol_mask] *= 0.5
                
            except Exception as e:
                print(f"Warning: Failed to apply volatility adjustment: {e}")
        
        # Apply signal decay (reduce strength of consecutive signals)
        decayed_signals = processed_signals.copy()
        for i in range(1, len(decayed_signals)):
            if (abs(decayed_signals.iloc[i]) > 0 and 
                abs(decayed_signals.iloc[i-1]) > 0 and
                np.sign(decayed_signals.iloc[i]) == np.sign(decayed_signals.iloc[i-1])):
                decayed_signals.iloc[i] *= self.signal_decay_rate
        
        # Apply maximum signal strength bounds
        final_signals = decayed_signals.clip(-self.max_signal_strength, self.max_signal_strength)
        
        return final_signals
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get comprehensive strategy summary."""
        summary = {
            'strategy_name': self.name,
            'is_fitted': self.is_fitted,
            'base_strategies': list(self.base_strategies.keys()),
            'num_features': len(self.feature_names),
            'config': self.config.copy()
        }
        
        if self.is_fitted:
            summary.update({
                'training_performance': self.performance_metrics.copy(),
                'feature_importance': self.ensemble_combiner.get_feature_importance(),
                'ensemble_summary': self.ensemble_combiner.get_ensemble_summary(),
                'feature_selection_summary': self.feature_selector.get_selection_summary()
            })
        
        return summary
    
    def get_position_size(self, signal: float, price: float, portfolio_value: float) -> float:
        """Calculate position size with enhanced risk management."""
        if abs(signal) < self.confidence_threshold:
            return 0.0
        
        # Base position sizing
        base_position = super().get_position_size(signal, price, portfolio_value)
        
        # Apply additional risk adjustments
        if self.is_fitted and 'overfitting_score' in self.performance_metrics:
            overfitting_score = self.performance_metrics['overfitting_score']
            # Reduce position size if model shows signs of overfitting
            if overfitting_score > 0.3:
                base_position *= (1.0 - overfitting_score)
        
        return base_position


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Unified Trading Strategy")
    print("=" * 50)
    
    # Create sample data
    import yfinance as yf
    
    symbols = ['AAPL', 'MSFT']
    print(f"📊 Fetching data for {symbols}...")
    
    historical_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1y')
            if not data.empty:
                historical_data[symbol] = data
                print(f"  ✅ {symbol}: {len(data)} days")
            else:
                print(f"  ❌ {symbol}: No data available")
        except Exception as e:
            print(f"  ❌ {symbol}: Error fetching data - {e}")
    
    if not historical_data:
        print("❌ No data available for testing")
        exit(1)
    
    # Test unified strategy
    config = {
        'feature_selection': {
            'max_features': 20,
            'correlation_threshold': 0.9
        },
        'ensemble': {
            'ensemble_method': 'voting',
            'cv_folds': 3
        },
        'risk_management': {
            'confidence_threshold': 0.2,
            'volatility_adjustment': True
        }
    }
    
    strategy = UnifiedTradingStrategy(config)
    
    # Test unfitted strategy (simple mode)
    print(f"\n🔍 Testing unfitted strategy on {symbols[0]}...")
    test_data = historical_data[symbols[0]]
    simple_signals = strategy.generate_signals(test_data)
    print(f"  Generated {len(simple_signals)} signals")
    print(f"  Signal range: [{simple_signals.min():.3f}, {simple_signals.max():.3f}]")
    print(f"  Non-zero signals: {(simple_signals != 0).sum()}")
    
    # Test fitted strategy (advanced mode)
    if len(historical_data) > 1:
        print(f"\n🚀 Testing fitted strategy...")
        try:
            strategy.fit(historical_data)
            
            # Generate signals for test data
            fitted_signals = strategy.generate_signals(test_data)
            print(f"  Generated {len(fitted_signals)} fitted signals")
            print(f"  Signal range: [{fitted_signals.min():.3f}, {fitted_signals.max():.3f}]")
            print(f"  Non-zero signals: {(fitted_signals != 0).sum()}")
            
            # Show strategy summary
            summary = strategy.get_strategy_summary()
            print(f"\n📋 Strategy Summary:")
            print(f"  Fitted: {summary['is_fitted']}")
            print(f"  Features: {summary['num_features']}")
            print(f"  Base strategies: {summary['base_strategies']}")
            
        except Exception as e:
            print(f"  ❌ Error training strategy: {e}")
    
    print("\n✅ Unified strategy testing completed!")