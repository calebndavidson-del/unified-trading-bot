#!/usr/bin/env python3
"""
Ensemble Model for Unified Trading Strategy

This module implements ensemble methods to combine signals from multiple trading
strategies and models while avoiding overfitting through regularization and
cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
)
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class EnsembleSignalCombiner:
    """
    Advanced ensemble model for combining trading signals from multiple strategies
    with built-in regularization and overfitting prevention.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ensemble signal combiner.
        
        Args:
            config: Configuration dictionary with ensemble parameters
        """
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.feature_weights = {}
        self.ensemble_weights = {}
        self.validation_scores = {}
        
        # Configuration parameters
        self.ensemble_method = self.config.get('ensemble_method', 'voting')
        self.regularization_strength = self.config.get('regularization_strength', 0.01)
        self.cv_folds = self.config.get('cv_folds', 5)
        self.min_samples_for_training = self.config.get('min_samples_for_training', 100)
        self.overfitting_threshold = self.config.get('overfitting_threshold', 0.2)
        
        # Initialize base models with regularization
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Initialize base models with proper regularization."""
        self.base_models = {
            'ridge': RidgeCV(
                alphas=np.logspace(-3, 2, 20),
                cv=self.cv_folds
            ),
            'elastic_net': ElasticNetCV(
                alphas=np.logspace(-3, 1, 20),
                l1_ratio=[0.1, 0.5, 0.7, 0.9],
                cv=self.cv_folds,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=5,  # Limit depth to prevent overfitting
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,  # Shallow trees
                learning_rate=0.1,
                subsample=0.8,  # Regularization through subsampling
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(50, 25),
                alpha=self.regularization_strength,  # L2 regularization
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42,
                max_iter=500
            )
        }
    
    def fit(self, 
            signals: pd.DataFrame, 
            target: pd.Series,
            strategy_weights: Dict[str, float] = None) -> 'EnsembleSignalCombiner':
        """
        Fit ensemble model on trading signals and target returns.
        
        Args:
            signals: DataFrame with signals from different strategies
            target: Target returns
            strategy_weights: Optional weights for different strategies
            
        Returns:
            Fitted ensemble model
        """
        if len(signals) < self.min_samples_for_training:
            raise ValueError(f"Insufficient data: need at least {self.min_samples_for_training} samples")
        
        # Clean and prepare data
        signals_clean, target_clean = self._prepare_data(signals, target)
        
        # Validate data quality
        self._validate_data_quality(signals_clean, target_clean)
        
        # Apply strategy weights if provided
        if strategy_weights:
            signals_weighted = self._apply_strategy_weights(signals_clean, strategy_weights)
        else:
            signals_weighted = signals_clean
        
        # Fit base models with cross-validation
        self._fit_base_models(signals_weighted, target_clean)
        
        # Create ensemble based on method
        if self.ensemble_method == 'voting':
            self._create_voting_ensemble(signals_weighted, target_clean)
        elif self.ensemble_method == 'stacking':
            self._create_stacking_ensemble(signals_weighted, target_clean)
        elif self.ensemble_method == 'weighted_average':
            self._create_weighted_average_ensemble(signals_weighted, target_clean)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return self
    
    def predict(self, signals: pd.DataFrame) -> pd.Series:
        """
        Predict trading signals using ensemble model.
        
        Args:
            signals: DataFrame with signals from different strategies
            
        Returns:
            Combined ensemble signals
        """
        if not hasattr(self, 'ensemble_model'):
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare signals
        signals_clean = self._prepare_signals_for_prediction(signals)
        
        if hasattr(self.ensemble_model, 'predict'):
            # Single model prediction
            predictions = self.ensemble_model.predict(signals_clean)
        else:
            # Custom ensemble prediction
            predictions = self._predict_ensemble(signals_clean)
        
        # Apply bounds to prevent extreme signals
        predictions = np.clip(predictions, -1.0, 1.0)
        
        return pd.Series(predictions, index=signals.index)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble model."""
        importance = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importance[name] = np.abs(model.coef_)
        
        # Combine importances with ensemble weights
        combined_importance = np.zeros(len(list(importance.values())[0]))
        total_weight = 0
        
        for name, imp in importance.items():
            weight = self.ensemble_weights.get(name, 1.0)
            combined_importance += weight * imp
            total_weight += weight
        
        if total_weight > 0:
            combined_importance /= total_weight
        
        return dict(zip(self._get_feature_names(), combined_importance))
    
    def evaluate_performance(self, 
                           signals: pd.DataFrame, 
                           target: pd.Series) -> Dict[str, float]:
        """
        Evaluate ensemble performance with multiple metrics.
        
        Args:
            signals: Test signals
            target: Test target returns
            
        Returns:
            Dictionary of performance metrics
        """
        predictions = self.predict(signals)
        
        # Align predictions and target
        common_index = predictions.index.intersection(target.index)
        pred_aligned = predictions.loc[common_index]
        target_aligned = target.loc[common_index]
        
        if len(pred_aligned) == 0:
            return {'error': 'No overlapping data for evaluation'}
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(target_aligned, pred_aligned),
            'mae': mean_absolute_error(target_aligned, pred_aligned),
            'correlation': np.corrcoef(target_aligned, pred_aligned)[0, 1],
            'directional_accuracy': self._calculate_directional_accuracy(
                target_aligned, pred_aligned
            ),
            'information_ratio': self._calculate_information_ratio(
                target_aligned, pred_aligned
            )
        }
        
        # Check for overfitting
        metrics['overfitting_score'] = self._calculate_overfitting_score()
        
        return metrics
    
    def _prepare_data(self, 
                     signals: pd.DataFrame, 
                     target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare and clean input data."""
        # Align indices
        common_index = signals.index.intersection(target.index)
        signals_aligned = signals.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Remove rows with missing values
        mask = ~(signals_aligned.isna().any(axis=1) | target_aligned.isna())
        signals_clean = signals_aligned[mask]
        target_clean = target_aligned[mask]
        
        # Remove infinite values
        finite_mask = np.isfinite(signals_clean).all(axis=1) & np.isfinite(target_clean)
        signals_clean = signals_clean[finite_mask]
        target_clean = target_clean[finite_mask]
        
        return signals_clean, target_clean
    
    def _validate_data_quality(self, signals: pd.DataFrame, target: pd.Series):
        """Validate data quality for training."""
        # Check for sufficient variance in target
        if target.std() < 1e-6:
            raise ValueError("Target variable has insufficient variance")
        
        # Check for sufficient variance in signals
        low_variance_signals = signals.columns[signals.std() < 1e-6]
        if len(low_variance_signals) > 0:
            print(f"Warning: Low variance signals detected: {list(low_variance_signals)}")
        
        # Check for extreme correlations between signals
        corr_matrix = signals.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print(f"Warning: High correlation pairs detected: {high_corr_pairs[:3]}")
    
    def _apply_strategy_weights(self, 
                              signals: pd.DataFrame, 
                              strategy_weights: Dict[str, float]) -> pd.DataFrame:
        """Apply weights to different strategy signals."""
        weighted_signals = signals.copy()
        
        for strategy, weight in strategy_weights.items():
            strategy_cols = [col for col in signals.columns if strategy.lower() in col.lower()]
            for col in strategy_cols:
                if col in weighted_signals.columns:
                    weighted_signals[col] *= weight
        
        return weighted_signals
    
    def _fit_base_models(self, signals: pd.DataFrame, target: pd.Series):
        """Fit base models with cross-validation."""
        # Scale features for models that need it
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        signals_scaled_std = self.scalers['standard'].fit_transform(signals)
        signals_scaled_rob = self.scalers['robust'].fit_transform(signals)
        
        # Fit each base model
        for name, model in self.base_models.items():
            try:
                if name in ['ridge', 'elastic_net', 'neural_network']:
                    # Use standard scaling for linear and neural models
                    model.fit(signals_scaled_std, target)
                    self.models[name] = model
                    
                    # Evaluate with cross-validation
                    cv_scores = cross_val_score(
                        model, signals_scaled_std, target,
                        cv=TimeSeriesSplit(n_splits=self.cv_folds),
                        scoring='neg_mean_squared_error'
                    )
                    self.validation_scores[name] = -cv_scores.mean()
                    
                else:
                    # Use original features for tree-based models
                    model.fit(signals, target)
                    self.models[name] = model
                    
                    # Evaluate with cross-validation
                    cv_scores = cross_val_score(
                        model, signals, target,
                        cv=TimeSeriesSplit(n_splits=self.cv_folds),
                        scoring='neg_mean_squared_error'
                    )
                    self.validation_scores[name] = -cv_scores.mean()
                    
            except Exception as e:
                print(f"Warning: Failed to fit {name} model: {e}")
                continue
    
    def _create_voting_ensemble(self, signals: pd.DataFrame, target: pd.Series):
        """Create voting ensemble from base models."""
        valid_models = [(name, model) for name, model in self.models.items()]
        
        if not valid_models:
            raise ValueError("No valid base models for ensemble")
        
        # Calculate weights based on validation performance
        weights = []
        for name, _ in valid_models:
            score = self.validation_scores.get(name, float('inf'))
            # Lower MSE is better, so use inverse
            weight = 1.0 / (score + 1e-6) if score != float('inf') else 0.0
            weights.append(weight)
            self.ensemble_weights[name] = weight
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            for i, (name, _) in enumerate(valid_models):
                self.ensemble_weights[name] = weights[i]
        
        # Create voting regressor
        self.ensemble_model = VotingRegressor(
            estimators=valid_models,
            weights=weights if total_weight > 0 else None
        )
        
        # Fit ensemble (this is redundant but needed for the interface)
        try:
            self.ensemble_model.fit(signals, target)
        except Exception as e:
            print(f"Warning: Ensemble fitting failed, using simple average: {e}")
            self.ensemble_method = 'weighted_average'
            self._create_weighted_average_ensemble(signals, target)
    
    def _create_stacking_ensemble(self, signals: pd.DataFrame, target: pd.Series):
        """Create stacking ensemble with meta-learner."""
        # Generate out-of-fold predictions from base models
        n_models = len(self.models)
        meta_features = np.zeros((len(signals), n_models))
        
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        for i, (name, model) in enumerate(self.models.items()):
            for train_idx, val_idx in tscv.split(signals):
                train_signals = signals.iloc[train_idx]
                train_target = target.iloc[train_idx]
                val_signals = signals.iloc[val_idx]
                
                # Fit model on training fold
                model_copy = type(model)(**model.get_params())
                
                if name in ['ridge', 'elastic_net', 'neural_network']:
                    train_scaled = self.scalers['standard'].fit_transform(train_signals)
                    val_scaled = self.scalers['standard'].transform(val_signals)
                    model_copy.fit(train_scaled, train_target)
                    meta_features[val_idx, i] = model_copy.predict(val_scaled)
                else:
                    model_copy.fit(train_signals, train_target)
                    meta_features[val_idx, i] = model_copy.predict(val_signals)
        
        # Train meta-learner on meta-features
        self.meta_learner = RidgeCV(alphas=np.logspace(-3, 2, 20))
        self.meta_learner.fit(meta_features, target)
        
        self.ensemble_model = 'stacking'  # Custom identifier
    
    def _create_weighted_average_ensemble(self, signals: pd.DataFrame, target: pd.Series):
        """Create weighted average ensemble."""
        # Calculate weights based on validation performance
        total_score = 0
        for name, score in self.validation_scores.items():
            if name in self.models and score != float('inf'):
                # Use inverse of MSE as weight
                weight = 1.0 / (score + 1e-6)
                self.ensemble_weights[name] = weight
                total_score += weight
        
        # Normalize weights
        if total_score > 0:
            for name in self.ensemble_weights:
                self.ensemble_weights[name] /= total_score
        else:
            # Equal weights fallback
            n_models = len(self.models)
            for name in self.models:
                self.ensemble_weights[name] = 1.0 / n_models
        
        self.ensemble_model = 'weighted_average'  # Custom identifier
    
    def _prepare_signals_for_prediction(self, signals: pd.DataFrame) -> np.ndarray:
        """Prepare signals for prediction."""
        # Handle missing values
        signals_filled = signals.fillna(method='ffill').fillna(0)
        
        # Get the method used during training
        if hasattr(self, 'training_method'):
            if self.training_method in ['ridge', 'elastic_net', 'neural_network']:
                return self.scalers['standard'].transform(signals_filled)
            else:
                return signals_filled.values
        
        # Default: return as is
        return signals_filled.values
    
    def _predict_ensemble(self, signals: np.ndarray) -> np.ndarray:
        """Make predictions using custom ensemble methods."""
        if self.ensemble_model == 'weighted_average':
            predictions = np.zeros(len(signals))
            
            for name, model in self.models.items():
                weight = self.ensemble_weights.get(name, 0.0)
                
                if name in ['ridge', 'elastic_net', 'neural_network']:
                    # These models expect scaled features
                    pred = model.predict(signals)
                else:
                    # Tree models use original features
                    pred = model.predict(signals)
                
                predictions += weight * pred
            
            return predictions
        
        elif self.ensemble_model == 'stacking':
            # Generate meta-features
            meta_features = np.zeros((len(signals), len(self.models)))
            
            for i, (name, model) in enumerate(self.models.items()):
                if name in ['ridge', 'elastic_net', 'neural_network']:
                    meta_features[:, i] = model.predict(signals)
                else:
                    meta_features[:, i] = model.predict(signals)
            
            # Predict using meta-learner
            return self.meta_learner.predict(meta_features)
        
        else:
            raise ValueError(f"Unknown custom ensemble method: {self.ensemble_model}")
    
    def _calculate_directional_accuracy(self, actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate directional accuracy of predictions."""
        actual_direction = np.sign(actual)
        predicted_direction = np.sign(predicted)
        correct_direction = (actual_direction == predicted_direction).sum()
        return correct_direction / len(actual) if len(actual) > 0 else 0.0
    
    def _calculate_information_ratio(self, actual: pd.Series, predicted: pd.Series) -> float:
        """Calculate information ratio of predictions."""
        excess_returns = predicted - actual.mean()
        tracking_error = excess_returns.std()
        return excess_returns.mean() / tracking_error if tracking_error != 0 else 0.0
    
    def _calculate_overfitting_score(self) -> float:
        """Calculate overfitting score based on validation performance."""
        if not self.validation_scores:
            return 0.0
        
        # Calculate the spread of validation scores
        scores = list(self.validation_scores.values())
        scores = [s for s in scores if s != float('inf')]
        
        if len(scores) < 2:
            return 0.0
        
        score_std = np.std(scores)
        score_mean = np.mean(scores)
        
        # High standard deviation relative to mean indicates potential overfitting
        return score_std / (score_mean + 1e-6) if score_mean != 0 else 1.0
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names used in training."""
        if hasattr(self, '_feature_names'):
            return self._feature_names
        else:
            # Return generic names if original names not stored
            return [f'feature_{i}' for i in range(len(self.models))]
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get summary of ensemble model."""
        return {
            'ensemble_method': self.ensemble_method,
            'base_models': list(self.models.keys()),
            'ensemble_weights': self.ensemble_weights.copy(),
            'validation_scores': self.validation_scores.copy(),
            'overfitting_score': self._calculate_overfitting_score(),
            'config': self.config.copy()
        }


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Ensemble Signal Combiner")
    print("=" * 50)
    
    # Create sample signals from different strategies
    np.random.seed(42)
    n_samples = 500
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Simulate signals from different strategies
    signals = pd.DataFrame({
        'technical_analysis_signal': np.random.normal(0, 0.5, n_samples),
        'mean_reversion_signal': np.random.normal(0, 0.3, n_samples),
        'momentum_signal': np.random.normal(0, 0.4, n_samples),
        'pattern_recognition_signal': np.random.normal(0, 0.2, n_samples),
        'earnings_signal': np.random.normal(0, 0.1, n_samples),
    }, index=dates)
    
    # Create target returns with some relationship to signals
    target = (
        0.3 * signals['technical_analysis_signal'] +
        0.2 * signals['mean_reversion_signal'] +
        0.25 * signals['momentum_signal'] +
        0.15 * signals['pattern_recognition_signal'] +
        0.1 * signals['earnings_signal'] +
        np.random.normal(0, 0.02, n_samples)  # noise
    )
    target.name = 'returns'
    
    # Split data
    train_size = int(0.8 * len(signals))
    train_signals = signals[:train_size]
    train_target = target[:train_size]
    test_signals = signals[train_size:]
    test_target = target[train_size:]
    
    print(f"📊 Training samples: {len(train_signals)}")
    print(f"📊 Test samples: {len(test_signals)}")
    
    # Test ensemble combiner
    ensemble = EnsembleSignalCombiner({
        'ensemble_method': 'voting',
        'cv_folds': 5
    })
    
    # Fit ensemble
    print("\n🚀 Training ensemble...")
    ensemble.fit(train_signals, train_target)
    
    # Make predictions
    print("🔮 Making predictions...")
    predictions = ensemble.predict(test_signals)
    
    # Evaluate performance
    print("📊 Evaluating performance...")
    performance = ensemble.evaluate_performance(test_signals, test_target)
    
    print(f"\n📈 Performance Metrics:")
    for metric, value in performance.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Show feature importance
    importance = ensemble.get_feature_importance()
    print(f"\n🎯 Feature Importance:")
    for feature, imp in importance.items():
        print(f"  {feature}: {imp:.4f}")
    
    # Show ensemble summary
    summary = ensemble.get_ensemble_summary()
    print(f"\n📋 Ensemble Summary:")
    print(f"  Method: {summary['ensemble_method']}")
    print(f"  Models: {summary['base_models']}")
    print(f"  Overfitting Score: {summary['overfitting_score']:.4f}")