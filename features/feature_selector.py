#!/usr/bin/env python3
"""
Feature Selection and Aggregation for Unified Trading Strategy

This module provides advanced feature selection, aggregation, and noise reduction
techniques to create robust feature sets for trading models while avoiding overfitting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.feature_selection import (
    SelectKBest, RFE, f_regression, mutual_info_regression
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Advanced feature selection with multiple methods to avoid overfitting
    and reduce noise in trading signals.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize feature selector with configuration.
        
        Args:
            config: Configuration dictionary with selection parameters
        """
        self.config = config or {}
        self.selection_results = {}
        self.feature_importance_ = {}
        self.selected_features_ = []
        self.scaler = StandardScaler()
        
        # Configuration parameters
        self.max_features = self.config.get('max_features', 50)
        self.regularization_alpha = self.config.get('regularization_alpha', 0.01)
        self.stability_threshold = self.config.get('stability_threshold', 0.8)
        self.correlation_threshold = self.config.get('correlation_threshold', 0.95)
        self.noise_reduction_window = self.config.get('noise_reduction_window', 5)
    
    def select_features(self, 
                       features: pd.DataFrame, 
                       target: pd.Series,
                       method: str = 'combined') -> pd.DataFrame:
        """
        Select optimal features using specified method.
        
        Args:
            features: Input feature matrix
            target: Target variable (future returns)
            method: Selection method ('lasso', 'elastic_net', 'recursive', 
                   'mutual_info', 'combined')
        
        Returns:
            DataFrame with selected features
        """
        if features.empty or target.empty:
            return features
        
        # Clean and align data
        features_clean, target_clean = self._clean_and_align_data(features, target)
        
        if method == 'combined':
            return self._combined_selection(features_clean, target_clean)
        elif method == 'lasso':
            return self._lasso_selection(features_clean, target_clean)
        elif method == 'elastic_net':
            return self._elastic_net_selection(features_clean, target_clean)
        elif method == 'recursive':
            return self._recursive_selection(features_clean, target_clean)
        elif method == 'mutual_info':
            return self._mutual_info_selection(features_clean, target_clean)
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def reduce_noise(self, 
                    data: pd.DataFrame, 
                    feature_types: Dict[str, List[str]] = None) -> pd.DataFrame:
        """
        Reduce noise in features using various techniques.
        
        Args:
            data: Input data with features
            feature_types: Dictionary mapping feature types to column names
        
        Returns:
            DataFrame with noise-reduced features
        """
        result_df = data.copy()
        
        # Default feature type mapping
        if feature_types is None:
            feature_types = self._infer_feature_types(data.columns)
        
        # Apply noise reduction by feature type
        for feature_type, columns in feature_types.items():
            if feature_type == 'sentiment':
                result_df = self._reduce_sentiment_noise(result_df, columns)
            elif feature_type == 'price':
                result_df = self._reduce_price_noise(result_df, columns)
            elif feature_type == 'volume':
                result_df = self._reduce_volume_noise(result_df, columns)
            elif feature_type == 'technical':
                result_df = self._reduce_technical_noise(result_df, columns)
        
        # Remove highly correlated features
        result_df = self._remove_correlated_features(result_df)
        
        return result_df
    
    def create_composite_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite features that combine multiple signals.
        
        Args:
            data: Input data
            
        Returns:
            DataFrame with additional composite features
        """
        result_df = data.copy()
        
        # Technical analysis composite
        technical_cols = [col for col in data.columns if any(
            indicator in col.lower() for indicator in 
            ['rsi', 'macd', 'bb_', 'stoch', 'williams', 'atr']
        )]
        
        if len(technical_cols) >= 3:
            # Normalize technical indicators to 0-1 scale
            tech_normalized = self._normalize_features(data[technical_cols])
            result_df['technical_composite'] = tech_normalized.mean(axis=1)
            result_df['technical_momentum'] = tech_normalized.diff().mean(axis=1)
        
        # Trend strength composite
        trend_cols = [col for col in data.columns if 'trend' in col.lower()]
        if trend_cols:
            result_df['trend_composite'] = data[trend_cols].mean(axis=1)
        
        # Volatility composite
        vol_cols = [col for col in data.columns if any(
            vol_term in col.lower() for vol_term in 
            ['volatility', 'atr', 'bb_width']
        )]
        if vol_cols:
            result_df['volatility_composite'] = data[vol_cols].mean(axis=1)
        
        # Market sentiment composite (if available)
        sentiment_cols = [col for col in data.columns if any(
            sent_term in col.lower() for sent_term in 
            ['sentiment', 'fear', 'greed', 'news']
        )]
        if sentiment_cols:
            result_df['sentiment_composite'] = data[sentiment_cols].mean(axis=1)
        
        # Earnings-related composite (if available)
        earnings_cols = [col for col in data.columns if 'earnings' in col.lower()]
        if earnings_cols:
            result_df['earnings_composite'] = data[earnings_cols].mean(axis=1)
        
        return result_df
    
    def _clean_and_align_data(self, 
                             features: pd.DataFrame, 
                             target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean and align features and target data."""
        # Align indices
        common_index = features.index.intersection(target.index)
        features_aligned = features.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Remove infinite and NaN values
        mask = np.isfinite(features_aligned).all(axis=1) & np.isfinite(target_aligned)
        features_clean = features_aligned[mask]
        target_clean = target_aligned[mask]
        
        # Remove constant features
        feature_variance = features_clean.var()
        non_constant_features = feature_variance[feature_variance > 1e-8].index
        features_clean = features_clean[non_constant_features]
        
        return features_clean, target_clean
    
    def _combined_selection(self, 
                          features: pd.DataFrame, 
                          target: pd.Series) -> pd.DataFrame:
        """Combine multiple selection methods for robust feature selection."""
        if len(features) < 50:  # Not enough data for robust selection
            return features
        
        # Method 1: Lasso selection
        lasso_features = self._lasso_selection(features, target, return_names=True)
        
        # Method 2: Recursive feature elimination
        rfe_features = self._recursive_selection(features, target, return_names=True)
        
        # Method 3: Mutual information
        mi_features = self._mutual_info_selection(features, target, return_names=True)
        
        # Combine results using voting
        all_methods = [lasso_features, rfe_features, mi_features]
        feature_votes = {}
        
        for method_features in all_methods:
            for feature in method_features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Select features that appear in at least 2 methods
        selected_features = [
            feature for feature, votes in feature_votes.items() 
            if votes >= 2
        ]
        
        # If too few features selected, use top features by vote count
        if len(selected_features) < 10:
            sorted_features = sorted(
                feature_votes.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            selected_features = [f[0] for f in sorted_features[:self.max_features//2]]
        
        # Limit to max_features
        selected_features = selected_features[:self.max_features]
        
        self.selected_features_ = selected_features
        return features[selected_features] if selected_features else features
    
    def _lasso_selection(self, 
                        features: pd.DataFrame, 
                        target: pd.Series,
                        return_names: bool = False) -> Union[pd.DataFrame, List[str]]:
        """Select features using Lasso regularization."""
        try:
            # Scale features for Lasso
            features_scaled = self.scaler.fit_transform(features)
            
            # Lasso with cross-validation
            lasso = LassoCV(
                alphas=np.logspace(-4, 0, 20),
                cv=max(3, min(5, len(features) // 10)),
                random_state=42
            )
            lasso.fit(features_scaled, target)
            
            # Select features with non-zero coefficients
            selected_mask = lasso.coef_ != 0
            selected_features = features.columns[selected_mask].tolist()
            
            self.feature_importance_['lasso'] = dict(zip(
                features.columns, np.abs(lasso.coef_)
            ))
            
            if return_names:
                return selected_features
            return features[selected_features] if selected_features else features
            
        except Exception as e:
            print(f"Warning: Lasso selection failed: {e}")
            if return_names:
                return features.columns.tolist()
            return features
    
    def _elastic_net_selection(self, 
                             features: pd.DataFrame, 
                             target: pd.Series,
                             return_names: bool = False) -> Union[pd.DataFrame, List[str]]:
        """Select features using Elastic Net regularization."""
        try:
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Elastic Net with cross-validation
            elastic_net = ElasticNetCV(
                alphas=np.logspace(-4, 0, 20),
                l1_ratio=[0.1, 0.5, 0.7, 0.9],
                cv=min(5, len(features) // 10),
                random_state=42
            )
            elastic_net.fit(features_scaled, target)
            
            # Select features with non-zero coefficients
            selected_mask = elastic_net.coef_ != 0
            selected_features = features.columns[selected_mask].tolist()
            
            self.feature_importance_['elastic_net'] = dict(zip(
                features.columns, np.abs(elastic_net.coef_)
            ))
            
            if return_names:
                return selected_features
            return features[selected_features] if selected_features else features
            
        except Exception as e:
            print(f"Warning: Elastic Net selection failed: {e}")
            if return_names:
                return features.columns.tolist()
            return features
    
    def _recursive_selection(self, 
                           features: pd.DataFrame, 
                           target: pd.Series,
                           return_names: bool = False) -> Union[pd.DataFrame, List[str]]:
        """Select features using Recursive Feature Elimination."""
        try:
            # Use Random Forest as base estimator
            estimator = RandomForestRegressor(
                n_estimators=50, 
                random_state=42, 
                n_jobs=-1
            )
            
            # Recursive feature elimination
            n_features_to_select = min(self.max_features, len(features.columns) // 2)
            rfe = RFE(estimator, n_features_to_select=n_features_to_select)
            rfe.fit(features, target)
            
            selected_features = features.columns[rfe.support_].tolist()
            
            # Get feature importance from the estimator
            self.feature_importance_['rfe'] = dict(zip(
                selected_features, 
                estimator.feature_importances_[rfe.support_]
            ))
            
            if return_names:
                return selected_features
            return features[selected_features] if selected_features else features
            
        except Exception as e:
            print(f"Warning: RFE selection failed: {e}")
            if return_names:
                return features.columns.tolist()
            return features
    
    def _mutual_info_selection(self, 
                             features: pd.DataFrame, 
                             target: pd.Series,
                             return_names: bool = False) -> Union[pd.DataFrame, List[str]]:
        """Select features using mutual information."""
        try:
            # Calculate mutual information
            mi_scores = mutual_info_regression(features, target, random_state=42)
            
            # Select top features by mutual information
            n_features_to_select = min(self.max_features, len(features.columns) // 2)
            selector = SelectKBest(
                score_func=mutual_info_regression, 
                k=n_features_to_select
            )
            selector.fit(features, target)
            
            selected_features = features.columns[selector.get_support()].tolist()
            
            self.feature_importance_['mutual_info'] = dict(zip(
                features.columns, mi_scores
            ))
            
            if return_names:
                return selected_features
            return features[selected_features] if selected_features else features
            
        except Exception as e:
            print(f"Warning: Mutual information selection failed: {e}")
            if return_names:
                return features.columns.tolist()
            return features
    
    def _infer_feature_types(self, columns: List[str]) -> Dict[str, List[str]]:
        """Infer feature types from column names."""
        feature_types = {
            'sentiment': [],
            'price': [],
            'volume': [],
            'technical': []
        }
        
        for col in columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['sentiment', 'news', 'fear', 'greed']):
                feature_types['sentiment'].append(col)
            elif any(term in col_lower for term in ['price', 'close', 'open', 'high', 'low']):
                feature_types['price'].append(col)
            elif any(term in col_lower for term in ['volume', 'obv', 'vwap']):
                feature_types['volume'].append(col)
            else:
                feature_types['technical'].append(col)
        
        return feature_types
    
    def _reduce_sentiment_noise(self, 
                              data: pd.DataFrame, 
                              columns: List[str]) -> pd.DataFrame:
        """Reduce noise in sentiment features."""
        if not columns:
            return data
        
        result_df = data.copy()
        
        for col in columns:
            if col in result_df.columns:
                # Apply rolling average to smooth sentiment
                result_df[f'{col}_smoothed'] = (
                    result_df[col].rolling(window=self.noise_reduction_window).mean()
                )
                
                # Create sentiment momentum
                result_df[f'{col}_momentum'] = (
                    result_df[col].diff().rolling(window=3).mean()
                )
        
        return result_df
    
    def _reduce_price_noise(self, 
                          data: pd.DataFrame, 
                          columns: List[str]) -> pd.DataFrame:
        """Reduce noise in price features."""
        if not columns:
            return data
        
        result_df = data.copy()
        
        for col in columns:
            if col in result_df.columns:
                # Apply rolling median to reduce price spikes
                result_df[f'{col}_denoised'] = (
                    result_df[col].rolling(window=self.noise_reduction_window).median()
                )
        
        return result_df
    
    def _reduce_volume_noise(self, 
                           data: pd.DataFrame, 
                           columns: List[str]) -> pd.DataFrame:
        """Reduce noise in volume features."""
        if not columns:
            return data
        
        result_df = data.copy()
        
        for col in columns:
            if col in result_df.columns:
                # Log transform and smooth volume
                volume_log = np.log1p(result_df[col])
                result_df[f'{col}_log_smoothed'] = (
                    volume_log.rolling(window=self.noise_reduction_window).mean()
                )
        
        return result_df
    
    def _reduce_technical_noise(self, 
                              data: pd.DataFrame, 
                              columns: List[str]) -> pd.DataFrame:
        """Reduce noise in technical indicator features."""
        if not columns:
            return data
        
        result_df = data.copy()
        
        for col in columns:
            if col in result_df.columns:
                # Apply EMA smoothing for technical indicators
                alpha = 2 / (self.noise_reduction_window + 1)
                result_df[f'{col}_ema'] = (
                    result_df[col].ewm(alpha=alpha).mean()
                )
        
        return result_df
    
    def _remove_correlated_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return data
        
        # Calculate correlation matrix
        corr_matrix = data[numeric_cols].corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Identify features to drop
        to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > self.correlation_threshold)
        ]
        
        # Keep the original columns that aren't dropped
        columns_to_keep = [col for col in data.columns if col not in to_drop]
        
        return data[columns_to_keep]
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to 0-1 scale."""
        return (data - data.min()) / (data.max() - data.min()).replace(0, 1)
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from all selection methods."""
        return self.feature_importance_.copy()
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of feature selection process."""
        return {
            'selected_features': self.selected_features_.copy(),
            'num_selected_features': len(self.selected_features_),
            'feature_importance': self.feature_importance_.copy(),
            'config': self.config.copy()
        }


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Feature Selector")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create features with noise and correlations
    features = pd.DataFrame({
        'rsi': np.random.uniform(0, 100, n_samples),
        'macd': np.random.normal(0, 1, n_samples),
        'bb_position': np.random.uniform(0, 1, n_samples),
        'volume_ratio': np.random.lognormal(0, 0.5, n_samples),
        'sentiment_score': np.random.normal(0, 1, n_samples),
        'news_count': np.random.poisson(5, n_samples),
    }, index=dates)
    
    # Add correlated features
    features['rsi_correlated'] = features['rsi'] * 0.95 + np.random.normal(0, 5, n_samples)
    features['macd_correlated'] = features['macd'] * 0.98 + np.random.normal(0, 0.1, n_samples)
    
    # Create target (future returns)
    target = (
        0.001 * features['rsi'] - 
        0.002 * features['macd'] + 
        0.001 * features['sentiment_score'] + 
        np.random.normal(0, 0.01, n_samples)
    )
    target.name = 'returns'
    
    # Test feature selector
    selector = FeatureSelector({
        'max_features': 6,
        'correlation_threshold': 0.9
    })
    
    print(f"📊 Original features: {len(features.columns)}")
    
    # Test noise reduction
    features_denoised = selector.reduce_noise(features)
    print(f"🔧 After noise reduction: {len(features_denoised.columns)}")
    
    # Test composite features
    features_composite = selector.create_composite_features(features_denoised)
    print(f"🎯 After composite features: {len(features_composite.columns)}")
    
    # Test feature selection
    features_selected = selector.select_features(features_composite, target)
    print(f"✨ After feature selection: {len(features_selected.columns)}")
    
    # Show selected features
    print(f"\n📋 Selected features: {list(features_selected.columns)}")
    
    # Show feature importance
    importance = selector.get_feature_importance()
    for method, features_imp in importance.items():
        print(f"\n🎯 {method.upper()} importance:")
        sorted_features = sorted(features_imp.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_features[:5]:
            print(f"  {feature}: {imp:.4f}")