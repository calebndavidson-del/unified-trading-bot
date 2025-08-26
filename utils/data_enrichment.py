#!/usr/bin/env python3
"""
Data Enrichment Utilities
Feature engineering, normalization, and meta-data enhancement for trading data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')


class DataEnricher:
    """Comprehensive data enrichment and feature engineering"""
    
    def __init__(self):
        self.enrichment_log = []
        self.feature_cache = {}
    
    def add_rolling_features(self, data: pd.DataFrame, windows: List[int] = None,
                            columns: List[str] = None) -> pd.DataFrame:
        """Add rolling statistical features"""
        if data.empty:
            return data
        
        if windows is None:
            windows = [5, 10, 20, 50]
        
        if columns is None:
            # Default to price columns
            columns = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                      if col in data.columns]
        
        df = data.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                if len(df) < window:
                    continue
                
                # Rolling mean
                df[f'{col}_sma_{window}'] = df[col].rolling(window=window).mean()
                
                # Rolling standard deviation
                df[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
                
                # Rolling min/max
                df[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
                df[f'{col}_max_{window}'] = df[col].rolling(window=window).max()
                
                # Rolling percentage change
                df[f'{col}_pct_change_{window}'] = df[col].pct_change(periods=window)
                
                # Position relative to rolling window
                if col in ['Close', 'Open']:
                    df[f'{col}_position_{window}'] = (
                        (df[col] - df[f'{col}_min_{window}']) / 
                        (df[f'{col}_max_{window}'] - df[f'{col}_min_{window}'])
                    )
        
        feature_count = len([col for col in df.columns if any(w in col for w in [f'_{w}' for w in windows])])
        self._log_action(f"Added {feature_count} rolling features for windows {windows}")
        
        return df
    
    def add_volatility_features(self, data: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """Add volatility-based features"""
        if data.empty or 'Close' not in data.columns:
            return data
        
        if windows is None:
            windows = [10, 20, 50]
        
        df = data.copy()
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        for window in windows:
            if len(df) < window:
                continue
            
            # Realized volatility
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
            
            # GARCH-like volatility (exponentially weighted)
            df[f'ewm_volatility_{window}'] = df['returns'].ewm(span=window).std() * np.sqrt(252)
            
            # Parkinson volatility (using High-Low)
            if all(col in df.columns for col in ['High', 'Low']):
                hl_ratio = np.log(df['High'] / df['Low'])
                df[f'parkinson_vol_{window}'] = np.sqrt(hl_ratio.rolling(window=window).mean() / (4 * np.log(2)))
            
            # Volatility regime (high/low volatility periods)
            vol_median = df[f'volatility_{window}'].rolling(window=window*2).median()
            df[f'vol_regime_{window}'] = (df[f'volatility_{window}'] > vol_median).astype(int)
        
        self._log_action(f"Added volatility features for windows {windows}")
        
        return df
    
    def add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features"""
        if data.empty:
            return data
        
        df = data.copy()
        price_cols = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns]
        
        if not price_cols:
            return df
        
        # Price-based features
        if 'Close' in df.columns:
            # Gap analysis
            if 'Open' in df.columns:
                df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
                df['gap_up'] = (df['gap'] > 0.02).astype(int)
                df['gap_down'] = (df['gap'] < -0.02).astype(int)
            
            # True Range
            if all(col in df.columns for col in ['High', 'Low', 'Close']):
                df['true_range'] = np.maximum(
                    df['High'] - df['Low'],
                    np.maximum(
                        np.abs(df['High'] - df['Close'].shift(1)),
                        np.abs(df['Low'] - df['Close'].shift(1))
                    )
                )
                
                # Average True Range
                df['atr_14'] = df['true_range'].rolling(window=14).mean()
                df['atr_20'] = df['true_range'].rolling(window=20).mean()
        
        # Momentum features
        if 'Close' in df.columns:
            # Rate of Change
            for period in [5, 10, 20]:
                df[f'roc_{period}'] = ((df['Close'] - df['Close'].shift(period)) / 
                                      df['Close'].shift(period)) * 100
            
            # Relative Strength Index approximation
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD approximation
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Volume features
        if 'Volume' in df.columns and 'Close' in df.columns:
            # Volume-weighted features
            df['vwap_20'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
            
            # Volume rate of change
            df['volume_roc'] = df['Volume'].pct_change(periods=5)
            
            # On-Balance Volume approximation
            price_change = df['Close'].diff()
            df['obv'] = (np.sign(price_change) * df['Volume']).cumsum()
            
            # Volume trend
            df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_trend'] = df['Volume'] / df['volume_sma_20']
        
        # Market structure features
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            # Higher highs, lower lows
            df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
            df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
            df['higher_low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
            df['lower_high'] = (df['High'] < df['High'].shift(1)).astype(int)
            
            # Trend structure score
            df['trend_score'] = (df['higher_high'] + df['higher_low'] - 
                               df['lower_low'] - df['lower_high'])
            df['trend_score_5'] = df['trend_score'].rolling(window=5).sum()
        
        feature_count = len(df.columns) - len(data.columns)
        self._log_action(f"Added {feature_count} technical analysis features")
        
        return df
    
    def add_regime_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        if data.empty or 'Close' not in data.columns:
            return data
        
        df = data.copy()
        
        # Trend regime based on moving averages
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        
        # Trend regime
        df['trend_regime'] = 0  # Neutral
        df.loc[df['Close'] > df['sma_20'], 'trend_regime'] = 1  # Bullish
        df.loc[df['Close'] < df['sma_20'], 'trend_regime'] = -1  # Bearish
        
        # Long-term trend
        df['long_trend'] = 0
        if 'sma_200' in df.columns:
            df.loc[df['sma_20'] > df['sma_200'], 'long_trend'] = 1
            df.loc[df['sma_20'] < df['sma_200'], 'long_trend'] = -1
        
        # Volatility regime
        if 'volatility_20' in df.columns:
            vol_median = df['volatility_20'].rolling(window=100).median()
            df['vol_regime'] = (df['volatility_20'] > vol_median).astype(int)
        
        # Market phase (accumulation, markup, distribution, markdown)
        if all(col in df.columns for col in ['Volume', 'Close']):
            # Simplified Wyckoff-inspired phases
            price_trend = df['Close'].rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            volume_trend = df['Volume'].rolling(window=20).mean()
            volume_avg = df['Volume'].rolling(window=50).mean()
            
            df['market_phase'] = 0  # Neutral
            
            # Accumulation: sideways price, high volume
            accumulation = (abs(price_trend) < 0.01) & (volume_trend > volume_avg)
            df.loc[accumulation, 'market_phase'] = 1
            
            # Markup: rising price, declining volume
            markup = (price_trend > 0.01) & (volume_trend < volume_avg)
            df.loc[markup, 'market_phase'] = 2
            
            # Distribution: sideways price, high volume
            distribution = (abs(price_trend) < 0.01) & (volume_trend > volume_avg) & (df['Close'] > df['sma_20'])
            df.loc[distribution, 'market_phase'] = 3
            
            # Markdown: falling price, increasing volume
            markdown = (price_trend < -0.01)
            df.loc[markdown, 'market_phase'] = 4
        
        self._log_action("Added market regime detection features")
        
        return df
    
    def add_sentiment_features(self, data: pd.DataFrame, news_data: pd.DataFrame = None,
                              social_data: Dict = None) -> pd.DataFrame:
        """Add sentiment-based features"""
        if data.empty:
            return data
        
        df = data.copy()
        
        # VIX-like fear/greed indicator from price data
        if 'Close' in df.columns:
            returns = df['Close'].pct_change()
            fear_greed = returns.rolling(window=20).std() / returns.rolling(window=20).mean()
            df['fear_greed_index'] = fear_greed.rolling(window=10).mean()
        
        # News sentiment features
        if news_data is not None and not news_data.empty:
            # Aggregate news sentiment by date
            if 'datetime' in news_data.columns and 'sentiment' in news_data.columns:
                news_daily = news_data.set_index('datetime').resample('D')['sentiment'].agg([
                    'mean', 'count', 'std'
                ]).fillna(0)
                
                # Merge with price data
                df = df.join(news_daily, rsuffix='_news')
                df = df.rename(columns={
                    'mean': 'news_sentiment',
                    'count': 'news_count',
                    'std': 'news_sentiment_std'
                })
        
        # Social media sentiment features
        if social_data:
            df['social_sentiment'] = social_data.get('sentiment_score', 0)
            df['social_mention_count'] = social_data.get('total_mention', 0)
            df['social_positive_ratio'] = social_data.get('positive_mention', 0) / max(social_data.get('total_mention', 1), 1)
        
        # Market sentiment indicators from price action
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            # Doji-like indecision
            body_size = abs(df['Close'] - df['Open']) if 'Open' in df.columns else abs(df['Close'] - df['Close'].shift(1))
            total_range = df['High'] - df['Low']
            df['indecision_ratio'] = body_size / total_range.replace(0, np.nan)
            
            # Buying/selling pressure
            df['buying_pressure'] = (df['Close'] - df['Low']) / total_range.replace(0, np.nan)
            df['selling_pressure'] = (df['High'] - df['Close']) / total_range.replace(0, np.nan)
        
        feature_count = len(df.columns) - len(data.columns)
        self._log_action(f"Added {feature_count} sentiment features")
        
        return df
    
    def add_meta_tags(self, data: pd.DataFrame, symbol_info: Dict = None) -> pd.DataFrame:
        """Add meta-data tags for asset classification"""
        if data.empty:
            return data
        
        df = data.copy()
        
        if symbol_info:
            # Add basic asset information
            df['sector'] = symbol_info.get('sector', 'Unknown')
            df['industry'] = symbol_info.get('industry', 'Unknown')
            df['country'] = symbol_info.get('country', 'Unknown')
            df['currency'] = symbol_info.get('currency', 'USD')
            df['exchange'] = symbol_info.get('exchange', 'Unknown')
            df['market_cap_category'] = self._categorize_market_cap(symbol_info.get('market_cap'))
            df['asset_type'] = self._determine_asset_type(symbol_info.get('symbol', ''))
        
        # Time-based tags
        try:
            if hasattr(df, 'index') and hasattr(df.index, 'dayofweek'):
                df['day_of_week'] = df.index.dayofweek
                df['month'] = df.index.month
                df['quarter'] = df.index.quarter
                df['is_month_end'] = df.index.is_month_end.astype(int)
                df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
                df['is_year_end'] = df.index.is_year_end.astype(int)
        except (AttributeError, TypeError):
            # Handle non-datetime indices
            pass
        
        # Market session tags (simplified)
        try:
            if hasattr(df, 'index') and hasattr(df.index, 'hour'):
                df['market_session'] = 0  # Pre-market
                market_hours_mask = (df.index.hour >= 9) & (df.index.hour <= 16)
                after_hours_mask = (df.index.hour >= 16) & (df.index.hour <= 20)
                df.loc[market_hours_mask, 'market_session'] = 1  # Market hours
                df.loc[after_hours_mask, 'market_session'] = 2  # After hours
            elif hasattr(df, 'index'):
                # For daily data, assume regular market hours
                df['market_session'] = 1
        except (AttributeError, TypeError):
            # Fallback for any index issues
            df['market_session'] = 1
        
        # Volatility environment tags
        if 'volatility_20' in df.columns:
            vol_percentile = df['volatility_20'].rolling(window=252).rank(pct=True)
            df['vol_environment'] = 0  # Low vol
            df.loc[vol_percentile > 0.66, 'vol_environment'] = 1  # Medium vol
            df.loc[vol_percentile > 0.90, 'vol_environment'] = 2  # High vol
        
        self._log_action("Added meta-data tags")
        
        return df
    
    def normalize_features(self, data: pd.DataFrame, method: str = 'minmax',
                          columns: List[str] = None, fit_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict]:
        """Normalize features using various methods"""
        if data.empty:
            return data, {}
        
        df = data.copy()
        
        if columns is None:
            # Auto-select numeric columns, excluding meta tags
            exclude_patterns = ['sector', 'industry', 'country', 'currency', 'exchange', 
                              'asset_type', 'market_session', 'day_of_week', 'month', 'quarter']
            columns = [col for col in df.select_dtypes(include=[np.number]).columns
                      if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        normalization_params = {}
        
        # Use fit_data for parameters if provided, otherwise use data itself
        fit_df = fit_data if fit_data is not None else df
        
        for col in columns:
            if col not in df.columns or col not in fit_df.columns:
                continue
            
            if method == 'minmax':
                min_val = fit_df[col].min()
                max_val = fit_df[col].max()
                
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    normalization_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                
            elif method == 'zscore':
                mean_val = fit_df[col].mean()
                std_val = fit_df[col].std()
                
                if std_val != 0:
                    df[col] = (df[col] - mean_val) / std_val
                    normalization_params[col] = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
                
            elif method == 'robust':
                median_val = fit_df[col].median()
                mad_val = np.median(np.abs(fit_df[col] - median_val))
                
                if mad_val != 0:
                    df[col] = (df[col] - median_val) / mad_val
                    normalization_params[col] = {'method': 'robust', 'median': median_val, 'mad': mad_val}
        
        normalized_count = len(normalization_params)
        self._log_action(f"Normalized {normalized_count} features using {method} method")
        
        return df, normalization_params
    
    def create_feature_interactions(self, data: pd.DataFrame, max_interactions: int = 50) -> pd.DataFrame:
        """Create feature interactions and polynomial features"""
        if data.empty:
            return data
        
        df = data.copy()
        
        # Select numeric columns for interactions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove already computed features to avoid redundancy
        base_cols = [col for col in numeric_cols if not any(
            suffix in col for suffix in ['_sma_', '_std_', '_min_', '_max_', '_roc_', '_vol_']
        )][:10]  # Limit to first 10 base columns
        
        interaction_count = 0
        
        # Create pairwise interactions
        for i, col1 in enumerate(base_cols):
            if interaction_count >= max_interactions:
                break
                
            for j, col2 in enumerate(base_cols[i+1:], i+1):
                if interaction_count >= max_interactions:
                    break
                
                # Multiplicative interaction
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                interaction_count += 1
                
                # Ratio interaction (if col2 is not zero)
                if (df[col2] != 0).all():
                    df[f'{col1}_div_{col2}'] = df[col1] / df[col2]
                    interaction_count += 1
        
        # Create polynomial features for key variables
        key_vars = ['Close', 'Volume', 'volatility_20']
        for var in key_vars:
            if var in df.columns and interaction_count < max_interactions:
                df[f'{var}_squared'] = df[var] ** 2
                df[f'{var}_log'] = np.log(df[var].replace(0, np.nan))
                interaction_count += 2
        
        self._log_action(f"Created {interaction_count} feature interactions")
        
        return df
    
    def _categorize_market_cap(self, market_cap: Optional[float]) -> str:
        """Categorize market capitalization"""
        if market_cap is None:
            return 'Unknown'
        
        if market_cap >= 200e9:  # $200B+
            return 'Mega Cap'
        elif market_cap >= 10e9:  # $10B+
            return 'Large Cap'
        elif market_cap >= 2e9:   # $2B+
            return 'Mid Cap'
        elif market_cap >= 300e6: # $300M+
            return 'Small Cap'
        else:
            return 'Micro Cap'
    
    def _determine_asset_type(self, symbol: str) -> str:
        """Determine asset type from symbol"""
        symbol = symbol.upper()
        
        if any(crypto in symbol for crypto in ['BTC', 'ETH', 'ADA', 'SOL', 'DOGE']):
            return 'Cryptocurrency'
        elif any(etf in symbol for etf in ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO']):
            return 'ETF'
        elif any(forex in symbol for forex in ['USD', 'EUR', 'GBP', 'JPY']):
            return 'Forex'
        elif any(commodity in symbol for commodity in ['GLD', 'SLV', 'OIL', 'GAS']):
            return 'Commodity'
        else:
            return 'Stock'
    
    def _log_action(self, message: str):
        """Log enrichment actions"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.enrichment_log.append(log_entry)
    
    def get_enrichment_log(self) -> List[str]:
        """Get the enrichment action log"""
        return self.enrichment_log.copy()
    
    def clear_log(self):
        """Clear the enrichment log"""
        self.enrichment_log.clear()


if __name__ == "__main__":
    # Example usage
    enricher = DataEnricher()
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    print("Original data shape:", sample_data.shape)
    
    # Add features
    enriched_data = enricher.add_rolling_features(sample_data)
    enriched_data = enricher.add_volatility_features(enriched_data)
    enriched_data = enricher.add_technical_features(enriched_data)
    
    print("Enriched data shape:", enriched_data.shape)
    print(f"Added {enriched_data.shape[1] - sample_data.shape[1]} features")
    
    print("\nEnrichment log:")
    for entry in enricher.get_enrichment_log():
        print(entry)