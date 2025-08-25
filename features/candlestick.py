#!/usr/bin/env python3
"""
Candlestick Pattern Feature Extraction Utilities
Advanced pattern recognition for trading signals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
# import talib  # Not available - using manual calculations
try:
    from scipy.signal import find_peaks
except ImportError:
    # Fallback implementation if scipy is not available
    def find_peaks(data, distance=None):
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
        return np.array(peaks), {}

import warnings
warnings.filterwarnings('ignore')


class CandlestickPatternExtractor:
    """Extract candlestick patterns and related features"""
    
    def __init__(self):
        # Since talib is not available, we'll implement simplified pattern detection
        self.pattern_functions = {
            'doji': self._detect_doji,
            'hammer': self._detect_hammer,
            'shooting_star': self._detect_shooting_star,
            'engulfing': self._detect_engulfing,
            'harami': self._detect_harami
        }
    
    def _detect_doji(self, open_prices, high_prices, low_prices, close_prices):
        """Simplified doji detection"""
        body_size = np.abs(close_prices - open_prices)
        range_size = high_prices - low_prices
        doji_threshold = 0.1  # 10% of the range
        return np.where(body_size <= doji_threshold * range_size, 100, 0)
    
    def _detect_hammer(self, open_prices, high_prices, low_prices, close_prices):
        """Simplified hammer detection"""
        body_size = np.abs(close_prices - open_prices)
        lower_shadow = np.minimum(open_prices, close_prices) - low_prices
        upper_shadow = high_prices - np.maximum(open_prices, close_prices)
        
        # Hammer conditions: small body, long lower shadow, small upper shadow
        hammer_condition = (
            (lower_shadow >= 2 * body_size) & 
            (upper_shadow <= 0.1 * lower_shadow) &
            (close_prices > open_prices)  # Bullish hammer
        )
        return np.where(hammer_condition, 100, 0)
    
    def _detect_shooting_star(self, open_prices, high_prices, low_prices, close_prices):
        """Simplified shooting star detection"""
        body_size = np.abs(close_prices - open_prices)
        lower_shadow = np.minimum(open_prices, close_prices) - low_prices
        upper_shadow = high_prices - np.maximum(open_prices, close_prices)
        
        # Shooting star conditions: small body, long upper shadow, small lower shadow
        shooting_star_condition = (
            (upper_shadow >= 2 * body_size) & 
            (lower_shadow <= 0.1 * upper_shadow) &
            (close_prices < open_prices)  # Bearish shooting star
        )
        return np.where(shooting_star_condition, -100, 0)
    
    def _detect_engulfing(self, open_prices, high_prices, low_prices, close_prices):
        """Simplified engulfing pattern detection"""
        results = np.zeros(len(close_prices))
        
        for i in range(1, len(close_prices)):
            current_body = close_prices[i] - open_prices[i]
            prev_body = close_prices[i-1] - open_prices[i-1]
            
            # Bullish engulfing
            if (current_body > 0 and prev_body < 0 and 
                open_prices[i] < close_prices[i-1] and 
                close_prices[i] > open_prices[i-1]):
                results[i] = 100
            # Bearish engulfing
            elif (current_body < 0 and prev_body > 0 and 
                  open_prices[i] > close_prices[i-1] and 
                  close_prices[i] < open_prices[i-1]):
                results[i] = -100
        
        return results
    
    def _detect_harami(self, open_prices, high_prices, low_prices, close_prices):
        """Simplified harami pattern detection"""
        results = np.zeros(len(close_prices))
        
        for i in range(1, len(close_prices)):
            current_high = np.maximum(open_prices[i], close_prices[i])
            current_low = np.minimum(open_prices[i], close_prices[i])
            prev_high = np.maximum(open_prices[i-1], close_prices[i-1])
            prev_low = np.minimum(open_prices[i-1], close_prices[i-1])
            
            # Harami condition: current candle is inside previous candle
            if (current_high < prev_high and current_low > prev_low):
                results[i] = 50  # Neutral harami
        
        return results
    
    def extract_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all candlestick patterns"""
        patterns_df = df.copy()
        
        # Ensure required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Extract each pattern
        for pattern_name, pattern_func in self.pattern_functions.items():
            try:
                patterns_df[f'pattern_{pattern_name}'] = pattern_func(
                    df['Open'].values,
                    df['High'].values,
                    df['Low'].values,
                    df['Close'].values
                )
            except Exception as e:
                print(f"Error extracting pattern {pattern_name}: {e}")
                patterns_df[f'pattern_{pattern_name}'] = 0
        
        return patterns_df
    
    def get_pattern_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern strength and reliability scores"""
        result_df = df.copy()
        
        # Bullish pattern count
        bullish_patterns = [
            'hammer', 'inverted_hammer', 'engulfing', 'harami', 'piercing',
            'morning_star', 'three_white_soldiers', 'three_inside_up'
        ]
        
        # Bearish pattern count
        bearish_patterns = [
            'shooting_star', 'hanging_man', 'dark_cloud', 'evening_star',
            'three_black_crows', 'tweezer_tops'
        ]
        
        # Calculate bullish/bearish scores
        bullish_score = 0
        bearish_score = 0
        
        for pattern in bullish_patterns:
            col_name = f'pattern_{pattern}'
            if col_name in result_df.columns:
                bullish_score += (result_df[col_name] > 0).astype(int)
        
        for pattern in bearish_patterns:
            col_name = f'pattern_{pattern}'
            if col_name in result_df.columns:
                bearish_score += (result_df[col_name] < 0).astype(int)
        
        result_df['bullish_pattern_score'] = bullish_score
        result_df['bearish_pattern_score'] = bearish_score
        result_df['net_pattern_score'] = bullish_score - bearish_score
        
        return result_df
    
    def extract_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional candlestick-based features"""
        features_df = df.copy()
        
        # Basic candle metrics
        features_df['body_size'] = abs(features_df['Close'] - features_df['Open'])
        features_df['upper_shadow'] = features_df['High'] - np.maximum(features_df['Open'], features_df['Close'])
        features_df['lower_shadow'] = np.minimum(features_df['Open'], features_df['Close']) - features_df['Low']
        features_df['candle_range'] = features_df['High'] - features_df['Low']
        
        # Ratios
        features_df['body_to_range'] = features_df['body_size'] / (features_df['candle_range'] + 1e-8)
        features_df['upper_shadow_to_range'] = features_df['upper_shadow'] / (features_df['candle_range'] + 1e-8)
        features_df['lower_shadow_to_range'] = features_df['lower_shadow'] / (features_df['candle_range'] + 1e-8)
        
        # Candle type
        features_df['is_bullish'] = (features_df['Close'] > features_df['Open']).astype(int)
        features_df['is_bearish'] = (features_df['Close'] < features_df['Open']).astype(int)
        features_df['is_doji'] = (abs(features_df['Close'] - features_df['Open']) < 0.1 * features_df['candle_range']).astype(int)
        
        # Gap analysis
        features_df['gap_up'] = (features_df['Open'] > features_df['High'].shift(1)).astype(int)
        features_df['gap_down'] = (features_df['Open'] < features_df['Low'].shift(1)).astype(int)
        
        # Price position within range
        features_df['close_position'] = (features_df['Close'] - features_df['Low']) / (features_df['candle_range'] + 1e-8)
        
        return features_df
    
    def detect_support_resistance(self, df: pd.DataFrame, window: int = 20, 
                                 min_touches: int = 2) -> pd.DataFrame:
        """Detect support and resistance levels using candlestick data"""
        result_df = df.copy()
        
        # Find local peaks and troughs
        high_peaks, _ = find_peaks(df['High'].values, distance=window//2)
        low_peaks, _ = find_peaks(-df['Low'].values, distance=window//2)
        
        # Initialize support and resistance columns
        result_df['resistance_level'] = np.nan
        result_df['support_level'] = np.nan
        result_df['near_resistance'] = 0
        result_df['near_support'] = 0
        
        # Identify resistance levels
        if len(high_peaks) > 0:
            resistance_levels = df['High'].iloc[high_peaks].values
            for i, price in enumerate(df['Close']):
                closest_resistance = min(resistance_levels, key=lambda x: abs(x - price) if x > price else float('inf'))
                if closest_resistance > price:
                    result_df.iloc[i, result_df.columns.get_loc('resistance_level')] = closest_resistance
                    if abs(price - closest_resistance) / price < 0.02:  # Within 2%
                        result_df.iloc[i, result_df.columns.get_loc('near_resistance')] = 1
        
        # Identify support levels
        if len(low_peaks) > 0:
            support_levels = df['Low'].iloc[low_peaks].values
            for i, price in enumerate(df['Close']):
                closest_support = max(support_levels, key=lambda x: x if x < price else 0)
                if closest_support < price:
                    result_df.iloc[i, result_df.columns.get_loc('support_level')] = closest_support
                    if abs(price - closest_support) / price < 0.02:  # Within 2%
                        result_df.iloc[i, result_df.columns.get_loc('near_support')] = 1
        
        return result_df
    
    def get_pattern_signals(self, df: pd.DataFrame, 
                           lookback_window: int = 5) -> pd.DataFrame:
        """Generate trading signals based on candlestick patterns"""
        signals_df = df.copy()
        
        # Strong bullish signals
        strong_bullish = [
            'pattern_hammer', 'pattern_engulfing', 'pattern_morning_star',
            'pattern_three_white_soldiers', 'pattern_piercing'
        ]
        
        # Strong bearish signals
        strong_bearish = [
            'pattern_shooting_star', 'pattern_dark_cloud', 'pattern_evening_star',
            'pattern_three_black_crows', 'pattern_hanging_man'
        ]
        
        # Calculate signal strength
        signals_df['strong_bullish_signal'] = 0
        signals_df['strong_bearish_signal'] = 0
        
        for pattern in strong_bullish:
            if pattern in signals_df.columns:
                signals_df['strong_bullish_signal'] += (signals_df[pattern] > 0).astype(int)
        
        for pattern in strong_bearish:
            if pattern in signals_df.columns:
                signals_df['strong_bearish_signal'] += (signals_df[pattern] < 0).astype(int)
        
        # Overall signal
        signals_df['pattern_signal'] = (
            signals_df['strong_bullish_signal'] - signals_df['strong_bearish_signal']
        )
        
        # Signal confidence based on volume and volatility
        if 'Volume' in signals_df.columns:
            avg_volume = signals_df['Volume'].rolling(window=lookback_window).mean()
            signals_df['volume_confirmation'] = (signals_df['Volume'] > avg_volume * 1.2).astype(int)
        else:
            signals_df['volume_confirmation'] = 0
        
        # Volatility confirmation
        signals_df['volatility'] = signals_df['candle_range'] / signals_df['Close']
        avg_volatility = signals_df['volatility'].rolling(window=lookback_window).mean()
        signals_df['volatility_confirmation'] = (signals_df['volatility'] > avg_volatility * 1.1).astype(int)
        
        # Final confirmed signal
        signals_df['confirmed_signal'] = signals_df['pattern_signal'] * (
            0.5 + 0.3 * signals_df['volume_confirmation'] + 0.2 * signals_df['volatility_confirmation']
        )
        
        return signals_df
    
    def calculate_pattern_performance(self, df: pd.DataFrame, 
                                    forward_days: int = 5) -> Dict[str, float]:
        """Calculate historical performance of each pattern"""
        performance = {}
        
        # Calculate forward returns
        df['forward_return'] = df['Close'].pct_change(forward_days).shift(-forward_days)
        
        # Analyze each pattern
        pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
        
        for pattern_col in pattern_cols:
            pattern_name = pattern_col.replace('pattern_', '')
            
            # Bullish pattern performance
            bullish_mask = df[pattern_col] > 0
            if bullish_mask.sum() > 0:
                bullish_returns = df.loc[bullish_mask, 'forward_return'].mean()
                performance[f'{pattern_name}_bullish_return'] = bullish_returns
                performance[f'{pattern_name}_bullish_count'] = bullish_mask.sum()
            
            # Bearish pattern performance
            bearish_mask = df[pattern_col] < 0
            if bearish_mask.sum() > 0:
                bearish_returns = df.loc[bearish_mask, 'forward_return'].mean()
                performance[f'{pattern_name}_bearish_return'] = bearish_returns
                performance[f'{pattern_name}_bearish_count'] = bearish_mask.sum()
        
        return performance


def extract_all_candlestick_features(df: pd.DataFrame, 
                                   include_patterns: bool = True,
                                   include_signals: bool = True) -> pd.DataFrame:
    """Extract comprehensive candlestick features"""
    extractor = CandlestickPatternExtractor()
    
    # Start with basic features
    result_df = extractor.extract_candlestick_features(df)
    
    # Add patterns if requested
    if include_patterns:
        result_df = extractor.extract_patterns(result_df)
        result_df = extractor.get_pattern_strength(result_df)
    
    # Add support/resistance
    result_df = extractor.detect_support_resistance(result_df)
    
    # Add signals if requested
    if include_signals:
        result_df = extractor.get_pattern_signals(result_df)
    
    return result_df


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1y")
    
    # Extract candlestick features
    extractor = CandlestickPatternExtractor()
    features_df = extract_all_candlestick_features(data)
    
    print(f"Extracted {len([col for col in features_df.columns if 'pattern_' in col])} candlestick patterns")
    print(f"Data shape: {features_df.shape}")
    
    # Show pattern performance
    performance = extractor.calculate_pattern_performance(features_df)
    print("\nTop performing bullish patterns:")
    bullish_patterns = {k: v for k, v in performance.items() if 'bullish_return' in k}
    for pattern, ret in sorted(bullish_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{pattern}: {ret:.4f}")