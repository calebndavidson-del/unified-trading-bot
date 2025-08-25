#!/usr/bin/env python3
"""
Market Trend Indicators and Technical Analysis Computation
Comprehensive technical analysis for trading decisions
"""

import numpy as np
import pandas as pd
Comprehensive technical analysis for trading decisions.

Note: Some third-party libraries (e.g., TA-Lib, scikit-learn) are not available in this environment.
Manual calculations are used for technical indicators and feature engineering where necessary.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
try:
    from scipy import stats
except ImportError:
    # Fallback for basic stats
    import numpy as np
    class stats:
        @staticmethod
        def skew(data):
            return 0.0
        @staticmethod
        def kurtosis(data):
            data = np.asarray(data)
            n = data.size
            mean = np.mean(data)
            std = np.std(data, ddof=0)
            if std == 0 or n < 3:
                return 0.0
            skewness = np.sum(((data - mean) / std) ** 3) * n / ((n - 1) * (n - 2))
            return skewness
        @staticmethod
        def kurtosis(data):
            data = np.asarray(data)
            n = data.size
            mean = np.mean(data)
            std = np.std(data, ddof=0)
            if std == 0 or n < 4:
                return 0.0
            kurt = np.sum(((data - mean) / std) ** 4) * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))
            kurt -= 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
            return kurt

# from sklearn.preprocessing import StandardScaler  # Not available
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """Comprehensive technical indicator calculations"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index - Manual calculation"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD Indicator - Manual calculation"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands - Manual calculation"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std)
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator - Manual calculation"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R - Manual calculation"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range - Manual calculation"""
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average Directional Index - Simplified calculation"""
        # Simplified ADX calculation
        tr = TechnicalIndicators.atr(high, low, close, 1)
        plus_dm = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        minus_dm = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        plus_dm_smooth = pd.Series(plus_dm, index=high.index).rolling(window=window).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=high.index).rolling(window=window).mean()
        tr_smooth = tr.rolling(window=window).mean()
        
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index - Manual calculation"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=window).mean()
        mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume - Manual calculation"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()


class TrendAnalyzer:
    """Analyze market trends and patterns"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def identify_trend_direction(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Identify overall trend direction"""
        result_df = data.copy()
        
        # Multiple timeframe moving averages
        short_ma = self.indicators.sma(data['Close'], window // 2)
        medium_ma = self.indicators.sma(data['Close'], window)
        long_ma = self.indicators.sma(data['Close'], window * 2)
        
        # Trend direction based on MA alignment
        result_df['trend_short'] = np.where(data['Close'] > short_ma, 1, -1)
        result_df['trend_medium'] = np.where(short_ma > medium_ma, 1, -1)
        result_df['trend_long'] = np.where(medium_ma > long_ma, 1, -1)
        
        # Overall trend strength
        result_df['trend_strength'] = (
            result_df['trend_short'] + result_df['trend_medium'] + result_df['trend_long']
        ) / 3
        
        # Trend classification
        result_df['trend_classification'] = pd.cut(
            result_df['trend_strength'],
            bins=[-1.1, -0.6, -0.2, 0.2, 0.6, 1.1],
            labels=['strong_bearish', 'bearish', 'sideways', 'bullish', 'strong_bullish']
        )
        
        return result_df
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various momentum indicators"""
        result_df = data.copy()
        
        # RSI
        result_df['rsi'] = self.indicators.rsi(data['Close'])
        
        # MACD
        macd_data = self.indicators.macd(data['Close'])
        result_df['macd'] = macd_data['macd']
        result_df['macd_signal'] = macd_data['signal']
        result_df['macd_histogram'] = macd_data['histogram']
        
        # Stochastic
        if all(col in data.columns for col in ['High', 'Low']):
            stoch_data = self.indicators.stochastic(data['High'], data['Low'], data['Close'])
            result_df['stoch_k'] = stoch_data['k']
            result_df['stoch_d'] = stoch_data['d']
            
            # Williams %R
            result_df['williams_r'] = self.indicators.williams_r(data['High'], data['Low'], data['Close'])
        
        # Rate of Change
        result_df['roc_1d'] = data['Close'].pct_change(1)
        result_df['roc_5d'] = data['Close'].pct_change(5)
        result_df['roc_20d'] = data['Close'].pct_change(20)
        
        # Momentum
        result_df['momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
        result_df['momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
        
        return result_df
    
    def calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based indicators"""
        result_df = data.copy()
        
        # Bollinger Bands
        bb_data = self.indicators.bollinger_bands(data['Close'])
        result_df['bb_upper'] = bb_data['upper']
        result_df['bb_middle'] = bb_data['middle']
        result_df['bb_lower'] = bb_data['lower']
        result_df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        result_df['bb_position'] = (data['Close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # Average True Range
        if all(col in data.columns for col in ['High', 'Low']):
            result_df['atr'] = self.indicators.atr(data['High'], data['Low'], data['Close'])
            result_df['atr_normalized'] = result_df['atr'] / data['Close']
        
        # Historical volatility
        returns = data['Close'].pct_change()
        result_df['volatility_10d'] = returns.rolling(window=10).std() * np.sqrt(252)
        result_df['volatility_20d'] = returns.rolling(window=20).std() * np.sqrt(252)
        result_df['volatility_60d'] = returns.rolling(window=60).std() * np.sqrt(252)
        
        # Volatility ratio
        result_df['volatility_ratio'] = result_df['volatility_10d'] / result_df['volatility_60d']
        
        return result_df
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        result_df = data.copy()
        
        if 'Volume' in data.columns:
            # On Balance Volume
            result_df['obv'] = self.indicators.obv(data['Close'], data['Volume'])
            
            # Volume moving averages
            result_df['volume_sma_10'] = data['Volume'].rolling(window=10).mean()
            result_df['volume_sma_20'] = data['Volume'].rolling(window=20).mean()
            
            # Volume ratio
            result_df['volume_ratio'] = data['Volume'] / result_df['volume_sma_20']
            
            # Price Volume Trend
            result_df['pvt'] = (data['Close'].pct_change() * data['Volume']).cumsum()
            
            # VWAP
            if all(col in data.columns for col in ['High', 'Low']):
                result_df['vwap'] = self.indicators.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
            
            # Volume oscillator
            result_df['volume_oscillator'] = (
                (result_df['volume_sma_10'] - result_df['volume_sma_20']) / result_df['volume_sma_20'] * 100
            )
        
        return result_df
    
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate dynamic support and resistance levels"""
        result_df = data.copy()
        
        # Rolling highs and lows
        result_df['resistance'] = data['High'].rolling(window=window).max()
        result_df['support'] = data['Low'].rolling(window=window).min()
        
        # Pivot points (traditional)
        if all(col in data.columns for col in ['High', 'Low', 'Close']):
            pivot = (data['High'] + data['Low'] + data['Close']) / 3
            result_df['pivot_point'] = pivot
            result_df['resistance_1'] = 2 * pivot - data['Low']
            result_df['support_1'] = 2 * pivot - data['High']
            result_df['resistance_2'] = pivot + (data['High'] - data['Low'])
            result_df['support_2'] = pivot - (data['High'] - data['Low'])
        
        # Distance to key levels
        result_df['distance_to_resistance'] = (result_df['resistance'] - data['Close']) / data['Close']
        result_df['distance_to_support'] = (data['Close'] - result_df['support']) / data['Close']
        
        return result_df
    
    def detect_chart_patterns(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Detect common chart patterns"""
        result_df = data.copy()
        
        # Higher highs and higher lows (uptrend)
        high_rolling_max = data['High'].rolling(window=window).max()
        low_rolling_min = data['Low'].rolling(window=window).min()
        
        result_df['higher_high'] = (data['High'] > high_rolling_max.shift(1)).astype(int)
        result_df['higher_low'] = (data['Low'] > low_rolling_min.shift(1)).astype(int)
        result_df['lower_high'] = (data['High'] < high_rolling_max.shift(1)).astype(int)
        result_df['lower_low'] = (data['Low'] < low_rolling_min.shift(1)).astype(int)
        
        # Pattern scoring
        result_df['uptrend_pattern'] = (result_df['higher_high'] + result_df['higher_low']) / 2
        result_df['downtrend_pattern'] = (result_df['lower_high'] + result_df['lower_low']) / 2
        
        # Breakout detection
        result_df['resistance_breakout'] = (data['Close'] > high_rolling_max.shift(1)).astype(int)
        result_df['support_breakdown'] = (data['Close'] < low_rolling_min.shift(1)).astype(int)
        
        # Consolidation detection
        price_range = (data['High'] - data['Low']) / data['Close']
        result_df['consolidation'] = (price_range < price_range.rolling(window=window).quantile(0.3)).astype(int)
        
        return result_df
    
    def calculate_market_regime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Identify market regime (trending vs. mean-reverting)"""
        result_df = data.copy()
        
        returns = data['Close'].pct_change()
        
        # Hurst exponent for trend persistence
        def hurst_exponent(ts, max_lag=20):
            """Calculate Hurst exponent"""
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        # Rolling Hurst exponent
        result_df['hurst_exponent'] = returns.rolling(window=60).apply(
            lambda x: hurst_exponent(x.dropna()) if len(x.dropna()) > 20 else np.nan
        )
        
        # Market regime classification
        result_df['market_regime'] = np.where(
            result_df['hurst_exponent'] > 0.5, 'trending', 'mean_reverting'
        )
        
        # Trend strength based on ADX
        if all(col in data.columns for col in ['High', 'Low']):
            result_df['adx'] = self.indicators.adx(data['High'], data['Low'], data['Close'])
            result_df['trend_strength_adx'] = pd.cut(
                result_df['adx'],
                bins=[0, 25, 50, 75, 100],
                labels=['weak', 'moderate', 'strong', 'very_strong']
            )
        
        return result_df


class TrendSignalGenerator:
    """Generate trading signals based on trend analysis"""
    
    def __init__(self):
        self.analyzer = TrendAnalyzer()
    
    def generate_momentum_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on momentum indicators"""
        result_df = data.copy()
        
        # RSI signals
        if 'rsi' in result_df.columns:
            result_df['rsi_oversold'] = (result_df['rsi'] < 30).astype(int)
            result_df['rsi_overbought'] = (result_df['rsi'] > 70).astype(int)
            result_df['rsi_bullish_divergence'] = self._detect_rsi_divergence(result_df, 'bullish')
            result_df['rsi_bearish_divergence'] = self._detect_rsi_divergence(result_df, 'bearish')
        
        # MACD signals
        if all(col in result_df.columns for col in ['macd', 'macd_signal']):
            result_df['macd_bullish_cross'] = (
                (result_df['macd'] > result_df['macd_signal']) & 
                (result_df['macd'].shift(1) <= result_df['macd_signal'].shift(1))
            ).astype(int)
            
            result_df['macd_bearish_cross'] = (
                (result_df['macd'] < result_df['macd_signal']) & 
                (result_df['macd'].shift(1) >= result_df['macd_signal'].shift(1))
            ).astype(int)
        
        # Stochastic signals
        if all(col in result_df.columns for col in ['stoch_k', 'stoch_d']):
            result_df['stoch_oversold'] = ((result_df['stoch_k'] < 20) & (result_df['stoch_d'] < 20)).astype(int)
            result_df['stoch_overbought'] = ((result_df['stoch_k'] > 80) & (result_df['stoch_d'] > 80)).astype(int)
        
        return result_df
    
    def generate_trend_following_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-following signals"""
        result_df = data.copy()
        
        # Moving average crossovers
        sma_short = TechnicalIndicators.sma(data['Close'], 10)
        sma_long = TechnicalIndicators.sma(data['Close'], 30)
        
        result_df['ma_bullish_cross'] = (
            (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
        ).astype(int)
        
        result_df['ma_bearish_cross'] = (
            (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))
        ).astype(int)
        
        # Breakout signals
        if 'resistance_breakout' in result_df.columns:
            result_df['breakout_signal'] = result_df['resistance_breakout']
        
        if 'support_breakdown' in result_df.columns:
            result_df['breakdown_signal'] = result_df['support_breakdown']
        
        # Trend strength signals
        if 'trend_strength' in result_df.columns:
            result_df['strong_uptrend'] = (result_df['trend_strength'] > 0.6).astype(int)
            result_df['strong_downtrend'] = (result_df['trend_strength'] < -0.6).astype(int)
        
        return result_df
    
    def generate_mean_reversion_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals"""
        result_df = data.copy()
        
        # Bollinger Band signals
        if all(col in result_df.columns for col in ['bb_upper', 'bb_lower', 'bb_position']):
            result_df['bb_oversold'] = (result_df['bb_position'] < 0.1).astype(int)
            result_df['bb_overbought'] = (result_df['bb_position'] > 0.9).astype(int)
            
            # Mean reversion from extreme levels
            result_df['bb_mean_reversion_long'] = (
                (result_df['bb_position'] < 0.2) & 
                (result_df['bb_position'].shift(1) < result_df['bb_position'])
            ).astype(int)
            
            result_df['bb_mean_reversion_short'] = (
                (result_df['bb_position'] > 0.8) & 
                (result_df['bb_position'].shift(1) > result_df['bb_position'])
            ).astype(int)
        
        return result_df
    
    def generate_composite_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate composite signals combining multiple indicators"""
        result_df = data.copy()
        
        # Bullish composite signal
        bullish_components = []
        bearish_components = []
        
        # Add available signals
        if 'rsi_oversold' in result_df.columns:
            bullish_components.append(result_df['rsi_oversold'])
        if 'macd_bullish_cross' in result_df.columns:
            bullish_components.append(result_df['macd_bullish_cross'])
        if 'ma_bullish_cross' in result_df.columns:
            bullish_components.append(result_df['ma_bullish_cross'])
        if 'bb_oversold' in result_df.columns:
            bullish_components.append(result_df['bb_oversold'])
        
        if 'rsi_overbought' in result_df.columns:
            bearish_components.append(result_df['rsi_overbought'])
        if 'macd_bearish_cross' in result_df.columns:
            bearish_components.append(result_df['macd_bearish_cross'])
        if 'ma_bearish_cross' in result_df.columns:
            bearish_components.append(result_df['ma_bearish_cross'])
        if 'bb_overbought' in result_df.columns:
            bearish_components.append(result_df['bb_overbought'])
        
        # Calculate composite scores
        if bullish_components:
            result_df['bullish_signal_strength'] = sum(bullish_components) / len(bullish_components)
        else:
            result_df['bullish_signal_strength'] = 0
        
        if bearish_components:
            result_df['bearish_signal_strength'] = sum(bearish_components) / len(bearish_components)
        else:
            result_df['bearish_signal_strength'] = 0
        
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
    
    def _detect_rsi_divergence(self, data: pd.DataFrame, divergence_type: str, window: int = 14) -> pd.Series:
        """Detect RSI divergence"""
        # Simplified divergence detection
        # In practice, this would be more sophisticated
        
        if 'rsi' not in data.columns:
            return pd.Series(0, index=data.index)
        
        price_trend = data['Close'].diff(window)
        rsi_trend = data['rsi'].diff(window)
        
        if divergence_type == 'bullish':
            # Price declining but RSI rising
            divergence = (price_trend < 0) & (rsi_trend > 0)
        else:  # bearish
            # Price rising but RSI declining
            divergence = (price_trend > 0) & (rsi_trend < 0)
        
        return divergence.astype(int)


def create_comprehensive_trend_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive trend analysis features"""
    
    analyzer = TrendAnalyzer()
    signal_generator = TrendSignalGenerator()
    
    # Start with input data
    result_df = data.copy()
    
    # Add trend direction analysis
    result_df = analyzer.identify_trend_direction(result_df)
    
    # Add momentum indicators
    result_df = analyzer.calculate_momentum_indicators(result_df)
    
    # Add volatility indicators
    result_df = analyzer.calculate_volatility_indicators(result_df)
    
    # Add volume indicators
    result_df = analyzer.calculate_volume_indicators(result_df)
    
    # Add support/resistance levels
    result_df = analyzer.calculate_support_resistance(result_df)
    
    # Add chart patterns
    result_df = analyzer.detect_chart_patterns(result_df)
    
    # Add market regime analysis
    result_df = analyzer.calculate_market_regime(result_df)
    
    # Generate trading signals
    result_df = signal_generator.generate_momentum_signals(result_df)
    result_df = signal_generator.generate_trend_following_signals(result_df)
    result_df = signal_generator.generate_mean_reversion_signals(result_df)
    result_df = signal_generator.generate_composite_signals(result_df)
    
    return result_df


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1y")
    
    # Create comprehensive trend features
    trend_features = create_comprehensive_trend_features(data)
    
    print(f"Created trend features for AAPL")
    print(f"Data shape: {trend_features.shape}")
    print(f"Number of features: {len(trend_features.columns)}")
    
    # Show some trend signals
    if 'composite_signal' in trend_features.columns:
        print(f"Average composite signal: {trend_features['composite_signal'].mean():.4f}")
        
    # Show signal distribution
    if 'signal_category' in trend_features.columns:
        signal_dist = trend_features['signal_category'].value_counts()
        print(f"Signal distribution:\n{signal_dist}")