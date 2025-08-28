#!/usr/bin/env python3
"""
Backtesting Engine for Trading Strategies
Simulate trades, positions, and portfolio performance using historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import yfinance as yf
import pytz
import traceback
import re

# Import existing modules
from features.market_trend import create_comprehensive_trend_features
from features.candlestick import extract_all_candlestick_features
from utils.risk import RiskMetrics, PositionSizing
from model_config import TradingBotConfig

warnings.filterwarnings('ignore')


@dataclass
class MissingDataConfig:
    """Configuration for missing data handling"""
    # Crypto tolerance settings
    crypto_daily_tolerance_hours: float = 6.0  # Max hours gap for crypto daily data
    
    # Strict mode settings
    strict_mode: bool = False
    max_missing_data_ratio: float = 0.1  # Max 10% missing data before error in strict mode
    
    # Asset type detection patterns
    crypto_patterns: List[str] = field(default_factory=lambda: [
        r'.*-USD$', r'.*USD$', r'BTC', r'ETH', r'SOL', r'ADA', r'DOGE'
    ])
    
    # Market hours for different asset types
    traditional_market_days: List[str] = field(default_factory=lambda: [
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'
    ])


@dataclass 
class MissingDataEntry:
    """Track a missing data occurrence"""
    symbol: str
    date: datetime
    asset_type: str  # 'stock', 'etf', 'index', 'crypto'
    gap_hours: Optional[float] = None  # Hours since nearest data (for crypto)
    is_expected: bool = False  # Weekend, holiday, etc.
    reason: str = ""  # Explanation for the missing data


@dataclass
class MissingDataSummary:
    """Summary of missing data for the entire backtest"""
    total_expected_gaps: int = 0
    total_unexpected_gaps: int = 0
    crypto_tolerance_violations: int = 0
    by_symbol: Dict[str, List[MissingDataEntry]] = field(default_factory=dict)
    by_asset_type: Dict[str, int] = field(default_factory=dict)
    
    def add_entry(self, entry: MissingDataEntry, config: Optional[MissingDataConfig] = None):
        """Add a missing data entry to the summary"""
        if entry.symbol not in self.by_symbol:
            self.by_symbol[entry.symbol] = []
        self.by_symbol[entry.symbol].append(entry)
        
        if entry.asset_type not in self.by_asset_type:
            self.by_asset_type[entry.asset_type] = 0
        self.by_asset_type[entry.asset_type] += 1
        
        if entry.is_expected:
            self.total_expected_gaps += 1
        else:
            self.total_unexpected_gaps += 1
            
        # Use config parameter to access crypto tolerance setting
        if self._is_crypto_tolerance_violation(entry, config):
            self.crypto_tolerance_violations += 1
    
    def _is_crypto_tolerance_violation(self, entry: MissingDataEntry, config: Optional[MissingDataConfig] = None) -> bool:
        """Check if a missing data entry represents a crypto tolerance violation"""
        # Only crypto assets can have tolerance violations
        if entry.asset_type != 'crypto':
            return False
        
        # If no config provided, cannot determine violation
        if config is None:
            return False
        
        # If no gap_hours specified, cannot determine violation
        if entry.gap_hours is None:
            return False
        
        # Check if gap exceeds tolerance threshold
        return entry.gap_hours > config.crypto_daily_tolerance_hours


class AssetTypeDetector:
    """Utility class for detecting asset types"""
    
    @staticmethod
    def detect_asset_type(symbol: str) -> str:
        """
        Detect asset type based on symbol format.
        Returns: 'crypto', 'index', 'etf', or 'stock'
        """
        symbol_upper = symbol.upper()
        
        # Crypto patterns
        crypto_patterns = [
            r'.*-USD$', r'.*USD$', r'BTC', r'ETH', r'SOL', r'ADA', r'DOGE',
            r'USDT', r'USDC', r'BNB', r'XRP', r'STETH'
        ]
        
        for pattern in crypto_patterns:
            if re.search(pattern, symbol_upper):
                return 'crypto'
        
        # Index patterns (usually start with ^)
        if symbol_upper.startswith('^'):
            return 'index'
        
        # Common ETF patterns
        etf_patterns = [
            'SPY', 'QQQ', 'IWM', 'EFA', 'VTI', 'VOO', 'VEA', 'IEFA',
            'AGG', 'BND', 'LQD', 'HYG', 'TLT', 'GLD', 'SLV'
        ]
        
        if symbol_upper in etf_patterns or symbol_upper.endswith('ETF'):
            return 'etf'
        
        # Default to stock
        return 'stock'
    
    @staticmethod
    def is_weekend(date: datetime) -> bool:
        """Check if date falls on weekend"""
        return date.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    @staticmethod
    def is_traditional_market_hours(date: datetime) -> bool:
        """Check if date is during traditional market hours (Mon-Fri)"""
        return date.weekday() < 5  # Monday = 0, Friday = 4




class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on data"""
        raise NotImplementedError("Strategy must implement generate_signals method")
    
    def get_position_size(self, signal: float, price: float, portfolio_value: float) -> float:
        """Calculate position size based on signal strength"""
        base_position = portfolio_value * 0.1  # 10% base position
        return base_position * abs(signal)


class TechnicalAnalysisStrategy(TradingStrategy):
    """Technical analysis strategy using RSI and moving averages"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Technical Analysis", config)
        self.rsi_oversold = config.get('rsi_oversold', 30) if config else 30
        self.rsi_overbought = config.get('rsi_overbought', 70) if config else 70
        self.ma_short = config.get('ma_short', 10) if config else 10
        self.ma_long = config.get('ma_long', 50) if config else 50
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI and moving average crossover"""
        signals = pd.Series(0.0, index=data.index)
        
        if 'rsi' not in data.columns:
            # Calculate RSI if not present
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            data['rsi'] = rsi
        
        # Calculate moving averages
        ma_short = data['Close'].rolling(window=self.ma_short).mean()
        ma_long = data['Close'].rolling(window=self.ma_long).mean()
        
        # Generate signals
        for i in range(len(data)):
            if pd.isna(data['rsi'].iloc[i]) or pd.isna(ma_short.iloc[i]) or pd.isna(ma_long.iloc[i]):
                continue
                
            # RSI signals
            if data['rsi'].iloc[i] < self.rsi_oversold and ma_short.iloc[i] > ma_long.iloc[i]:
                signals.iloc[i] = 1.0  # Strong buy
            elif data['rsi'].iloc[i] < self.rsi_oversold:
                signals.iloc[i] = 0.5  # Weak buy
            elif data['rsi'].iloc[i] > self.rsi_overbought and ma_short.iloc[i] < ma_long.iloc[i]:
                signals.iloc[i] = -1.0  # Strong sell
            elif data['rsi'].iloc[i] > self.rsi_overbought:
                signals.iloc[i] = -0.5  # Weak sell
            elif ma_short.iloc[i] > ma_long.iloc[i] * 1.02:  # 2% above
                signals.iloc[i] = 0.3  # Weak buy on momentum
            elif ma_short.iloc[i] < ma_long.iloc[i] * 0.98:  # 2% below
                signals.iloc[i] = -0.3  # Weak sell on momentum
        
        return signals


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion strategy using Bollinger Bands"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Mean Reversion", config)
        self.bb_period = config.get('bb_period', 20) if config else 20
        self.bb_std = config.get('bb_std', 2) if config else 2
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on Bollinger Bands"""
        signals = pd.Series(0.0, index=data.index)
        
        # Calculate Bollinger Bands
        bb_middle = data['Close'].rolling(window=self.bb_period).mean()
        bb_std = data['Close'].rolling(window=self.bb_period).std()
        bb_upper = bb_middle + (bb_std * self.bb_std)
        bb_lower = bb_middle - (bb_std * self.bb_std)
        
        # Generate signals
        for i in range(len(data)):
            if pd.isna(bb_upper.iloc[i]) or pd.isna(bb_lower.iloc[i]):
                continue
                
            price = data['Close'].iloc[i]
            
            # Mean reversion signals
            if price <= bb_lower.iloc[i]:
                signals.iloc[i] = 1.0  # Buy at lower band
            elif price >= bb_upper.iloc[i]:
                signals.iloc[i] = -1.0  # Sell at upper band
            elif price < bb_middle.iloc[i] * 0.99:  # 1% below middle
                signals.iloc[i] = 0.5  # Weak buy
            elif price > bb_middle.iloc[i] * 1.01:  # 1% above middle
                signals.iloc[i] = -0.5  # Weak sell
        
        return signals


class MomentumStrategy(TradingStrategy):
    """Momentum strategy using moving average convergence"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Momentum", config)
        self.ma_fast = config.get('ma_fast', 12) if config else 12
        self.ma_slow = config.get('ma_slow', 26) if config else 26
        self.signal_line = config.get('signal_line', 9) if config else 9
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on MACD momentum"""
        signals = pd.Series(0.0, index=data.index)
        
        # Calculate MACD
        ema_fast = data['Close'].ewm(span=self.ma_fast).mean()
        ema_slow = data['Close'].ewm(span=self.ma_slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal_line).mean()
        histogram = macd - signal_line
        
        # Generate signals
        for i in range(1, len(data)):
            if pd.isna(macd.iloc[i]) or pd.isna(signal_line.iloc[i]):
                continue
                
            # MACD crossover signals
            if macd.iloc[i] > signal_line.iloc[i] and macd.iloc[i-1] <= signal_line.iloc[i-1]:
                signals.iloc[i] = 1.0  # Bullish crossover
            elif macd.iloc[i] < signal_line.iloc[i] and macd.iloc[i-1] >= signal_line.iloc[i-1]:
                signals.iloc[i] = -1.0  # Bearish crossover
            elif histogram.iloc[i] > 0 and histogram.iloc[i] > histogram.iloc[i-1]:
                signals.iloc[i] = 0.5  # Momentum building
            elif histogram.iloc[i] < 0 and histogram.iloc[i] < histogram.iloc[i-1]:
                signals.iloc[i] = -0.5  # Negative momentum building
        
        return signals


class PatternRecognitionStrategy(TradingStrategy):
    """Strategy based on candlestick patterns"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Pattern Recognition", config)
        self.pattern_weight = config.get('pattern_weight', 0.5) if config else 0.5
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on candlestick patterns"""
        signals = pd.Series(0.0, index=data.index)
        
        # Extract candlestick patterns
        pattern_data = extract_all_candlestick_features(data)
        
        # Get pattern columns
        pattern_cols = [col for col in pattern_data.columns if 'pattern_' in col]
        
        if not pattern_cols:
            return signals
        
        # Generate signals based on patterns
        for i in range(len(pattern_data)):
            total_signal = 0.0
            pattern_count = 0
            
            for pattern_col in pattern_cols:
                pattern_value = pattern_data[pattern_col].iloc[i]
                if not pd.isna(pattern_value) and pattern_value != 0:
                    total_signal += pattern_value * self.pattern_weight
                    pattern_count += 1
            
            if pattern_count > 0:
                signals.iloc[i] = np.clip(total_signal / pattern_count, -1.0, 1.0)
        
        return signals


class Trade:
    """Represents a single trade"""
    
    def __init__(self, symbol: str, entry_date: datetime, entry_price: float, 
                 quantity: float, direction: str):
        self.symbol = symbol
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.quantity = abs(quantity)
        self.direction = direction  # 'long' or 'short'
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0.0
        self.commission = 0.0
        self.is_open = True
    
    def close_trade(self, exit_date: datetime, exit_price: float, commission: float = 0.0):
        """Close the trade and calculate P&L"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.commission = commission
        self.is_open = False
        
        if self.direction == 'long':
            self.pnl = (self.exit_price - self.entry_price) * self.quantity - commission
        else:  # short
            self.pnl = (self.entry_price - self.exit_price) * self.quantity - commission
    
    def get_current_pnl(self, current_price: float) -> float:
        """Get current unrealized P&L"""
        if not self.is_open:
            return self.pnl
        
        if self.direction == 'long':
            return (current_price - self.entry_price) * self.quantity
        else:  # short
            return (self.entry_price - current_price) * self.quantity


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config: TradingBotConfig = None, missing_data_config: MissingDataConfig = None):
        self.config = config or TradingBotConfig()
        self.missing_data_config = missing_data_config or MissingDataConfig()
        self.strategies = {
            'Technical Analysis': TechnicalAnalysisStrategy(),
            'Mean Reversion': MeanReversionStrategy(),
            'Momentum': MomentumStrategy(),
            'Pattern Recognition': PatternRecognitionStrategy()
        }
        
        # Add unified strategy if available
        try:
            from features.unified_strategy import UnifiedTradingStrategy
            self.strategies['Unified Strategy'] = UnifiedTradingStrategy()
        except ImportError:
            pass  # Unified strategy not available
        self.reset()
    
    def reset(self):
        """Reset the backtest state"""
        self.trades = []
        self.portfolio_value_history = []
        self.cash_history = []
        self.positions = {}  # symbol -> Trade
        self.initial_capital = self.config.risk.initial_capital
        self.current_cash = self.initial_capital
        self.current_portfolio_value = self.initial_capital
        self.commission_rate = 0.001  # 0.1% commission
        
        # Missing data tracking
        self.missing_data_summary = MissingDataSummary()
        self.asset_types = {}  # symbol -> asset_type cache
    
    def fetch_current_year_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for current year only with timezone normalization"""
        current_year = datetime.now().year
        start_date = f"{current_year}-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        data_dict = {}
        
        for symbol in symbols:
            try:
                # Fetch data using yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if data.empty:
                    # Check if symbol is invalid using ticker info
                    info = getattr(ticker, 'info', {})
                    
                    # Check if the request includes future dates or is likely API lag
                    now = datetime.now()
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    
                    # Check for future dates in request
                    if start_dt > now:
                        print(f"⚠️ No price data available for {symbol}: requested period extends into the future.")
                        continue
                    
                    # Check for valid ticker info indicating a real security
                    has_valid_info = (info and ('longName' in info or 'shortName' in info or 
                                               'symbol' in info or 'marketCap' in info or
                                               'regularMarketPrice' in info or 'quoteType' in info))
                    
                    if has_valid_info:
                        # Valid security but no data - likely API lag or market closure
                        print(f"⚠️ No price data available for {symbol} for requested period (possible API lag or market closure).")
                    else:
                        # No valid ticker info - likely invalid symbol
                        print(f"❌ No data available for {symbol}: symbol appears to be invalid or delisted.")
                    continue
                
                # Ensure required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_cols):
                    print(f"⚠️ Missing required columns for {symbol}: {set(required_cols) - set(data.columns)}")
                    continue
                
                # Normalize timezone to UTC for consistent handling
                if data.index.tz is not None:
                    # Convert to UTC and then localize to avoid ambiguous times
                    data.index = data.index.tz_convert('UTC')
                    print(f"✅ Fetched {len(data)} days of data for {symbol} (converted to UTC)")
                else:
                    # If no timezone, assume UTC
                    data.index = data.index.tz_localize('UTC')
                    print(f"✅ Fetched {len(data)} days of data for {symbol} (localized to UTC)")
                
                # Validate data quality
                if data['Close'].isna().any():
                    print(f"⚠️ Warning: {symbol} has missing Close prices on {data['Close'].isna().sum()} days")
                
                data_dict[symbol] = data
                    
            except Exception as e:
                print(f"❌ Error fetching data for {symbol}: {e}")
                print(f"   Details: {traceback.format_exc()}")
        
        return data_dict

    def fetch_comprehensive_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch comprehensive historical data including multiple time periods and sources"""
        print("📊 Fetching comprehensive historical data (current year + 5 years + live data)...")
        
        # Define multiple time periods for comprehensive coverage
        periods = [
            ("5y", "5 years"),
            ("2y", "2 years"),
            ("1y", "1 year"), 
            ("6mo", "6 months"),
            ("3mo", "3 months"),
            ("1mo", "1 month")
        ]
        
        data_dict = {}
        successful_symbols = []
        failed_symbols = []
        
        for symbol in symbols:
            print(f"🔍 Fetching comprehensive data for {symbol}...")
            
            # Try to get data with multiple periods, starting with longest
            symbol_data = None
            
            for period, description in periods:
                try:
                    print(f"  📈 Trying {description} data for {symbol}...")
                    
                    # Retry mechanism for each period
                    for retry in range(3):  # Try up to 3 times
                        try:
                            ticker = yf.Ticker(symbol)
                            
                            # Fetch data for this period
                            data = ticker.history(period=period, interval='1d', timeout=15)
                            
                            if not data.empty and len(data) > 50:  # Need at least 50 days
                                # Ensure required columns
                                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                                if all(col in data.columns for col in required_cols):
                                    # Normalize timezone to UTC
                                    if data.index.tz is not None:
                                        data.index = data.index.tz_convert('UTC')
                                    else:
                                        data.index = data.index.tz_localize('UTC')
                                    
                                    # Validate data quality
                                    if not data['Close'].isna().all() and data['Close'].notna().sum() > len(data) * 0.9:
                                        # Clean the data - forward fill small gaps
                                        data = self._clean_fetched_data(data, symbol)
                                        symbol_data = data
                                        print(f"  ✅ Successfully fetched {len(data)} days from {description}")
                                        break
                                    else:
                                        print(f"  ⚠️ Too many NaN Close prices for {description}")
                                else:
                                    print(f"  ⚠️ Missing required columns for {description}")
                            else:
                                print(f"  ⚠️ Insufficient data for {description}: {len(data) if not data.empty else 0} days")
                                
                        except Exception as e:
                            if retry < 2:  # Not last retry
                                print(f"  🔄 Retry {retry + 1}/3 for {description} due to: {e}")
                                continue
                            else:
                                print(f"  ❌ Final retry failed for {description}: {e}")
                                break
                    
                    # If we got data for this period, break out of periods loop
                    if symbol_data is not None:
                        break
                        
                except Exception as e:
                    print(f"  ❌ Error fetching {description} for {symbol}: {e}")
                    continue
            
            # Try fallback with alternative symbols or data sources
            if symbol_data is None:
                symbol_data = self._try_fallback_data_sources(symbol)
            
            if symbol_data is not None:
                data_dict[symbol] = symbol_data
                successful_symbols.append(symbol)
                
                # Additional validation
                if symbol_data['Close'].isna().any():
                    na_count = symbol_data['Close'].isna().sum()
                    print(f"  ⚠️ Warning: {symbol} has {na_count} missing Close prices")
            else:
                failed_symbols.append(symbol)
                print(f"  ❌ Failed to fetch any data for {symbol}")
        
        print(f"\n📊 Data fetching summary:")
        print(f"  ✅ Successful: {len(successful_symbols)} symbols")
        print(f"  ❌ Failed: {len(failed_symbols)} symbols")
        
        if failed_symbols:
            print(f"  Failed symbols: {failed_symbols}")
        
        return data_dict

    def _clean_fetched_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean fetched data by handling common data quality issues"""
        cleaned_data = data.copy()
        
        # Normalize dates to remove time component for consistent matching
        # This ensures all dates are at 00:00:00 UTC regardless of original timezone
        if hasattr(cleaned_data.index, 'tz'):
            if cleaned_data.index.tz is not None:
                # If index is timezone-aware, convert to UTC and normalize (preserves tz)
                cleaned_data.index = cleaned_data.index.tz_convert('UTC').normalize()
            else:
                # If index is naive, normalize and localize to UTC
                cleaned_data.index = cleaned_data.index.normalize().tz_localize('UTC')
        
        # Forward fill small gaps (up to 3 consecutive missing values)
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in cleaned_data.columns:
                # Identify small gaps
                na_groups = cleaned_data[col].isna().groupby((~cleaned_data[col].isna()).cumsum()).sum()
                small_gaps = na_groups[na_groups <= 3].index
                
                for gap_group in small_gaps:
                    mask = (cleaned_data[col].isna().groupby((~cleaned_data[col].isna()).cumsum()).cumsum() == gap_group) & cleaned_data[col].isna()
                    if mask.any():
                        cleaned_data.loc[mask, col] = cleaned_data[col].ffill().loc[mask]
        
        # Ensure Volume is non-negative
        if 'Volume' in cleaned_data.columns:
            cleaned_data['Volume'] = cleaned_data['Volume'].clip(lower=0)
        
        # Validate price consistency (High >= Low, Close between High and Low)
        if all(col in cleaned_data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Fix obvious data errors
            cleaned_data['High'] = cleaned_data[['High', 'Low', 'Open', 'Close']].max(axis=1)
            cleaned_data['Low'] = cleaned_data[['High', 'Low', 'Open', 'Close']].min(axis=1)
        
        return cleaned_data

    def _try_fallback_data_sources(self, symbol: str) -> Optional[pd.DataFrame]:
        """Try alternative data sources and symbol variations for problematic assets"""
        print(f"  🔄 Trying fallback data sources for {symbol}...")
        
        # Try common symbol variations
        symbol_variations = self._get_symbol_variations(symbol)
        
        for variant in symbol_variations:
            if variant == symbol:
                continue  # Already tried
                
            try:
                print(f"    🔍 Trying symbol variation: {variant}")
                ticker = yf.Ticker(variant)
                data = ticker.history(period="1y", interval='1d')
                
                if not data.empty and len(data) > 50:
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in data.columns for col in required_cols):
                        # Normalize timezone
                        if data.index.tz is not None:
                            data.index = data.index.tz_convert('UTC')
                        else:
                            data.index = data.index.tz_localize('UTC')
                        
                        print(f"    ✅ Found data using variation {variant}: {len(data)} days")
                        return data
                        
            except Exception as e:
                print(f"    ❌ Variation {variant} failed: {e}")
                continue
        
        # Try with data pipeline if available
        try:
            from features.data_pipeline import DataPipeline
            pipeline = DataPipeline()
            
            print(f"    🔄 Trying multi-source data pipeline...")
            multi_data = pipeline.fetch_multi_source_data([symbol], period="1y")
            
            if symbol in multi_data:
                for source, data in multi_data[symbol].items():
                    if not data.empty and len(data) > 50:
                        print(f"    ✅ Found data from {source}: {len(data)} days")
                        return data
                        
        except Exception as e:
            print(f"    ❌ Data pipeline fallback failed: {e}")
        
        return None

    def _get_symbol_variations(self, symbol: str) -> List[str]:
        """Generate common symbol variations for fallback attempts"""
        variations = [symbol]
        
        # Common ticker transformations
        if '-' in symbol:
            # Try without dash (BTC-USD -> BTCUSD)
            variations.append(symbol.replace('-', ''))
            # Try with equals (BTC-USD -> BTC=X)
            base = symbol.split('-')[0]
            variations.append(f"{base}=X")
        
        # Try with exchange suffixes
        if '.' not in symbol:
            common_exchanges = ['.TO', '.L', '.F', '.HK', '.T']
            for exchange in common_exchanges:
                variations.append(f"{symbol}{exchange}")
        
        # Try without exchange suffix
        if '.' in symbol:
            variations.append(symbol.split('.')[0])
        
        # ETF variations
        if symbol.startswith('^'):
            # Index symbols (^GSPC -> SPY as ETF proxy)
            index_to_etf = {
                '^GSPC': 'SPY',
                '^DJI': 'DIA', 
                '^IXIC': 'QQQ',
                '^RUT': 'IWM'
            }
            if symbol in index_to_etf:
                variations.append(index_to_etf[symbol])
        
        return list(set(variations))  # Remove duplicates

    def _validate_comprehensive_data_coverage(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate that we have comprehensive data coverage for all assets"""
        print("  🔍 Checking data coverage...")
        
        validation_issues = []
        coverage_stats = {}
        
        min_days_required = 50
        total_symbols = len(data_dict)
        valid_symbols = 0
        
        for symbol, data in data_dict.items():
            asset_type = AssetTypeDetector.detect_asset_type(symbol)
            
            # Basic validation
            if data.empty:
                validation_issues.append(f"{symbol}: No data available")
                continue
                
            if len(data) < min_days_required:
                validation_issues.append(f"{symbol}: Insufficient data ({len(data)} days, need {min_days_required})")
                continue
                
            # Check for excessive missing data
            missing_closes = data['Close'].isna().sum()
            missing_ratio = missing_closes / len(data)
            
            if missing_ratio > 0.1:  # More than 10% missing
                validation_issues.append(f"{symbol}: Too many missing Close prices ({missing_ratio:.1%})")
                continue
                
            # Check date range coverage
            date_range = (data.index[-1] - data.index[0]).days
            expected_days = date_range if asset_type == 'crypto' else date_range * 5/7  # Adjust for weekends
            
            coverage_ratio = len(data) / expected_days if expected_days > 0 else 0
            
            coverage_stats[symbol] = {
                'days': len(data),
                'date_range': date_range,
                'coverage_ratio': coverage_ratio,
                'missing_ratio': missing_ratio,
                'asset_type': asset_type,
                'start_date': data.index[0].date(),
                'end_date': data.index[-1].date()
            }
            
            # Additional validation for date continuity
            if asset_type == 'crypto' and coverage_ratio < 0.9:
                validation_issues.append(f"{symbol}: Crypto data has poor coverage ({coverage_ratio:.1%})")
                continue
            elif asset_type != 'crypto' and coverage_ratio < 0.8:
                validation_issues.append(f"{symbol}: Stock data has poor coverage ({coverage_ratio:.1%})")
                continue
                
            valid_symbols += 1
        
        # Overall validation
        if valid_symbols == 0:
            return {
                'valid': False,
                'reason': 'No symbols have valid data',
                'issues': validation_issues,
                'stats': coverage_stats
            }
        
        if valid_symbols < total_symbols * 0.5:  # Less than 50% symbols valid
            return {
                'valid': False,
                'reason': f'Too many symbols failed validation ({valid_symbols}/{total_symbols} valid)',
                'issues': validation_issues,
                'stats': coverage_stats
            }
        
        # Check for date range consistency
        all_start_dates = [stats['start_date'] for stats in coverage_stats.values()]
        all_end_dates = [stats['end_date'] for stats in coverage_stats.values()]
        
        earliest_start = min(all_start_dates)
        latest_end = max(all_end_dates)
        common_range_days = (latest_end - earliest_start).days
        
        if common_range_days < min_days_required:
            return {
                'valid': False,
                'reason': f'Common date range too short ({common_range_days} days)',
                'issues': validation_issues,
                'stats': coverage_stats
            }
        
        # Success
        return {
            'valid': True,
            'summary': f'{valid_symbols}/{total_symbols} symbols valid, {common_range_days} days common range',
            'issues': validation_issues,
            'stats': coverage_stats,
            'valid_symbols': valid_symbols,
            'total_symbols': total_symbols,
            'common_range_days': common_range_days
        }

    def _filter_relevant_dates(self, all_dates: List[pd.Timestamp], asset_types: set) -> List[pd.Timestamp]:
        """Filter dates to only include those relevant for the assets in the portfolio"""
        
        # For now, include all dates and let the individual asset logic handle missing data
        # This prevents us from filtering out important dates
        return all_dates

    def _find_common_date_range(self, data_dict: Dict[str, pd.DataFrame], 
                              backtest_period: str = "1y") -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Find the common date range where all assets have meaningful data coverage
        
        Args:
            data_dict: Dictionary of symbol data
            backtest_period: Period for backtest window ("1mo", "6mo", "1y", "3y", "5y")
        """
        
        # Calculate target start date based on backtest period
        end_date = pd.Timestamp.now(tz=pytz.UTC)
        
        # Map period strings to days
        period_to_days = {
            "1mo": 30,
            "6mo": 180,
            "1y": 365,
            "3y": 365 * 3,
            "5y": 365 * 5
        }
        
        if backtest_period in period_to_days:
            target_start_date = end_date - timedelta(days=period_to_days[backtest_period])
        else:
            # Default to 1 year if invalid period
            target_start_date = end_date - timedelta(days=365)
            print(f"⚠️ Unknown backtest period '{backtest_period}', defaulting to 1 year")
        
        print(f"🎯 Target backtest period: {backtest_period} (from {target_start_date.date()} to {end_date.date()})")
        
        # Get the actual start and end dates with data for each symbol
        symbol_ranges = {}
        
        for symbol, data in data_dict.items():
            if not data.empty:
                # Find first and last non-NaN close prices
                valid_data = data[data['Close'].notna()]
                if not valid_data.empty:
                    symbol_ranges[symbol] = {
                        'start': valid_data.index[0],
                        'end': valid_data.index[-1],
                        'asset_type': AssetTypeDetector.detect_asset_type(symbol)
                    }
        
        if not symbol_ranges:
            return None, None
            
        # Find the common overlap period, respecting the target backtest period
        # Use the latest start date (but not earlier than target) and earliest end date
        latest_start = max(ranges['start'] for ranges in symbol_ranges.values())
        earliest_end = min(ranges['end'] for ranges in symbol_ranges.values())
        
        # Ensure we don't start earlier than the target start date
        effective_start = max(latest_start, target_start_date)
        
        # Use the earlier of: earliest data end, or current time
        effective_end = min(earliest_end, end_date)
        
        # Validate we have sufficient data for the requested period
        if (effective_end - effective_start).days < 30:
            # Check if we can get any meaningful period with available data
            available_start = min(ranges['start'] for ranges in symbol_ranges.values())
            available_end = max(ranges['end'] for ranges in symbol_ranges.values())
            
            if (available_end - available_start).days >= 30:
                # Use what's available but warn user
                print(f"⚠️ Requested period {backtest_period} not fully available")
                print(f"⚠️ Using available data from {available_start.date()} to {available_end.date()}")
                return available_start, available_end
            else:
                # Really insufficient data
                return None, None
        
        return effective_start, effective_end
    
    def _should_check_date_for_asset(self, date: pd.Timestamp, asset_type: str) -> bool:
        """Determine if we should check for data on this date for this asset type"""
        
        # Crypto assets trade 24/7, so check all dates
        if asset_type == 'crypto':
            return True
        
        # Traditional assets (stocks, ETFs, indexes) only trade on weekdays
        # Skip weekends to avoid false missing data entries
        if AssetTypeDetector.is_weekend(date):
            return False
        
        # Check weekdays for traditional assets
        return True
    
    def _validate_date_access(self, date: pd.Timestamp, data: pd.DataFrame, symbol: str) -> Optional[float]:
        """
        Safely validate and access data for a specific date with intelligent missing data handling.
        
        Args:
            date: The date to access (should be timezone-aware)
            data: The DataFrame containing the data
            symbol: Symbol name for error reporting
            
        Returns:
            Close price if found, None if date is invalid/missing
        """
        try:
            # Get or cache asset type for this symbol
            if symbol not in self.asset_types:
                self.asset_types[symbol] = AssetTypeDetector.detect_asset_type(symbol)
            asset_type = self.asset_types[symbol]
            
            # Ensure date is timezone-aware and matches data timezone
            normalized_date = self._normalize_date_timezone(date, data)
            
            # Check if date exists in index
            if normalized_date not in data.index:
                return self._handle_missing_date(normalized_date, data, symbol, asset_type)
            
            # Access the data
            close_price = data.loc[normalized_date, 'Close']
            
            # Validate the price
            if pd.isna(close_price):
                self._log_missing_data(
                    symbol, normalized_date, asset_type, 
                    reason="NaN close price", is_expected=False
                )
                return None
            
            return float(close_price)
            
        except KeyError as e:
            self._log_missing_data(
                symbol, date, asset_type if 'asset_type' in locals() else 'unknown',
                reason=f"KeyError: {e}", is_expected=False
            )
            return None
        except Exception as e:
            self._log_missing_data(
                symbol, date, asset_type if 'asset_type' in locals() else 'unknown',
                reason=f"Unexpected error: {e}", is_expected=False
            )
            return None
    
    def _normalize_date_timezone(self, date: pd.Timestamp, data: pd.DataFrame) -> pd.Timestamp:
        """Normalize date timezone to match data timezone"""
        if date.tz is None and data.index.tz is not None:
            return date.tz_localize('UTC')
        elif date.tz is not None and data.index.tz is None:
            return date.tz_localize(None)
        elif date.tz != data.index.tz and data.index.tz is not None:
            return date.tz_convert(data.index.tz)
        return date
    
    def _handle_missing_date(self, date: pd.Timestamp, data: pd.DataFrame, 
                           symbol: str, asset_type: str) -> Optional[float]:
        """Handle missing date with asset-type specific logic"""
        
        # For crypto, check tolerance before finding closest date
        if asset_type == 'crypto':
            return self._handle_crypto_missing_date(date, data, symbol)
        
        # For traditional assets (stocks, ETFs, indexes), check if missing data is expected
        is_expected = self._is_expected_missing_data(date, asset_type)
        
        # Try to find the closest business day
        search_window = 7 if asset_type != 'crypto' else 1
        available_dates = self._find_closest_dates(date, data, search_window)
        
        if len(available_dates) == 0:
            self._log_missing_data(
                symbol, date, asset_type,
                reason="No data in search window", is_expected=is_expected
            )
            return None
        
        # Use the closest available date
        closest_date = min(available_dates, key=lambda x: abs((x - date).total_seconds()))
        
        # Only log if unexpected or in debug mode
        if not is_expected:
            self._log_missing_data(
                symbol, date, asset_type,
                reason=f"Using closest date {closest_date}", is_expected=False
            )
        else:
            # Still track expected gaps but don't log verbosely
            self._log_missing_data(
                symbol, date, asset_type,
                reason="Expected market closure", is_expected=True
            )
        
        return float(data.loc[closest_date, 'Close'])
    
    def _handle_crypto_missing_date(self, date: pd.Timestamp, data: pd.DataFrame, 
                                  symbol: str) -> Optional[float]:
        """Handle missing crypto data with tolerance checking"""
        
        # Find closest available dates
        available_dates = self._find_closest_dates(date, data, search_window=3)
        
        if len(available_dates) == 0:
            self._log_missing_data(
                symbol, date, 'crypto',
                reason="No crypto data in 3-day window", is_expected=False
            )
            return None
        
        closest_date = min(available_dates, key=lambda x: abs((x - date).total_seconds()))
        gap_hours = abs((closest_date - date).total_seconds()) / 3600
        
        # Check if gap exceeds tolerance
        if gap_hours > self.missing_data_config.crypto_daily_tolerance_hours:
            self._log_missing_data(
                symbol, date, 'crypto',
                gap_hours=gap_hours,
                reason=f"Gap {gap_hours:.1f}h exceeds tolerance", is_expected=False
            )
            # In strict mode, we might want to return None here
            if self.missing_data_config.strict_mode:
                return None
        else:
            # Gap is within tolerance, don't log unless debugging
            self._log_missing_data(
                symbol, date, 'crypto',
                gap_hours=gap_hours,
                reason=f"Gap {gap_hours:.1f}h within tolerance", is_expected=True
            )
        
        return float(data.loc[closest_date, 'Close'])
    
    def _find_closest_dates(self, target_date: pd.Timestamp, data: pd.DataFrame, 
                          search_window: int) -> pd.DatetimeIndex:
        """Find available dates within search window"""
        start_date = target_date - timedelta(days=search_window)
        end_date = target_date + timedelta(days=search_window)
        
        # For traditional assets, use business days
        if search_window > 1:
            search_range = pd.bdate_range(start=start_date, end=end_date, tz=data.index.tz)
        else:
            # For crypto, use all days
            search_range = pd.date_range(start=start_date, end=end_date, freq='D', tz=data.index.tz)
        
        return data.index.intersection(search_range)
    
    def _is_expected_missing_data(self, date: pd.Timestamp, asset_type: str) -> bool:
        """Determine if missing data is expected (weekends, holidays)"""
        
        # Crypto trades 24/7, so missing data is generally unexpected
        if asset_type == 'crypto':
            return False
        
        # Traditional assets: weekends are expected to be missing
        if AssetTypeDetector.is_weekend(date):
            return True
        
        # TODO: Could add holiday detection here using pandas market calendars
        # For now, assume weekdays should have data
        return False
    
    def _log_missing_data(self, symbol: str, date: pd.Timestamp, asset_type: str,
                         gap_hours: Optional[float] = None, reason: str = "", 
                         is_expected: bool = False):
        """Log missing data entry with appropriate verbosity"""
        
        entry = MissingDataEntry(
            symbol=symbol,
            date=date,
            asset_type=asset_type,
            gap_hours=gap_hours,
            is_expected=is_expected,
            reason=reason
        )
        
        self.missing_data_summary.add_entry(entry, self.missing_data_config)
        
        # Only print verbose logs for unexpected missing data or crypto tolerance violations
        if not is_expected or (asset_type == 'crypto' and gap_hours and 
                              gap_hours > self.missing_data_config.crypto_daily_tolerance_hours):
            if asset_type == 'crypto' and gap_hours:
                print(f"⚠️ {symbol} ({asset_type}): {reason} on {date.strftime('%Y-%m-%d')}")
            else:
                print(f"⚠️ {symbol} ({asset_type}): {reason} on {date.strftime('%Y-%m-%d')}")
    
    def run_backtest(self, symbols: List[str], strategy_name: str, 
                    model_name: str = "LSTM Neural Network", 
                    confidence_threshold: float = 0.75,
                    backtest_period: str = "1y") -> Dict[str, Any]:
        """Run a complete backtest with enhanced error handling and timezone management
        
        Args:
            symbols: List of symbols to backtest
            strategy_name: Trading strategy to use
            model_name: ML model name (future integration)
            confidence_threshold: Minimum signal confidence
            backtest_period: Period for backtest window ("1mo", "6mo", "1y", "3y", "5y")
        """
        self.reset()
        
        try:
            # Fetch comprehensive data with multiple periods and sources
            print("📊 Fetching comprehensive historical data...")
            data_dict = self.fetch_comprehensive_data(symbols)
            
            if not data_dict:
                return {"error": "No data available for selected symbols. Check symbols are valid and markets are open."}
            
            # Get strategy
            if strategy_name not in self.strategies:
                available_strategies = list(self.strategies.keys())
                return {"error": f"Strategy '{strategy_name}' not found. Available strategies: {available_strategies}"}
            
            strategy = self.strategies[strategy_name]
            
            # Validate comprehensive data coverage before proceeding
            print("🔍 Validating data coverage...")
            validation_result = self._validate_comprehensive_data_coverage(data_dict)
            
            if not validation_result['valid']:
                print(f"❌ Data validation failed: {validation_result['reason']}")
                return {"error": f"Data validation failed: {validation_result['reason']}"}
            
            print(f"✅ Data validation passed: {validation_result['summary']}")
            
            # Find the optimal common date range to minimize missing data issues
            print("📅 Finding optimal date range for all symbols...")
            common_start, common_end = self._find_common_date_range(data_dict, backtest_period)
            
            if common_start is None or common_end is None:
                return {"error": "Could not find a common date range for all symbols"}
            
            print(f"📊 Common date range: {common_start.date()} to {common_end.date()}")
            
            # Collect and filter dates intelligently based on asset types and common range
            print("📅 Consolidating trading dates across all symbols...")
            all_dates = set()
            asset_types_in_portfolio = set()
            
            for symbol, data in data_dict.items():
                # Ensure all data is in UTC
                if data.index.tz != pytz.UTC:
                    if data.index.tz is not None:
                        data.index = data.index.tz_convert('UTC')
                    else:
                        data.index = data.index.tz_localize('UTC')
                    data_dict[symbol] = data
                
                # Filter data to common date range
                data_in_range = data[(data.index >= common_start) & (data.index <= common_end)]
                
                # Track asset types in the portfolio
                asset_type = AssetTypeDetector.detect_asset_type(symbol)
                asset_types_in_portfolio.add(asset_type)
                
                all_dates.update(data_in_range.index)
                print(f"  {symbol} ({asset_type}): {len(data_in_range)} trading days in common range")
            
            # Filter dates intelligently based on portfolio composition
            all_dates = sorted(list(all_dates))
            filtered_dates = self._filter_relevant_dates(all_dates, asset_types_in_portfolio)
            
            print(f"📈 Original dates in range: {len(all_dates)}, Filtered dates: {len(filtered_dates)}")
            print(f"📈 Processing {len(filtered_dates)} relevant trading days")
            print(f"📊 Portfolio asset types: {sorted(asset_types_in_portfolio)}")
            
            if len(filtered_dates) < 50:
                return {"error": f"Insufficient relevant trading data in common range. Need at least 50 trading days, got {len(filtered_dates)}"}
            
            # Track portfolio value over time with enhanced error handling
            successful_days = 0
            skipped_days = 0
            
            for i, date in enumerate(filtered_dates):
                try:
                    daily_portfolio_value = self.current_cash
                    
                    # Process each symbol for this date
                    for symbol, data in data_dict.items():
                        # Get or cache asset type for this symbol
                        if symbol not in self.asset_types:
                            self.asset_types[symbol] = AssetTypeDetector.detect_asset_type(symbol)
                        asset_type = self.asset_types[symbol]
                        
                        # Skip dates that are not relevant for this asset type
                        if not self._should_check_date_for_asset(date, asset_type):
                            continue
                        
                        # Safely get current price with validation
                        current_price = self._validate_date_access(date, data, symbol)
                        if current_price is None:
                            continue  # Skip this symbol for this date
                        
                        # Get historical data up to current date
                        try:
                            historical_data = data.loc[:date]
                            if len(historical_data) < 50:  # Need minimum history
                                continue
                        except Exception as e:
                            print(f"⚠️ Error getting historical data for {symbol} on {date}: {e}")
                            continue
                        
                        # Generate trading signal with error handling
                        try:
                            signals = strategy.generate_signals(historical_data)
                            if date not in signals.index:
                                continue
                            
                            signal = signals.loc[date]
                            
                            # Apply confidence threshold
                            if abs(signal) < confidence_threshold:
                                signal = 0
                            
                            # Execute trades
                            self._execute_trade(symbol, date, current_price, signal, strategy)
                            
                        except Exception as e:
                            print(f"⚠️ Error generating signals for {symbol} on {date}: {e}")
                            continue
                        
                        # Add position value to portfolio
                        if symbol in self.positions:
                            trade = self.positions[symbol]
                            daily_portfolio_value += trade.get_current_pnl(current_price)
                    
                    # Record portfolio value
                    self.current_portfolio_value = daily_portfolio_value
                    self.portfolio_value_history.append({
                        'date': date,
                        'portfolio_value': daily_portfolio_value,
                        'cash': self.current_cash
                    })
                    
                    successful_days += 1
                    
                    # Progress reporting for long backtests
                    if i % 50 == 0 or i == len(filtered_dates) - 1:
                        progress = (i + 1) / len(filtered_dates) * 100
                        print(f"  Progress: {progress:.1f}% ({i+1}/{len(filtered_dates)} days)")
                        
                except Exception as e:
                    print(f"⚠️ Error processing date {date}: {e}")
                    skipped_days += 1
                    continue
            
            print(f"📊 Backtest completed: {successful_days} successful days, {skipped_days} skipped days")
            
            if successful_days == 0:
                return {"error": "No trading days were successfully processed. Check data availability and symbol validity."}
            
            # Close all open positions with enhanced error handling
            if all_dates:
                final_date = all_dates[-1]
                print(f"💼 Closing open positions as of {final_date.date()}")
                
                for symbol, trade in list(self.positions.items()):
                    if trade.is_open:
                        try:
                            final_price = self._validate_date_access(final_date, data_dict[symbol], symbol)
                            if final_price is not None:
                                commission = trade.quantity * final_price * self.commission_rate
                                trade.close_trade(final_date, final_price, commission)
                                if trade.direction == 'short':
                                    self.current_cash -= (trade.quantity * final_price + commission)
                                else:
                                    self.current_cash += (trade.quantity * final_price - commission)
                            else:
                                print(f"⚠️ Could not get final price for {symbol}, leaving position open")
                        except Exception as e:
                            print(f"⚠️ Error closing position for {symbol}: {e}")
                
                self.positions.clear()
            
            # Calculate metrics
            results = self._calculate_results()
            
            if "error" in results:
                return results
            
            # Check strict mode violations
            if self.missing_data_config.strict_mode:
                total_gaps = self.missing_data_summary.total_unexpected_gaps
                total_data_points = len(all_dates) * len(symbols)
                missing_ratio = total_gaps / total_data_points if total_data_points > 0 else 0
                
                if missing_ratio > self.missing_data_config.max_missing_data_ratio:
                    return {
                        "error": f"Strict mode: Missing data ratio {missing_ratio:.2%} exceeds threshold {self.missing_data_config.max_missing_data_ratio:.2%}"
                    }
            
            # Generate missing data summary report
            missing_data_report = self._generate_missing_data_report()
            
            results.update({
                'strategy': strategy_name,
                'model': model_name,
                'symbols': symbols,
                'start_date': all_dates[0] if all_dates else None,
                'end_date': all_dates[-1] if all_dates else None,
                'total_days': len(all_dates),
                'successful_days': successful_days,
                'skipped_days': skipped_days,
                'data_quality': f"{successful_days}/{len(all_dates)} days processed successfully",
                'missing_data_summary': missing_data_report
            })
            
            return results
            
        except Exception as e:
            error_msg = f"Error running backtest: {str(e)}"
            print(f"❌ {error_msg}")
            print(f"   Traceback: {traceback.format_exc()}")
            return {"error": error_msg}
    
    def _execute_trade(self, symbol: str, date: datetime, price: float, 
                      signal: float, strategy: TradingStrategy):
        """Execute a trade based on signal"""
        if signal == 0:
            return
        
        # Close existing position if signal changes direction
        if symbol in self.positions:
            existing_trade = self.positions[symbol]
            existing_direction = existing_trade.direction
            new_direction = 'long' if signal > 0 else 'short'
            
            if existing_direction != new_direction:
                # Close existing position
                commission = existing_trade.quantity * price * self.commission_rate
                existing_trade.close_trade(date, price, commission)
                self.trades.append(existing_trade)
                self.current_cash += existing_trade.quantity * price - commission
                if existing_direction == 'short':
                    self.current_cash += existing_trade.quantity * existing_trade.entry_price
                del self.positions[symbol]
        
        # Open new position if we don't have one
        if symbol not in self.positions and abs(signal) > 0:
            direction = 'long' if signal > 0 else 'short'
            
            # Calculate position size
            position_value = strategy.get_position_size(signal, price, self.current_portfolio_value)
            max_position_value = self.current_cash * 0.9  # Use max 90% of cash
            position_value = min(position_value, max_position_value)
            
            if position_value < price:  # Can't afford even 1 share
                return
            
            quantity = position_value // price
            total_cost = quantity * price
            commission = total_cost * self.commission_rate
            
            if total_cost + commission > self.current_cash:
                return
            
            # Create trade
            trade = Trade(symbol, date, price, quantity, direction)
            self.positions[symbol] = trade
            
            # Update cash
            if direction == 'long':
                self.current_cash -= (total_cost + commission)
            else:  # short - receive cash but need to track obligation
                self.current_cash += (total_cost - commission)
    
    def _generate_missing_data_report(self) -> Dict[str, Any]:
        """Generate a comprehensive missing data summary report"""
        summary = self.missing_data_summary
        
        # Print summary to console
        print("\n📊 Missing Data Summary Report")
        print("=" * 40)
        
        if summary.total_expected_gaps == 0 and summary.total_unexpected_gaps == 0:
            print("✅ No missing data issues detected")
            return {
                'total_expected_gaps': 0,
                'total_unexpected_gaps': 0,
                'crypto_tolerance_violations': 0,
                'by_symbol': {},
                'by_asset_type': {},
                'status': 'clean'
            }
        
        print(f"📈 Expected gaps (weekends/holidays): {summary.total_expected_gaps}")
        print(f"⚠️ Unexpected gaps: {summary.total_unexpected_gaps}")
        print(f"🔴 Crypto tolerance violations: {summary.crypto_tolerance_violations}")
        
        # Report by asset type
        if summary.by_asset_type:
            print("\n📊 Missing data by asset type:")
            for asset_type, count in summary.by_asset_type.items():
                print(f"  {asset_type}: {count} gaps")
        
        # Report problematic symbols
        problematic_symbols = []
        for symbol, entries in summary.by_symbol.items():
            unexpected_count = sum(1 for entry in entries if not entry.is_expected)
            if unexpected_count > 0:
                problematic_symbols.append((symbol, unexpected_count))
        
        if problematic_symbols:
            print("\n⚠️ Symbols with unexpected missing data:")
            for symbol, count in sorted(problematic_symbols, key=lambda x: x[1], reverse=True):
                asset_type = self.asset_types.get(symbol, 'unknown')
                print(f"  {symbol} ({asset_type}): {count} unexpected gaps")
        
        # Crypto tolerance violations details
        if summary.crypto_tolerance_violations > 0:
            print(f"\n🔴 Crypto tolerance violations (>{self.missing_data_config.crypto_daily_tolerance_hours}h gaps):")
            for symbol, entries in summary.by_symbol.items():
                if self.asset_types.get(symbol) == 'crypto':
                    violations = [e for e in entries if e.gap_hours and 
                                e.gap_hours > self.missing_data_config.crypto_daily_tolerance_hours]
                    if violations:
                        print(f"  {symbol}: {len(violations)} violations")
                        for violation in violations[:3]:  # Show first 3
                            print(f"    {violation.date.strftime('%Y-%m-%d')}: {violation.gap_hours:.1f}h gap")
                        if len(violations) > 3:
                            print(f"    ... and {len(violations) - 3} more")
        
        return {
            'total_expected_gaps': summary.total_expected_gaps,
            'total_unexpected_gaps': summary.total_unexpected_gaps,
            'crypto_tolerance_violations': summary.crypto_tolerance_violations,
            'by_symbol': {k: len(v) for k, v in summary.by_symbol.items()},
            'by_asset_type': dict(summary.by_asset_type),
            'problematic_symbols': problematic_symbols,
            'status': 'issues_found' if summary.total_unexpected_gaps > 0 else 'clean'
        }
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest results and metrics"""
        # Convert portfolio history to DataFrame
        if not self.portfolio_value_history:
            return {"error": "No portfolio history available"}
        
        portfolio_df = pd.DataFrame(self.portfolio_value_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.set_index('date')
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change().fillna(0)
        
        # Basic metrics
        initial_value = self.initial_capital
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Risk metrics
        returns_series = portfolio_df['returns']
        sharpe_ratio = RiskMetrics.sharpe_ratio(returns_series)
        
        # Maximum drawdown
        portfolio_values = portfolio_df['portfolio_value']
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        completed_trades = [t for t in self.trades if not t.is_open]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Volatility (annualized)
        volatility = returns_series.std() * np.sqrt(252)
        
        return {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'portfolio_history': portfolio_df,
            'trade_log': completed_trades,
            'equity_curve': portfolio_values
        }
    
    def get_trade_details(self) -> pd.DataFrame:
        """Get detailed trade log as DataFrame"""
        completed_trades = [t for t in self.trades if not t.is_open]
        
        if not completed_trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in completed_trades:
            trade_data.append({
                'Symbol': trade.symbol,
                'Direction': trade.direction.title(),
                'Entry Date': trade.entry_date.strftime('%Y-%m-%d'),
                'Entry Price': f"${trade.entry_price:.2f}",
                'Exit Date': trade.exit_date.strftime('%Y-%m-%d'),
                'Exit Price': f"${trade.exit_price:.2f}",
                'Quantity': int(trade.quantity),
                'P&L': f"${trade.pnl:.2f}",
                'Return %': f"{((trade.exit_price - trade.entry_price) / trade.entry_price * 100):.2f}%" if trade.direction == 'long' else f"{((trade.entry_price - trade.exit_price) / trade.entry_price * 100):.2f}%"
            })
        
        return pd.DataFrame(trade_data)


# ========== AUTOMATED OPTIMIZATION FRAMEWORK ==========

import optuna
from typing import Callable
from dataclasses import dataclass, field
import time
import logging
from features.models import AutoModelSelector


@dataclass
class OptimizationConfig:
    """Configuration for automated optimization"""
    n_trials: int = 100
    study_name: str = "automated_backtest_optimization"
    direction: str = "maximize"  # maximize Sharpe ratio
    symbols: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL'])
    timeout: Optional[int] = None
    n_jobs: int = 1
    objective_metric: str = "sharpe_ratio"  # sharpe_ratio, total_return, profit_factor
    

@dataclass
class OptimizationResult:
    """Result from an optimization trial"""
    trial_number: int
    score: float
    model_config: Dict[str, Any]
    strategy_config: Dict[str, Any]
    backtest_config: Dict[str, Any]
    backtest_results: Dict[str, Any]
    optimization_time: float
    

class AutomatedOptimizationBacktest:
    """
    Automated optimization backtesting framework that:
    - Automatically selects model types and strategies
    - Optimizes all parameters using Bayesian optimization
    - Tracks detailed metrics for each configuration
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.model_selector = AutoModelSelector()
        self.study = None
        self.optimization_results = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def optimize(self, symbols: List[str] = None, n_trials: int = None) -> Dict[str, Any]:
        """
        Run automated optimization across models, strategies, and parameters
        
        Args:
            symbols: List of symbols to optimize on (defaults to config symbols)
            n_trials: Number of optimization trials (defaults to config n_trials)
        
        Returns:
            Dictionary with optimization results and leaderboard
        """
        start_time = time.time()
        
        # Use provided symbols or fall back to config
        symbols = symbols or self.config.symbols
        n_trials = n_trials or self.config.n_trials
        
        self.logger.info(f"🚀 Starting automated optimization with {n_trials} trials")
        self.logger.info(f"📊 Optimizing on symbols: {symbols}")
        self.logger.info(f"🎯 Objective: Maximize {self.config.objective_metric}")
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction=self.config.direction,
            study_name=self.config.study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
        )
        
        # Run optimization
        self.study.optimize(
            lambda trial: self._objective_function(trial, symbols),
            n_trials=n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True
        )
        
        optimization_time = time.time() - start_time
        
        # Generate results
        results = self._generate_optimization_results(optimization_time)
        
        self.logger.info(f"✅ Optimization completed in {optimization_time:.2f} seconds")
        self.logger.info(f"📈 Best {self.config.objective_metric}: {self.study.best_value:.4f}")
        
        return results
    
    def _objective_function(self, trial: optuna.Trial, symbols: List[str]) -> float:
        """
        Objective function for optimization trials
        
        Args:
            trial: Optuna trial object
            symbols: List of symbols to backtest
            
        Returns:
            Objective score (higher is better for maximization)
        """
        try:
            # Get configurations from model selector
            model_config = self.model_selector.suggest_model_config(trial)
            strategy_config = self.model_selector.suggest_strategy_config(trial)
            backtest_config = self.model_selector.suggest_backtest_config(trial)
            
            # Create backtest engine with trial-specific configuration
            trading_config = TradingBotConfig()
            trading_config.risk.initial_capital = 100000  # Fixed capital for fair comparison
            
            engine = BacktestEngine(trading_config)
            
            # Run backtest with trial configuration
            results = engine.run_backtest(
                symbols=symbols,
                strategy_name=strategy_config['strategy_name'],
                model_name=model_config['model_name'],
                confidence_threshold=strategy_config.get('confidence_threshold', 0.75),
                backtest_period=backtest_config.get('backtest_period', '1y')
            )
            
            # Handle backtest errors
            if "error" in results:
                self.logger.warning(f"Trial {trial.number} failed: {results['error']}")
                return -np.inf  # Worst possible score
            
            # Calculate objective score
            score = self._calculate_objective_score(results, trial)
            
            # Store detailed results
            optimization_result = OptimizationResult(
                trial_number=trial.number,
                score=score,
                model_config=model_config,
                strategy_config=strategy_config,
                backtest_config=backtest_config,
                backtest_results=results,
                optimization_time=time.time() - trial._start_time if hasattr(trial, '_start_time') else 0
            )
            self.optimization_results.append(optimization_result)
            
            # Log progress
            if trial.number % 10 == 0:
                self.logger.info(f"Trial {trial.number}: {self.config.objective_metric}={score:.4f}")
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}")
            return -np.inf  # Return worst possible score for failed trials
    
    def _calculate_objective_score(self, results: Dict[str, Any], trial: optuna.Trial) -> float:
        """Calculate the objective score based on backtest results"""
        
        if self.config.objective_metric == "sharpe_ratio":
            base_score = results.get('sharpe_ratio', 0)
            
            # Add penalties for poor performance
            win_rate = results.get('win_rate', 0)
            max_drawdown = abs(results.get('max_drawdown', 0))
            total_trades = results.get('total_trades', 0)
            
            # Penalty for low win rate
            if win_rate < 0.4:
                base_score *= 0.8
            
            # Penalty for high drawdown
            if max_drawdown > 0.2:  # More than 20% drawdown
                base_score *= 0.7
            
            # Penalty for too few trades (might be overfitting)
            if total_trades < 5:
                base_score *= 0.9
            
            return base_score
            
        elif self.config.objective_metric == "total_return":
            return results.get('total_return', 0)
            
        elif self.config.objective_metric == "profit_factor":
            return results.get('profit_factor', 0)
            
        else:
            # Default to Sharpe ratio
            return results.get('sharpe_ratio', 0)
    
    def _generate_optimization_results(self, optimization_time: float) -> Dict[str, Any]:
        """Generate comprehensive optimization results"""
        
        if not self.optimization_results:
            return {"error": "No optimization results available"}
        
        # Sort results by score (best first)
        sorted_results = sorted(self.optimization_results, key=lambda x: x.score, reverse=True)
        
        # Create leaderboard (top 10 configurations)
        leaderboard = []
        for i, result in enumerate(sorted_results[:10]):
            leaderboard_entry = {
                'rank': i + 1,
                'trial': result.trial_number,
                'score': result.score,
                'model': result.model_config.get('model_name', 'Unknown'),
                'strategy': result.strategy_config.get('strategy_name', 'Unknown'),
                'total_return_pct': result.backtest_results.get('total_return_pct', 0),
                'sharpe_ratio': result.backtest_results.get('sharpe_ratio', 0),
                'max_drawdown_pct': result.backtest_results.get('max_drawdown_pct', 0),
                'win_rate_pct': result.backtest_results.get('win_rate_pct', 0),
                'total_trades': result.backtest_results.get('total_trades', 0),
                'profit_factor': result.backtest_results.get('profit_factor', 0),
                'volatility_pct': result.backtest_results.get('volatility_pct', 0),
                'backtest_period': result.backtest_config.get('backtest_period', '1y'),
                'confidence_threshold': result.strategy_config.get('confidence_threshold', 0.75),
            }
            leaderboard.append(leaderboard_entry)
        
        # Best configuration details
        best_result = sorted_results[0]
        best_config = {
            'model': best_result.model_config,
            'strategy': best_result.strategy_config,
            'backtest': best_result.backtest_config,
            'full_results': best_result.backtest_results
        }
        
        # Optimization statistics
        all_scores = [r.score for r in self.optimization_results if r.score > -np.inf]
        optimization_stats = {
            'total_trials': len(self.optimization_results),
            'successful_trials': len(all_scores),
            'failed_trials': len(self.optimization_results) - len(all_scores),
            'best_score': max(all_scores) if all_scores else 0,
            'worst_score': min(all_scores) if all_scores else 0,
            'avg_score': np.mean(all_scores) if all_scores else 0,
            'score_std': np.std(all_scores) if all_scores else 0,
            'optimization_time': optimization_time,
            'avg_trial_time': optimization_time / len(self.optimization_results) if self.optimization_results else 0
        }
        
        # Model/strategy performance analysis
        model_performance = {}
        strategy_performance = {}
        
        for result in self.optimization_results:
            if result.score > -np.inf:
                model_name = result.model_config.get('model_name', 'Unknown')
                strategy_name = result.strategy_config.get('strategy_name', 'Unknown')
                
                if model_name not in model_performance:
                    model_performance[model_name] = []
                model_performance[model_name].append(result.score)
                
                if strategy_name not in strategy_performance:
                    strategy_performance[strategy_name] = []
                strategy_performance[strategy_name].append(result.score)
        
        # Calculate averages
        model_avg_performance = {model: np.mean(scores) 
                               for model, scores in model_performance.items()}
        strategy_avg_performance = {strategy: np.mean(scores) 
                                  for strategy, scores in strategy_performance.items()}
        
        return {
            'leaderboard': leaderboard,
            'best_configuration': best_config,
            'optimization_stats': optimization_stats,
            'model_performance': model_avg_performance,
            'strategy_performance': strategy_avg_performance,
            'study': self.study,
            'all_results': sorted_results,
            'config': {
                'objective_metric': self.config.objective_metric,
                'n_trials': self.config.n_trials,
                'symbols': self.config.symbols
            }
        }
    
    def get_best_configuration(self) -> Dict[str, Any]:
        """Get the best configuration found during optimization"""
        if not self.optimization_results:
            return {}
        
        best_result = max(self.optimization_results, key=lambda x: x.score)
        return {
            'model_config': best_result.model_config,
            'strategy_config': best_result.strategy_config,
            'backtest_config': best_result.backtest_config,
            'score': best_result.score,
            'results': best_result.backtest_results
        }
    
    def get_trial_details(self, trial_number: int) -> Optional[OptimizationResult]:
        """Get detailed results for a specific trial"""
        for result in self.optimization_results:
            if result.trial_number == trial_number:
                return result
        return None