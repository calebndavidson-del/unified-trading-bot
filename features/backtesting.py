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
        if (entry.asset_type == 'crypto' and entry.gap_hours and config and 
            entry.gap_hours > config.crypto_daily_tolerance_hours):
            self.crypto_tolerance_violations += 1


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
                    confidence_threshold: float = 0.75) -> Dict[str, Any]:
        """Run a complete backtest with enhanced error handling and timezone management"""
        self.reset()
        
        try:
            # Fetch data
            print("📊 Fetching current year historical data...")
            data_dict = self.fetch_current_year_data(symbols)
            
            if not data_dict:
                return {"error": "No data available for selected symbols. Check symbols are valid and markets are open."}
            
            # Get strategy
            if strategy_name not in self.strategies:
                available_strategies = list(self.strategies.keys())
                return {"error": f"Strategy '{strategy_name}' not found. Available strategies: {available_strategies}"}
            
            strategy = self.strategies[strategy_name]
            
            # Collect all unique dates across all symbols (in UTC)
            print("📅 Consolidating trading dates across all symbols...")
            all_dates = set()
            for symbol, data in data_dict.items():
                # Ensure all data is in UTC
                if data.index.tz != pytz.UTC:
                    if data.index.tz is not None:
                        data.index = data.index.tz_convert('UTC')
                    else:
                        data.index = data.index.tz_localize('UTC')
                    data_dict[symbol] = data
                
                all_dates.update(data.index)
                print(f"  {symbol}: {len(data)} trading days")
            
            all_dates = sorted(list(all_dates))
            print(f"📈 Processing {len(all_dates)} unique trading days from {all_dates[0].date()} to {all_dates[-1].date()}")
            
            if len(all_dates) < 50:
                return {"error": f"Insufficient data for backtesting. Need at least 50 trading days, got {len(all_dates)}"}
            
            # Track portfolio value over time with enhanced error handling
            successful_days = 0
            skipped_days = 0
            
            for i, date in enumerate(all_dates):
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
                    if i % 50 == 0 or i == len(all_dates) - 1:
                        progress = (i + 1) / len(all_dates) * 100
                        print(f"  Progress: {progress:.1f}% ({i+1}/{len(all_dates)} days)")
                        
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