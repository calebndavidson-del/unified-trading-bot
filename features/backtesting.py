#!/usr/bin/env python3
"""
Backtesting Engine for Trading Strategies
Simulate trades, positions, and portfolio performance using historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import yfinance as yf
import pytz

# Import existing modules
from features.market_trend import create_comprehensive_trend_features
from features.candlestick import extract_all_candlestick_features
from utils.risk import RiskMetrics, PositionSizing
from model_config import TradingBotConfig

warnings.filterwarnings('ignore')


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
    
    def __init__(self, config: TradingBotConfig = None):
        self.config = config or TradingBotConfig()
        self.strategies = {
            'Technical Analysis': TechnicalAnalysisStrategy(),
            'Mean Reversion': MeanReversionStrategy(),
            'Momentum': MomentumStrategy(),
            'Pattern Recognition': PatternRecognitionStrategy()
        }
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
                    if not info or ('quoteType' not in info and 'regularMarketPrice' not in info):
                        print(f"❌ No data available for {symbol}: symbol appears to be invalid.")
                    else:
                        print(f"⚠️ No data available for {symbol}: market may be closed for the requested period.")
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
    
    def _validate_date_access(self, date: pd.Timestamp, data: pd.DataFrame, symbol: str) -> Optional[float]:
        """
        Safely validate and access data for a specific date with detailed error handling.
        
        Args:
            date: The date to access (should be timezone-aware)
            data: The DataFrame containing the data
            symbol: Symbol name for error reporting
            
        Returns:
            Close price if found, None if date is invalid/missing
        """
        try:
            # Ensure date is timezone-aware and matches data timezone
            if date.tz is None and data.index.tz is not None:
                date = date.tz_localize('UTC')
            elif date.tz is not None and data.index.tz is None:
                date = date.tz_localize(None)
            elif date.tz != data.index.tz and data.index.tz is not None:
                date = date.tz_convert(data.index.tz)
            
            # Check if date exists in index
            if date not in data.index:
                # Try to find the closest business day
                business_days = pd.bdate_range(start=date - timedelta(days=7), 
                                             end=date + timedelta(days=7), 
                                             tz=data.index.tz)
                available_dates = data.index.intersection(business_days)
                
                if len(available_dates) == 0:
                    print(f"⚠️ No data available for {symbol} around {date} (likely holiday period)")
                    return None
                else:
                    # Use the closest available date
                    closest_date = min(available_dates, key=lambda x: abs((x - date).total_seconds()))
                    print(f"⚠️ Date {date} not available for {symbol}, using closest date {closest_date}")
                    date = closest_date
            
            # Access the data
            close_price = data.loc[date, 'Close']
            
            # Validate the price
            if pd.isna(close_price):
                print(f"⚠️ Close price is NaN for {symbol} on {date}")
                return None
            
            return float(close_price)
            
        except KeyError as e:
            print(f"❌ KeyError accessing {symbol} data for {date}: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error accessing {symbol} data for {date}: {e}")
            import traceback
            print(f"   Details: {traceback.format_exc()}")
            return None
    
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
            
            results.update({
                'strategy': strategy_name,
                'model': model_name,
                'symbols': symbols,
                'start_date': all_dates[0] if all_dates else None,
                'end_date': all_dates[-1] if all_dates else None,
                'total_days': len(all_dates),
                'successful_days': successful_days,
                'skipped_days': skipped_days,
                'data_quality': f"{successful_days}/{len(all_dates)} days processed successfully"
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