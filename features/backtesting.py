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
        """Fetch historical data for current year only"""
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
                    print(f"No data available for {symbol}")
                    continue
                
                # Ensure required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in data.columns for col in required_cols):
                    data_dict[symbol] = data
                    print(f"✅ Fetched {len(data)} days of data for {symbol}")
                else:
                    print(f"⚠️ Missing required columns for {symbol}")
                    
            except Exception as e:
                print(f"❌ Error fetching data for {symbol}: {e}")
        
        return data_dict
    
    def run_backtest(self, symbols: List[str], strategy_name: str, 
                    model_name: str = "LSTM Neural Network", 
                    confidence_threshold: float = 0.75) -> Dict[str, Any]:
        """Run a complete backtest"""
        self.reset()
        
        # Fetch data
        print("📊 Fetching current year historical data...")
        data_dict = self.fetch_current_year_data(symbols)
        
        if not data_dict:
            return {"error": "No data available for selected symbols"}
        
        # Get strategy
        if strategy_name not in self.strategies:
            return {"error": f"Strategy '{strategy_name}' not found"}
        
        strategy = self.strategies[strategy_name]
        
        # Run backtest for each symbol
        all_dates = set()
        for data in data_dict.values():
            all_dates.update(data.index)
        
        all_dates = sorted(list(all_dates))
        
        # Track portfolio value over time
        for date in all_dates:
            daily_portfolio_value = self.current_cash
            
            # Process each symbol
            for symbol, data in data_dict.items():
                if date not in data.index:
                    continue
                
                current_price = data.loc[date, 'Close']
                
                # Get historical data up to current date
                historical_data = data.loc[:date]
                if len(historical_data) < 50:  # Need minimum history
                    continue
                
                # Generate trading signal
                signals = strategy.generate_signals(historical_data)
                if date not in signals.index:
                    continue
                
                signal = signals.loc[date]
                
                # Apply confidence threshold
                if abs(signal) < confidence_threshold:
                    signal = 0
                
                # Execute trades
                self._execute_trade(symbol, date, current_price, signal, strategy)
                
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
        
        # Close all open positions
        final_date = all_dates[-1] if all_dates else datetime.now()
        for symbol, trade in list(self.positions.items()):
            if trade.is_open:
                final_price = data_dict[symbol].loc[final_date, 'Close']
                commission = trade.quantity * final_price * self.commission_rate
                trade.close_trade(final_date, final_price, commission)
                if trade.direction == 'short':
                    self.current_cash -= (trade.quantity * final_price + commission)
                else:
                    self.current_cash += (trade.quantity * final_price - commission)
        self.positions.clear()
        
        # Calculate metrics
        results = self._calculate_results()
        results.update({
            'strategy': strategy_name,
            'model': model_name,
            'symbols': symbols,
            'start_date': all_dates[0] if all_dates else None,
            'end_date': all_dates[-1] if all_dates else None,
            'total_days': len(all_dates)
        })
        
        return results
    
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