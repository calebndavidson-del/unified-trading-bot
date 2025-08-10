#!/usr/bin/env python3
"""
Formal Parameter Schema for Algorithmic Bot Backtesting

This module provides a comprehensive parameter schema for algorithmic trading bot
backtesting, optimization, and live parameter tracking. All parameters are grouped
by category and include sensible default, minimum, and maximum values for grid
search and random search optimization.

Usage:
    from parameters import BacktestParameters, get_parameter_ranges
    
    # Get default parameters
    params = BacktestParameters()
    
    # Get parameter ranges for optimization
    ranges = get_parameter_ranges()
    
    # Create parameters from dictionary
    custom_params = BacktestParameters.from_dict({
        'ema_fast': 10,
        'ema_slow': 25,
        'risk_per_trade_pct': 1.5
    })
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Union, Tuple
import json


@dataclass
class MarketTimeframeParams:
    """Market and Timeframe Parameters"""
    asset_class: str = "equity"  # equity, crypto, forex, futures
    candle_timeframe: str = "5m"  # 1m, 5m, 15m, 1h, 4h, 1d
    session_start: str = "09:30"  # Market session start time (HH:MM)
    session_end: str = "16:00"    # Market session end time (HH:MM)


@dataclass
class EntryParams:
    """Entry Signal Parameters"""
    volatility_atr_min: float = 0.5      # Minimum ATR for trade entry
    ema_fast: int = 12                   # Fast EMA period
    ema_slow: int = 26                   # Slow EMA period
    rsi_period: int = 14                 # RSI calculation period
    rsi_overbought: float = 70.0         # RSI overbought threshold
    rsi_oversold: float = 30.0           # RSI oversold threshold
    breakout_lookback: int = 20          # Lookback period for breakout detection


@dataclass
class ExitParams:
    """Exit Signal Parameters"""
    profit_target_mult: float = 2.0     # Profit target as multiple of ATR
    stop_loss_pct: float = 2.0          # Stop loss percentage
    trailing_stop: bool = True           # Enable trailing stop
    exit_on_signal: bool = True          # Exit on opposite signal


@dataclass
class PositionSizingParams:
    """Position Sizing Parameters"""
    risk_per_trade_pct: float = 1.0     # Risk per trade as % of capital
    leverage: float = 1.0                # Trading leverage
    max_open_positions: int = 5          # Maximum concurrent positions


@dataclass
class FrequencyControlParams:
    """Trade Frequency Control Parameters"""
    max_trades_per_day: int = 10         # Maximum trades per day
    cooldown_minutes: int = 30           # Cooldown period between trades (minutes)


@dataclass
class ExecutionParams:
    """Execution Quality Parameters"""
    spread_max: float = 0.05            # Maximum spread tolerance (%)
    volume_min: int = 100000            # Minimum volume requirement


@dataclass
class BacktestConstraintParams:
    """Backtest Performance Constraints"""
    max_drawdown_pct: float = 15.0      # Maximum acceptable drawdown (%)
    min_sharpe_ratio: float = 1.0       # Minimum Sharpe ratio requirement
    min_profit_factor: float = 1.2      # Minimum profit factor requirement


@dataclass
class AdditionalParams:
    """Additional Parameters"""
    lookback: int = 252                 # Lookback period for walk-forward/rolling optimization (days)


@dataclass
class BacktestParameters:
    """
    Complete parameter schema for algorithmic bot backtesting.
    
    This dataclass consolidates all trading parameters into a single structure
    that can be used for backtesting, optimization, and live trading parameter
    tracking.
    """
    
    # Parameter categories
    market_timeframe: MarketTimeframeParams = field(default_factory=MarketTimeframeParams)
    entry: EntryParams = field(default_factory=EntryParams)
    exit: ExitParams = field(default_factory=ExitParams)
    position_sizing: PositionSizingParams = field(default_factory=PositionSizingParams)
    frequency_control: FrequencyControlParams = field(default_factory=FrequencyControlParams)
    execution: ExecutionParams = field(default_factory=ExecutionParams)
    backtest_constraints: BacktestConstraintParams = field(default_factory=BacktestConstraintParams)
    additional: AdditionalParams = field(default_factory=AdditionalParams)
    
    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'BacktestParameters':
        """
        Create BacktestParameters from a dictionary.
        
        Args:
            params_dict: Dictionary with parameter names as keys
            
        Returns:
            BacktestParameters instance with updated values
        """
        instance = cls()
        
        # Map of parameter names to their category and attribute
        param_mapping = {
            # Market & Timeframe
            'asset_class': ('market_timeframe', 'asset_class'),
            'candle_timeframe': ('market_timeframe', 'candle_timeframe'),
            'session_start': ('market_timeframe', 'session_start'),
            'session_end': ('market_timeframe', 'session_end'),
            
            # Entry
            'volatility_atr_min': ('entry', 'volatility_atr_min'),
            'ema_fast': ('entry', 'ema_fast'),
            'ema_slow': ('entry', 'ema_slow'),
            'rsi_period': ('entry', 'rsi_period'),
            'rsi_overbought': ('entry', 'rsi_overbought'),
            'rsi_oversold': ('entry', 'rsi_oversold'),
            'breakout_lookback': ('entry', 'breakout_lookback'),
            
            # Exit
            'profit_target_mult': ('exit', 'profit_target_mult'),
            'stop_loss_pct': ('exit', 'stop_loss_pct'),
            'trailing_stop': ('exit', 'trailing_stop'),
            'exit_on_signal': ('exit', 'exit_on_signal'),
            
            # Position Sizing
            'risk_per_trade_pct': ('position_sizing', 'risk_per_trade_pct'),
            'leverage': ('position_sizing', 'leverage'),
            'max_open_positions': ('position_sizing', 'max_open_positions'),
            
            # Frequency Control
            'max_trades_per_day': ('frequency_control', 'max_trades_per_day'),
            'cooldown_minutes': ('frequency_control', 'cooldown_minutes'),
            
            # Execution
            'spread_max': ('execution', 'spread_max'),
            'volume_min': ('execution', 'volume_min'),
            
            # Backtest Constraints
            'max_drawdown_pct': ('backtest_constraints', 'max_drawdown_pct'),
            'min_sharpe_ratio': ('backtest_constraints', 'min_sharpe_ratio'),
            'min_profit_factor': ('backtest_constraints', 'min_profit_factor'),
            
            # Additional
            'lookback': ('additional', 'lookback'),
        }
        
        for param_name, value in params_dict.items():
            if param_name in param_mapping:
                category, attr = param_mapping[param_name]
                category_obj = getattr(instance, category)
                setattr(category_obj, attr, value)
        
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert BacktestParameters to a flat dictionary.
        
        Returns:
            Dictionary with all parameters as key-value pairs
        """
        result = {}
        
        # Market & Timeframe
        result['asset_class'] = self.market_timeframe.asset_class
        result['candle_timeframe'] = self.market_timeframe.candle_timeframe
        result['session_start'] = self.market_timeframe.session_start
        result['session_end'] = self.market_timeframe.session_end
        
        # Entry
        result['volatility_atr_min'] = self.entry.volatility_atr_min
        result['ema_fast'] = self.entry.ema_fast
        result['ema_slow'] = self.entry.ema_slow
        result['rsi_period'] = self.entry.rsi_period
        result['rsi_overbought'] = self.entry.rsi_overbought
        result['rsi_oversold'] = self.entry.rsi_oversold
        result['breakout_lookback'] = self.entry.breakout_lookback
        
        # Exit
        result['profit_target_mult'] = self.exit.profit_target_mult
        result['stop_loss_pct'] = self.exit.stop_loss_pct
        result['trailing_stop'] = self.exit.trailing_stop
        result['exit_on_signal'] = self.exit.exit_on_signal
        
        # Position Sizing
        result['risk_per_trade_pct'] = self.position_sizing.risk_per_trade_pct
        result['leverage'] = self.position_sizing.leverage
        result['max_open_positions'] = self.position_sizing.max_open_positions
        
        # Frequency Control
        result['max_trades_per_day'] = self.frequency_control.max_trades_per_day
        result['cooldown_minutes'] = self.frequency_control.cooldown_minutes
        
        # Execution
        result['spread_max'] = self.execution.spread_max
        result['volume_min'] = self.execution.volume_min
        
        # Backtest Constraints
        result['max_drawdown_pct'] = self.backtest_constraints.max_drawdown_pct
        result['min_sharpe_ratio'] = self.backtest_constraints.min_sharpe_ratio
        result['min_profit_factor'] = self.backtest_constraints.min_profit_factor
        
        # Additional
        result['lookback'] = self.additional.lookback
        
        return result
    
    def save_to_json(self, filepath: str) -> None:
        """Save parameters to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'BacktestParameters':
        """Load parameters from JSON file."""
        with open(filepath, 'r') as f:
            params_dict = json.load(f)
        return cls.from_dict(params_dict)


def get_parameter_ranges() -> Dict[str, Dict[str, Union[Tuple[float, float], Tuple[int, int], list]]]:
    """
    Get parameter ranges for optimization (grid search, random search).
    
    Returns:
        Dictionary with parameter names as keys and range specifications as values.
        Each range specification contains 'min', 'max', and 'default' values.
        For categorical parameters, 'choices' list is provided instead.
    """
    return {
        # Market & Timeframe
        'asset_class': {
            'choices': ['equity', 'crypto', 'forex', 'futures'],
            'default': 'equity'
        },
        'candle_timeframe': {
            'choices': ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
            'default': '5m'
        },
        'session_start': {
            'choices': ['09:30', '09:00', '08:30', '00:00'],  # Different market sessions
            'default': '09:30'
        },
        'session_end': {
            'choices': ['16:00', '16:30', '17:00', '23:59'],  # Different market sessions
            'default': '16:00'
        },
        
        # Entry Parameters
        'volatility_atr_min': {
            'min': 0.1,
            'max': 2.0,
            'default': 0.5
        },
        'ema_fast': {
            'min': 5,
            'max': 50,
            'default': 12
        },
        'ema_slow': {
            'min': 15,
            'max': 100,
            'default': 26
        },
        'rsi_period': {
            'min': 5,
            'max': 50,
            'default': 14
        },
        'rsi_overbought': {
            'min': 60.0,
            'max': 90.0,
            'default': 70.0
        },
        'rsi_oversold': {
            'min': 10.0,
            'max': 40.0,
            'default': 30.0
        },
        'breakout_lookback': {
            'min': 5,
            'max': 100,
            'default': 20
        },
        
        # Exit Parameters
        'profit_target_mult': {
            'min': 1.0,
            'max': 5.0,
            'default': 2.0
        },
        'stop_loss_pct': {
            'min': 0.5,
            'max': 10.0,
            'default': 2.0
        },
        'trailing_stop': {
            'choices': [True, False],
            'default': True
        },
        'exit_on_signal': {
            'choices': [True, False],
            'default': True
        },
        
        # Position Sizing
        'risk_per_trade_pct': {
            'min': 0.1,
            'max': 5.0,
            'default': 1.0
        },
        'leverage': {
            'min': 1.0,
            'max': 10.0,
            'default': 1.0
        },
        'max_open_positions': {
            'min': 1,
            'max': 20,
            'default': 5
        },
        
        # Frequency Control
        'max_trades_per_day': {
            'min': 1,
            'max': 50,
            'default': 10
        },
        'cooldown_minutes': {
            'min': 0,
            'max': 240,
            'default': 30
        },
        
        # Execution
        'spread_max': {
            'min': 0.01,
            'max': 1.0,
            'default': 0.05
        },
        'volume_min': {
            'min': 1000,
            'max': 1000000,
            'default': 100000
        },
        
        # Backtest Constraints
        'max_drawdown_pct': {
            'min': 5.0,
            'max': 50.0,
            'default': 15.0
        },
        'min_sharpe_ratio': {
            'min': 0.5,
            'max': 3.0,
            'default': 1.0
        },
        'min_profit_factor': {
            'min': 1.0,
            'max': 3.0,
            'default': 1.2
        },
        
        # Additional
        'lookback': {
            'min': 30,
            'max': 1000,
            'default': 252
        }
    }


def generate_random_parameters(seed: int = None) -> BacktestParameters:
    """
    Generate random parameters within valid ranges for optimization.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        BacktestParameters with randomized values
    """
    import random
    
    if seed is not None:
        random.seed(seed)
    
    ranges = get_parameter_ranges()
    params = {}
    
    for param_name, range_spec in ranges.items():
        if 'choices' in range_spec:
            params[param_name] = random.choice(range_spec['choices'])
        elif 'min' in range_spec and 'max' in range_spec:
            min_val, max_val = range_spec['min'], range_spec['max']
            if isinstance(min_val, int):
                params[param_name] = random.randint(min_val, max_val)
            else:
                params[param_name] = random.uniform(min_val, max_val)
    
    return BacktestParameters.from_dict(params)


def get_parameter_grid(param_subset: Dict[str, list] = None) -> list:
    """
    Generate a parameter grid for grid search optimization.
    
    Args:
        param_subset: Optional subset of parameters to include in grid.
                     If None, uses a sensible default subset.
                     
    Returns:
        List of parameter dictionaries for grid search
    """
    if param_subset is None:
        # Default subset for grid search (avoid too many combinations)
        param_subset = {
            'ema_fast': [8, 12, 16],
            'ema_slow': [21, 26, 30],
            'rsi_period': [10, 14, 18],
            'rsi_overbought': [65, 70, 75],
            'rsi_oversold': [25, 30, 35],
            'profit_target_mult': [1.5, 2.0, 2.5],
            'stop_loss_pct': [1.0, 2.0, 3.0],
            'risk_per_trade_pct': [0.5, 1.0, 1.5]
        }
    
    from itertools import product
    
    param_names = list(param_subset.keys())
    param_values = list(param_subset.values())
    
    grid = []
    for combination in product(*param_values):
        params = dict(zip(param_names, combination))
        grid.append(params)
    
    return grid


# Convenience function for backward compatibility
def get_default_parameters() -> BacktestParameters:
    """Get default parameters for backtesting."""
    return BacktestParameters()


if __name__ == "__main__":
    # Example usage
    print("=== Formal Parameter Schema Example ===")
    
    # Create default parameters
    params = BacktestParameters()
    print(f"Default RSI period: {params.entry.rsi_period}")
    print(f"Default risk per trade: {params.position_sizing.risk_per_trade_pct}%")
    
    # Create from dictionary
    custom_params = BacktestParameters.from_dict({
        'ema_fast': 10,
        'ema_slow': 25,
        'rsi_period': 21,
        'risk_per_trade_pct': 1.5
    })
    print(f"Custom EMA fast: {custom_params.entry.ema_fast}")
    
    # Generate random parameters
    random_params = generate_random_parameters(seed=42)
    print(f"Random profit target: {random_params.exit.profit_target_mult}")
    
    # Get parameter ranges
    ranges = get_parameter_ranges()
    print(f"RSI period range: {ranges['rsi_period']['min']}-{ranges['rsi_period']['max']}")
    
    print("\n=== Parameter Categories ===")
    print("Market/Timeframe:", list(vars(params.market_timeframe).keys()))
    print("Entry:", list(vars(params.entry).keys()))
    print("Exit:", list(vars(params.exit).keys()))
    print("Position Sizing:", list(vars(params.position_sizing).keys()))
    print("Frequency Control:", list(vars(params.frequency_control).keys()))
    print("Execution:", list(vars(params.execution).keys()))
    print("Backtest Constraints:", list(vars(params.backtest_constraints).keys()))
    print("Additional:", list(vars(params.additional).keys()))