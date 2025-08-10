#!/usr/bin/env python3
"""
Advanced Optimization Engine for Trading Bot
Provides parallel parameter optimization with progress tracking and caching
"""

import time
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import yfinance as yf
import os
import logging

from parameter_manager import ParameterManager


@dataclass
class OptimizationResult:
    """Single optimization result containing all performance metrics"""
    symbol: str
    parameters: Dict[str, Any]
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trading metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Portfolio metrics
    final_value: float
    equity_curve: pd.Series
    trades_list: List[Dict[str, Any]]
    
    # Timing and metadata
    optimization_time: float
    backtest_start: str
    backtest_end: str
    data_quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'parameters': self.parameters,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'final_value': self.final_value,
            'trades_count': len(self.trades_list),
            'optimization_time': self.optimization_time,
            'backtest_start': self.backtest_start,
            'backtest_end': self.backtest_end,
            'data_quality_score': self.data_quality_score
        }


@dataclass
class OptimizationSummary:
    """Summary of complete optimization run"""
    total_combinations: int
    successful_runs: int
    failed_runs: int
    best_result: Optional[OptimizationResult]
    worst_result: Optional[OptimizationResult]
    average_metrics: Dict[str, float]
    total_time: float
    results: List[OptimizationResult] = field(default_factory=list)
    
    def get_top_results(self, n: int = 10, metric: str = 'sharpe_ratio') -> List[OptimizationResult]:
        """Get top N results by specified metric"""
        return sorted(self.results, key=lambda x: getattr(x, metric, 0), reverse=True)[:n]


class OptimizationEngine:
    """Advanced optimization engine with parallel processing and caching"""
    
    def __init__(self, cache_dir: str = "cache/optimization", max_workers: int = 8, log_level: int = logging.INFO):
        self.cache_dir = cache_dir
        self.max_workers = max_workers
        self.cache_enabled = True
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(cache_dir, 'optimization.log'))
            ]
        )
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Progress tracking
        self.progress_callback: Optional[Callable] = None
        self.current_progress = 0
        self.total_progress = 0
        
        # Test Yahoo Finance connection at startup
        self.yahoo_connection_ok = self._test_yahoo_finance_connection()
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """Set callback for progress updates (current, total, status)"""
        self.progress_callback = callback
    
    def run_optimization(self, 
                         parameter_manager: ParameterManager,
                         symbols: List[str],
                         days: int,
                         objective: str = 'sharpe_ratio',
                         max_combinations: Optional[int] = None) -> OptimizationSummary:
        """
        Run comprehensive parameter optimization
        
        Args:
            parameter_manager: ParameterManager with defined parameters
            symbols: List of symbols to optimize
            days: Number of days for backtesting
            objective: Optimization objective ('sharpe_ratio', 'total_return', 'calmar_ratio', etc.)
            max_combinations: Limit number of combinations to test (for large parameter spaces)
            
        Returns:
            OptimizationSummary with all results
        """
        start_time = time.time()
        
        # Get parameter combinations
        combinations = parameter_manager.get_parameter_combinations()
        
        # Limit combinations if specified
        if max_combinations and len(combinations) > max_combinations:
            # Use random sampling to get diverse parameter combinations
            import random
            combinations = random.sample(combinations, max_combinations)
        
        total_combinations = len(combinations) * len(symbols)
        self.total_progress = total_combinations
        self.current_progress = 0
        
        if self.progress_callback:
            self.progress_callback(0, total_combinations, "Starting optimization...")
        
        # Run optimization in parallel
        results = []
        failed_runs = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all optimization tasks
            future_to_params = {}
            
            for symbol in symbols:
                for params in combinations:
                    future = executor.submit(self._optimize_single_combination, symbol, params, days)
                    future_to_params[future] = (symbol, params)
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                symbol, params = future_to_params[future]
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    else:
                        failed_runs += 1
                except Exception as e:
                    print(f"Optimization failed for {symbol} with params {params}: {e}")
                    failed_runs += 1
                
                self.current_progress += 1
                if self.progress_callback:
                    progress_pct = (self.current_progress / total_combinations) * 100
                    status = f"Completed {self.current_progress}/{total_combinations} ({progress_pct:.1f}%)"
                    self.progress_callback(self.current_progress, total_combinations, status)
        
        # Create optimization summary
        total_time = time.time() - start_time
        summary = self._create_optimization_summary(results, total_combinations, failed_runs, total_time)
        
        if self.progress_callback:
            self.progress_callback(total_combinations, total_combinations, 
                                   f"Optimization complete! Best {objective}: {getattr(summary.best_result, objective, 0):.3f}")
        
        return summary
    
    def _optimize_single_combination(self, symbol: str, params: Dict[str, Any], days: int) -> Optional[OptimizationResult]:
        """Optimize a single parameter combination"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(symbol, params, days)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Run backtest
            backtest_result = self._run_backtest(symbol, params, days)
            if not backtest_result:
                return None
            
            # Calculate comprehensive metrics
            result = self._calculate_comprehensive_metrics(symbol, params, backtest_result, time.time() - start_time)
            
            # Cache the result
            if self.cache_enabled:
                self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            print(f"Error optimizing {symbol} with {params}: {e}")
            return None
    
    def _run_backtest(self, symbol: str, params: Dict[str, Any], days: int) -> Optional[Dict[str, Any]]:
        """Run backtest with given parameters using robust data retrieval"""
        try:
            # Get market data with retry logic
            df = self._get_yahoo_data_with_retry(symbol, days)
            if df is None or df.empty:
                logging.warning(f"No data available for {symbol} after retries")
                return None
            
            # Validate data quality
            if not self._validate_data_quality(df, symbol):
                logging.warning(f"Poor data quality for {symbol}, skipping backtest")
                return None
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(df, params)
            
            # Run trading simulation
            simulation_result = self._run_trading_simulation(df, indicators, params)
            
            logging.info(f"Successful backtest for {symbol}: {len(df)} data points")
            return simulation_result
            
        except Exception as e:
            # Log the error with details but don't fall back to mock data
            logging.error(f"Backtest failed for {symbol}: {type(e).__name__}: {e}")
            return None
    
    def _get_yahoo_data_with_retry(self, symbol: str, days: int, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Get Yahoo Finance data with exponential backoff retry logic"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for attempt in range(max_retries):
            try:
                logging.info(f"Attempting to fetch data for {symbol} (attempt {attempt + 1}/{max_retries})")
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if not df.empty and len(df) > 0:
                    logging.info(f"Successfully retrieved {len(df)} data points for {symbol}")
                    return df
                else:
                    raise ValueError(f"Empty dataset returned for {symbol}")
                    
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed for {symbol}: {type(e).__name__}: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logging.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All {max_retries} attempts failed for {symbol}")
                    return None
        
        return None
    
    def _validate_data_quality(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate market data quality before proceeding with backtest"""
        try:
            # Check if DataFrame is empty
            if df.empty:
                logging.warning(f"Data validation failed for {symbol}: Empty dataset")
                return False
            
            # Check minimum number of trading days
            min_days = 20
            if len(df) < min_days:
                logging.warning(f"Data validation failed for {symbol}: Only {len(df)} days, need at least {min_days}")
                return False
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.warning(f"Data validation failed for {symbol}: Missing columns {missing_columns}")
                return False
            
            # Check for valid price data (no zeros or negative prices)
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if (df[col] <= 0).any():
                    logging.warning(f"Data validation failed for {symbol}: Invalid prices in {col} column")
                    return False
            
            # Check for volume data (at least some non-zero volume)
            if df['Volume'].sum() == 0:
                logging.warning(f"Data validation failed for {symbol}: No volume data")
                return False
            
            # Check for too many missing values
            missing_pct = df.isnull().sum().max() / len(df)
            if missing_pct > 0.1:  # More than 10% missing
                logging.warning(f"Data validation failed for {symbol}: {missing_pct:.1%} missing data")
                return False
            
            # Check for realistic price ranges (no extreme outliers)
            for col in price_columns:
                price_std = df[col].std()
                price_mean = df[col].mean()
                if price_std > price_mean * 2:  # Very high volatility might indicate bad data
                    logging.warning(f"Data validation failed for {symbol}: Extreme volatility in {col}")
                    return False
            
            logging.info(f"Data validation passed for {symbol}: {len(df)} days of quality data")
            return True
            
        except Exception as e:
            logging.error(f"Data validation error for {symbol}: {e}")
            return False
    
    def _test_yahoo_finance_connection(self) -> bool:
        """Test connection to Yahoo Finance API"""
        try:
            logging.info("Testing Yahoo Finance API connection...")
            test_ticker = yf.Ticker("SPY")
            test_data = test_ticker.history(period="5d")
            
            if not test_data.empty:
                logging.info("Yahoo Finance API connection test passed")
                return True
            else:
                logging.warning("Yahoo Finance API connection test failed: empty data")
                return False
                
        except Exception as e:
            logging.warning(f"Yahoo Finance API connection test failed: {e}")
            return False
    
    def _calculate_technical_indicators(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Calculate technical indicators based on parameters"""
        indicators = {}
        
        # RSI
        rsi_period = params.get('rsi_period', 14)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2.0)
        bb_ma = df['Close'].rolling(window=bb_period).mean()
        bb_upper = bb_ma + (df['Close'].rolling(window=bb_period).std() * bb_std)
        bb_lower = bb_ma - (df['Close'].rolling(window=bb_period).std() * bb_std)
        
        indicators['bb_upper'] = bb_upper
        indicators['bb_lower'] = bb_lower
        indicators['bb_ma'] = bb_ma
        
        # Moving averages
        indicators['ma_short'] = df['Close'].rolling(window=10).mean()
        indicators['ma_long'] = df['Close'].rolling(window=20).mean()
        
        return indicators
    
    def _run_trading_simulation(self, df: pd.DataFrame, indicators: Dict[str, pd.Series], 
                                params: Dict[str, Any]) -> Dict[str, Any]:
        """Run trading simulation with given indicators and parameters"""
        
        position = 0
        cash = params.get('starting_capital', 100000)
        initial_capital = cash
        shares = 0
        
        portfolio_values = []
        trades = []
        
        # Parameters
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)
        position_size = params.get('position_size', 0.1)
        stop_loss = params.get('stop_loss', 0.02)
        
        # Start simulation after indicators are calculated
        start_idx = max(params.get('bb_period', 20), params.get('rsi_period', 14))
        
        for i in range(start_idx, len(df)):
            current_price = df['Close'].iloc[i]
            current_date = df.index[i]
            
            # Get indicator values
            rsi_val = indicators['rsi'].iloc[i]
            bb_upper = indicators['bb_upper'].iloc[i]
            bb_lower = indicators['bb_lower'].iloc[i]
            ma_short = indicators['ma_short'].iloc[i]
            ma_long = indicators['ma_long'].iloc[i]
            
            # Skip if any indicators are NaN
            if pd.isna(rsi_val) or pd.isna(bb_upper) or pd.isna(bb_lower):
                portfolio_values.append(cash + (shares * current_price if shares > 0 else 0))
                continue
            
            # Generate signals - more practical momentum/mean reversion strategy
            # Buy signal: Either momentum (MA crossover) OR oversold conditions
            momentum_buy = (ma_short > ma_long and position == 0)
            oversold_buy = (rsi_val < rsi_oversold and current_price < bb_lower and position == 0)
            
            buy_signal = momentum_buy or oversold_buy
            
            # Sell signal: Either overbought, stop loss, or momentum reversal
            overbought_sell = (rsi_val > rsi_overbought or current_price > bb_upper)
            momentum_sell = (ma_short < ma_long)
            stop_loss_sell = (position > 0 and current_price < position * (1 - stop_loss))
            
            sell_signal = position > 0 and (overbought_sell or momentum_sell or stop_loss_sell)
            
            # Execute trades
            if buy_signal and cash > 0:
                shares_to_buy = (cash * position_size) // current_price
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    cash -= cost
                    shares += shares_to_buy
                    position = current_price
                    
                    trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'cost': cost
                    })
            
            elif sell_signal and shares > 0:
                proceeds = shares * current_price
                profit = proceeds - (shares * position) if position > 0 else 0
                cash += proceeds
                
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': shares,
                    'proceeds': proceeds,
                    'profit': profit
                })
                
                shares = 0
                position = 0
            
            # Calculate portfolio value
            portfolio_value = cash + (shares * current_price)
            portfolio_values.append(portfolio_value)
        
        # Create portfolio series
        portfolio_series = pd.Series(portfolio_values, index=df.index[start_idx:])
        
        return {
            'portfolio_series': portfolio_series,
            'trades': trades,
            'final_value': portfolio_series.iloc[-1] if len(portfolio_series) > 0 else initial_capital,
            'initial_capital': initial_capital,
            'start_date': df.index[start_idx],
            'end_date': df.index[-1]
        }
    
    def _calculate_comprehensive_metrics(self, symbol: str, params: Dict[str, Any], 
                                         backtest_result: Dict[str, Any], optimization_time: float) -> OptimizationResult:
        """Calculate comprehensive performance metrics"""
        
        portfolio_series = backtest_result['portfolio_series']
        trades = backtest_result['trades']
        initial_capital = backtest_result['initial_capital']
        final_value = backtest_result['final_value']
        
        # Basic returns
        returns = portfolio_series.pct_change().fillna(0)
        total_return = (final_value / initial_capital) - 1
        
        # Annualized metrics
        trading_days = len(portfolio_series)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
        
        # Volatility metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Downside deviation for Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 1 else 0
        
        # Risk-adjusted returns
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        running_max = portfolio_series.cummax()
        drawdown = (portfolio_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade analysis
        winning_trades = [t for t in trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit', 0) < 0]
        
        total_trades = len([t for t in trades if t['action'] == 'SELL'])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t['profit']) for t in losing_trades]) if losing_trades else 0
        
        total_wins = sum(t['profit'] for t in winning_trades)
        total_losses = sum(abs(t['profit']) for t in losing_trades)
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        # Data quality score
        data_quality_score = min(100, (trading_days / 60) * 100)  # Prefer 60+ days
        
        return OptimizationResult(
            symbol=symbol,
            parameters=params,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            final_value=final_value,
            equity_curve=portfolio_series,
            trades_list=trades,
            optimization_time=optimization_time,
            backtest_start=str(backtest_result['start_date']),
            backtest_end=str(backtest_result['end_date']),
            data_quality_score=data_quality_score
        )
    
    def _create_optimization_summary(self, results: List[OptimizationResult], 
                                     total_combinations: int, failed_runs: int, 
                                     total_time: float) -> OptimizationSummary:
        """Create comprehensive optimization summary"""
        
        if not results:
            return OptimizationSummary(
                total_combinations=total_combinations,
                successful_runs=0,
                failed_runs=failed_runs,
                best_result=None,
                worst_result=None,
                average_metrics={},
                total_time=total_time,
                results=[]
            )
        
        # Find best and worst results by Sharpe ratio
        best_result = max(results, key=lambda x: x.sharpe_ratio)
        worst_result = min(results, key=lambda x: x.sharpe_ratio)
        
        # Calculate average metrics
        average_metrics = {
            'total_return': np.mean([r.total_return for r in results]),
            'sharpe_ratio': np.mean([r.sharpe_ratio for r in results]),
            'max_drawdown': np.mean([r.max_drawdown for r in results]),
            'win_rate': np.mean([r.win_rate for r in results]),
            'volatility': np.mean([r.volatility for r in results])
        }
        
        return OptimizationSummary(
            total_combinations=total_combinations,
            successful_runs=len(results),
            failed_runs=failed_runs,
            best_result=best_result,
            worst_result=worst_result,
            average_metrics=average_metrics,
            total_time=total_time,
            results=results
        )
    
    def _generate_cache_key(self, symbol: str, params: Dict[str, Any], days: int) -> str:
        """Generate unique cache key for parameter combination"""
        # Sort parameters for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        key_string = f"{symbol}_{days}_{sorted_params}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _save_to_cache(self, cache_key: str, result: OptimizationResult) -> None:
        """Save optimization result to cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            # Convert result to dict, excluding non-serializable fields
            result_dict = result.to_dict()
            result_dict['equity_curve_values'] = result.equity_curve.tolist()
            result_dict['equity_curve_index'] = [str(d) for d in result.equity_curve.index]
            
            with open(cache_file, 'w') as f:
                json.dump(result_dict, f)
        except Exception as e:
            print(f"Error saving to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[OptimizationResult]:
        """Load optimization result from cache"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            if not os.path.exists(cache_file):
                return None
            
            # Check if cache is recent (within 24 hours)
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age > 24 * 3600:  # 24 hours
                return None
            
            with open(cache_file, 'r') as f:
                result_dict = json.load(f)
            
            # Reconstruct equity curve
            equity_curve = pd.Series(
                result_dict['equity_curve_values'],
                index=pd.to_datetime(result_dict['equity_curve_index'])
            )
            
            # Create OptimizationResult object
            return OptimizationResult(
                symbol=result_dict['symbol'],
                parameters=result_dict['parameters'],
                total_return=result_dict['total_return'],
                annualized_return=result_dict['annualized_return'],
                volatility=result_dict['volatility'],
                sharpe_ratio=result_dict['sharpe_ratio'],
                sortino_ratio=result_dict['sortino_ratio'],
                max_drawdown=result_dict['max_drawdown'],
                calmar_ratio=result_dict['calmar_ratio'],
                total_trades=result_dict['total_trades'],
                win_rate=result_dict['win_rate'],
                avg_win=result_dict['avg_win'],
                avg_loss=result_dict['avg_loss'],
                profit_factor=result_dict['profit_factor'],
                final_value=result_dict['final_value'],
                equity_curve=equity_curve,
                trades_list=result_dict.get('trades_list', []),
                optimization_time=result_dict['optimization_time'],
                backtest_start=result_dict['backtest_start'],
                backtest_end=result_dict['backtest_end'],
                data_quality_score=result_dict['data_quality_score']
            )
            
        except Exception as e:
            print(f"Error loading from cache: {e}")
            return None


# Example usage
if __name__ == "__main__":
    from parameter_manager import create_default_parameters
    
    # Create parameter manager with smart ranges
    params = create_default_parameters("rsi_bollinger")
    
    # Create optimization engine
    engine = OptimizationEngine(max_workers=4)
    
    # Set up progress callback
    def progress_callback(current, total, status):
        print(f"Progress: {current}/{total} - {status}")
    
    engine.set_progress_callback(progress_callback)
    
    # Run optimization
    summary = engine.run_optimization(
        parameter_manager=params,
        symbols=['AAPL'],
        days=60,
        objective='sharpe_ratio',
        max_combinations=50  # Limit for testing
    )
    
    print(f"\nOptimization completed in {summary.total_time:.2f} seconds")
    print(f"Best Sharpe Ratio: {summary.best_result.sharpe_ratio:.3f}")
    print(f"Best Parameters: {summary.best_result.parameters}")