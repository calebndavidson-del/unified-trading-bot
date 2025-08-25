#!/usr/bin/env python3
"""
Risk Metrics and Stop-Loss Helper Functions
Comprehensive risk management for trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
try:
    from scipy import stats
    from scipy.optimize import minimize
except ImportError:
    # Fallback implementations
    class stats:
        @staticmethod
        def skew(data):
            return 0.0
        @staticmethod
        def kurtosis(data):
            return 0.0
    
    def minimize(*args, **kwargs):
        raise NotImplementedError("scipy.optimize.minimize is not available. Please install scipy.")

import warnings
warnings.filterwarnings('ignore')


class RiskMetrics:
    """Calculate various risk metrics for portfolio and individual positions"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate returns from price series"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if excess_returns.std() == 0:
            return 0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        return excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        max_dd = RiskMetrics.max_drawdown(returns)
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0
        return annual_return / abs(max_dd)
    
    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def conditional_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)"""
        var = RiskMetrics.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def beta(returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta relative to market"""
        aligned_data = pd.concat([returns, market_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 1.0
        
        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data.iloc[:, 1].var()
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    @staticmethod
    def volatility(returns: pd.Series, annualized: bool = True) -> float:
        """Calculate volatility"""
        vol = returns.std()
        return vol * np.sqrt(252) if annualized else vol
    
    @staticmethod
    def skewness(returns: pd.Series) -> float:
        """Calculate skewness"""
        return stats.skew(returns.dropna())
    
    @staticmethod
    def kurtosis(returns: pd.Series) -> float:
        """Calculate kurtosis"""
        return stats.kurtosis(returns.dropna())
    
    @staticmethod
    def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio"""
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std()
        if tracking_error == 0:
            return 0
        return excess_returns.mean() / tracking_error * np.sqrt(252)


class PositionSizing:
    """Position sizing methods"""
    
    @staticmethod
    def fixed_percentage(capital: float, percentage: float) -> float:
        """Fixed percentage position sizing"""
        return capital * percentage
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly criterion for optimal position size"""
        if avg_loss == 0:
            return 0
        
        b = avg_win / abs(avg_loss)  # Win/loss ratio
        p = win_rate  # Probability of winning
        q = 1 - p  # Probability of losing
        
        kelly_fraction = (b * p - q) / b
        return max(0, min(kelly_fraction, 0.25))  # Cap at 25%
    
    @staticmethod
    def volatility_targeting(capital: float, target_volatility: float, 
                           current_volatility: float, base_position: float) -> float:
        """Volatility targeting position sizing"""
        if current_volatility == 0:
            return base_position
        
        vol_ratio = target_volatility / current_volatility
        return capital * base_position * vol_ratio
    
    @staticmethod
    def risk_parity_position(capital: float, volatilities: List[float], 
                           correlations: np.ndarray) -> List[float]:
        """Risk parity position sizing"""
        n_assets = len(volatilities)
        target_risk = 1.0 / n_assets
        
        # Convert to numpy arrays
        vols = np.array(volatilities)
        
        # Equal risk contribution weights
        weights = 1.0 / vols
        weights = weights / weights.sum()
        
        # Adjust for capital
        positions = [capital * w for w in weights]
        
        return positions


class StopLossManager:
    """Advanced stop-loss management"""
    
    def __init__(self, initial_stop_pct: float = 0.05):
        self.initial_stop_pct = initial_stop_pct
        self.stops = {}
    
    def set_initial_stop(self, symbol: str, entry_price: float, 
                        position_type: str = 'long') -> float:
        """Set initial stop-loss level"""
        if position_type.lower() == 'long':
            stop_price = entry_price * (1 - self.initial_stop_pct)
        else:  # short
            stop_price = entry_price * (1 + self.initial_stop_pct)
        
        self.stops[symbol] = {
            'stop_price': stop_price,
            'entry_price': entry_price,
            'position_type': position_type,
            'trailing_high': entry_price if position_type.lower() == 'long' else entry_price
        }
        
        return stop_price
    
    def update_trailing_stop(self, symbol: str, current_price: float,
                           trailing_pct: float = 0.03) -> float:
        """Update trailing stop-loss"""
        if symbol not in self.stops:
            return current_price
        
        stop_data = self.stops[symbol]
        position_type = stop_data['position_type'].lower()
        
        if position_type == 'long':
            # Update trailing high
            if current_price > stop_data['trailing_high']:
                stop_data['trailing_high'] = current_price
            
            # Calculate new stop
            new_stop = stop_data['trailing_high'] * (1 - trailing_pct)
            
            # Only move stop up, never down
            if new_stop > stop_data['stop_price']:
                stop_data['stop_price'] = new_stop
        
        else:  # short position
            # Update trailing low
            if current_price < stop_data['trailing_high']:  # Using 'trailing_high' as trailing_low for shorts
                stop_data['trailing_high'] = current_price
            
            # Calculate new stop
            new_stop = stop_data['trailing_high'] * (1 + trailing_pct)
            
            # Only move stop down, never up
            if new_stop < stop_data['stop_price']:
                stop_data['stop_price'] = new_stop
        
        return stop_data['stop_price']
    
    def atr_based_stop(self, symbol: str, current_price: float, atr: float,
                      atr_multiplier: float = 2.0, position_type: str = 'long') -> float:
        """ATR-based dynamic stop-loss"""
        if position_type.lower() == 'long':
            stop_price = current_price - (atr * atr_multiplier)
        else:  # short
            stop_price = current_price + (atr * atr_multiplier)
        
        # Update stored stop
        if symbol in self.stops:
            if position_type.lower() == 'long' and stop_price > self.stops[symbol]['stop_price']:
                self.stops[symbol]['stop_price'] = stop_price
            elif position_type.lower() == 'short' and stop_price < self.stops[symbol]['stop_price']:
                self.stops[symbol]['stop_price'] = stop_price
        else:
            self.stops[symbol] = {
                'stop_price': stop_price,
                'entry_price': current_price,
                'position_type': position_type,
                'trailing_high': current_price
            }
        
        return self.stops[symbol]['stop_price']
    
    def volatility_based_stop(self, symbol: str, current_price: float,
                            volatility: float, vol_multiplier: float = 2.0,
                            position_type: str = 'long') -> float:
        """Volatility-based stop-loss"""
        if position_type.lower() == 'long':
            stop_price = current_price * (1 - volatility * vol_multiplier)
        else:  # short
            stop_price = current_price * (1 + volatility * vol_multiplier)
        
        # Update stored stop
        if symbol in self.stops:
            if position_type.lower() == 'long' and stop_price > self.stops[symbol]['stop_price']:
                self.stops[symbol]['stop_price'] = stop_price
            elif position_type.lower() == 'short' and stop_price < self.stops[symbol]['stop_price']:
                self.stops[symbol]['stop_price'] = stop_price
        else:
            self.stops[symbol] = {
                'stop_price': stop_price,
                'entry_price': current_price,
                'position_type': position_type,
                'trailing_high': current_price
            }
        
        return self.stops[symbol]['stop_price']
    
    def check_stop_triggered(self, symbol: str, current_price: float) -> bool:
        """Check if stop-loss is triggered"""
        if symbol not in self.stops:
            return False
        
        stop_data = self.stops[symbol]
        position_type = stop_data['position_type'].lower()
        stop_price = stop_data['stop_price']
        
        if position_type == 'long':
            return current_price <= stop_price
        else:  # short
            return current_price >= stop_price


class PortfolioRiskManager:
    """Portfolio-level risk management"""
    
    def __init__(self, max_portfolio_risk: float = 0.02, max_position_size: float = 0.1):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_size = max_position_size
        self.positions = {}
    
    def calculate_portfolio_var(self, positions: Dict[str, float], 
                               returns_data: pd.DataFrame,
                               confidence_level: float = 0.05) -> float:
        """Calculate portfolio Value at Risk"""
        if not positions or returns_data.empty:
            return 0.0
        
        # Align positions with returns data
        common_symbols = set(positions.keys()) & set(returns_data.columns)
        if not common_symbols:
            return 0.0
        
        # Create weights vector
        weights = np.array([positions[symbol] for symbol in common_symbols])
        weights = weights / np.sum(np.abs(weights))  # Normalize
        
        # Calculate portfolio returns
        aligned_returns = returns_data[list(common_symbols)].dropna()
        portfolio_returns = (aligned_returns * weights).sum(axis=1)
        
        # Calculate VaR
        return RiskMetrics.value_at_risk(portfolio_returns, confidence_level)
    
    def calculate_portfolio_beta(self, positions: Dict[str, float],
                               returns_data: pd.DataFrame,
                               market_returns: pd.Series) -> float:
        """Calculate portfolio beta"""
        if not positions or returns_data.empty:
            return 1.0
        
        common_symbols = set(positions.keys()) & set(returns_data.columns)
        if not common_symbols:
            return 1.0
        
        # Calculate weighted average beta
        total_weight = 0
        weighted_beta = 0
        
        for symbol in common_symbols:
            weight = abs(positions[symbol])
            symbol_returns = returns_data[symbol].dropna()
            
            # Align with market returns
            aligned_data = pd.concat([symbol_returns, market_returns], axis=1).dropna()
            if len(aligned_data) > 1:
                beta = RiskMetrics.beta(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
                weighted_beta += weight * beta
                total_weight += weight
        
        return weighted_beta / total_weight if total_weight > 0 else 1.0
    
    def calculate_correlation_risk(self, positions: Dict[str, float],
                                 returns_data: pd.DataFrame) -> float:
        """Calculate portfolio concentration risk based on correlations"""
        if not positions or returns_data.empty:
            return 0.0
        
        common_symbols = list(set(positions.keys()) & set(returns_data.columns))
        if len(common_symbols) < 2:
            return 0.0
        
        # Calculate correlation matrix
        corr_matrix = returns_data[common_symbols].corr()
        
        # Calculate average correlation weighted by position sizes
        weights = np.array([abs(positions[symbol]) for symbol in common_symbols])
        weights = weights / weights.sum()
        
        # Weighted average correlation
        total_corr = 0
        total_weight = 0
        
        for i in range(len(common_symbols)):
            for j in range(i + 1, len(common_symbols)):
                corr = corr_matrix.iloc[i, j]
                weight = weights[i] * weights[j]
                total_corr += corr * weight
                total_weight += weight
        
        return total_corr / total_weight if total_weight > 0 else 0.0
    
    def check_position_limits(self, symbol: str, new_position_size: float,
                            current_portfolio_value: float) -> Tuple[bool, str]:
        """Check if new position violates limits"""
        # Check individual position size
        position_pct = abs(new_position_size) / current_portfolio_value
        if position_pct > self.max_position_size:
            return False, f"Position size {position_pct:.1%} exceeds limit {self.max_position_size:.1%}"
        
        # Check total portfolio risk (simplified)
        total_exposure = sum(abs(pos) for pos in self.positions.values()) + abs(new_position_size)
        total_risk = total_exposure / current_portfolio_value
        
        if total_risk > 1.0:  # More than 100% exposed
            return False, f"Total exposure {total_risk:.1%} exceeds 100%"
        
        return True, "Position within limits"
    
    def suggest_position_size(self, symbol: str, entry_price: float,
                            stop_loss_price: float, risk_per_trade: float,
                            portfolio_value: float) -> float:
        """Suggest optimal position size based on risk parameters"""
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
        
        # Calculate position size based on risk
        risk_amount = portfolio_value * risk_per_trade
        shares = risk_amount / risk_per_share
        
        # Apply position size limits
        max_position_value = portfolio_value * self.max_position_size
        max_shares = max_position_value / entry_price
        
        return min(shares, max_shares)


class RiskReporting:
    """Risk reporting and monitoring"""
    
    @staticmethod
    def generate_risk_report(portfolio_returns: pd.Series,
                           benchmark_returns: pd.Series = None,
                           risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Generate comprehensive risk report"""
        report = {}
        
        # Basic metrics
        report['total_return'] = (1 + portfolio_returns).prod() - 1
        report['annual_return'] = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
        report['volatility'] = RiskMetrics.volatility(portfolio_returns)
        report['sharpe_ratio'] = RiskMetrics.sharpe_ratio(portfolio_returns, risk_free_rate)
        report['sortino_ratio'] = RiskMetrics.sortino_ratio(portfolio_returns, risk_free_rate)
        report['calmar_ratio'] = RiskMetrics.calmar_ratio(portfolio_returns)
        
        # Drawdown metrics
        report['max_drawdown'] = RiskMetrics.max_drawdown(portfolio_returns)
        
        # Risk metrics
        report['var_95'] = RiskMetrics.value_at_risk(portfolio_returns, 0.05)
        report['cvar_95'] = RiskMetrics.conditional_var(portfolio_returns, 0.05)
        report['skewness'] = RiskMetrics.skewness(portfolio_returns)
        report['kurtosis'] = RiskMetrics.kurtosis(portfolio_returns)
        
        # Benchmark comparison
        if benchmark_returns is not None:
            aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
            if len(aligned_data) > 1:
                report['beta'] = RiskMetrics.beta(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
                report['information_ratio'] = RiskMetrics.information_ratio(
                    aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
                )
        
        return report
    
    @staticmethod
    def risk_alerts(current_metrics: Dict[str, float],
                   thresholds: Dict[str, float]) -> List[str]:
        """Generate risk alerts based on thresholds"""
        alerts = []
        
        for metric, current_value in current_metrics.items():
            if metric in thresholds:
                threshold = thresholds[metric]
                
                # Define alert conditions
                if metric == 'max_drawdown' and current_value < -threshold:
                    alerts.append(f"Maximum drawdown {current_value:.1%} exceeds threshold {threshold:.1%}")
                elif metric == 'volatility' and current_value > threshold:
                    alerts.append(f"Volatility {current_value:.1%} exceeds threshold {threshold:.1%}")
                elif metric == 'var_95' and current_value < -threshold:
                    alerts.append(f"VaR {current_value:.1%} exceeds threshold {threshold:.1%}")
                elif metric == 'sharpe_ratio' and current_value < threshold:
                    alerts.append(f"Sharpe ratio {current_value:.2f} below threshold {threshold:.2f}")
        
        return alerts


def calculate_comprehensive_risk_metrics(returns: pd.Series, 
                                       benchmark_returns: pd.Series = None,
                                       risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Calculate comprehensive risk metrics for a return series"""
    
    risk_metrics = RiskMetrics()
    
    metrics = {
        'sharpe_ratio': risk_metrics.sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': risk_metrics.sortino_ratio(returns, risk_free_rate),
        'calmar_ratio': risk_metrics.calmar_ratio(returns),
        'max_drawdown': risk_metrics.max_drawdown(returns),
        'volatility': risk_metrics.volatility(returns),
        'var_95': risk_metrics.value_at_risk(returns, 0.05),
        'cvar_95': risk_metrics.conditional_var(returns, 0.05),
        'skewness': risk_metrics.skewness(returns),
        'kurtosis': risk_metrics.kurtosis(returns)
    }
    
    if benchmark_returns is not None:
        metrics['beta'] = risk_metrics.beta(returns, benchmark_returns)
        metrics['information_ratio'] = risk_metrics.information_ratio(returns, benchmark_returns)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="2y")
    returns = data['Close'].pct_change().dropna()
    
    # Calculate risk metrics
    risk_metrics = calculate_comprehensive_risk_metrics(returns)
    
    print("Risk Metrics for AAPL:")
    for metric, value in risk_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test stop-loss manager
    stop_manager = StopLossManager()
    entry_price = 150.0
    stop_price = stop_manager.set_initial_stop("AAPL", entry_price)
    print(f"\nInitial stop-loss set at: ${stop_price:.2f}")
    
    # Test trailing stop
    current_price = 160.0
    new_stop = stop_manager.update_trailing_stop("AAPL", current_price)
    print(f"Updated trailing stop at: ${new_stop:.2f}")
    
    # Test position sizing
    kelly_size = PositionSizing.kelly_criterion(0.6, 0.1, 0.05)
    print(f"Kelly criterion position size: {kelly_size:.1%}")