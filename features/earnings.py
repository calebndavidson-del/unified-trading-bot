#!/usr/bin/env python3
"""
Earnings Data API Fetch and Feature Engineering
Comprehensive earnings analysis for trading decisions
"""

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class EarningsDataFetcher:
    """Fetch earnings data from multiple sources"""
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.base_url_av = "https://www.alphavantage.co/query"
    
    def fetch_earnings_calendar(self, symbol: str, 
                               horizon: str = "3month") -> pd.DataFrame:
        """Fetch earnings calendar for a symbol"""
        try:
            # Use yfinance as primary source
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is not None and not calendar.empty:
                # Convert to standardized format
                earnings_df = pd.DataFrame({
                    'symbol': symbol,
                    'earnings_date': calendar.index,
                    'eps_estimate': calendar.get('EPS Estimate', np.nan),
                    'reported_eps': calendar.get('Reported EPS', np.nan),
                    'surprise': calendar.get('Surprise(%)', np.nan)
                })
                return earnings_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching earnings calendar for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_earnings_history(self, symbol: str, 
                              quarters: int = 8) -> pd.DataFrame:
        """Fetch historical earnings data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get quarterly financials
            quarterly_financials = ticker.quarterly_financials
            quarterly_earnings = ticker.quarterly_earnings
            
            if quarterly_earnings is not None and not quarterly_earnings.empty:
                earnings_df = quarterly_earnings.T  # Transpose to have dates as index
                earnings_df['symbol'] = symbol
                earnings_df.index.name = 'date'
                earnings_df = earnings_df.reset_index()
                
                # Limit to requested quarters
                earnings_df = earnings_df.head(quarters)
                
                return earnings_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching earnings history for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_analyst_estimates(self, symbol: str) -> pd.DataFrame:
        """Fetch analyst estimates and recommendations"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get analyst info
            analyst_info = ticker.info
            recommendations = ticker.recommendations
            
            estimates_data = {
                'symbol': symbol,
                'target_mean_price': analyst_info.get('targetMeanPrice', np.nan),
                'target_high_price': analyst_info.get('targetHighPrice', np.nan),
                'target_low_price': analyst_info.get('targetLowPrice', np.nan),
                'recommendation_mean': analyst_info.get('recommendationMean', np.nan),
                'number_of_analyst_opinions': analyst_info.get('numberOfAnalystOpinions', 0)
            }
            
            # Add recommendation history if available
            if recommendations is not None and not recommendations.empty:
                latest_rec = recommendations.iloc[-1]
                estimates_data['latest_strong_buy'] = latest_rec.get('strongBuy', 0)
                estimates_data['latest_buy'] = latest_rec.get('buy', 0)
                estimates_data['latest_hold'] = latest_rec.get('hold', 0)
                estimates_data['latest_sell'] = latest_rec.get('sell', 0)
                estimates_data['latest_strong_sell'] = latest_rec.get('strongSell', 0)
            
            return pd.DataFrame([estimates_data])
            
        except Exception as e:
            print(f"Error fetching analyst estimates for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_earnings_from_alpha_vantage(self, symbol: str) -> pd.DataFrame:
        """Fetch earnings data from Alpha Vantage API"""
        if not self.alpha_vantage_key:
            return pd.DataFrame()
        
        try:
            params = {
                'function': 'EARNINGS',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(self.base_url_av, params=params)
            data = response.json()
            
            if 'quarterlyEarnings' in data:
                earnings_df = pd.DataFrame(data['quarterlyEarnings'])
                earnings_df['symbol'] = symbol
                earnings_df['fiscalDateEnding'] = pd.to_datetime(earnings_df['fiscalDateEnding'])
                return earnings_df
            
        except Exception as e:
            print(f"Error fetching Alpha Vantage earnings for {symbol}: {e}")
        
        return pd.DataFrame()


class EarningsFeatureEngineer:
    """Engineer earnings-related features for trading models"""
    
    def __init__(self):
        self.fetcher = EarningsDataFetcher()
    
    def calculate_earnings_surprise(self, earnings_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate earnings surprise metrics"""
        result_df = earnings_df.copy()
        
        if 'Actual' in result_df.columns and 'Estimate' in result_df.columns:
            # EPS surprise percentage
            result_df['eps_surprise_pct'] = (
                (result_df['Actual'] - result_df['Estimate']) / 
                abs(result_df['Estimate'] + 1e-8) * 100
            )
            
            # Absolute surprise
            result_df['eps_surprise_abs'] = result_df['Actual'] - result_df['Estimate']
            
            # Beat/miss classification
            result_df['earnings_beat'] = (result_df['Actual'] > result_df['Estimate']).astype(int)
            result_df['earnings_miss'] = (result_df['Actual'] < result_df['Estimate']).astype(int)
            
            # Surprise magnitude categories
            result_df['surprise_magnitude'] = pd.cut(
                result_df['eps_surprise_pct'],
                bins=[-np.inf, -10, -5, 5, 10, np.inf],
                labels=['large_miss', 'small_miss', 'inline', 'small_beat', 'large_beat']
            )
        
        return result_df
    
    def calculate_earnings_growth(self, earnings_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate earnings growth metrics"""
        result_df = earnings_df.copy()
        
        if 'Actual' in result_df.columns:
            # Sort by date
            result_df = result_df.sort_values('date')
            
            # Quarter-over-quarter growth
            result_df['eps_qoq_growth'] = result_df['Actual'].pct_change()
            
            # Year-over-year growth (assuming quarterly data)
            result_df['eps_yoy_growth'] = result_df['Actual'].pct_change(periods=4)
            
            # Growth acceleration
            result_df['eps_growth_acceleration'] = result_df['eps_qoq_growth'].diff()
            
            # Earnings trend (improving/deteriorating)
            result_df['eps_trend_score'] = (
                result_df['eps_qoq_growth'].rolling(window=4).mean()
            )
            
            # Consistency score (lower is more consistent)
            result_df['eps_consistency'] = (
                result_df['eps_qoq_growth'].rolling(window=4).std()
            )
        
        return result_df
    
    def calculate_revenue_metrics(self, financials_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate revenue-related metrics"""
        result_df = financials_df.copy()
        
        if 'Total Revenue' in result_df.columns:
            # Revenue growth
            result_df['revenue_qoq_growth'] = result_df['Total Revenue'].pct_change()
            result_df['revenue_yoy_growth'] = result_df['Total Revenue'].pct_change(periods=4)
            
            # Revenue surprise (if estimates available)
            if 'Revenue Estimate' in result_df.columns:
                result_df['revenue_surprise_pct'] = (
                    (result_df['Total Revenue'] - result_df['Revenue Estimate']) /
                    abs(result_df['Revenue Estimate'] + 1e-8) * 100
                )
        
        return result_df
    
    def create_earnings_event_features(self, price_df: pd.DataFrame,
                                     earnings_dates: List[datetime],
                                     event_window: int = 5) -> pd.DataFrame:
        """Create features around earnings announcement dates"""
        result_df = price_df.copy()
        result_df.index = pd.to_datetime(result_df.index)
        
        # Initialize event features
        result_df['days_to_earnings'] = np.nan
        result_df['days_since_earnings'] = np.nan
        result_df['in_earnings_window'] = 0
        result_df['pre_earnings_period'] = 0
        result_df['post_earnings_period'] = 0
        
        for earnings_date in earnings_dates:
            earnings_date = pd.to_datetime(earnings_date)
            
            # Calculate days to/since earnings for each date
            for idx, date in enumerate(result_df.index):
                days_diff = (earnings_date - date).days
                
                if days_diff >= 0:  # Before earnings
                    if pd.isna(result_df.iloc[idx]['days_to_earnings']) or days_diff < result_df.iloc[idx]['days_to_earnings']:
                        result_df.iloc[idx, result_df.columns.get_loc('days_to_earnings')] = days_diff
                        
                        if days_diff <= event_window:
                            result_df.iloc[idx, result_df.columns.get_loc('pre_earnings_period')] = 1
                            result_df.iloc[idx, result_df.columns.get_loc('in_earnings_window')] = 1
                
                else:  # After earnings
                    days_since = abs(days_diff)
                    if pd.isna(result_df.iloc[idx]['days_since_earnings']) or days_since < result_df.iloc[idx]['days_since_earnings']:
                        result_df.iloc[idx, result_df.columns.get_loc('days_since_earnings')] = days_since
                        
                        if days_since <= event_window:
                            result_df.iloc[idx, result_df.columns.get_loc('post_earnings_period')] = 1
                            result_df.iloc[idx, result_df.columns.get_loc('in_earnings_window')] = 1
        
        return result_df
    
    def calculate_pre_earnings_momentum(self, price_df: pd.DataFrame,
                                      earnings_dates: List[datetime],
                                      lookback_days: int = 20) -> pd.DataFrame:
        """Calculate momentum leading up to earnings"""
        result_df = price_df.copy()
        result_df.index = pd.to_datetime(result_df.index)
        
        # Initialize momentum features
        result_df['pre_earnings_return'] = np.nan
        result_df['pre_earnings_volatility'] = np.nan
        result_df['pre_earnings_volume_avg'] = np.nan
        
        for earnings_date in earnings_dates:
            earnings_date = pd.to_datetime(earnings_date)
            
            # Find the lookback period
            start_date = earnings_date - timedelta(days=lookback_days)
            end_date = earnings_date
            
            # Get data for the period
            period_data = result_df[(result_df.index >= start_date) & (result_df.index < end_date)]
            
            if len(period_data) > 0:
                # Calculate return
                period_return = (period_data['Close'].iloc[-1] / period_data['Close'].iloc[0] - 1)
                
                # Calculate volatility
                period_volatility = period_data['Close'].pct_change().std()
                
                # Calculate average volume
                if 'Volume' in period_data.columns:
                    period_volume = period_data['Volume'].mean()
                else:
                    period_volume = np.nan
                
                # Assign to the earnings date
                earnings_idx = result_df.index.get_indexer([earnings_date], method='nearest')[0]
                if earnings_idx < len(result_df):
                    result_df.iloc[earnings_idx, result_df.columns.get_loc('pre_earnings_return')] = period_return
                    result_df.iloc[earnings_idx, result_df.columns.get_loc('pre_earnings_volatility')] = period_volatility
                    result_df.iloc[earnings_idx, result_df.columns.get_loc('pre_earnings_volume_avg')] = period_volume
        
        return result_df
    
    def create_analyst_revision_features(self, estimates_df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on analyst estimate revisions"""
        result_df = estimates_df.copy()
        
        if 'Estimate' in result_df.columns:
            # Estimate revision trends
            result_df['estimate_revision'] = result_df['Estimate'].diff()
            result_df['estimate_revision_pct'] = result_df['Estimate'].pct_change()
            
            # Revision momentum
            result_df['revision_momentum'] = (
                result_df['estimate_revision'].rolling(window=3).mean()
            )
            
            # Estimate dispersion (if multiple estimates available)
            if 'High_Estimate' in result_df.columns and 'Low_Estimate' in result_df.columns:
                result_df['estimate_dispersion'] = (
                    (result_df['High_Estimate'] - result_df['Low_Estimate']) / 
                    result_df['Estimate']
                )
        
        return result_df
    
    def create_earnings_trading_signals(self, price_df: pd.DataFrame,
                                      earnings_df: pd.DataFrame,
                                      estimates_df: pd.DataFrame) -> pd.DataFrame:
        """Create trading signals based on earnings analysis"""
        result_df = price_df.copy()
        
        # Initialize signal columns
        result_df['earnings_bullish_signal'] = 0
        result_df['earnings_bearish_signal'] = 0
        result_df['earnings_signal_strength'] = 0
        
        # Merge earnings data
        if not earnings_df.empty:
            earnings_signals = self._generate_earnings_signals(earnings_df)
            
            # Map signals to price data (simplified approach)
            for idx, row in earnings_signals.iterrows():
                if 'date' in row and not pd.isna(row['date']):
                    signal_date = pd.to_datetime(row['date'])
                    price_idx = result_df.index.get_indexer([signal_date], method='nearest')[0]
                    
                    if price_idx < len(result_df):
                        result_df.iloc[price_idx, result_df.columns.get_loc('earnings_bullish_signal')] = row.get('bullish_signal', 0)
                        result_df.iloc[price_idx, result_df.columns.get_loc('earnings_bearish_signal')] = row.get('bearish_signal', 0)
                        result_df.iloc[price_idx, result_df.columns.get_loc('earnings_signal_strength')] = row.get('signal_strength', 0)
        
        return result_df
    
    def _generate_earnings_signals(self, earnings_df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from earnings data"""
        signals_df = earnings_df.copy()
        
        # Initialize signals
        signals_df['bullish_signal'] = 0
        signals_df['bearish_signal'] = 0
        signals_df['signal_strength'] = 0
        
        # EPS surprise signals
        if 'eps_surprise_pct' in signals_df.columns:
            # Strong beat
            strong_beat_mask = signals_df['eps_surprise_pct'] > 10
            signals_df.loc[strong_beat_mask, 'bullish_signal'] = 1
            signals_df.loc[strong_beat_mask, 'signal_strength'] = 0.8
            
            # Strong miss
            strong_miss_mask = signals_df['eps_surprise_pct'] < -10
            signals_df.loc[strong_miss_mask, 'bearish_signal'] = 1
            signals_df.loc[strong_miss_mask, 'signal_strength'] = 0.8
        
        # Growth signals
        if 'eps_yoy_growth' in signals_df.columns:
            # Accelerating growth
            accel_growth_mask = (signals_df['eps_yoy_growth'] > 0.2) & (signals_df['eps_growth_acceleration'] > 0)
            signals_df.loc[accel_growth_mask, 'bullish_signal'] = 1
            signals_df.loc[accel_growth_mask, 'signal_strength'] += 0.3
            
            # Decelerating growth
            decel_growth_mask = (signals_df['eps_yoy_growth'] < 0) & (signals_df['eps_growth_acceleration'] < 0)
            signals_df.loc[decel_growth_mask, 'bearish_signal'] = 1
            signals_df.loc[decel_growth_mask, 'signal_strength'] += 0.3
        
        # Cap signal strength
        signals_df['signal_strength'] = signals_df['signal_strength'].clip(0, 1)
        
        return signals_df


def create_comprehensive_earnings_features(symbol: str, 
                                         price_df: pd.DataFrame,
                                         alpha_vantage_key: Optional[str] = None) -> pd.DataFrame:
    """Create comprehensive earnings features for a symbol"""
    
    # Initialize components
    fetcher = EarningsDataFetcher(alpha_vantage_key)
    engineer = EarningsFeatureEngineer()
    
    # Fetch earnings data
    earnings_history = fetcher.fetch_earnings_history(symbol)
    earnings_calendar = fetcher.fetch_earnings_calendar(symbol)
    analyst_estimates = fetcher.fetch_analyst_estimates(symbol)
    
    # Start with price data
    result_df = price_df.copy()
    
    # Process earnings history
    if not earnings_history.empty:
        earnings_with_features = engineer.calculate_earnings_surprise(earnings_history)
        earnings_with_features = engineer.calculate_earnings_growth(earnings_with_features)
        
        # Create event features
        if 'date' in earnings_with_features.columns:
            earnings_dates = earnings_with_features['date'].tolist()
            result_df = engineer.create_earnings_event_features(result_df, earnings_dates)
            result_df = engineer.calculate_pre_earnings_momentum(result_df, earnings_dates)
    
    # Create trading signals
    if not earnings_history.empty:
        result_df = engineer.create_earnings_trading_signals(
            result_df, earnings_history, analyst_estimates
        )
    
    return result_df


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    symbol = "AAPL"
    ticker = yf.Ticker(symbol)
    price_data = ticker.history(period="2y")
    
    # Create earnings features
    earnings_features = create_comprehensive_earnings_features(symbol, price_data)
    
    print(f"Created earnings features for {symbol}")
    print(f"Data shape: {earnings_features.shape}")
    
    # Show earnings-related columns
    earnings_cols = [col for col in earnings_features.columns if 'earnings' in col.lower()]
    print(f"Earnings columns: {earnings_cols}")
    
    # Show any earnings signals
    if 'earnings_bullish_signal' in earnings_features.columns:
        bullish_signals = earnings_features['earnings_bullish_signal'].sum()
        bearish_signals = earnings_features['earnings_bearish_signal'].sum()
        print(f"Bullish earnings signals: {bullish_signals}")
        print(f"Bearish earnings signals: {bearish_signals}")