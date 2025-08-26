#!/usr/bin/env python3
"""
Data Cleaning Utilities
Comprehensive data cleaning, validation, and preprocessing functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Comprehensive data cleaning and validation"""
    
    def __init__(self):
        self.cleaning_log = []
    
    def clean_ohlcv_data(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Clean OHLCV (Open, High, Low, Close, Volume) data"""
        if data.empty:
            return data
        
        df = data.copy()
        original_rows = len(df)
        
        # Log original state
        self._log_action(f"Starting cleanup for {symbol or 'unknown'}: {original_rows} rows")
        
        # 1. Remove rows with all NaN values
        df = df.dropna(how='all')
        if len(df) < original_rows:
            self._log_action(f"Removed {original_rows - len(df)} completely empty rows")
        
        # 2. Handle price columns
        price_cols = ['Open', 'High', 'Low', 'Close']
        existing_price_cols = [col for col in price_cols if col in df.columns]
        
        if existing_price_cols:
            # Remove rows where all price columns are NaN
            df = df.dropna(subset=existing_price_cols, how='all')
            
            # Remove negative or zero prices
            for col in existing_price_cols:
                invalid_prices = (df[col] <= 0) | df[col].isnull()
                if invalid_prices.any():
                    self._log_action(f"Found {invalid_prices.sum()} invalid prices in {col}")
                    df.loc[invalid_prices, col] = np.nan
            
            # Validate OHLC relationships
            df = self._validate_ohlc_relationships(df, existing_price_cols)
        
        # 3. Handle volume column
        if 'Volume' in df.columns:
            # Set negative volumes to 0
            negative_volume = df['Volume'] < 0
            if negative_volume.any():
                self._log_action(f"Found {negative_volume.sum()} negative volume values")
                df.loc[negative_volume, 'Volume'] = 0
            
            # Handle extremely high volume (potential outliers)
            volume_threshold = df['Volume'].quantile(0.99) * 10
            extreme_volume = df['Volume'] > volume_threshold
            if extreme_volume.any():
                self._log_action(f"Found {extreme_volume.sum()} extreme volume values")
                # Cap at 99th percentile * 5
                df.loc[extreme_volume, 'Volume'] = df['Volume'].quantile(0.99) * 5
        
        # 4. Remove duplicate timestamps
        if df.index.duplicated().any():
            duplicates = df.index.duplicated().sum()
            self._log_action(f"Removing {duplicates} duplicate timestamps")
            df = df[~df.index.duplicated(keep='first')]
        
        # 5. Sort by index (time)
        df = df.sort_index()
        
        # 6. Fill small gaps in price data
        df = self._fill_price_gaps(df, existing_price_cols)
        
        self._log_action(f"Cleaning complete: {len(df)} rows remaining")
        
        return df
    
    def detect_outliers(self, data: pd.DataFrame, columns: List[str] = None, 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Detect outliers using various methods"""
        if data.empty:
            return pd.DataFrame()
        
        df = data.copy()
        
        if columns is None:
            # Auto-detect numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) < 10:  # Need minimum data for outlier detection
                continue
            
            if method == 'iqr':
                # Interquartile Range method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                # Z-score method
                mean = series.mean()
                std = series.std()
                z_scores = np.abs((df[col] - mean) / std)
                outlier_mask[col] = z_scores > threshold
                
            elif method == 'modified_zscore':
                # Modified Z-score using median
                median = series.median()
                mad = np.median(np.abs(series - median))
                modified_z_scores = 0.6745 * (df[col] - median) / mad
                outlier_mask[col] = np.abs(modified_z_scores) > threshold
        
        # Create summary of outliers
        outlier_summary = outlier_mask.sum()
        for col, count in outlier_summary.items():
            if count > 0:
                self._log_action(f"Detected {count} outliers in {col} using {method} method")
        
        return outlier_mask
    
    def handle_missing_values(self, data: pd.DataFrame, method: str = 'interpolate',
                             columns: List[str] = None) -> pd.DataFrame:
        """Handle missing values using various methods"""
        if data.empty:
            return data
        
        df = data.copy()
        
        if columns is None:
            columns = df.columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            self._log_action(f"Handling {missing_count} missing values in {col} using {method}")
            
            if method == 'interpolate':
                # Linear interpolation for time series
                df[col] = df[col].interpolate(method='linear')
                
            elif method == 'forward_fill':
                # Forward fill
                df[col] = df[col].fillna(method='ffill')
                
            elif method == 'backward_fill':
                # Backward fill
                df[col] = df[col].fillna(method='bfill')
                
            elif method == 'mean':
                # Fill with mean
                mean_value = df[col].mean()
                df[col] = df[col].fillna(mean_value)
                
            elif method == 'median':
                # Fill with median
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                
            elif method == 'rolling_mean':
                # Fill with rolling mean
                rolling_mean = df[col].rolling(window=5, min_periods=1).mean()
                df[col] = df[col].fillna(rolling_mean)
        
        return df
    
    def remove_outliers(self, data: pd.DataFrame, outlier_mask: pd.DataFrame,
                       action: str = 'remove') -> pd.DataFrame:
        """Remove or replace outliers"""
        if data.empty or outlier_mask.empty:
            return data
        
        df = data.copy()
        
        for col in outlier_mask.columns:
            if col not in df.columns:
                continue
                
            outliers = outlier_mask[col]
            outlier_count = outliers.sum()
            
            if outlier_count == 0:
                continue
            
            if action == 'remove':
                # Set outliers to NaN
                df.loc[outliers, col] = np.nan
                self._log_action(f"Removed {outlier_count} outliers from {col}")
                
            elif action == 'cap':
                # Cap outliers at percentiles
                lower_cap = df[col].quantile(0.01)
                upper_cap = df[col].quantile(0.99)
                
                df.loc[outliers & (df[col] < lower_cap), col] = lower_cap
                df.loc[outliers & (df[col] > upper_cap), col] = upper_cap
                
                self._log_action(f"Capped {outlier_count} outliers in {col}")
                
            elif action == 'median':
                # Replace with median
                median_value = df[col].median()
                df.loc[outliers, col] = median_value
                self._log_action(f"Replaced {outlier_count} outliers in {col} with median")
        
        return df
    
    def align_time_series(self, data_dict: Dict[str, pd.DataFrame],
                         frequency: str = 'D') -> Dict[str, pd.DataFrame]:
        """Align multiple time series to common time index"""
        if not data_dict:
            return data_dict
        
        # Find common time range
        start_dates = []
        end_dates = []
        
        for name, df in data_dict.items():
            if not df.empty and hasattr(df, 'index'):
                start_dates.append(df.index.min())
                end_dates.append(df.index.max())
        
        if not start_dates or not end_dates:
            return data_dict
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        # Create common time index
        if frequency == 'D':
            common_index = pd.date_range(start=common_start, end=common_end, freq='D')
        elif frequency == 'H':
            common_index = pd.date_range(start=common_start, end=common_end, freq='H')
        elif frequency == 'min':
            common_index = pd.date_range(start=common_start, end=common_end, freq='min')
        else:
            common_index = pd.date_range(start=common_start, end=common_end, freq=frequency)
        
        # Align all data to common index
        aligned_data = {}
        for name, df in data_dict.items():
            if df.empty:
                aligned_data[name] = df
                continue
            
            # Reindex to common time index
            aligned_df = df.reindex(common_index)
            
            # Forward fill missing values for better alignment
            aligned_df = aligned_df.fillna(method='ffill')
            
            aligned_data[name] = aligned_df
            
            self._log_action(f"Aligned {name}: {len(df)} -> {len(aligned_df)} rows")
        
        return aligned_data
    
    def validate_data_quality(self, data: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        if data.empty:
            return {'status': 'empty', 'issues': ['Dataset is empty']}
        
        quality_report = {
            'symbol': symbol,
            'total_rows': len(data),
            'date_range': {
                'start': data.index.min() if hasattr(data, 'index') else None,
                'end': data.index.max() if hasattr(data, 'index') else None
            },
            'issues': [],
            'warnings': [],
            'quality_score': 100.0
        }
        
        # Check for missing values
        missing_data = data.isnull().sum()
        total_missing = missing_data.sum()
        if total_missing > 0:
            quality_report['missing_values'] = missing_data.to_dict()
            missing_percentage = (total_missing / (len(data) * len(data.columns))) * 100
            quality_report['missing_percentage'] = missing_percentage
            
            if missing_percentage > 20:
                quality_report['issues'].append(f"High missing data: {missing_percentage:.1f}%")
                quality_report['quality_score'] -= 30
            elif missing_percentage > 5:
                quality_report['warnings'].append(f"Moderate missing data: {missing_percentage:.1f}%")
                quality_report['quality_score'] -= 10
        
        # Check for duplicates
        if hasattr(data, 'index') and data.index.duplicated().any():
            duplicate_count = data.index.duplicated().sum()
            quality_report['issues'].append(f"Duplicate timestamps: {duplicate_count}")
            quality_report['quality_score'] -= 20
        
        # Check price data validity
        price_cols = ['Open', 'High', 'Low', 'Close']
        existing_price_cols = [col for col in price_cols if col in data.columns]
        
        if existing_price_cols:
            # Check for negative prices
            negative_prices = (data[existing_price_cols] <= 0).any().any()
            if negative_prices:
                quality_report['issues'].append("Negative or zero prices found")
                quality_report['quality_score'] -= 25
            
            # Check OHLC relationships
            if all(col in data.columns for col in price_cols):
                invalid_ohlc = self._check_ohlc_validity(data[price_cols])
                if invalid_ohlc > 0:
                    quality_report['warnings'].append(f"Invalid OHLC relationships: {invalid_ohlc}")
                    quality_report['quality_score'] -= 5
        
        # Check for data gaps
        if hasattr(data, 'index') and len(data) > 1:
            time_diff = data.index.to_series().diff()
            expected_freq = time_diff.mode()[0] if not time_diff.mode().empty else None
            
            if expected_freq:
                large_gaps = time_diff > expected_freq * 3
                if large_gaps.any():
                    gap_count = large_gaps.sum()
                    quality_report['warnings'].append(f"Large time gaps: {gap_count}")
                    quality_report['quality_score'] -= min(gap_count * 2, 15)
        
        # Overall quality assessment
        if quality_report['quality_score'] >= 90:
            quality_report['status'] = 'excellent'
        elif quality_report['quality_score'] >= 80:
            quality_report['status'] = 'good'
        elif quality_report['quality_score'] >= 70:
            quality_report['status'] = 'acceptable'
        elif quality_report['quality_score'] >= 50:
            quality_report['status'] = 'poor'
        else:
            quality_report['status'] = 'very_poor'
        
        return quality_report
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """Validate OHLC price relationships"""
        if len(price_cols) < 4:
            return df
        
        # Check High >= Open, Close, Low
        # Check Low <= Open, Close, High
        invalid_count = 0
        
        for idx in df.index:
            o, h, l, c = df.loc[idx, ['Open', 'High', 'Low', 'Close']]
            
            if pd.isna([o, h, l, c]).any():
                continue
            
            # Fix invalid relationships
            if h < max(o, c, l):
                df.loc[idx, 'High'] = max(o, c, l)
                invalid_count += 1
            
            if l > min(o, c, h):
                df.loc[idx, 'Low'] = min(o, c, h)
                invalid_count += 1
        
        if invalid_count > 0:
            self._log_action(f"Fixed {invalid_count} invalid OHLC relationships")
        
        return df
    
    def _fill_price_gaps(self, df: pd.DataFrame, price_cols: List[str]) -> pd.DataFrame:
        """Fill small gaps in price data"""
        for col in price_cols:
            if col not in df.columns:
                continue
            
            # Fill gaps of 1-3 missing values with interpolation
            missing_mask = df[col].isnull()
            
            if missing_mask.any():
                # Group consecutive missing values
                groups = (missing_mask != missing_mask.shift()).cumsum()
                missing_groups = groups[missing_mask]
                
                filled_count = 0
                for group_id in missing_groups.unique():
                    group_size = (missing_groups == group_id).sum()
                    
                    # Only fill small gaps
                    if group_size <= 3:
                        group_mask = groups == group_id
                        df.loc[group_mask, col] = df[col].interpolate().loc[group_mask]
                        filled_count += group_size
                
                if filled_count > 0:
                    self._log_action(f"Filled {filled_count} small gaps in {col}")
        
        return df
    
    def _check_ohlc_validity(self, ohlc_data: pd.DataFrame) -> int:
        """Check OHLC relationship validity"""
        invalid_count = 0
        
        for idx in ohlc_data.index:
            o, h, l, c = ohlc_data.loc[idx, ['Open', 'High', 'Low', 'Close']]
            
            if pd.isna([o, h, l, c]).any():
                continue
            
            if not (h >= max(o, c) and l <= min(o, c) and h >= l):
                invalid_count += 1
        
        return invalid_count
    
    def _log_action(self, message: str):
        """Log cleaning actions"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.cleaning_log.append(log_entry)
        # print(log_entry)  # Optional: print to console
    
    def get_cleaning_log(self) -> List[str]:
        """Get the cleaning action log"""
        return self.cleaning_log.copy()
    
    def clear_log(self):
        """Clear the cleaning log"""
        self.cleaning_log.clear()


if __name__ == "__main__":
    # Example usage
    cleaner = DataCleaner()
    
    # Create sample data with issues
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Introduce some issues
    sample_data.iloc[10:15] = np.nan  # Missing values
    sample_data.iloc[20, 1] = -5  # Negative price
    sample_data.iloc[30, 4] = 1000000  # Extreme volume
    
    print("Original data shape:", sample_data.shape)
    
    # Clean the data
    cleaned_data = cleaner.clean_ohlcv_data(sample_data, "SAMPLE")
    
    print("Cleaned data shape:", cleaned_data.shape)
    print("\nCleaning log:")
    for entry in cleaner.get_cleaning_log():
        print(entry)