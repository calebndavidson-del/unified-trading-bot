#!/usr/bin/env python3
"""
Data Quality Assurance Utilities
Bias reduction, quality monitoring, and visualization tools for trading data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')


class DataQualityAssurance:
    """Comprehensive data quality monitoring and bias reduction"""
    
    def __init__(self):
        self.qa_log = []
        self.quality_metrics = {}
        
    def balance_asset_selection(self, data_dict: Dict[str, pd.DataFrame],
                               balance_criteria: Dict[str, Any] = None) -> Dict[str, pd.DataFrame]:
        """Balance asset selection across sectors, regions, and market caps"""
        if not data_dict:
            return data_dict
        
        if balance_criteria is None:
            balance_criteria = {
                'max_per_sector': 5,
                'max_per_region': 10,
                'market_cap_distribution': {
                    'Large Cap': 0.4,
                    'Mid Cap': 0.3,
                    'Small Cap': 0.2,
                    'Micro Cap': 0.1
                }
            }
        
        # Extract metadata from datasets
        asset_metadata = []
        for symbol, df in data_dict.items():
            if df.empty:
                continue
                
            metadata = {
                'symbol': symbol,
                'sector': df.get('sector', pd.Series(['Unknown'])).iloc[0] if 'sector' in df.columns else 'Unknown',
                'country': df.get('country', pd.Series(['Unknown'])).iloc[0] if 'country' in df.columns else 'Unknown',
                'market_cap_category': df.get('market_cap_category', pd.Series(['Unknown'])).iloc[0] if 'market_cap_category' in df.columns else 'Unknown',
                'data_quality': len(df),
                'completeness': 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
            }
            asset_metadata.append(metadata)
        
        metadata_df = pd.DataFrame(asset_metadata)
        
        if metadata_df.empty:
            return data_dict
        
        # Balance by sector
        balanced_symbols = set()
        
        # Sector balancing
        sector_counts = metadata_df['sector'].value_counts()
        for sector in sector_counts.index:
            sector_assets = metadata_df[metadata_df['sector'] == sector]
            
            # Sort by data quality and take top N
            sector_assets = sector_assets.sort_values(['completeness', 'data_quality'], ascending=False)
            selected = sector_assets.head(balance_criteria['max_per_sector'])['symbol'].tolist()
            balanced_symbols.update(selected)
        
        # Market cap balancing
        target_distribution = balance_criteria['market_cap_distribution']
        total_assets = len(balanced_symbols)
        
        cap_balanced_symbols = set()
        for cap_category, target_ratio in target_distribution.items():
            target_count = int(total_assets * target_ratio)
            cap_assets = metadata_df[
                (metadata_df['symbol'].isin(balanced_symbols)) &
                (metadata_df['market_cap_category'] == cap_category)
            ]
            
            # Sort by quality and select
            cap_assets = cap_assets.sort_values(['completeness', 'data_quality'], ascending=False)
            selected = cap_assets.head(target_count)['symbol'].tolist()
            cap_balanced_symbols.update(selected)
        
        # Combine balanced selection
        final_balanced_symbols = list(cap_balanced_symbols) if cap_balanced_symbols else list(balanced_symbols)
        
        # Create balanced dataset
        balanced_data = {symbol: data_dict[symbol] for symbol in final_balanced_symbols if symbol in data_dict}
        
        self._log_action(f"Balanced selection: {len(data_dict)} -> {len(balanced_data)} assets")
        self._log_action(f"Sector distribution: {metadata_df[metadata_df['symbol'].isin(final_balanced_symbols)]['sector'].value_counts().to_dict()}")
        
        return balanced_data
    
    def stratified_sampling(self, data: pd.DataFrame, target_column: str = None,
                          train_ratio: float = 0.7, val_ratio: float = 0.15,
                          test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """Perform stratified sampling for model training and evaluation"""
        if data.empty:
            return {'train': data, 'val': data, 'test': data}
        
        # If no target column specified, use temporal stratification
        if target_column is None or target_column not in data.columns:
            return self._temporal_stratified_split(data, train_ratio, val_ratio, test_ratio)
        
        # Create strata based on target variable
        if data[target_column].dtype in ['object', 'category']:
            # Categorical stratification
            strata = data[target_column]
        else:
            # Numerical stratification - create quartiles
            strata = pd.qcut(data[target_column], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        
        # Stratified split
        train_indices = []
        val_indices = []
        test_indices = []
        
        for stratum in strata.unique():
            if pd.isna(stratum):
                continue
                
            stratum_data = data[strata == stratum]
            n_stratum = len(stratum_data)
            
            if n_stratum < 3:  # Too few samples
                train_indices.extend(stratum_data.index.tolist())
                continue
            
            # Calculate split sizes
            n_train = int(n_stratum * train_ratio)
            n_val = int(n_stratum * val_ratio)
            n_test = n_stratum - n_train - n_val
            
            # Random split within stratum
            shuffled_indices = stratum_data.sample(frac=1).index.tolist()
            
            train_indices.extend(shuffled_indices[:n_train])
            val_indices.extend(shuffled_indices[n_train:n_train + n_val])
            test_indices.extend(shuffled_indices[n_train + n_val:])
        
        splits = {
            'train': data.loc[train_indices],
            'val': data.loc[val_indices],
            'test': data.loc[test_indices]
        }
        
        self._log_action(f"Stratified sampling: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        
        return splits
    
    def monitor_data_drift(self, reference_data: pd.DataFrame, new_data: pd.DataFrame,
                          columns: List[str] = None, threshold: float = 0.1) -> Dict[str, Any]:
        """Monitor for data drift between reference and new datasets"""
        if reference_data.empty or new_data.empty:
            return {'drift_detected': False, 'details': 'Insufficient data'}
        
        if columns is None:
            columns = [col for col in reference_data.select_dtypes(include=[np.number]).columns
                      if col in new_data.columns]
        
        drift_results = {
            'drift_detected': False,
            'drift_scores': {},
            'drift_columns': [],
            'summary': {}
        }
        
        for col in columns:
            if col not in reference_data.columns or col not in new_data.columns:
                continue
            
            # Calculate statistical distance (KL divergence approximation)
            ref_values = reference_data[col].dropna()
            new_values = new_data[col].dropna()
            
            if len(ref_values) < 10 or len(new_values) < 10:
                continue
            
            # Kolmogorov-Smirnov test
            from scipy import stats
            try:
                ks_stat, p_value = stats.ks_2samp(ref_values, new_values)
                
                drift_score = ks_stat
                drift_results['drift_scores'][col] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_score': drift_score
                }
                
                if drift_score > threshold:
                    drift_results['drift_detected'] = True
                    drift_results['drift_columns'].append(col)
                
            except Exception as e:
                drift_results['drift_scores'][col] = {'error': str(e)}
        
        # Summary statistics
        drift_results['summary'] = {
            'total_columns_checked': len(columns),
            'columns_with_drift': len(drift_results['drift_columns']),
            'max_drift_score': max([score.get('drift_score', 0) for score in drift_results['drift_scores'].values()]),
            'average_drift_score': np.mean([score.get('drift_score', 0) for score in drift_results['drift_scores'].values()])
        }
        
        if drift_results['drift_detected']:
            self._log_action(f"Data drift detected in {len(drift_results['drift_columns'])} columns")
        else:
            self._log_action("No significant data drift detected")
        
        return drift_results
    
    def visualize_feature_distributions(self, data: pd.DataFrame, 
                                      columns: List[str] = None,
                                      save_path: str = None) -> go.Figure:
        """Create comprehensive visualization of feature distributions"""
        if data.empty:
            return go.Figure()
        
        if columns is None:
            # Select numeric columns, limit to reasonable number for visualization
            columns = data.select_dtypes(include=[np.number]).columns.tolist()[:20]
        
        # Create subplots
        n_cols = min(4, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=columns,
            specs=[[{"secondary_y": False} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        for i, col in enumerate(columns):
            row = (i // n_cols) + 1
            col_pos = (i % n_cols) + 1
            
            if col not in data.columns:
                continue
            
            values = data[col].dropna()
            
            if len(values) == 0:
                continue
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=col,
                    nbinsx=50,
                    opacity=0.7,
                    showlegend=False
                ),
                row=row, col=col_pos
            )
            
            # Add statistics annotation
            stats_text = f"Mean: {values.mean():.3f}<br>Std: {values.std():.3f}<br>Skew: {values.skew():.3f}"
            fig.add_annotation(
                text=stats_text,
                xref=f"x{i+1}" if i > 0 else "x",
                yref=f"y{i+1}" if i > 0 else "y",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=8),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title="Feature Distribution Analysis",
            height=300 * n_rows,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        self._log_action(f"Generated distribution plots for {len(columns)} features")
        
        return fig
    
    def visualize_correlation_matrix(self, data: pd.DataFrame,
                                   columns: List[str] = None,
                                   method: str = 'pearson',
                                   save_path: str = None) -> go.Figure:
        """Create correlation matrix visualization"""
        if data.empty:
            return go.Figure()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()[:50]  # Limit for readability
        
        # Calculate correlation matrix
        corr_data = data[columns].corr(method=method)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation"),
            text=np.round(corr_data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f"Feature Correlation Matrix ({method.capitalize()})",
            xaxis_title="Features",
            yaxis_title="Features",
            height=600,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        # Identify highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_data.columns)):
            for j in range(i+1, len(corr_data.columns)):
                corr_val = corr_data.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((corr_data.columns[i], corr_data.columns[j], corr_val))
        
        self._log_action(f"Generated correlation matrix with {len(high_corr_pairs)} highly correlated pairs")
        
        return fig
    
    def detect_anomalies(self, data: pd.DataFrame, columns: List[str] = None,
                        method: str = 'isolation_forest') -> pd.DataFrame:
        """Detect anomalies in the dataset"""
        if data.empty:
            return pd.DataFrame()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        anomaly_scores = pd.DataFrame(index=data.index)
        
        for col in columns:
            if col not in data.columns:
                continue
            
            values = data[col].dropna()
            if len(values) < 10:
                continue
            
            if method == 'zscore':
                # Z-score method
                z_scores = np.abs((values - values.mean()) / values.std())
                anomaly_scores[f'{col}_anomaly'] = z_scores > 3
                
            elif method == 'iqr':
                # IQR method
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies = (data[col] < lower_bound) | (data[col] > upper_bound)
                anomaly_scores[f'{col}_anomaly'] = anomalies
                
            elif method == 'isolation_forest':
                try:
                    from sklearn.ensemble import IsolationForest
                    
                    # Fit isolation forest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_pred = iso_forest.fit_predict(values.values.reshape(-1, 1))
                    
                    # Map to boolean (isolation forest returns -1 for anomalies, 1 for normal)
                    anomaly_mask = pd.Series(anomaly_pred == -1, index=values.index)
                    anomaly_scores[f'{col}_anomaly'] = anomaly_mask.reindex(data.index, fill_value=False)
                    
                except ImportError:
                    # Fallback to IQR method
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    anomalies = (data[col] < lower_bound) | (data[col] > upper_bound)
                    anomaly_scores[f'{col}_anomaly'] = anomalies
        
        # Overall anomaly score
        if not anomaly_scores.empty:
            anomaly_scores['total_anomaly_score'] = anomaly_scores.sum(axis=1)
            anomaly_scores['is_anomaly'] = anomaly_scores['total_anomaly_score'] > len(columns) * 0.3
        
        total_anomalies = anomaly_scores['is_anomaly'].sum() if 'is_anomaly' in anomaly_scores.columns else 0
        self._log_action(f"Detected {total_anomalies} anomalies using {method} method")
        
        return anomaly_scores
    
    def generate_quality_report(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        if not data_dict:
            return {'status': 'error', 'message': 'No data provided'}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
            'overall_summary': {},
            'recommendations': []
        }
        
        total_rows = 0
        total_missing = 0
        quality_scores = []
        
        for symbol, data in data_dict.items():
            if data.empty:
                continue
            
            # Basic statistics
            dataset_stats = {
                'rows': len(data),
                'columns': len(data.columns),
                'missing_values': data.isnull().sum().sum(),
                'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
                'duplicate_rows': data.duplicated().sum(),
                'date_range': {
                    'start': data.index.min().isoformat() if hasattr(data, 'index') and len(data) > 0 else None,
                    'end': data.index.max().isoformat() if hasattr(data, 'index') and len(data) > 0 else None
                }
            }
            
            # Quality score calculation
            quality_score = 100
            if dataset_stats['missing_percentage'] > 20:
                quality_score -= 30
            elif dataset_stats['missing_percentage'] > 5:
                quality_score -= 10
            
            if dataset_stats['duplicate_rows'] > 0:
                quality_score -= 15
            
            dataset_stats['quality_score'] = quality_score
            quality_scores.append(quality_score)
            
            report['datasets'][symbol] = dataset_stats
            total_rows += dataset_stats['rows']
            total_missing += dataset_stats['missing_values']
        
        # Overall summary
        report['overall_summary'] = {
            'total_datasets': len(report['datasets']),
            'total_rows': total_rows,
            'total_missing_values': total_missing,
            'average_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'datasets_with_issues': sum(1 for stats in report['datasets'].values() 
                                      if stats['missing_percentage'] > 5 or stats['duplicate_rows'] > 0)
        }
        
        # Recommendations
        if report['overall_summary']['average_quality_score'] < 80:
            report['recommendations'].append("Overall data quality is below recommended threshold")
        
        if total_missing / total_rows > 0.05:
            report['recommendations'].append("High missing data percentage - consider imputation strategies")
        
        low_quality_datasets = [symbol for symbol, stats in report['datasets'].items() 
                               if stats['quality_score'] < 70]
        if low_quality_datasets:
            report['recommendations'].append(f"Low quality datasets requiring attention: {low_quality_datasets}")
        
        self._log_action(f"Generated quality report for {len(report['datasets'])} datasets")
        
        return report
    
    def _temporal_stratified_split(self, data: pd.DataFrame, train_ratio: float,
                                 val_ratio: float, test_ratio: float) -> Dict[str, pd.DataFrame]:
        """Perform temporal stratified split preserving time order"""
        total_rows = len(data)
        
        train_end = int(total_rows * train_ratio)
        val_end = int(total_rows * (train_ratio + val_ratio))
        
        splits = {
            'train': data.iloc[:train_end],
            'val': data.iloc[train_end:val_end],
            'test': data.iloc[val_end:]
        }
        
        return splits
    
    def _log_action(self, message: str):
        """Log QA actions"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.qa_log.append(log_entry)
    
    def get_qa_log(self) -> List[str]:
        """Get the QA action log"""
        return self.qa_log.copy()
    
    def clear_log(self):
        """Clear the QA log"""
        self.qa_log.clear()


if __name__ == "__main__":
    # Example usage
    qa = DataQualityAssurance()
    
    # Create sample datasets
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data1 = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100),
        'sector': 'Technology',
        'country': 'US'
    }, index=dates)
    
    sample_data2 = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 50,
        'High': np.random.randn(100).cumsum() + 55,
        'Low': np.random.randn(100).cumsum() + 45,
        'Close': np.random.randn(100).cumsum() + 50,
        'Volume': np.random.randint(500, 5000, 100),
        'sector': 'Finance',
        'country': 'US'
    }, index=dates)
    
    data_dict = {'TECH_STOCK': sample_data1, 'FIN_STOCK': sample_data2}
    
    # Generate quality report
    report = qa.generate_quality_report(data_dict)
    print(f"Quality Report: Average score = {report['overall_summary']['average_quality_score']:.1f}")
    
    # Test stratified sampling
    splits = qa.stratified_sampling(sample_data1)
    print(f"Data splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    print("\nQA log:")
    for entry in qa.get_qa_log():
        print(entry)