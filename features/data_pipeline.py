#!/usr/bin/env python3
"""
Comprehensive Data Pipeline
Multi-source data integration, cleaning, enrichment, and quality assurance pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

# Import data sources
from features.data_sources import (
    YahooFinanceAPI, IEXCloudAPI, AlphaVantageAPI, 
    QuandlAPI, FinnhubAPI, BinanceAPI
)

# Import utilities
from utils.data_cleaning import DataCleaner
from utils.data_enrichment import DataEnricher
from utils.data_quality import DataQualityAssurance

warnings.filterwarnings('ignore')


class DataPipeline:
    """Comprehensive data pipeline for multi-source financial data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize pipeline state first
        self.pipeline_log = []
        self.cached_data = {}
        self.processing_stats = {}
        
        # Initialize components
        self.data_sources = self._initialize_data_sources()
        self.cleaner = DataCleaner()
        self.enricher = DataEnricher()
        self.qa = DataQualityAssurance()
        
    def _initialize_data_sources(self) -> Dict[str, Any]:
        """Initialize all data source APIs"""
        sources = {}
        
        # Always initialize Yahoo Finance (free)
        sources['yahoo_finance'] = YahooFinanceAPI()
        
        # Initialize other sources if API keys are provided
        av_key = self.config.get('alpha_vantage_key')
        if av_key:
            sources['alpha_vantage'] = AlphaVantageAPI(api_key=av_key)
        
        iex_key = self.config.get('iex_cloud_key')
        if iex_key:
            sources['iex_cloud'] = IEXCloudAPI(api_key=iex_key, sandbox=self.config.get('iex_sandbox', True))
        
        quandl_key = self.config.get('quandl_key')
        if quandl_key:
            sources['quandl'] = QuandlAPI(api_key=quandl_key)
        
        finnhub_key = self.config.get('finnhub_key')
        if finnhub_key:
            sources['finnhub'] = FinnhubAPI(api_key=finnhub_key)
        
        # Binance doesn't require API key for public data
        sources['binance'] = BinanceAPI()
        
        self._log_action(f"Initialized {len(sources)} data sources: {list(sources.keys())}")
        
        return sources
    
    def fetch_multi_source_data(self, symbols: List[str], period: str = "1y",
                               sources: List[str] = None, parallel: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch data from multiple sources for given symbols"""
        if sources is None:
            sources = list(self.data_sources.keys())
        
        # Filter available sources
        available_sources = [src for src in sources if src in self.data_sources]
        
        if not available_sources:
            self._log_action("No available data sources")
            return {}
        
        self._log_action(f"Fetching data for {len(symbols)} symbols from {len(available_sources)} sources")
        
        if parallel:
            return self._fetch_parallel(symbols, period, available_sources)
        else:
            return self._fetch_sequential(symbols, period, available_sources)
    
    def _fetch_parallel(self, symbols: List[str], period: str, sources: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch data in parallel using ThreadPoolExecutor"""
        results = {symbol: {} for symbol in symbols}
        
        with ThreadPoolExecutor(max_workers=min(len(sources) * len(symbols), 20)) as executor:
            # Submit all fetch tasks
            future_to_info = {}
            
            for symbol in symbols:
                for source in sources:
                    if source in self.data_sources:
                        future = executor.submit(self._fetch_single_source, source, symbol, period)
                        future_to_info[future] = (symbol, source)
            
            # Collect results
            for future in as_completed(future_to_info):
                symbol, source = future_to_info[future]
                try:
                    data = future.result(timeout=30)  # 30 second timeout per fetch
                    if not data.empty:
                        results[symbol][source] = data
                except Exception as e:
                    self._log_action(f"Error fetching {symbol} from {source}: {e}")
        
        return results
    
    def _fetch_sequential(self, symbols: List[str], period: str, sources: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch data sequentially"""
        results = {symbol: {} for symbol in symbols}
        
        for symbol in symbols:
            for source in sources:
                try:
                    data = self._fetch_single_source(source, symbol, period)
                    if not data.empty:
                        results[symbol][source] = data
                except Exception as e:
                    self._log_action(f"Error fetching {symbol} from {source}: {e}")
        
        return results
    
    def _fetch_single_source(self, source: str, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from a single source"""
        api = self.data_sources.get(source)
        if not api:
            return pd.DataFrame()
        
        try:
            # Special handling for crypto symbols on Binance
            if source == 'binance' and any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA', 'SOL']):
                # Convert to Binance format
                if not symbol.upper().endswith('USDT'):
                    symbol = symbol.replace('-USD', '-USDT')
            
            data = api.fetch_market_data(symbol, period=period)
            
            if not data.empty:
                # Add source metadata
                data['data_source'] = source
                data['fetch_timestamp'] = datetime.now()
            
            return data
            
        except Exception as e:
            print(f"Error in _fetch_single_source {source}/{symbol}: {e}")
            return pd.DataFrame()
    
    def process_pipeline(self, symbols: List[str], period: str = "1y",
                        enable_cleaning: bool = True, enable_enrichment: bool = True,
                        enable_qa: bool = True) -> Dict[str, pd.DataFrame]:
        """Run the complete data pipeline"""
        self._log_action(f"Starting pipeline for {len(symbols)} symbols")
        pipeline_start = datetime.now()
        
        # Step 1: Fetch multi-source data
        raw_data = self.fetch_multi_source_data(symbols, period)
        
        # Step 2: Merge and consolidate data
        consolidated_data = self._consolidate_multi_source_data(raw_data)
        
        # Step 3: Data cleaning
        if enable_cleaning:
            consolidated_data = self._clean_pipeline_data(consolidated_data)
        
        # Step 4: Data enrichment
        if enable_enrichment:
            consolidated_data = self._enrich_pipeline_data(consolidated_data)
        
        # Step 5: Quality assurance
        if enable_qa:
            qa_results = self._qa_pipeline_data(consolidated_data)
            self.processing_stats['qa_results'] = qa_results
        
        # Step 6: Balance and stratify data
        consolidated_data = self._balance_pipeline_data(consolidated_data)
        
        # Pipeline completion
        pipeline_end = datetime.now()
        processing_time = (pipeline_end - pipeline_start).total_seconds()
        
        self.processing_stats.update({
            'processing_time_seconds': processing_time,
            'total_symbols_processed': len(consolidated_data),
            'total_features_per_symbol': len(next(iter(consolidated_data.values())).columns) if consolidated_data else 0,
            'pipeline_completion_time': pipeline_end.isoformat()
        })
        
        self._log_action(f"Pipeline completed in {processing_time:.2f} seconds")
        
        return consolidated_data
    
    def _consolidate_multi_source_data(self, raw_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Consolidate data from multiple sources for each symbol"""
        consolidated = {}
        
        for symbol, source_data in raw_data.items():
            if not source_data:
                continue
            
            # Priority order for data sources (most reliable first)
            source_priority = ['yahoo_finance', 'iex_cloud', 'alpha_vantage', 'finnhub', 'binance', 'quandl']
            
            # Find the best available source
            primary_source = None
            primary_data = None
            
            for source in source_priority:
                if source in source_data and not source_data[source].empty:
                    primary_source = source
                    primary_data = source_data[source].copy()
                    break
            
            if primary_data is None:
                self._log_action(f"No valid data found for {symbol}")
                continue
            
            # Merge additional data from other sources
            for source, data in source_data.items():
                if source == primary_source or data.empty:
                    continue
                
                # Merge complementary data (news, sentiment, etc.)
                try:
                    if source == 'alpha_vantage' and 'sector' not in primary_data.columns:
                        # Add fundamental data from Alpha Vantage
                        if hasattr(self.data_sources['alpha_vantage'], 'fetch_company_overview'):
                            overview = self.data_sources['alpha_vantage'].fetch_company_overview(symbol)
                            for key, value in overview.items():
                                if key not in ['symbol', 'source'] and value is not None:
                                    primary_data[key] = value
                    
                    # Add volume-weighted features from high-frequency sources
                    if source in ['binance', 'iex_cloud'] and 'Volume' in data.columns:
                        if data['Volume'].mean() > primary_data['Volume'].mean() * 1.1:
                            # Use higher volume data if significantly better
                            primary_data['Volume'] = data['Volume'].reindex(primary_data.index, method='nearest')
                
                except Exception as e:
                    self._log_action(f"Error merging {source} data for {symbol}: {e}")
            
            # Add metadata
            primary_data['primary_source'] = primary_source
            primary_data['sources_used'] = ','.join(source_data.keys())
            
            consolidated[symbol] = primary_data
        
        self._log_action(f"Consolidated data for {len(consolidated)} symbols")
        return consolidated
    
    def _clean_pipeline_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply data cleaning to all datasets"""
        cleaned_data = {}
        
        for symbol, data in data_dict.items():
            if data.empty:
                continue
            
            try:
                # Clean OHLCV data
                cleaned = self.cleaner.clean_ohlcv_data(data, symbol)
                
                # Detect and handle outliers
                outlier_mask = self.cleaner.detect_outliers(cleaned, method='iqr')
                if not outlier_mask.empty:
                    cleaned = self.cleaner.remove_outliers(cleaned, outlier_mask, action='cap')
                
                # Handle missing values
                cleaned = self.cleaner.handle_missing_values(cleaned, method='interpolate')
                
                cleaned_data[symbol] = cleaned
                
            except Exception as e:
                self._log_action(f"Error cleaning data for {symbol}: {e}")
                cleaned_data[symbol] = data  # Use original data if cleaning fails
        
        # Align time series
        try:
            aligned_data = self.cleaner.align_time_series(cleaned_data)
            self._log_action(f"Cleaned and aligned {len(aligned_data)} datasets")
            return aligned_data
        except Exception as e:
            self._log_action(f"Error aligning time series: {e}")
            return cleaned_data
    
    def _enrich_pipeline_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply data enrichment to all datasets"""
        enriched_data = {}
        
        for symbol, data in data_dict.items():
            if data.empty:
                continue
            
            try:
                enriched = data.copy()
                
                # Add rolling features
                enriched = self.enricher.add_rolling_features(enriched)
                
                # Add volatility features
                enriched = self.enricher.add_volatility_features(enriched)
                
                # Add technical features
                enriched = self.enricher.add_technical_features(enriched)
                
                # Add regime detection
                enriched = self.enricher.add_regime_detection(enriched)
                
                # Add meta-tags
                symbol_info = self._get_symbol_info(symbol, data)
                enriched = self.enricher.add_meta_tags(enriched, symbol_info)
                
                # Normalize features
                enriched, norm_params = self.enricher.normalize_features(enriched, method='minmax')
                
                # Store normalization parameters
                if not hasattr(self, 'normalization_params'):
                    self.normalization_params = {}
                self.normalization_params[symbol] = norm_params
                
                enriched_data[symbol] = enriched
                
            except Exception as e:
                self._log_action(f"Error enriching data for {symbol}: {e}")
                enriched_data[symbol] = data  # Use original data if enrichment fails
        
        self._log_action(f"Enriched {len(enriched_data)} datasets")
        return enriched_data
    
    def _qa_pipeline_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Apply quality assurance to pipeline data"""
        try:
            # Generate comprehensive quality report
            qa_report = self.qa.generate_quality_report(data_dict)
            
            # Detect anomalies in each dataset
            anomaly_results = {}
            for symbol, data in data_dict.items():
                if not data.empty:
                    anomalies = self.qa.detect_anomalies(data, method='isolation_forest')
                    if not anomalies.empty:
                        anomaly_results[symbol] = anomalies['is_anomaly'].sum() if 'is_anomaly' in anomalies.columns else 0
            
            qa_report['anomaly_summary'] = anomaly_results
            
            self._log_action(f"QA completed - Average quality score: {qa_report['overall_summary']['average_quality_score']:.1f}")
            
            return qa_report
            
        except Exception as e:
            self._log_action(f"Error in QA process: {e}")
            return {'error': str(e)}
    
    def _balance_pipeline_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Balance and stratify the final dataset"""
        try:
            # Balance asset selection
            balance_criteria = self.config.get('balance_criteria', {
                'max_per_sector': 5,
                'max_per_region': 10
            })
            
            balanced_data = self.qa.balance_asset_selection(data_dict, balance_criteria)
            
            self._log_action(f"Balanced selection: {len(data_dict)} -> {len(balanced_data)} assets")
            
            return balanced_data
            
        except Exception as e:
            self._log_action(f"Error in balancing data: {e}")
            return data_dict
    
    def _get_symbol_info(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract symbol information from data or fetch from APIs"""
        info = {'symbol': symbol}
        
        # Try to get info from existing data columns
        for col in ['sector', 'industry', 'country', 'currency', 'exchange']:
            if col in data.columns:
                value = data[col].iloc[0] if not data[col].empty else 'Unknown'
                info[col] = value
        
        # Try to fetch from Yahoo Finance
        try:
            if 'yahoo_finance' in self.data_sources:
                yf_info = self.data_sources['yahoo_finance'].fetch_info(symbol)
                info.update(yf_info)
        except Exception:
            pass
        
        return info
    
    def save_pipeline_results(self, data_dict: Dict[str, pd.DataFrame], output_dir: str = "data/processed"):
        """Save processed data and pipeline metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual datasets
        for symbol, data in data_dict.items():
            if not data.empty:
                file_path = os.path.join(output_dir, f"{symbol.replace('/', '_')}_processed.csv")
                data.to_csv(file_path)
        
        # Save pipeline metadata
        metadata = {
            'processing_stats': self.processing_stats,
            'pipeline_log': self.pipeline_log,
            'normalization_params': getattr(self, 'normalization_params', {}),
            'config': self.config
        }
        
        metadata_path = os.path.join(output_dir, "pipeline_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self._log_action(f"Saved results to {output_dir}")
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline execution summary"""
        return {
            'config': self.config,
            'available_sources': list(self.data_sources.keys()),
            'processing_stats': self.processing_stats,
            'pipeline_log': self.pipeline_log,
            'cleaner_log': self.cleaner.get_cleaning_log(),
            'enricher_log': self.enricher.get_enrichment_log(),
            'qa_log': self.qa.get_qa_log()
        }
    
    def _log_action(self, message: str):
        """Log pipeline actions"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.pipeline_log.append(log_entry)
        print(log_entry)  # Also print to console


# Convenience function for easy pipeline execution
def run_data_pipeline(symbols: List[str], config: Dict[str, Any] = None, 
                     period: str = "1y", save_results: bool = True) -> Dict[str, pd.DataFrame]:
    """Run the complete data pipeline with given configuration"""
    
    pipeline = DataPipeline(config)
    
    # Process the pipeline
    results = pipeline.process_pipeline(symbols, period)
    
    # Save results if requested
    if save_results:
        pipeline.save_pipeline_results(results)
    
    # Print summary
    summary = pipeline.get_pipeline_summary()
    print(f"\nPipeline Summary:")
    print(f"- Processed {len(results)} symbols")
    print(f"- Processing time: {summary['processing_stats'].get('processing_time_seconds', 0):.2f} seconds")
    print(f"- Average quality score: {summary['processing_stats'].get('qa_results', {}).get('overall_summary', {}).get('average_quality_score', 0):.1f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    test_symbols = ['AAPL', 'MSFT', 'BTC-USD', 'ETH-USD']
    test_config = {
        'alpha_vantage_key': None,  # Add your API key
        'iex_cloud_key': None,      # Add your API key
        'finnhub_key': None,        # Add your API key
        'balance_criteria': {
            'max_per_sector': 3
        }
    }
    
    print("Running test pipeline...")
    results = run_data_pipeline(test_symbols, test_config, period="3mo")
    
    print(f"\nResults: {len(results)} datasets processed")
    for symbol, data in results.items():
        print(f"- {symbol}: {data.shape[0]} rows, {data.shape[1]} features")