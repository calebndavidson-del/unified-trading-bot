#!/usr/bin/env python3
"""
Data Pipeline Test Suite
Comprehensive testing for the enhanced data pipeline
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features.data_pipeline import DataPipeline, run_data_pipeline
from features.data_sources import YahooFinanceAPI, BinanceAPI
from utils.data_cleaning import DataCleaner
from utils.data_enrichment import DataEnricher
from utils.data_quality import DataQualityAssurance
from model_config import TradingBotConfig


def test_data_sources():
    """Test individual data source APIs"""
    print("🔄 Testing Data Sources")
    print("=" * 50)
    
    # Test Yahoo Finance (always available)
    print("📊 Testing Yahoo Finance API...")
    yf_api = YahooFinanceAPI()
    try:
        yf_data = yf_api.fetch_market_data("AAPL", period="1mo")
        if not yf_data.empty:
            print(f"✅ Yahoo Finance: Fetched {len(yf_data)} rows for AAPL")
        else:
            print("⚠️ Yahoo Finance: No data returned")
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")
    
    # Test Binance (public API)
    print("🪙 Testing Binance API...")
    binance_api = BinanceAPI()
    try:
        btc_data = binance_api.fetch_market_data("BTC-USDT", period="1mo")
        if not btc_data.empty:
            print(f"✅ Binance: Fetched {len(btc_data)} rows for BTC-USDT")
        else:
            print("⚠️ Binance: No data returned")
    except Exception as e:
        print(f"❌ Binance error: {e}")
    
    print()


def test_data_cleaning():
    """Test data cleaning utilities"""
    print("🧹 Testing Data Cleaning")
    print("=" * 50)
    
    # Create sample data with issues
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Introduce issues
    sample_data.iloc[10:15] = np.nan  # Missing values
    sample_data.iloc[20, 1] = -5      # Negative price
    sample_data.iloc[30, 4] = 1000000 # Extreme volume
    
    cleaner = DataCleaner()
    
    print(f"📈 Original data: {sample_data.shape}")
    print(f"   Missing values: {sample_data.isnull().sum().sum()}")
    
    # Test cleaning
    cleaned_data = cleaner.clean_ohlcv_data(sample_data, "TEST")
    print(f"✨ Cleaned data: {cleaned_data.shape}")
    print(f"   Missing values: {cleaned_data.isnull().sum().sum()}")
    
    # Test outlier detection
    outlier_mask = cleaner.detect_outliers(cleaned_data, method='iqr')
    if not outlier_mask.empty:
        total_outliers = outlier_mask.sum().sum()
        print(f"🎯 Outliers detected: {total_outliers}")
    
    print(f"📝 Cleaning actions: {len(cleaner.get_cleaning_log())}")
    print()


def test_data_enrichment():
    """Test data enrichment utilities"""
    print("🚀 Testing Data Enrichment")
    print("=" * 50)
    
    # Create sample OHLCV data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    enricher = DataEnricher()
    
    print(f"📊 Original features: {sample_data.shape[1]}")
    
    # Add rolling features
    enriched = enricher.add_rolling_features(sample_data)
    print(f"📈 After rolling features: {enriched.shape[1]}")
    
    # Add volatility features
    enriched = enricher.add_volatility_features(enriched)
    print(f"📊 After volatility features: {enriched.shape[1]}")
    
    # Add technical features
    enriched = enricher.add_technical_features(enriched)
    print(f"🔧 After technical features: {enriched.shape[1]}")
    
    # Add regime detection
    enriched = enricher.add_regime_detection(enriched)
    print(f"🎯 After regime detection: {enriched.shape[1]}")
    
    # Add meta tags
    sample_info = {'symbol': 'TEST', 'sector': 'Technology', 'country': 'US'}
    enriched = enricher.add_meta_tags(enriched, sample_info)
    print(f"🏷️ After meta tags: {enriched.shape[1]}")
    
    # Normalization
    normalized, norm_params = enricher.normalize_features(enriched)
    print(f"📏 Normalized features: {len(norm_params)}")
    
    total_features_added = enriched.shape[1] - sample_data.shape[1]
    print(f"✨ Total features added: {total_features_added}")
    print()


def test_data_quality():
    """Test data quality assurance"""
    print("🔍 Testing Data Quality Assurance")
    print("=" * 50)
    
    qa = DataQualityAssurance()
    
    # Create test datasets
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    test_data = {
        'STOCK_A': pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100),
            'sector': 'Technology',
            'country': 'US'
        }, index=dates),
        'STOCK_B': pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 50,
            'Volume': np.random.randint(500, 5000, 100),
            'sector': 'Finance',
            'country': 'US'
        }, index=dates)
    }
    
    # Generate quality report
    quality_report = qa.generate_quality_report(test_data)
    avg_quality = quality_report['overall_summary']['average_quality_score']
    print(f"📊 Quality Report Generated")
    print(f"   Average Quality Score: {avg_quality:.1f}")
    print(f"   Datasets: {quality_report['overall_summary']['total_datasets']}")
    
    # Test stratified sampling
    sample_data = test_data['STOCK_A']
    splits = qa.stratified_sampling(sample_data)
    print(f"📈 Stratified Sampling:")
    print(f"   Train: {len(splits['train'])} rows")
    print(f"   Val: {len(splits['val'])} rows") 
    print(f"   Test: {len(splits['test'])} rows")
    
    # Test anomaly detection
    anomalies = qa.detect_anomalies(sample_data, method='iqr')
    if not anomalies.empty and 'is_anomaly' in anomalies.columns:
        total_anomalies = anomalies['is_anomaly'].sum()
        print(f"🎯 Anomalies detected: {total_anomalies}")
    
    print()


def test_full_pipeline():
    """Test the complete data pipeline"""
    print("🔄 Testing Complete Data Pipeline")
    print("=" * 50)
    
    # Test configuration
    test_config = {
        'alpha_vantage_key': None,
        'iex_cloud_key': None,
        'finnhub_key': None,
        'enable_multi_source': True,
        'save_pipeline_results': False,  # Don't save for test
        'balance_criteria': {
            'max_per_sector': 3
        }
    }
    
    # Test symbols (mix of stocks and crypto)
    test_symbols = ['AAPL', 'MSFT', 'BTC-USD']
    
    try:
        print(f"🚀 Running pipeline for {len(test_symbols)} symbols...")
        
        # Initialize pipeline
        pipeline = DataPipeline(test_config)
        
        # Run pipeline
        results = pipeline.process_pipeline(
            symbols=test_symbols,
            period="3mo",
            enable_cleaning=True,
            enable_enrichment=True,
            enable_qa=True
        )
        
        if results:
            print(f"✅ Pipeline completed successfully!")
            print(f"   Processed: {len(results)} symbols")
            
            for symbol, data in results.items():
                if not data.empty:
                    print(f"   {symbol}: {data.shape[0]} rows, {data.shape[1]} features")
                else:
                    print(f"   {symbol}: No data")
            
            # Get pipeline summary
            summary = pipeline.get_pipeline_summary()
            if 'processing_stats' in summary:
                stats = summary['processing_stats']
                processing_time = stats.get('processing_time_seconds', 0)
                print(f"   Processing time: {processing_time:.2f} seconds")
                
                if 'qa_results' in stats:
                    qa_score = stats['qa_results'].get('overall_summary', {}).get('average_quality_score', 0)
                    print(f"   Average quality score: {qa_score:.1f}")
        
        else:
            print("⚠️ Pipeline returned no results")
    
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def test_configuration():
    """Test configuration loading"""
    print("⚙️ Testing Configuration")
    print("=" * 50)
    
    try:
        # Test loading default config
        config = TradingBotConfig()
        print(f"✅ Default config loaded")
        print(f"   Symbols: {len(config.data.symbols)}")
        print(f"   Enabled sources: {len(config.data.enabled_sources)}")
        print(f"   Rolling windows: {config.data.rolling_windows}")
        print(f"   Data cleaning: {config.data.enable_data_cleaning}")
        print(f"   Data enrichment: {config.data.enable_data_enrichment}")
        print(f"   Quality assurance: {config.data.enable_quality_assurance}")
        
        # Test YAML config loading
        from model_config import load_config
        yaml_config = load_config('config.yaml')
        print(f"✅ YAML config loaded")
        print(f"   Multi-source enabled: {yaml_config.enable_multi_source}")
        print(f"   Parallel processing: {yaml_config.parallel_processing}")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
    
    print()


def main():
    """Run all tests"""
    print("🧪 Running Data Pipeline Test Suite")
    print("=" * 60)
    print()
    
    # Test individual components
    test_data_sources()
    test_data_cleaning()
    test_data_enrichment()
    test_data_quality()
    test_configuration()
    
    # Test full pipeline
    test_full_pipeline()
    
    print("=" * 60)
    print("✅ Test Suite Completed!")
    print()
    
    # Provide usage instructions
    print("📖 Usage Instructions:")
    print("1. Set your API keys in config.yaml or environment variables")
    print("2. Run: python -c \"from features.data_pipeline import run_data_pipeline; run_data_pipeline(['AAPL', 'BTC-USD'])\"")
    print("3. Check the 'data/processed' directory for results")
    print()


if __name__ == "__main__":
    main()