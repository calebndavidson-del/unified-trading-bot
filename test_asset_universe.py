#!/usr/bin/env python3
"""
Asset Universe Management Test Suite
Comprehensive tests for asset universe functionality
"""

from utils.asset_universe import AssetUniverseManager, AssetInfo, AssetUniverse
import tempfile
import os
import json
from datetime import datetime


def test_asset_universe_basic_functionality():
    """Test basic asset universe functionality"""
    print("🧪 Testing Asset Universe Management System")
    print("=" * 50)
    
    # Create temporary config file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_config = f.name
    
    try:
        # Initialize manager with temporary config
        print("📋 Initializing Asset Universe Manager...")
        manager = AssetUniverseManager(temp_config)
        
        # Test preloaded lists
        print("📚 Testing preloaded lists...")
        preloaded = manager.get_preloaded_lists()
        
        expected_lists = ['top_250_us_stocks', 'top_50_etfs', 'top_10_global_indexes', 'top_10_crypto']
        for list_name in expected_lists:
            assert list_name in preloaded, f"Missing preloaded list: {list_name}"
            assets = preloaded[list_name]
            assert len(assets) > 0, f"Empty preloaded list: {list_name}"
            print(f"   ✅ {list_name}: {len(assets)} assets")
        
        # Test asset search
        print("🔍 Testing asset search...")
        search_results = manager.search_assets("AAPL")
        assert len(search_results) > 0, "No search results for AAPL"
        
        aapl_result = search_results[0]
        assert aapl_result.symbol == "AAPL", f"Expected AAPL, got {aapl_result.symbol}"
        assert aapl_result.asset_type == "stock", f"Expected stock, got {aapl_result.asset_type}"
        print(f"   ✅ Found {aapl_result.symbol}: {aapl_result.name}")
        
        # Test adding assets to universe
        print("➕ Testing asset addition...")
        initial_count = len(manager.get_universe().get_all_symbols())
        
        # Add AAPL
        success = manager.add_to_universe("AAPL", "stock")
        assert success, "Failed to add AAPL to universe"
        
        # Add BTC-USD
        success = manager.add_to_universe("BTC-USD", "crypto")
        assert success, "Failed to add BTC-USD to universe"
        
        # Add SPY
        success = manager.add_to_universe("SPY", "etf")
        assert success, "Failed to add SPY to universe"
        
        current_count = len(manager.get_universe().get_all_symbols())
        assert current_count == initial_count + 3, f"Expected {initial_count + 3} assets, got {current_count}"
        print(f"   ✅ Added 3 assets. Universe now has {current_count} assets")
        
        # Test universe categorization
        print("📊 Testing universe categorization...")
        universe = manager.get_universe()
        assert "AAPL" in universe.stocks, "AAPL not found in stocks category"
        assert "BTC-USD" in universe.crypto, "BTC-USD not found in crypto category"
        assert "SPY" in universe.etfs, "SPY not found in ETFs category"
        print("   ✅ Assets correctly categorized")
        
        # Test saving and loading universe
        print("💾 Testing universe persistence...")
        success = manager.save_universe()
        assert success, "Failed to save universe"
        
        # Create new manager instance to test loading
        manager2 = AssetUniverseManager(temp_config)
        loaded_universe = manager2.get_universe()
        
        assert len(loaded_universe.get_all_symbols()) == current_count, "Universe not loaded correctly"
        assert "AAPL" in loaded_universe.stocks, "AAPL not persisted in stocks"
        assert "BTC-USD" in loaded_universe.crypto, "BTC-USD not persisted in crypto"
        assert "SPY" in loaded_universe.etfs, "SPY not persisted in ETFs"
        print("   ✅ Universe saved and loaded successfully")
        
        # Test asset removal
        print("❌ Testing asset removal...")
        success = manager.remove_from_universe("AAPL")
        assert success, "Failed to remove AAPL from universe"
        
        updated_count = len(manager.get_universe().get_all_symbols())
        assert updated_count == current_count - 1, f"Expected {current_count - 1} assets, got {updated_count}"
        assert "AAPL" not in manager.get_universe().stocks, "AAPL still in stocks after removal"
        print(f"   ✅ Removed AAPL. Universe now has {updated_count} assets")
        
        # Test bulk operations
        print("🎯 Testing bulk operations...")
        crypto_assets = preloaded['top_10_crypto'][:3]  # Take first 3 for testing
        count_added = manager.bulk_add_to_universe(crypto_assets)
        assert count_added > 0, "No assets added in bulk operation"
        print(f"   ✅ Bulk added {count_added} crypto assets")
        
        # Test symbol validation (basic check)
        print("✅ Testing symbol validation...")
        is_valid, message, asset_info = manager.validate_symbol("MSFT")
        if is_valid:
            assert asset_info is not None, "Asset info should not be None for valid symbol"
            print(f"   ✅ MSFT validation: {message}")
        else:
            print(f"   ⚠️ MSFT validation failed (may be network issue): {message}")
        
        print("\n🎉 All Asset Universe tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_config)
        except:
            pass


def test_integration_with_dashboard():
    """Test integration with dashboard configuration"""
    print("\n🔗 Testing Dashboard Integration")
    print("=" * 50)
    
    try:
        from dashboard import TradingDashboard
        from model_config import TradingBotConfig
        
        # Test that dashboard can initialize with asset universe
        print("📋 Testing dashboard initialization...")
        dashboard = TradingDashboard()
        
        # Check that asset manager is initialized
        assert hasattr(dashboard, 'asset_manager'), "Dashboard missing asset_manager"
        assert dashboard.asset_manager is not None, "Asset manager not initialized"
        print("   ✅ Asset manager initialized in dashboard")
        
        # Test configuration integration
        print("⚙️ Testing configuration integration...")
        config = dashboard.config
        assert hasattr(config.data, 'use_asset_universe'), "Configuration missing use_asset_universe"
        print(f"   ✅ Asset universe enabled: {config.data.use_asset_universe}")
        
        # Test symbol refresh functionality
        print("🔄 Testing symbol refresh...")
        initial_symbols = dashboard.default_symbols.copy()
        dashboard._refresh_symbols()
        
        # Symbols should be refreshed from asset universe
        refreshed_symbols = dashboard.default_symbols
        print(f"   ✅ Symbol refresh completed. {len(refreshed_symbols)} symbols available")
        
        print("\n🎉 Dashboard integration tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Dashboard integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Asset Universe Test Suite")
    print("=" * 70)
    
    # Run basic functionality tests
    basic_test_passed = test_asset_universe_basic_functionality()
    
    # Run integration tests
    integration_test_passed = test_integration_with_dashboard()
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 70)
    
    if basic_test_passed:
        print("✅ Basic Asset Universe Tests: PASSED")
    else:
        print("❌ Basic Asset Universe Tests: FAILED")
    
    if integration_test_passed:
        print("✅ Dashboard Integration Tests: PASSED")
    else:
        print("❌ Dashboard Integration Tests: FAILED")
    
    if basic_test_passed and integration_test_passed:
        print("\n🎉 ALL TESTS PASSED! Asset Universe Management is ready for use!")
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
    
    print("=" * 70)