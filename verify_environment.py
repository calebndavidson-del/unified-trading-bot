#!/usr/bin/env python3
"""
Environment verification script for the unified trading bot devcontainer.
This script verifies that all dependencies are properly installed and importable.
"""

import sys
import importlib.util

def check_import(module_name, display_name=None):
    """Check if a module can be imported successfully."""
    if display_name is None:
        display_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {display_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {display_name} import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠ {display_name} import had unexpected error: {e}")
        return False

def main():
    """Main verification function."""
    print("=" * 60)
    print("UNIFIED TRADING BOT - ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print()
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Required dependencies
    dependencies = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('yfinance', 'YFinance'),
        ('yaml', 'PyYAML'),
        ('requests', 'Requests'),
        ('dash', 'Dash'),
        ('plotly', 'Plotly'),
    ]
    
    print("Checking required dependencies:")
    print("-" * 40)
    
    success_count = 0
    total_count = len(dependencies)
    
    for module_name, display_name in dependencies:
        if check_import(module_name, display_name):
            success_count += 1
    
    print()
    print("Checking application modules:")
    print("-" * 40)
    
    app_modules = [
        ('dashboard', 'Dashboard'),
        ('quant_bot', 'Quant Bot'),
    ]
    
    for module_name, display_name in app_modules:
        if check_import(module_name, display_name):
            success_count += 1
    
    total_count += len(app_modules)
    
    print()
    print("=" * 60)
    print(f"VERIFICATION SUMMARY: {success_count}/{total_count} modules imported successfully")
    
    if success_count == total_count:
        print("🎉 ALL CHECKS PASSED! Environment is ready for development.")
        return 0
    else:
        print("❌ Some modules failed to import. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())