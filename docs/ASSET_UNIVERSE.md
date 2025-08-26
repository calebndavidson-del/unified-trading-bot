# Asset Universe Management

The Asset Universe Management feature allows users to fully customize the trading bot's asset universe through an intuitive web interface. This feature provides comprehensive tools for discovering, adding, removing, and managing stocks, cryptocurrencies, ETFs, and indexes.

## Overview

The Asset Universe Management system consists of:

1. **Asset Universe Manager** (`utils/asset_universe.py`) - Backend logic for asset management
2. **UI Interface** - Streamlit-based web interface in the dashboard
3. **Configuration Integration** - Seamless integration with existing model configuration
4. **Preloaded Asset Lists** - Curated lists of top assets for easy bulk selection

## Features

### 🔍 Search & Discovery
- **Real-time Search**: Search for assets by symbol or company name
- **Asset Type Filtering**: Filter results by stocks, ETFs, cryptocurrencies, or indexes
- **Validation**: Automatic validation using Yahoo Finance API
- **Smart Results**: Displays asset name, type, sector, and industry information

### 📊 Current Universe Management
- **Live Display**: Real-time view of current asset universe with categorization
- **Easy Removal**: One-click removal of individual assets
- **Bulk Operations**: Clear all assets or manage by category
- **Asset Counts**: Live counts for each asset category

### 📚 Preloaded Asset Lists
The system includes four carefully curated preloaded lists:

#### 🏢 Top 250 US Stocks by Market Cap
- **Mega Cap** (>$200B): AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, etc.
- **Large Cap** ($10B-$200B): JNJ, V, XOM, WMT, JPM, MA, PG, HD, etc.
- **Growth Leaders**: Technology, healthcare, and financial sectors
- **Value Stocks**: Established companies with strong fundamentals

#### 📊 Top 50 ETFs by Volume
- **Broad Market**: SPY, QQQ, IWM, VTI, VOO
- **Sector ETFs**: XLF, XLK, XLE, XLV, XLI, XLP, XLY, XLU, XLB, XLRE
- **Bond ETFs**: TLT, IEF, SHY, LQD, HYG, AGG, BND, TIP
- **International**: VEA, VWO, EFA, EEM, VGK, VPL, FXI, INDA
- **Thematic**: ARKK, ARKG, ICLN, SOXX, SMH

#### 🌍 Top 10 Global Indexes
- **US Indexes**: S&P 500 (^GSPC), Dow Jones (^DJI), NASDAQ (^IXIC), Russell 2000 (^RUT)
- **International**: FTSE 100 (^FTSE), DAX (^GDAXI), Nikkei 225 (^N225)
- **Asia-Pacific**: Hang Seng (^HSI), BSE Sensex (^BSESN), ASX 200 (^AXJO)

#### 💎 Top 10 Cryptocurrencies by Market Cap
- **Major Coins**: BTC-USD, ETH-USD, BNB-USD, SOL-USD, ADA-USD
- **Stablecoins**: USDT-USD, USDC-USD
- **DeFi/Staking**: STETH-USD (Lido Staked Ether)
- **Payments**: XRP-USD (Ripple)
- **Meme Coins**: DOGE-USD

### 🎯 Bulk Operations
- **Add All**: One-click addition of entire preloaded lists
- **Quick Add**: Fast addition of top assets from each category
- **Selective Addition**: Individual asset selection from comprehensive tables

## Technical Implementation

### Backend Architecture

```python
# Core Classes
class AssetInfo:
    """Information about a single asset"""
    symbol: str
    name: str
    asset_type: str  # 'stock', 'etf', 'crypto', 'index'
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    volume: Optional[float]
    exchange: Optional[str]
    country: Optional[str]

class AssetUniverse:
    """User's custom asset universe"""
    stocks: Set[str]
    crypto: Set[str] 
    etfs: Set[str]
    indexes: Set[str]
    custom: Set[str]
    last_updated: Optional[datetime]

class AssetUniverseManager:
    """Manager for asset universe operations"""
    # Handles search, validation, persistence, and preloaded lists
```

### Configuration Integration

The system integrates with the existing model configuration:

```python
# model_config.py
@dataclass
class DataConfig:
    use_asset_universe: bool = True  # Enable asset universe management
    symbols: List[str] = field(default_factory=lambda: [...])  # Fallback symbols
    crypto_symbols: List[str] = field(default_factory=lambda: [...])  # Fallback crypto
```

### UI Integration

The dashboard automatically refreshes symbols from the asset universe:

```python
def _refresh_symbols(self):
    """Refresh symbol list from asset universe"""
    if self.config.data.use_asset_universe:
        universe_symbols = self.asset_manager.get_universe().get_all_symbols()
        if universe_symbols:
            self.default_symbols = universe_symbols
```

## Usage Examples

### Basic Usage

1. **Navigate to Asset Universe Tab**: Click on "🌐 Asset Universe" in the main dashboard
2. **Search for Assets**: Type a symbol or company name in the search box
3. **Add Assets**: Click the ➕ button next to search results
4. **Manage Universe**: View and remove assets from the current universe
5. **Bulk Operations**: Use preloaded lists for quick universe setup

### Adding Individual Assets

```python
# Search and add IBM
search_results = manager.search_assets("IBM")
if search_results:
    asset = search_results[0]  # IBM result
    manager.add_to_universe(asset.symbol, asset.asset_type)
    manager.save_universe()
```

### Bulk Addition from Preloaded Lists

```python
# Add all top cryptocurrencies
preloaded = manager.get_preloaded_lists()
crypto_assets = preloaded['top_10_crypto']
count = manager.bulk_add_to_universe(crypto_assets)
manager.save_universe()
print(f"Added {count} cryptocurrencies to universe")
```

### Custom Symbol Validation

```python
# Validate and add custom symbol
is_valid, message, asset_info = manager.validate_symbol("TSLA")
if is_valid and asset_info:
    manager.add_to_universe(asset_info.symbol, asset_info.asset_type)
    manager.save_universe()
    print(f"Added {asset_info.name} to universe")
```

## Best Practices

### Portfolio Diversification
- **Sector Balance**: Include assets from multiple sectors to reduce sector-specific risk
- **Asset Class Mix**: Combine stocks, ETFs, and indexes for balanced exposure
- **Geographic Diversity**: Include international assets for global exposure
- **Market Cap Spread**: Mix large-cap stability with small-cap growth potential

### Performance Optimization
- **Reasonable Universe Size**: Keep universe between 20-100 assets for optimal performance
- **Quality Over Quantity**: Focus on liquid, well-established assets
- **Regular Review**: Periodically review and update universe based on market conditions

### Risk Management
- **Avoid Over-Concentration**: Limit single assets to <10% of universe
- **Correlation Awareness**: Monitor correlations between selected assets
- **Liquidity Considerations**: Ensure selected assets have adequate trading volume

## Data Persistence

Asset universes are automatically saved to `asset_universe.json`:

```json
{
  "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
  "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
  "etfs": ["SPY", "QQQ"],
  "indexes": ["^GSPC"],
  "custom": [],
  "last_updated": "2025-01-27T10:30:00"
}
```

## Error Handling

The system includes comprehensive error handling:
- **Invalid Symbols**: Validation against Yahoo Finance API
- **Network Issues**: Graceful fallback to default symbols
- **Data Corruption**: Automatic recovery with default universe
- **User Feedback**: Clear error messages and success notifications

## Integration with Trading Strategies

The selected asset universe automatically integrates with all trading strategies and models:

1. **Data Pipeline**: Fetches data only for selected assets
2. **Feature Engineering**: Applies technical indicators to universe assets
3. **Model Training**: Uses universe for training data generation
4. **Signal Generation**: Generates trading signals for universe assets only
5. **Risk Management**: Applies risk calculations across the entire universe

## Future Enhancements

Potential future improvements:
- **Custom Lists**: User-created and named asset lists
- **Sector Analysis**: Automatic sector balance recommendations
- **Performance Tracking**: Historical performance analysis of universe changes
- **Import/Export**: CSV import/export of asset universes
- **API Integration**: Integration with additional data providers
- **Social Features**: Sharing and importing community asset lists

## Troubleshooting

### Common Issues

**Assets not appearing in dropdown**:
- Ensure asset universe management is enabled in configuration
- Check that assets are successfully added to universe
- Refresh the page to reload symbols

**Search not working**:
- Verify internet connection for Yahoo Finance API access
- Check that symbols use correct format (e.g., "BTC-USD" for crypto)
- Try alternative symbol formats if search fails

**Universe not persisting**:
- Check file permissions for `asset_universe.json`
- Verify disk space availability
- Ensure application has write permissions

**Performance issues with large universes**:
- Reduce universe size to <100 assets
- Focus on most liquid assets
- Remove duplicate or highly correlated assets

This comprehensive asset universe management system provides traders with full control over their trading bot's asset selection while maintaining ease of use and professional-grade functionality.