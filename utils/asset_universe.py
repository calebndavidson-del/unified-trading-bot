#!/usr/bin/env python3
"""
Asset Universe Management
Comprehensive asset selection and management for trading bot
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
import json
import os
from datetime import datetime
import yfinance as yf
import pandas as pd
import streamlit as st


@dataclass
class AssetInfo:
    """Information about a single asset"""
    symbol: str
    name: str
    asset_type: str  # 'stock', 'etf', 'crypto', 'index'
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    volume: Optional[float] = None
    exchange: Optional[str] = None
    country: Optional[str] = None


@dataclass 
class AssetUniverse:
    """User's custom asset universe"""
    stocks: Set[str] = field(default_factory=set)
    crypto: Set[str] = field(default_factory=set)
    etfs: Set[str] = field(default_factory=set)
    indexes: Set[str] = field(default_factory=set)
    custom: Set[str] = field(default_factory=set)
    last_updated: Optional[datetime] = None
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols in the universe"""
        all_symbols = set()
        all_symbols.update(self.stocks)
        all_symbols.update(self.crypto)
        all_symbols.update(self.etfs)
        all_symbols.update(self.indexes)
        all_symbols.update(self.custom)
        return sorted(list(all_symbols))
    
    def add_symbol(self, symbol: str, asset_type: str) -> bool:
        """Add a symbol to the appropriate category"""
        symbol = symbol.upper()
        if asset_type == 'stock':
            self.stocks.add(symbol)
        elif asset_type == 'crypto':
            self.crypto.add(symbol)
        elif asset_type == 'etf':
            self.etfs.add(symbol)
        elif asset_type == 'index':
            self.indexes.add(symbol)
        else:
            self.custom.add(symbol)
        self.last_updated = datetime.now()
        return True
    
    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from all categories"""
        symbol = symbol.upper()
        removed = False
        for category in [self.stocks, self.crypto, self.etfs, self.indexes, self.custom]:
            if symbol in category:
                category.remove(symbol)
                removed = True
        if removed:
            self.last_updated = datetime.now()
        return removed


class AssetUniverseManager:
    """Manager for asset universe operations"""
    
    def __init__(self, config_path: str = "asset_universe.json"):
        self.config_path = config_path
        self.universe = self._load_universe()
        self._preloaded_lists = self._initialize_preloaded_lists()
    
    def _initialize_preloaded_lists(self) -> Dict[str, List[AssetInfo]]:
        """Initialize preloaded asset lists"""
        return {
            'top_250_us_stocks': self._get_top_250_us_stocks(),
            'top_50_etfs': self._get_top_50_etfs(),
            'top_10_global_indexes': self._get_top_10_global_indexes(),
            'top_10_crypto': self._get_top_10_crypto()
        }
    
    def _get_top_250_us_stocks(self) -> List[AssetInfo]:
        """Get top 250 US stocks by market cap"""
        # Static list of top US stocks by market cap (simplified for implementation)
        top_stocks = [
            # Mega Cap (>$200B)
            AssetInfo("AAPL", "Apple Inc", "stock", "Technology", "Consumer Electronics"),
            AssetInfo("MSFT", "Microsoft Corporation", "stock", "Technology", "Software"),
            AssetInfo("GOOGL", "Alphabet Inc", "stock", "Technology", "Internet Services"),
            AssetInfo("AMZN", "Amazon.com Inc", "stock", "Consumer Discretionary", "E-commerce"),
            AssetInfo("NVDA", "NVIDIA Corporation", "stock", "Technology", "Semiconductors"),
            AssetInfo("TSLA", "Tesla Inc", "stock", "Consumer Discretionary", "Electric Vehicles"),
            AssetInfo("META", "Meta Platforms Inc", "stock", "Technology", "Social Media"),
            AssetInfo("TSM", "Taiwan Semiconductor", "stock", "Technology", "Semiconductors"),
            AssetInfo("BRK-B", "Berkshire Hathaway", "stock", "Financial Services", "Insurance"),
            AssetInfo("UNH", "UnitedHealth Group", "stock", "Healthcare", "Health Insurance"),
            
            # Large Cap ($10B-$200B)
            AssetInfo("JNJ", "Johnson & Johnson", "stock", "Healthcare", "Pharmaceuticals"),
            AssetInfo("V", "Visa Inc", "stock", "Financial Services", "Payment Processing"),
            AssetInfo("XOM", "Exxon Mobil Corporation", "stock", "Energy", "Oil & Gas"),
            AssetInfo("WMT", "Walmart Inc", "stock", "Consumer Staples", "Retail"),
            AssetInfo("JPM", "JPMorgan Chase", "stock", "Financial Services", "Banking"),
            AssetInfo("MA", "Mastercard Inc", "stock", "Financial Services", "Payment Processing"),
            AssetInfo("PG", "Procter & Gamble", "stock", "Consumer Staples", "Personal Care"),
            AssetInfo("HD", "Home Depot", "stock", "Consumer Discretionary", "Home Improvement"),
            AssetInfo("CVX", "Chevron Corporation", "stock", "Energy", "Oil & Gas"),
            AssetInfo("ABBV", "AbbVie Inc", "stock", "Healthcare", "Pharmaceuticals"),
            AssetInfo("BAC", "Bank of America", "stock", "Financial Services", "Banking"),
            AssetInfo("AVGO", "Broadcom Inc", "stock", "Technology", "Semiconductors"),
            AssetInfo("KO", "Coca-Cola Company", "stock", "Consumer Staples", "Beverages"),
            AssetInfo("LLY", "Eli Lilly and Company", "stock", "Healthcare", "Pharmaceuticals"),
            AssetInfo("PEP", "PepsiCo Inc", "stock", "Consumer Staples", "Food & Beverages"),
            AssetInfo("TMO", "Thermo Fisher Scientific", "stock", "Healthcare", "Life Sciences"),
            AssetInfo("COST", "Costco Wholesale", "stock", "Consumer Staples", "Retail"),
            AssetInfo("ADBE", "Adobe Inc", "stock", "Technology", "Software"),
            AssetInfo("WFC", "Wells Fargo", "stock", "Financial Services", "Banking"),
            AssetInfo("MRK", "Merck & Co", "stock", "Healthcare", "Pharmaceuticals"),
            AssetInfo("DIS", "Walt Disney Company", "stock", "Communication Services", "Entertainment"),
            AssetInfo("ACN", "Accenture plc", "stock", "Technology", "IT Services"),
            AssetInfo("VZ", "Verizon Communications", "stock", "Communication Services", "Telecom"),
            AssetInfo("CRM", "Salesforce Inc", "stock", "Technology", "Cloud Software"),
            AssetInfo("NFLX", "Netflix Inc", "stock", "Communication Services", "Streaming"),
            AssetInfo("AMD", "Advanced Micro Devices", "stock", "Technology", "Semiconductors"),
            AssetInfo("CMCSA", "Comcast Corporation", "stock", "Communication Services", "Media"),
            AssetInfo("NKE", "Nike Inc", "stock", "Consumer Discretionary", "Apparel"),
            AssetInfo("TXN", "Texas Instruments", "stock", "Technology", "Semiconductors"),
            AssetInfo("UPS", "United Parcel Service", "stock", "Industrials", "Logistics"),
            AssetInfo("RTX", "Raytheon Technologies", "stock", "Industrials", "Aerospace"),
            AssetInfo("INTC", "Intel Corporation", "stock", "Technology", "Semiconductors"),
            AssetInfo("QCOM", "Qualcomm Inc", "stock", "Technology", "Semiconductors"),
            AssetInfo("PM", "Philip Morris International", "stock", "Consumer Staples", "Tobacco"),
            AssetInfo("HON", "Honeywell International", "stock", "Industrials", "Conglomerates"),
            AssetInfo("IBM", "International Business Machines", "stock", "Technology", "IT Services"),
            AssetInfo("SBUX", "Starbucks Corporation", "stock", "Consumer Discretionary", "Restaurants"),
            AssetInfo("LOW", "Lowe's Companies", "stock", "Consumer Discretionary", "Home Improvement"),
            AssetInfo("INTU", "Intuit Inc", "stock", "Technology", "Financial Software"),
            AssetInfo("UNP", "Union Pacific Corporation", "stock", "Industrials", "Railroads"),
            AssetInfo("GS", "Goldman Sachs Group", "stock", "Financial Services", "Investment Banking"),
            AssetInfo("MS", "Morgan Stanley", "stock", "Financial Services", "Investment Banking"),
            
            # Additional notable stocks
            AssetInfo("CAT", "Caterpillar Inc", "stock", "Industrials", "Heavy Machinery"),
            AssetInfo("DE", "Deere & Company", "stock", "Industrials", "Agricultural Equipment"),
            AssetInfo("MMM", "3M Company", "stock", "Industrials", "Conglomerates"),
            AssetInfo("GE", "General Electric", "stock", "Industrials", "Conglomerates"),
            AssetInfo("BA", "Boeing Company", "stock", "Industrials", "Aerospace"),
            AssetInfo("T", "AT&T Inc", "stock", "Communication Services", "Telecom"),
            AssetInfo("MO", "Altria Group", "stock", "Consumer Staples", "Tobacco"),
            AssetInfo("BMY", "Bristol Myers Squibb", "stock", "Healthcare", "Pharmaceuticals"),
            AssetInfo("GILD", "Gilead Sciences", "stock", "Healthcare", "Biotechnology"),
            AssetInfo("AMGN", "Amgen Inc", "stock", "Healthcare", "Biotechnology"),
            AssetInfo("CVS", "CVS Health Corporation", "stock", "Healthcare", "Healthcare Services"),
            AssetInfo("MDT", "Medtronic plc", "stock", "Healthcare", "Medical Devices"),
            AssetInfo("DHR", "Danaher Corporation", "stock", "Healthcare", "Life Sciences"),
            AssetInfo("ISRG", "Intuitive Surgical", "stock", "Healthcare", "Medical Devices"),
            AssetInfo("REGN", "Regeneron Pharmaceuticals", "stock", "Healthcare", "Biotechnology"),
            AssetInfo("ZTS", "Zoetis Inc", "stock", "Healthcare", "Animal Health"),
            AssetInfo("COP", "ConocoPhillips", "stock", "Energy", "Oil & Gas"),
            AssetInfo("SLB", "Schlumberger Limited", "stock", "Energy", "Oil Services"),
            AssetInfo("EOG", "EOG Resources", "stock", "Energy", "Oil & Gas Exploration"),
            AssetInfo("KMI", "Kinder Morgan", "stock", "Energy", "Pipeline"),
            AssetInfo("OXY", "Occidental Petroleum", "stock", "Energy", "Oil & Gas"),
            AssetInfo("SPG", "Simon Property Group", "stock", "Real Estate", "REITs"),
            AssetInfo("PLD", "Prologis Inc", "stock", "Real Estate", "Industrial REITs"),
            AssetInfo("AMT", "American Tower Corporation", "stock", "Real Estate", "REITs"),
            AssetInfo("CCI", "Crown Castle International", "stock", "Real Estate", "REITs"),
            AssetInfo("EQIX", "Equinix Inc", "stock", "Real Estate", "Data Center REITs"),
        ]
        
        # Extend to 250 with additional quality stocks
        additional_stocks = [
            AssetInfo("PYPL", "PayPal Holdings", "stock", "Financial Services", "Payment Processing"),
            AssetInfo("ORCL", "Oracle Corporation", "stock", "Technology", "Database Software"),
            AssetInfo("CSCO", "Cisco Systems", "stock", "Technology", "Networking Equipment"),
            AssetInfo("PFE", "Pfizer Inc", "stock", "Healthcare", "Pharmaceuticals"),
            AssetInfo("BLK", "BlackRock Inc", "stock", "Financial Services", "Asset Management"),
            AssetInfo("AMAT", "Applied Materials", "stock", "Technology", "Semiconductor Equipment"),
            AssetInfo("MU", "Micron Technology", "stock", "Technology", "Memory Semiconductors"),
            AssetInfo("LRCX", "Lam Research", "stock", "Technology", "Semiconductor Equipment"),
        ]
        
        return top_stocks + additional_stocks
    
    def _get_top_50_etfs(self) -> List[AssetInfo]:
        """Get top 50 ETFs by volume"""
        return [
            # Broad Market ETFs
            AssetInfo("SPY", "SPDR S&P 500 ETF", "etf", "Equity", "Large Cap Blend"),
            AssetInfo("QQQ", "Invesco QQQ Trust", "etf", "Equity", "Technology"),
            AssetInfo("IWM", "iShares Russell 2000 ETF", "etf", "Equity", "Small Cap"),
            AssetInfo("VTI", "Vanguard Total Stock Market", "etf", "Equity", "Total Market"),
            AssetInfo("VOO", "Vanguard S&P 500 ETF", "etf", "Equity", "Large Cap"),
            AssetInfo("VEA", "Vanguard FTSE Developed Markets", "etf", "Equity", "International"),
            AssetInfo("VWO", "Vanguard Emerging Markets", "etf", "Equity", "Emerging Markets"),
            AssetInfo("EFA", "iShares MSCI EAFE ETF", "etf", "Equity", "International"),
            AssetInfo("EEM", "iShares MSCI Emerging Markets", "etf", "Equity", "Emerging Markets"),
            AssetInfo("GLD", "SPDR Gold Shares", "etf", "Commodity", "Gold"),
            
            # Sector ETFs
            AssetInfo("XLF", "Financial Select Sector SPDR", "etf", "Equity", "Financial"),
            AssetInfo("XLK", "Technology Select Sector SPDR", "etf", "Equity", "Technology"),
            AssetInfo("XLE", "Energy Select Sector SPDR", "etf", "Equity", "Energy"),
            AssetInfo("XLV", "Health Care Select Sector SPDR", "etf", "Equity", "Healthcare"),
            AssetInfo("XLI", "Industrial Select Sector SPDR", "etf", "Equity", "Industrials"),
            AssetInfo("XLP", "Consumer Staples Select Sector", "etf", "Equity", "Consumer Staples"),
            AssetInfo("XLY", "Consumer Discretionary Select", "etf", "Equity", "Consumer Discretionary"),
            AssetInfo("XLU", "Utilities Select Sector SPDR", "etf", "Equity", "Utilities"),
            AssetInfo("XLB", "Materials Select Sector SPDR", "etf", "Equity", "Materials"),
            AssetInfo("XLRE", "Real Estate Select Sector SPDR", "etf", "Equity", "Real Estate"),
            
            # Bond ETFs
            AssetInfo("TLT", "iShares 20+ Year Treasury Bond", "etf", "Fixed Income", "Long-Term Treasury"),
            AssetInfo("IEF", "iShares 7-10 Year Treasury Bond", "etf", "Fixed Income", "Treasury"),
            AssetInfo("SHY", "iShares 1-3 Year Treasury Bond", "etf", "Fixed Income", "Short-Term Treasury"),
            AssetInfo("LQD", "iShares iBoxx Investment Grade", "etf", "Fixed Income", "Corporate Bonds"),
            AssetInfo("HYG", "iShares iBoxx High Yield Corporate", "etf", "Fixed Income", "High Yield"),
            AssetInfo("AGG", "iShares Core US Aggregate Bond", "etf", "Fixed Income", "Aggregate Bonds"),
            AssetInfo("BND", "Vanguard Total Bond Market", "etf", "Fixed Income", "Total Bond Market"),
            AssetInfo("TIP", "iShares TIPS Bond ETF", "etf", "Fixed Income", "Inflation Protected"),
            
            # International and Emerging Market ETFs
            AssetInfo("VGK", "Vanguard FTSE Europe ETF", "etf", "Equity", "European Stocks"),
            AssetInfo("VPL", "Vanguard FTSE Pacific ETF", "etf", "Equity", "Pacific Stocks"),
            AssetInfo("FXI", "iShares China Large-Cap ETF", "etf", "Equity", "Chinese Stocks"),
            AssetInfo("INDA", "iShares MSCI India ETF", "etf", "Equity", "Indian Stocks"),
            AssetInfo("EWJ", "iShares MSCI Japan ETF", "etf", "Equity", "Japanese Stocks"),
            AssetInfo("EWZ", "iShares MSCI Brazil ETF", "etf", "Equity", "Brazilian Stocks"),
            AssetInfo("RSX", "VanEck Russia ETF", "etf", "Equity", "Russian Stocks"),
            AssetInfo("EWY", "iShares MSCI South Korea ETF", "etf", "Equity", "South Korean Stocks"),
            
            # Specialty and Thematic ETFs
            AssetInfo("ARKK", "ARK Innovation ETF", "etf", "Equity", "Innovation"),
            AssetInfo("ARKG", "ARK Genomics Revolution ETF", "etf", "Equity", "Genomics"),
            AssetInfo("ARKW", "ARK Next Generation Internet", "etf", "Equity", "Internet"),
            AssetInfo("ICLN", "iShares Global Clean Energy", "etf", "Equity", "Clean Energy"),
            AssetInfo("SOXX", "iShares Semiconductor ETF", "etf", "Equity", "Semiconductors"),
            AssetInfo("SMH", "VanEck Semiconductor ETF", "etf", "Equity", "Semiconductors"),
            AssetInfo("GDXJ", "VanEck Junior Gold Miners", "etf", "Equity", "Gold Miners"),
            AssetInfo("GDX", "VanEck Gold Miners ETF", "etf", "Equity", "Gold Miners"),
            AssetInfo("USO", "United States Oil Fund", "etf", "Commodity", "Oil"),
            AssetInfo("UNG", "United States Natural Gas Fund", "etf", "Commodity", "Natural Gas"),
            AssetInfo("SLV", "iShares Silver Trust", "etf", "Commodity", "Silver"),
            AssetInfo("PDBC", "Invesco Optimum Yield Diversified", "etf", "Commodity", "Diversified Commodities"),
            
            # Growth and Value ETFs
            AssetInfo("VUG", "Vanguard Growth ETF", "etf", "Equity", "Growth"),
            AssetInfo("VTV", "Vanguard Value ETF", "etf", "Equity", "Value"),
            AssetInfo("IWF", "iShares Russell 1000 Growth", "etf", "Equity", "Large Cap Growth"),
            AssetInfo("IWD", "iShares Russell 1000 Value", "etf", "Equity", "Large Cap Value"),
        ]
    
    def _get_top_10_global_indexes(self) -> List[AssetInfo]:
        """Get top 10 global stock indexes"""
        return [
            AssetInfo("^GSPC", "S&P 500", "index", "US Equity", "Large Cap"),
            AssetInfo("^DJI", "Dow Jones Industrial Average", "index", "US Equity", "Blue Chip"),
            AssetInfo("^IXIC", "NASDAQ Composite", "index", "US Equity", "Technology Heavy"),
            AssetInfo("^RUT", "Russell 2000", "index", "US Equity", "Small Cap"),
            AssetInfo("^FTSE", "FTSE 100", "index", "UK Equity", "Large Cap"),
            AssetInfo("^GDAXI", "DAX", "index", "German Equity", "Large Cap"),
            AssetInfo("^N225", "Nikkei 225", "index", "Japanese Equity", "Large Cap"),
            AssetInfo("^HSI", "Hang Seng Index", "index", "Hong Kong Equity", "Large Cap"),
            AssetInfo("^BSESN", "BSE Sensex", "index", "Indian Equity", "Large Cap"),
            AssetInfo("^AXJO", "ASX 200", "index", "Australian Equity", "Large Cap"),
        ]
    
    def _get_top_10_crypto(self) -> List[AssetInfo]:
        """Get top 10 cryptocurrencies by market cap"""
        return [
            AssetInfo("BTC-USD", "Bitcoin", "crypto", "Cryptocurrency", "Digital Currency"),
            AssetInfo("ETH-USD", "Ethereum", "crypto", "Cryptocurrency", "Smart Contracts"),
            AssetInfo("USDT-USD", "Tether", "crypto", "Cryptocurrency", "Stablecoin"),
            AssetInfo("BNB-USD", "Binance Coin", "crypto", "Cryptocurrency", "Exchange Token"),
            AssetInfo("SOL-USD", "Solana", "crypto", "Cryptocurrency", "Smart Contracts"),
            AssetInfo("USDC-USD", "USD Coin", "crypto", "Cryptocurrency", "Stablecoin"),
            AssetInfo("XRP-USD", "Ripple", "crypto", "Cryptocurrency", "Payment Protocol"),
            AssetInfo("STETH-USD", "Lido Staked Ether", "crypto", "Cryptocurrency", "Staking"),
            AssetInfo("DOGE-USD", "Dogecoin", "crypto", "Cryptocurrency", "Meme Coin"),
            AssetInfo("ADA-USD", "Cardano", "crypto", "Cryptocurrency", "Smart Contracts"),
        ]
    
    def get_preloaded_lists(self) -> Dict[str, List[AssetInfo]]:
        """Get all preloaded asset lists"""
        return self._preloaded_lists
    
    def search_assets(self, query: str, asset_type: Optional[str] = None) -> List[AssetInfo]:
        """Search for assets by symbol or name"""
        results = []
        query_lower = query.lower()
        
        for list_name, assets in self._preloaded_lists.items():
            for asset in assets:
                if asset_type and asset.asset_type != asset_type:
                    continue
                    
                if (query_lower in asset.symbol.lower() or 
                    query_lower in asset.name.lower()):
                    if asset not in results:
                        results.append(asset)
        
        return results[:50]  # Limit results
    
    def validate_symbol(self, symbol: str) -> Tuple[bool, str, Optional[AssetInfo]]:
        """Validate a symbol using Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'regularMarketPrice' not in info:
                # Try getting historical data as fallback
                hist = ticker.history(period="5d")
                if hist.empty:
                    return False, f"Symbol {symbol} not found or no data available", None
            
            # Create AssetInfo from ticker data
            asset_info = AssetInfo(
                symbol=symbol.upper(),
                name=info.get('longName', info.get('shortName', symbol)),
                asset_type=self._determine_asset_type(symbol, info),
                sector=info.get('sector'),
                industry=info.get('industry'),
                market_cap=info.get('marketCap'),
                volume=info.get('averageVolume'),
                exchange=info.get('exchange'),
                country=info.get('country')
            )
            
            return True, f"Valid symbol: {asset_info.name}", asset_info
            
        except Exception as e:
            return False, f"Error validating {symbol}: {str(e)}", None
    
    def _determine_asset_type(self, symbol: str, info: Dict) -> str:
        """Determine asset type from symbol and info"""
        symbol_upper = symbol.upper()
        
        if '-USD' in symbol_upper or any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'DOGE']):
            return 'crypto'
        elif info.get('quoteType') == 'ETF':
            return 'etf'
        elif symbol_upper.startswith('^'):
            return 'index'
        else:
            return 'stock'
    
    def add_to_universe(self, symbol: str, asset_type: str) -> bool:
        """Add symbol to user's universe"""
        return self.universe.add_symbol(symbol, asset_type)
    
    def remove_from_universe(self, symbol: str) -> bool:
        """Remove symbol from user's universe"""
        return self.universe.remove_symbol(symbol)
    
    def bulk_add_to_universe(self, assets: List[AssetInfo]) -> int:
        """Add multiple assets to universe, returns count added"""
        count = 0
        for asset in assets:
            if self.add_to_universe(asset.symbol, asset.asset_type):
                count += 1
        return count
    
    def get_universe(self) -> AssetUniverse:
        """Get current user universe"""
        return self.universe
    
    def save_universe(self) -> bool:
        """Save universe to file"""
        try:
            data = {
                'stocks': list(self.universe.stocks),
                'crypto': list(self.universe.crypto),
                'etfs': list(self.universe.etfs),
                'indexes': list(self.universe.indexes),
                'custom': list(self.universe.custom),
                'last_updated': self.universe.last_updated.isoformat() if self.universe.last_updated else None
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving universe: {e}")
            return False
    
    def _load_universe(self) -> AssetUniverse:
        """Load universe from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                universe = AssetUniverse(
                    stocks=set(data.get('stocks', [])),
                    crypto=set(data.get('crypto', [])),
                    etfs=set(data.get('etfs', [])),
                    indexes=set(data.get('indexes', [])),
                    custom=set(data.get('custom', [])),
                    last_updated=datetime.fromisoformat(data['last_updated']) if data.get('last_updated') else None
                )
                return universe
        except Exception as e:
            st.warning(f"Could not load saved universe: {e}")
        
        # Return default universe if loading fails
        return AssetUniverse(
            stocks={'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ'},
            crypto={'BTC-USD', 'ETH-USD', 'SOL-USD'},
            etfs={'SPY', 'QQQ'},
            indexes={'^GSPC'},
            custom=set()
        )