# Data sources module for trading bot
"""
Multi-source data fetching and integration for enhanced trading analytics.
Supports Yahoo Finance, IEX Cloud, Alpha Vantage, Quandl, Finnhub, and Binance.
"""

from .yahoo_finance import YahooFinanceAPI
from .iex_cloud import IEXCloudAPI
from .alpha_vantage import AlphaVantageAPI
from .quandl import QuandlAPI
from .finnhub import FinnhubAPI
from .binance import BinanceAPI

__all__ = [
    'YahooFinanceAPI',
    'IEXCloudAPI', 
    'AlphaVantageAPI',
    'QuandlAPI',
    'FinnhubAPI',
    'BinanceAPI'
]