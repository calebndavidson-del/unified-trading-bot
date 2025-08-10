/**
 * Main App component for the Current Market Dashboard
 */

import React, { useState, useEffect } from 'react';
import MarketChart from './components/MarketChart';
import MarketCard from './components/MarketCard';
import LoadingSpinner from './components/LoadingSpinner';
import { getMarketData, checkApiHealth } from './services/api';
import './App.css';

function App() {
  const [marketData, setMarketData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [selectedPeriod, setSelectedPeriod] = useState('5d');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [apiHealth, setApiHealth] = useState(null);

  // Fetch market data
  const fetchMarketData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await getMarketData(selectedCategory, selectedPeriod, '1h');
      setMarketData(data);
      setLastUpdate(new Date());
    } catch (err) {
      console.error('Failed to fetch market data:', err);
      setError('Failed to load market data. Please check your connection and try again.');
    } finally {
      setLoading(false);
    }
  };

  // Check API health
  const checkHealth = async () => {
    try {
      const health = await checkApiHealth();
      setApiHealth(health);
    } catch (err) {
      console.error('API health check failed:', err);
      setApiHealth({ status: 'offline' });
    }
  };

  // Initial load
  useEffect(() => {
    checkHealth();
    fetchMarketData();
  }, [selectedPeriod, selectedCategory]);

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const interval = setInterval(() => {
      fetchMarketData();
    }, 300000); // 5 minutes

    return () => clearInterval(interval);
  }, [selectedPeriod, selectedCategory]);

  if (loading && !marketData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <LoadingSpinner size="large" message="Loading current market data..." />
      </div>
    );
  }

  if (error && !marketData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center p-8">
          <div className="text-red-500 text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Error Loading Data</h2>
          <p className="text-gray-600 mb-6">{error}</p>
          <button
            onClick={fetchMarketData}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  const indices = marketData?.data ? Object.entries(marketData.data).filter(([symbol]) => 
    marketData.symbols.indices[symbol]
  ) : [];

  const crypto = marketData?.data ? Object.entries(marketData.data).filter(([symbol]) => 
    marketData.symbols.crypto[symbol]
  ) : [];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">📊 Current Market</h1>
              <div className="ml-4 flex items-center space-x-2">
                {apiHealth?.status === 'online' ? (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    🟢 Online
                  </span>
                ) : (
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                    🔴 Offline
                  </span>
                )}
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Period selector */}
              <select
                value={selectedPeriod}
                onChange={(e) => setSelectedPeriod(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="1d">1 Day</option>
                <option value="5d">5 Days</option>
                <option value="1mo">1 Month</option>
                <option value="3mo">3 Months</option>
              </select>

              {/* Category selector */}
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Markets</option>
                <option value="indices">US Indices Only</option>
                <option value="crypto">Crypto Only</option>
              </select>

              {/* Refresh button */}
              <button
                onClick={fetchMarketData}
                disabled={loading}
                className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700 disabled:opacity-50 transition-colors"
              >
                {loading ? '⟳' : '🔄'} Refresh
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Last update info */}
        {lastUpdate && (
          <div className="mb-6 text-sm text-gray-500 text-center">
            Last updated: {lastUpdate.toLocaleString()}
            {marketData?.summary && (
              <span className="ml-4">
                📈 {marketData.summary.successful}/{marketData.summary.total_symbols} symbols loaded
              </span>
            )}
          </div>
        )}

        {/* US Indices Section */}
        {(selectedCategory === 'all' || selectedCategory === 'indices') && indices.length > 0 && (
          <section className="mb-12">
            <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center">
              🏛️ US Market Indices
            </h2>
            
            {/* Market cards grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              {indices.map(([symbol, data]) => (
                <MarketCard
                  key={symbol}
                  symbol={symbol}
                  data={data}
                  metadata={marketData.symbols.indices[symbol]}
                />
              ))}
            </div>

            {/* Charts grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {indices.map(([symbol, data]) => (
                <MarketChart
                  key={symbol}
                  symbol={symbol}
                  data={data?.data}
                  title={`${marketData.symbols.indices[symbol]?.name || symbol} (${selectedPeriod.toUpperCase()})`}
                  height={350}
                />
              ))}
            </div>
          </section>
        )}

        {/* Cryptocurrency Section */}
        {(selectedCategory === 'all' || selectedCategory === 'crypto') && crypto.length > 0 && (
          <section>
            <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center">
              🪙 Cryptocurrencies
            </h2>
            
            {/* Market cards grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
              {crypto.map(([symbol, data]) => (
                <MarketCard
                  key={symbol}
                  symbol={symbol}
                  data={data}
                  metadata={marketData.symbols.crypto[symbol]}
                />
              ))}
            </div>

            {/* Charts grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {crypto.map(([symbol, data]) => (
                <MarketChart
                  key={symbol}
                  symbol={symbol}
                  data={data?.data}
                  title={`${marketData.symbols.crypto[symbol]?.name || symbol} (${selectedPeriod.toUpperCase()})`}
                  height={350}
                />
              ))}
            </div>
          </section>
        )}

        {/* No data message */}
        {(!indices.length && !crypto.length) && (
          <div className="text-center py-12">
            <div className="text-gray-400 text-4xl mb-4">📊</div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No market data available</h3>
            <p className="text-gray-500">Try refreshing or check your connection</p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-500">
            <p>Unified Trading Bot - Current Market Dashboard</p>
            <p className="mt-1">Real-time market data provided by Yahoo Finance</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;