/**
 * API service for communicating with the FastAPI backend
 */

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout for market data requests
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Get market data for all symbols or specific category
 * @param {string} category - 'all', 'indices', or 'crypto'
 * @param {string} period - Time period (1d, 5d, 1mo, etc.)
 * @param {string} interval - Data interval (1h, 1d, etc.)
 * @returns {Promise} API response with market data
 */
export const getMarketData = async (category = 'all', period = '5d', interval = '1h') => {
  try {
    const response = await api.get('/market-data', {
      params: { category, period, interval }
    });
    return response.data;
  } catch (error) {
    console.error('Error fetching market data:', error);
    throw error;
  }
};

/**
 * Get data for a specific symbol
 * @param {string} symbol - Symbol to fetch (e.g., 'SPY', 'BTC-USD')
 * @param {string} period - Time period
 * @param {string} interval - Data interval
 * @returns {Promise} API response with symbol data
 */
export const getSymbolData = async (symbol, period = '5d', interval = '1h') => {
  try {
    const response = await api.get(`/market-data/${symbol}`, {
      params: { period, interval }
    });
    return response.data;
  } catch (error) {
    console.error(`Error fetching data for ${symbol}:`, error);
    throw error;
  }
};

/**
 * Get list of available symbols
 * @returns {Promise} API response with symbol metadata
 */
export const getSymbols = async () => {
  try {
    const response = await api.get('/symbols');
    return response.data;
  } catch (error) {
    console.error('Error fetching symbols:', error);
    throw error;
  }
};

/**
 * Check API health
 * @returns {Promise} API health status
 */
export const checkApiHealth = async () => {
  try {
    const response = await api.get('/');
    return response.data;
  } catch (error) {
    console.error('Error checking API health:', error);
    throw error;
  }
};

export default api;