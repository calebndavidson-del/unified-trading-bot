/**
 * MarketCard component for displaying individual market summary
 */

import React from 'react';

const MarketCard = ({ symbol, data, metadata }) => {
  if (!data || data.error) {
    return (
      <div className="bg-white rounded-lg border p-4 shadow-sm">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold text-gray-900">{symbol}</h3>
          <span className="text-sm text-gray-500">Error</span>
        </div>
        <div className="text-sm text-red-500">
          {data?.error || 'Failed to load data'}
        </div>
      </div>
    );
  }

  const {
    latest_price,
    price_change,
    price_change_pct,
    is_positive
  } = data;

  const changeColor = is_positive ? 'text-green-600' : 'text-red-600';
  const changeIcon = is_positive ? '↗' : '↘';
  const changeBg = is_positive ? 'bg-green-100' : 'bg-red-100';

  return (
    <div className="bg-white rounded-lg border p-4 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{symbol}</h3>
          <p className="text-sm text-gray-500">{metadata?.name || symbol}</p>
        </div>
        <div className={`p-2 rounded-full ${changeBg}`}>
          <span className={`text-lg ${changeColor}`}>{changeIcon}</span>
        </div>
      </div>
      
      <div className="space-y-2">
        <div className="flex items-baseline justify-between">
          <span className="text-2xl font-bold text-gray-900">
            ${latest_price?.toFixed(2) || 'N/A'}
          </span>
          <div className={`text-right ${changeColor}`}>
            <div className="text-sm font-medium">
              {price_change !== undefined ? 
                `${is_positive ? '+' : ''}$${price_change.toFixed(2)}` : 
                'N/A'
              }
            </div>
            <div className="text-xs">
              {price_change_pct !== undefined ? 
                `${is_positive ? '+' : ''}${price_change_pct.toFixed(2)}%` : 
                'N/A'
              }
            </div>
          </div>
        </div>
        
        {data.data && data.data.length > 0 && (
          <div className="text-xs text-gray-500 pt-2 border-t">
            Last updated: {new Date(data.timestamp).toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
};

export default MarketCard;