/**
 * MarketChart component for displaying interactive line charts
 * Uses Chart.js for high-performance charting
 */

import React, { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

const MarketChart = ({ 
  data, 
  symbol, 
  title, 
  height = 400,
  showVolume = false 
}) => {
  const chartRef = useRef();

  // Prepare chart data
  const chartData = {
    labels: data?.map(point => new Date(point.timestamp)) || [],
    datasets: [
      {
        label: `${symbol} Price`,
        data: data?.map(point => point.close) || [],
        borderColor: data?.length > 0 && data[data.length - 1]?.close > data[0]?.close 
          ? '#10B981' // Green for positive
          : '#EF4444', // Red for negative
        backgroundColor: data?.length > 0 && data[data.length - 1]?.close > data[0]?.close 
          ? 'rgba(16, 185, 129, 0.1)'
          : 'rgba(239, 68, 68, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 6,
      }
    ]
  };

  // Chart options
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
        }
      },
      title: {
        display: true,
        text: title || `${symbol} Price Chart`,
        font: {
          size: 16,
          weight: 'bold'
        },
        padding: 20
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: '#374151',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: false,
        callbacks: {
          title: (context) => {
            const date = new Date(context[0].parsed.x);
            return date.toLocaleString();
          },
          label: (context) => {
            const value = context.parsed.y;
            return `Price: $${value?.toFixed(2) || 'N/A'}`;
          },
          afterBody: (context) => {
            if (data && context[0]) {
              const index = context[0].dataIndex;
              const point = data[index];
              if (point) {
                return [
                  `Open: $${point.open?.toFixed(2) || 'N/A'}`,
                  `High: $${point.high?.toFixed(2) || 'N/A'}`,
                  `Low: $${point.low?.toFixed(2) || 'N/A'}`,
                  `Volume: ${point.volume?.toLocaleString() || 'N/A'}`
                ];
              }
            }
            return [];
          }
        }
      },
    },
    scales: {
      x: {
        type: 'time',
        time: {
          displayFormats: {
            hour: 'MMM dd HH:mm',
            day: 'MMM dd',
          }
        },
        grid: {
          color: 'rgba(156, 163, 175, 0.1)',
        },
        ticks: {
          color: '#6B7280',
          maxTicksLimit: 8,
        }
      },
      y: {
        grid: {
          color: 'rgba(156, 163, 175, 0.1)',
        },
        ticks: {
          color: '#6B7280',
          callback: function(value) {
            return '$' + value.toFixed(2);
          }
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    },
    elements: {
      point: {
        hoverBackgroundColor: '#fff',
        hoverBorderWidth: 2,
      }
    }
  };

  // Handle chart updates
  useEffect(() => {
    const chart = chartRef.current;
    if (chart) {
      chart.update('none'); // Update without animation for better performance
    }
  }, [data]);

  if (!data || data.length === 0) {
    return (
      <div 
        className="flex items-center justify-center bg-white rounded-lg border"
        style={{ height: `${height}px` }}
      >
        <div className="text-center">
          <div className="text-gray-400 text-lg mb-2">📊</div>
          <div className="text-gray-500">No data available for {symbol}</div>
        </div>
      </div>
    );
  }

  return (
    <div 
      className="bg-white rounded-lg border p-4 shadow-sm"
      style={{ height: `${height}px` }}
    >
      <Line ref={chartRef} data={chartData} options={options} />
    </div>
  );
};

export default MarketChart;