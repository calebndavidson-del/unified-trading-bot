#!/usr/bin/env python3
"""
Model Configuration and Hyperparameter Specification
Deep Learning Trading Bot Configuration Management
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class DataConfig:
    """Data fetching and preprocessing configuration"""
    # Market data settings
    symbols: List[str] = field(default_factory=lambda: [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ'
    ])
    crypto_symbols: List[str] = field(default_factory=lambda: [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD'
    ])
    
    # Time series parameters
    lookback_days: int = 252  # 1 year of trading days
    interval: str = '1d'  # 1d, 1h, 15m, 5m
    
    # Feature engineering
    technical_indicators: List[str] = field(default_factory=lambda: [
        'sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stoch', 'williams'
    ])
    candlestick_patterns: List[str] = field(default_factory=lambda: [
        'doji', 'hammer', 'shooting_star', 'engulfing', 'harami'
    ])
    
    # Data splitting
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class ModelConfig:
    """Deep learning model configuration"""
    # Architecture
    model_type: str = 'lstm'  # lstm, gru, transformer, cnn_lstm
    sequence_length: int = 60  # Number of time steps to look back
    
    # LSTM/GRU specific
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    bidirectional: bool = True
    
    # Transformer specific
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 6
    dim_feedforward: int = 1024
    
    # CNN-LSTM specific
    cnn_filters: List[int] = field(default_factory=lambda: [64, 32, 16])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    
    # Output layers
    dense_layers: List[int] = field(default_factory=lambda: [64, 32])
    output_size: int = 1  # Predict next day return
    activation: str = 'relu'
    
    # Regularization
    batch_norm: bool = True
    layer_norm: bool = False
    l1_reg: float = 0.0001
    l2_reg: float = 0.001


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings"""
    # Optimization
    learning_rate: float = 0.001
    optimizer: str = 'adam'  # adam, sgd, rmsprop
    weight_decay: float = 0.0001
    
    # Training schedule
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    
    # Loss function
    loss_function: str = 'mse'  # mse, mae, huber, custom
    loss_weights: Optional[Dict[str, float]] = None
    
    # Validation
    validation_freq: int = 1  # Validate every N epochs
    save_best_model: bool = True
    
    # Advanced training
    gradient_clipping: Optional[float] = 1.0
    mixed_precision: bool = False
    accumulate_grad_batches: int = 1


@dataclass
class RiskConfig:
    """Risk management and trading configuration"""
    # Portfolio management
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% max per position
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    margin_requirement: float = 0.3  # 30% margin requirement for available cash calculations
    
    # Stop loss and take profit
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    
    # Position sizing
    position_sizing_method: str = 'kelly'  # fixed, percent_risk, kelly, volatility
    volatility_lookback: int = 20
    
    # Risk metrics
    max_drawdown_threshold: float = 0.10  # 10% max drawdown
    var_confidence: float = 0.05  # 5% VaR confidence level
    sharpe_min_threshold: float = 1.0


@dataclass
class EarningsConfig:
    """Earnings data and analysis configuration"""
    # Data sources
    earnings_api_key: Optional[str] = None
    earnings_lookback_quarters: int = 8
    
    # Features to extract
    earnings_features: List[str] = field(default_factory=lambda: [
        'eps_surprise', 'revenue_surprise', 'eps_growth', 'revenue_growth',
        'guidance_change', 'analyst_revisions'
    ])
    
    # Event windows
    pre_earnings_days: int = 5
    post_earnings_days: int = 3
    
    # Integration
    earnings_weight: float = 0.2  # Weight in final model


@dataclass
class TradingBotConfig:
    """Complete trading bot configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    earnings: EarningsConfig = field(default_factory=EarningsConfig)
    
    # Global settings
    random_seed: int = 42
    device: str = 'auto'  # auto, cpu, cuda, mps
    
    # Paths
    data_dir: str = 'data'
    model_dir: str = 'models'
    logs_dir: str = 'logs'
    results_dir: str = 'results'
    
    # API keys and credentials
    alpha_vantage_key: Optional[str] = None
    quandl_key: Optional[str] = None


def load_config(config_path: str = 'config.yaml') -> TradingBotConfig:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dict to dataclass
        return TradingBotConfig(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            risk=RiskConfig(**config_dict.get('risk', {})),
            earnings=EarningsConfig(**config_dict.get('earnings', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['data', 'model', 'training', 'risk', 'earnings']}
        )
    else:
        return TradingBotConfig()


def save_config(config: TradingBotConfig, config_path: str = 'config.yaml'):
    """Save configuration to YAML file"""
    config_dict = {
        'data': config.data.__dict__,
        'model': config.model.__dict__,
        'training': config.training.__dict__,
        'risk': config.risk.__dict__,
        'earnings': config.earnings.__dict__,
        'random_seed': config.random_seed,
        'device': config.device,
        'data_dir': config.data_dir,
        'model_dir': config.model_dir,
        'logs_dir': config.logs_dir,
        'results_dir': config.results_dir,
        'alpha_vantage_key': config.alpha_vantage_key,
        'quandl_key': config.quandl_key
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def get_model_params(config: TradingBotConfig) -> Dict[str, Any]:
    """Extract model parameters for training"""
    return {
        'sequence_length': config.model.sequence_length,
        'hidden_size': config.model.hidden_size,
        'num_layers': config.model.num_layers,
        'dropout': config.model.dropout,
        'bidirectional': config.model.bidirectional,
        'batch_size': config.training.batch_size,
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.weight_decay
    }


def get_feature_config(config: TradingBotConfig) -> Dict[str, Any]:
    """Extract feature engineering configuration"""
    return {
        'technical_indicators': config.data.technical_indicators,
        'candlestick_patterns': config.data.candlestick_patterns,
        'earnings_features': config.earnings.earnings_features,
        'sequence_length': config.model.sequence_length,
        'lookback_days': config.data.lookback_days
    }


# Default configuration instance
DEFAULT_CONFIG = TradingBotConfig()

if __name__ == "__main__":
    # Example usage
    config = TradingBotConfig()
    save_config(config, 'default_config.yaml')
    print("Default configuration saved to 'default_config.yaml'")
    
    # Load and print configuration
    loaded_config = load_config('default_config.yaml')
    print(f"Loaded configuration with {len(loaded_config.data.symbols)} symbols")