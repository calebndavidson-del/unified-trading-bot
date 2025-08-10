#!/usr/bin/env python3
import os
import sys
import time
import yaml
import signal
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from typing import Dict, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0

@dataclass
class TradingParameters:
    """Dynamic trading parameters that can be updated"""
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    bb_period: int = 20
    bb_std: float = 2.0
    position_size: float = 0.1
    stop_loss: float = 0.02
    take_profit: float = 0.04
    starting_capital: float = 100000

@dataclass
class PaperBroker:
    equity: float
    positions: Dict[str, Position] = field(default_factory=dict)
    log_path: str = "logs/trades.csv"
    params_path: str = "logs/current_params.json"

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=["ts","symbol","side","qty","price","net","equity_after"]).to_csv(self.log_path, index=False)

    def market_order(self, symbol: str, side: str, qty: float, price: float):
        if qty == 0:
            return
        
        cost = price * qty
        if side.lower() == "buy":
            self.equity -= cost
            pos = self.positions.get(symbol, Position(symbol, 0, 0))
            new_qty = pos.qty + qty
            new_avg = (pos.avg_price * pos.qty + price * qty) / new_qty if new_qty > 0 else 0
            self.positions[symbol] = Position(symbol, new_qty, new_avg)
        else:
            self.equity += cost
            if symbol in self.positions:
                self.positions[symbol].qty -= qty
                if self.positions[symbol].qty <= 0:
                    del self.positions[symbol]

        rec = {
            "ts": pd.Timestamp.now().isoformat(),
            "symbol": symbol, "side": side, "qty": qty, "price": price,
            "net": cost if side == "sell" else -cost, "equity_after": self.equity
        }
        pd.DataFrame([rec]).to_csv(self.log_path, mode="a", header=False, index=False)
        logger.info(f"Trade: {side} {qty:.4f} {symbol} @ ${price:.2f}")

    def save_current_params(self, params: TradingParameters):
        """Save current trading parameters"""
        params_dict = {
            'rsi_period': params.rsi_period,
            'rsi_oversold': params.rsi_oversold,
            'rsi_overbought': params.rsi_overbought,
            'bb_period': params.bb_period,
            'bb_std': params.bb_std,
            'position_size': params.position_size,
            'stop_loss': params.stop_loss,
            'take_profit': params.take_profit,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        with open(self.params_path, 'w') as f:
            json.dump(params_dict, f, indent=2)

def read_symbol_list(csv_path: str) -> List[str]:
    symbols = []
    try:
        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    symbols.append(line)
    except FileNotFoundError:
        logger.error(f"Symbol file {csv_path} not found")
    return symbols

def calculate_technical_indicators(df, params: TradingParameters):
    """Calculate technical indicators with dynamic parameters"""
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=params.rsi_period).mean()
    avg_loss = loss.rolling(window=params.rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_ma = df['Close'].rolling(window=params.bb_period).mean()
    bb_upper = bb_ma + (df['Close'].rolling(window=params.bb_period).std() * params.bb_std)
    bb_lower = bb_ma - (df['Close'].rolling(window=params.bb_period).std() * params.bb_std)
    
    # Moving averages
    ma_short = df['Close'].rolling(window=10).mean()
    ma_long = df['Close'].rolling(window=20).mean()
    
    return {
        'rsi': rsi,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_ma': bb_ma,
        'ma_short': ma_short,
        'ma_long': ma_long
    }

class EnhancedBot:
    def __init__(self, config):
        self.config = config
        self.broker = PaperBroker(
            equity=config["broker"]["starting_equity"],
            log_path=config["general"]["log_file"]
        )
        self.symbols = read_symbol_list(config["universe"]["symbols_csv"])
        
        # Initialize with default parameters
        self.params = TradingParameters()
        self.load_current_params()

    def load_current_params(self):
        """Load current trading parameters from file"""
        try:
            if os.path.exists(self.broker.params_path):
                with open(self.broker.params_path, 'r') as f:
                    params_dict = json.load(f)
                    self.params.rsi_period = params_dict.get('rsi_period', 14)
                    self.params.rsi_oversold = params_dict.get('rsi_oversold', 30)
                    self.params.rsi_overbought = params_dict.get('rsi_overbought', 70)
                    self.params.bb_period = params_dict.get('bb_period', 20)
                    self.params.bb_std = params_dict.get('bb_std', 2.0)
                    self.params.position_size = params_dict.get('position_size', 0.1)
                    self.params.stop_loss = params_dict.get('stop_loss', 0.02)
                    self.params.take_profit = params_dict.get('take_profit', 0.04)
                    logger.info("Loaded updated trading parameters")
        except Exception as e:
            logger.warning(f"Could not load parameters: {e}, using defaults")

    def run_once(self):
        """Enhanced trading logic with dynamic parameters"""
        logger.info(f"Bot running with dynamic parameters - Portfolio: ${self.broker.equity:,.2f}")
        
        # Save current parameters
        self.broker.save_current_params(self.params)
        
        # Reload parameters in case they were updated
        self.load_current_params()
        
        # Demo trading logic with current parameters
        logger.info(f"Current RSI period: {self.params.rsi_period}")
        logger.info(f"Current position size: {self.params.position_size:.1%}")
        logger.info(f"Current stop loss: {self.params.stop_loss:.1%}")

def main():
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("config.yaml not found")
        return

    bot = EnhancedBot(config)
    logger.info("=== Enhanced Trading Bot Started ===")
    
    while True:
        try:
            bot.run_once()
            time.sleep(config["general"]["loop_seconds"])
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
