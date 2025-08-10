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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0

@dataclass
class PaperBroker:
    equity: float
    positions: Dict[str, Position] = field(default_factory=dict)
    log_path: str = "logs/trades.csv"

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

class Bot:
    def __init__(self, config):
        self.config = config
        self.broker = PaperBroker(
            equity=config["broker"]["starting_equity"],
            log_path=config["general"]["log_file"]
        )
        self.symbols = read_symbol_list(config["universe"]["symbols_csv"])

    def run_once(self):
        logger.info("Bot running - this is a demo version")
        logger.info(f"Portfolio: ${self.broker.equity:,.2f}")

def main():
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("config.yaml not found")
        return

    bot = Bot(config)
    logger.info("=== Trading Bot Started ===")
    
    while True:
        try:
            bot.run_once()
            time.sleep(config["general"]["loop_seconds"])
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()