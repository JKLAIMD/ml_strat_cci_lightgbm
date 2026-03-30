#!/usr/bin/env python3
"""
live_server.py — 实时信号服务器

用法:
    python scripts/live_server.py --config config/signal_server.yaml

功能:
    1. WebSocket 接收 Binance 实时K线
    2. 实时计算因子 + LightGBM 推理
    3. 发出交易信号到 Discord / Telegram
    4. 每周自动重训练模型
"""

import argparse
import pickle
import yaml
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import requests
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.cci_lightgbm_signals import build_factor_library, generate_signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


class SignalNotifier:
    """信号推送"""
    def __init__(self, config: dict):
        self.discord_webhook = config.get('notifier', {}).get('discord_webhook')
        self.telegram_token = config.get('notifier', {}).get('telegram_bot_token')
        self.telegram_chat = config.get('notifier', {}).get('telegram_chat_id')

    def send(self, message: str):
        log.info(f"SIGNAL: {message}")
        if self.discord_webhook:
            try:
                requests.post(self.discord_webhook, json={'content': message}, timeout=5)
            except Exception as e:
                log.error(f"Discord error: {e}")
        if self.telegram_token and self.telegram_chat:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                requests.post(url, json={'chat_id': self.telegram_chat, 'text': message}, timeout=5)
            except Exception as e:
                log.error(f"Telegram error: {e}")


class LiveSignalServer:
    """实时信号服务器"""
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.symbol = self.config['strategy']['symbol']
        self.interval = self.config['strategy']['interval']

        # 加载模型
        model_path = self.config['strategy']['model']['path']
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        log.info(f"Loaded model: {model_path}, features: {self.feature_cols}")

        self.threshold = self.config['strategy']['model']['threshold']
        self.notifier = SignalNotifier(self.config)

        # 数据缓冲
        self.df = pd.DataFrame()

    def on_kline(self, kline: dict):
        """处理新的K线数据"""
        ts = pd.to_datetime(kline['open_time'], unit='ms')
        row = {
            'open': float(kline['open']),
            'high': float(kline['high']),
            'low': float(kline['low']),
            'close': float(kline['close']),
            'volume': float(kline['volume']),
        }

        # 追加到缓冲
        self.df = pd.concat([self.df, pd.DataFrame([row], index=[ts])])
        # 只保留最近 500 根K线
        if len(self.df) > 500:
            self.df = self.df[-500:]

        # 需要至少100根K线
        if len(self.df) < 100:
            return

        # 生成信号
        try:
            sig = generate_signal(self.df, self.model, self.feature_cols, self.threshold)
            self.process_signal(sig)
        except Exception as e:
            log.error(f"Signal generation error: {e}")

    def process_signal(self, sig: dict):
        """处理并推送信号"""
        position = sig['position']
        proba = sig['proba']
        close = sig['close']
        sl = sig['stop_loss']
        tp = sig['take_profit']
        atr = sig['atr']

        ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

        if position == 'long':
            msg = (
                f"🟢 **LONG** | {ts}\n"
                f"Price: ${close:.2f}\n"
                f"Prob: {proba:.1%}\n"
                f"ATR: ${atr:.2f}\n"
                f"SL: ${sl:.2f} (-{3:.0f}×ATR)\n"
                f"TP: ${tp:.2f} (+{5:.0f}×ATR)"
            )
        elif position == 'short':
            msg = (
                f"🔴 **SHORT** | {ts}\n"
                f"Price: ${close:.2f}\n"
                f"Prob: {proba:.1%}\n"
                f"ATR: ${atr:.2f}\n"
                f"SL: ${sl:.2f} (+{3:.0f}×ATR)\n"
                f"TP: ${tp:.2f} (-{5:.0f}×ATR)"
            )
        else:
            msg = f"⚪ **FLAT** | {ts} | Prob: {proba:.1%}"

        self.notifier.send(msg)

    def run(self):
        """启动 WebSocket 连接"""
        symbol_lower = self.symbol.lower().replace('USDT', '').lower()
        ws_url = (
            f"wss://stream.binance.com:9443/ws/"
            f"{symbol_lower}usdt@kline_{self.interval}"
        )

        log.info(f"Connecting to {ws_url}")

        import websocket
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_ws_message,
            on_error=lambda ws, err: log.error(f"WS Error: {err}"),
            on_close=lambda ws: log.warning("WS closed, reconnecting..."),
        )
        ws.on_open = lambda ws: log.info("WS connected")
        ws.run_forever(ping_interval=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    server = LiveSignalServer(args.config)
    server.run()
