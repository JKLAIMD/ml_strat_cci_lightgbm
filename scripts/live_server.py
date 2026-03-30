#!/usr/bin/env python3
"""
signal_server_v3.py — CCI LightGBM 实时信号服务器
Flask API (port 6660) + 轮询推送
导入 c3_ml_live 做信号计算，自身只负责推送和 API。

结构完全对齐 signal_server_v2.py：
- Flask API on port 6660
- polling 轮询 Binance，60s 一次
- Discord webhook 推送
- LAST_STATE 状态管理
- check_all() 主逻辑

用法:
    python scripts/live_server.py
    curl http://localhost:6660/signals
"""

import datetime
import logging
import os
import sys
import threading
import time
import pickle
import pandas as pd
import numpy as np
import ccxt
import requests
from flask import Flask, jsonify
from flask_cors import CORS

# ─────────────────────────────────────────
# 项目路径
# ─────────────────────────────────────────
_share_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # → ml_strat_cci_lightgbm
_parent     = os.path.dirname(_share_root)  # → scripts (parent of repo root)
# 让 c3_ml_live 可导入（它在 ~/Desktop/share/ 下）
DESKTOP_SHARE = os.path.expanduser("~/Desktop/share")
sys.path.insert(0, DESKTOP_SHARE)

from c3_ml_live import get_c3_ml_signal

# ─────────────────────────────────────────
# 日志
# ─────────────────────────────────────────
LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
LOG_DIR  = os.path.normpath(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'c3_ml_signals.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# Flask
# ─────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
# 配置
# ─────────────────────────────────────────
WEBHOOK = os.environ.get(
    "DISCORD_C3_WEBHOOK",
    "https://discordapp.com/api/webhooks/1481886350973210780/CfAlW9gzDtRcW_ZaNH8Nz1SgifWXmVicDAgfODxvEWVhqJdpqbQhvu7N35wEpcwnkA3U"
)

# SYMBOLS 配置（与 signal_server_v2.py 结构一致）
SYMBOLS = [
    {
        'symbol': 'BTCUSDT',
        'strategy': 'c3_ml',
        'version': 'C3',
        'label': 'BTC C3_ML LightGBM (CCI)',
        # 模型配置
        'model_path': os.path.join(DESKTOP_SHARE, 'share_repository', 'data', 'c3_lgbm.pkl'),
        'feat_path':  os.path.join(DESKTOP_SHARE, 'share_repository', 'data', 'c3_lgbm_features.json'),
        'cache_path': os.path.join(DESKTOP_SHARE, 'share_repository', 'data', 'BTC_USDT_1h.pkl'),
        # 风控参数
        'atr_period': 14,
        'sl_atr_mult': 3.0,
        'tp_atr_mult': 5.0,
        # 信号阈值
        'threshold': 0.50,
        'min_confidence': 0.55,
    },
]

# 扩展 SYMBOLS 未来可加更多币种
# {'symbol': 'ETHUSDT', 'strategy': 'c3_ml', ...}

# LAST_STATE[key] = {'signal': int, 'action': str, 'kline_time': int, 'price': float}
LAST_STATE: dict = {}
_state_lock = threading.Lock()
_is_first_run = True

# ─────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────

def _beijing_now() -> str:
    return datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=8)
    ).strftime('%H:%M:%S')


def _action_cn(action: str) -> str:
    return {
        'open_long':  '开多',
        'open_short': '开空',
        'none':       '空仓',
    }.get(action, action)


def send_discord(msg: str) -> None:
    try:
        requests.post(WEBHOOK, json={'content': msg}, timeout=10)
    except Exception as e:
        logger.error(f'Discord 推送失败: {e}')


# ─────────────────────────────────────────
# 推送逻辑
# ─────────────────────────────────────────

def _push_appear(cfg: dict, action: str, price: float,
                  proba: float = 0, confidence: float = 0,
                  sl: float = 0, tp: float = 0,
                  atr: float = 0) -> None:
    direction = _action_cn(action)
    emoji = '🟢' if action == 'open_long' else '🔴'
    version_tag = f"[{cfg.get('version', 'C3')}]"

    msg = (
        f"{emoji} {version_tag}信号出现: {direction} | {cfg['label']}\n"
        f"  价格: ${price:.2f} | 概率: {proba:.1%} | 置信度: {confidence:.1%}\n"
        f"  ATR: ${atr:.2f} | SL: ${sl:.2f} ({cfg.get('sl_atr_mult', 3):.0f}×ATR)"
        f" | TP: ${tp:.2f} ({cfg.get('tp_atr_mult', 5):.0f}×ATR)\n"
        f"  北京时间: {_beijing_now()}"
    )
    logger.info(f'[SIGNAL] {msg}')
    send_discord(msg)


def _push_disappear(cfg: dict, prev_action: str, price: float) -> None:
    direction = _action_cn(prev_action)
    version_tag = f"[{cfg.get('version', 'C3')}]"
    msg = (
        f"⚪ {version_tag}信号消失 | {cfg['label']}\n"
        f"  原信号: {direction} | 价格: ${price:.2f} | 北京时间: {_beijing_now()}"
    )
    logger.info(f'[SIGNAL DISAPPER] {msg}')
    send_discord(msg)


# ─────────────────────────────────────────
# 主检查逻辑
# ─────────────────────────────────────────

def check_all(is_init: bool = False) -> None:
    """
    is_init=True 时为启动初始化：填充 LAST_STATE，并推送一次当前方向。
    """
    for cfg in SYMBOLS:
        key = f"{cfg['symbol']}_{cfg['strategy']}"
        try:
            ml_result = get_c3_ml_signal()
            curr_sig   = ml_result['signal']       # 1=LONG, -1=SHORT, 0=FLAT
            curr_price = ml_result['price']
            proba_up   = ml_result.get('proba_up', 0.5)
            confidence  = ml_result.get('confidence', 0)
            atr        = ml_result.get('atr', 0)
            sl = curr_price - cfg['sl_atr_mult'] * atr
            tp = curr_price + cfg['tp_atr_mult'] * atr

            curr_ktime = int(datetime.datetime.now(
                datetime.timezone(datetime.timedelta(hours=8))
            ).timestamp())
            curr_act = ('open_long' if curr_sig == 1
                        else 'open_short' if curr_sig == -1
                        else 'none')

            if is_init:
                init_state = {
                    'signal': curr_sig,
                    'action': curr_act,
                    'kline_time': curr_ktime,
                    'price': curr_price,
                    'proba_up': proba_up,
                    'confidence': confidence,
                    'atr': atr,
                    'sl': sl,
                    'tp': tp,
                }
                with _state_lock:
                    LAST_STATE[key] = init_state
                if curr_sig != 0:
                    _push_appear(cfg, curr_act, curr_price, proba_up, confidence, sl, tp, atr)
                continue

            with _state_lock:
                prev = LAST_STATE.get(key, {
                    'signal': 0, 'action': 'none',
                    'kline_time': None, 'price': 0.0
                })
                prev_sig = prev.get('signal', 0)

            push_fn   = None
            new_state = None

            # 场景1：信号出现（0 → 非0）
            if prev_sig == 0 and curr_sig != 0:
                push_fn = lambda: _push_appear(cfg, curr_act, curr_price, proba_up, confidence, sl, tp, atr)
                new_state = {
                    'signal': curr_sig, 'action': curr_act,
                    'kline_time': curr_ktime, 'price': curr_price,
                    'proba_up': proba_up, 'confidence': confidence,
                    'atr': atr, 'sl': sl, 'tp': tp,
                }

            # 场景2：方向反转（非0 → 非0，方向变）
            elif prev_sig != 0 and curr_sig != 0 and prev_sig != curr_sig:
                push_fn = lambda: _push_appear(cfg, curr_act, curr_price, proba_up, confidence, sl, tp, atr)
                new_state = {
                    'signal': curr_sig, 'action': curr_act,
                    'kline_time': curr_ktime, 'price': curr_price,
                    'proba_up': proba_up, 'confidence': confidence,
                    'atr': atr, 'sl': sl, 'tp': tp,
                }

            # 场景3：信号消失（非0 → 0）
            elif prev_sig != 0 and curr_sig == 0:
                push_fn = lambda: _push_disappear(cfg, prev.get('action', 'none'), curr_price)
                new_state = {
                    'signal': 0, 'action': 'none',
                    'kline_time': curr_ktime, 'price': curr_price,
                }

            # 场景4：信号不变，仅更新价格/时间
            elif prev_sig != 0 and curr_sig != 0 and prev_sig == curr_sig:
                new_state = {
                    'signal': curr_sig, 'action': curr_act,
                    'kline_time': curr_ktime, 'price': curr_price,
                    'proba_up': proba_up, 'confidence': confidence,
                    'atr': atr, 'sl': sl, 'tp': tp,
                }

            # 场景5：空仓状态，仅更新时间
            elif prev_sig == 0 and curr_sig == 0:
                new_state = {
                    'signal': 0, 'action': 'none',
                    'kline_time': curr_ktime, 'price': curr_price,
                }

            if new_state is not None:
                with _state_lock:
                    LAST_STATE[key] = new_state
            if push_fn is not None:
                push_fn()

        except Exception as e:
            logger.error(f'C3 ML error [{cfg.get("label","?")}]: {e}')
            import traceback
            traceback.print_exc()


# ─────────────────────────────────────────
# Flask API
# ─────────────────────────────────────────

@app.route('/signals')
def signals():
    """返回所有币种的当前信号状态"""
    with _state_lock:
        return jsonify({
            'ok': True,
            'timestamp': datetime.datetime.now(
                datetime.timezone(datetime.timedelta(hours=8))
            .isoformat(),
            'signals': dict(LAST_STATE),
        })


@app.route('/health')
def health():
    return jsonify({'ok': True, 'service': 'c3_ml_signal_server'})


# ─────────────────────────────────────────
# 轮询线程
# ─────────────────────────────────────────

def polling_loop():
    global _is_first_run
    while True:
        check_all(is_init=_is_first_run)
        if _is_first_run:
            _is_first_run = False
        time.sleep(60)  # 60s 轮询一次


if __name__ == '__main__':
    # 启动轮询线程
    t = threading.Thread(target=polling_loop, daemon=True)
    t.start()

    # 启动 Flask（默认 port 6660）
    logger.info('C3 ML Signal Server 启动 on :6660')
    app.run(host='0.0.0.0', port=6660, debug=False, threaded=True)
