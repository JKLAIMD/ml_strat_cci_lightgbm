#!/usr/bin/env python3
"""
ML CCI LightGBM 实盘交易进程（自包含，不 import first_trading_server）。

信号：scripts/c3_ml_signal_asymmetric.get_ml_cci_bar
交易所：Binance USDT 永续，ccxt，单向持仓，市价单。

环境变量（节选）：
  BINANCE_API_KEY / BINANCE_SECRET
  ENABLE_TRADING=0|1
  ML_LONG_THRESHOLD / ML_SHORT_THRESHOLD
  ORDER_MARGIN_USD, FIXED_LEVERAGE, MARGIN_MODE
  ML_TRADING_PORT（默认 6671）
  TRADING_DISCORD_WEBHOOK（可选）
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import requests
from flask import Flask, jsonify

# ── path：本仓库 scripts + 根目录
_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS = pathlib.Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))

from c3_ml_signal_asymmetric import get_ml_cci_bar  # noqa: E402


def _load_env_file() -> None:
    env_path = _ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and not os.getenv(key):
            os.environ[key] = value


_load_env_file()

BJ_TZ = timezone(timedelta(hours=8))


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def now_ts_ms() -> int:
    return int(time.time() * 1000)


# ── Logging
LOG_DIR = _ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "ml_cci_trading_server.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

WEBHOOK_URL = (os.getenv("TRADING_DISCORD_WEBHOOK", "") or "").strip()

ORDER_MARGIN_USD = float(os.getenv("ORDER_MARGIN_USD", "60"))
FIXED_LEVERAGE = int(os.getenv("FIXED_LEVERAGE", "50"))
ORDER_NOTIONAL_USD = float(os.getenv("ORDER_NOTIONAL_USD", str(ORDER_MARGIN_USD * FIXED_LEVERAGE)))
MIN_OPEN_NOTIONAL_USD = float(os.getenv("MIN_OPEN_NOTIONAL_USD", "5.0"))
MARGIN_PCT = float(os.getenv("MARGIN_PCT", "0.30"))
MARGIN_MODE = os.getenv("MARGIN_MODE", "isolated").strip().lower()
ENABLE_TRADING = env_bool("ENABLE_TRADING", False)
RECONCILE_ON_STARTUP = env_bool("RECONCILE_ON_STARTUP", True)
AUTO_SET_ONE_WAY_MODE = env_bool("AUTO_SET_ONE_WAY_MODE", True)
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "120"))
TAKER_FEE_RATE = float(os.getenv("TAKER_FEE_RATE", "0.0005"))
POSITION_CONFIRM_RETRIES = int(os.getenv("POSITION_CONFIRM_RETRIES", "20"))
POSITION_CONFIRM_SLEEP = float(os.getenv("POSITION_CONFIRM_SLEEP", "0.35"))
PROXY_URL = (os.getenv("PROXY_URL", "") or "").strip()
ML_TRADING_PORT = int(os.getenv("ML_TRADING_PORT", "6671"))

PROXY_KWARGS: Dict[str, Any] = {}
if PROXY_URL:
    PROXY_KWARGS = {
        "proxies": {"http": PROXY_URL, "https": PROXY_URL},
        "aiohttp_proxy": PROXY_URL,
    }

app = Flask(__name__)

# ── 单任务
ML_JOB: Dict[str, Any] = {
    "id": "btc_ml_cci",
    "label": "BTC ML CCI LightGBM (asymmetric)",
    "trade_symbol": os.getenv("ML_TRADE_SYMBOL", "BTC/USDT:USDT"),
    "margin_pct": float(os.getenv("ML_MARGIN_PCT", "0.23")),
    "order_margin_usd": float(os.getenv("ORDER_MARGIN_USD", str(ORDER_MARGIN_USD))),
    "order_notional_usd": float(os.getenv("ORDER_NOTIONAL_USD", str(ORDER_NOTIONAL_USD))),
    "leverage": FIXED_LEVERAGE,
    "check_seconds": float(os.getenv("ML_CHECK_SECONDS", "120")),
    "enabled": True,
}
ML_JOB["order_notional_usd"] = ML_JOB["order_margin_usd"] * FIXED_LEVERAGE

STATE_LOCK = threading.Lock()
JOB_STATE: Dict[str, Dict[str, Any]] = {}

_trade_exchange: Optional[ccxt.Exchange] = None
_trade_exchange_lock = threading.Lock()


def _bj_now_str() -> str:
    return datetime.now(BJ_TZ).strftime("%H:%M:%S")


def send_discord(msg: str) -> None:
    if not WEBHOOK_URL:
        return
    try:
        requests.post(WEBHOOK_URL, json={"content": msg}, timeout=10)
    except Exception as exc:
        logger.warning("Discord 推送失败: %s", exc)


def get_trade_exchange() -> ccxt.Exchange:
    global _trade_exchange
    with _trade_exchange_lock:
        if _trade_exchange is None:
            cfg: Dict[str, Any] = {
                "apiKey": os.getenv("BINANCE_API_KEY", "").strip(),
                "secret": os.getenv("BINANCE_SECRET", "").strip(),
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {"defaultType": "swap", "defaultSubType": "linear"},
            }
            cfg.update(PROXY_KWARGS)
            _trade_exchange = ccxt.binance(cfg)
            _trade_exchange.load_markets()
        return _trade_exchange


def has_trade_credentials() -> bool:
    return bool(os.getenv("BINANCE_API_KEY", "").strip() and os.getenv("BINANCE_SECRET", "").strip())


def fetch_wallet_balance() -> float:
    ex = get_trade_exchange()
    bal = ex.fetch_balance()
    info = bal.get("info", {})
    total = to_float(info.get("totalWalletBalance"), 0.0)
    if total <= 0:
        for asset in info.get("assets", []):
            if asset.get("asset") == "USDT":
                total = to_float(asset.get("walletBalance"), 0.0)
                break
    return total


def compute_dynamic_margin(margin_pct: float) -> Tuple[float, float]:
    try:
        balance = fetch_wallet_balance()
        if balance > 0:
            margin = balance * margin_pct
            min_margin = MIN_OPEN_NOTIONAL_USD / FIXED_LEVERAGE
            margin = max(margin, min_margin)
            notional = margin * FIXED_LEVERAGE
            return margin, notional
    except Exception as exc:
        logger.warning("动态保证金失败: %s", exc)
    return ORDER_MARGIN_USD, ORDER_NOTIONAL_USD


def fetch_position_mode_safe(exchange: ccxt.Exchange, symbol: Optional[str] = None) -> Dict[str, Any]:
    if hasattr(exchange, "fetch_position_mode"):
        return exchange.fetch_position_mode(symbol)
    getter = getattr(exchange, "fapiPrivateGetPositionSideDual", None)
    if callable(getter):
        raw = getter({"timestamp": now_ts_ms()})
        dual = raw.get("dualSidePosition")
        hedged = bool(dual) if dual is not None else False
        return {"hedged": hedged}
    return {"hedged": False}


def set_one_way_mode_safe(exchange: ccxt.Exchange, symbol: Optional[str] = None) -> Dict[str, Any]:
    setter = getattr(exchange, "set_position_mode", None)
    if callable(setter):
        return setter(False, symbol)
    raise RuntimeError("ccxt does not support set_position_mode for Binance futures")


def ensure_one_way_mode(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    mode = fetch_position_mode_safe(exchange, symbol)
    if not mode.get("hedged", False):
        return mode
    if not AUTO_SET_ONE_WAY_MODE:
        raise RuntimeError("account is in hedge mode; switch to one-way first")
    logger.warning("检测到对冲模式, 切换单向…")
    set_one_way_mode_safe(exchange, symbol)
    mode = fetch_position_mode_safe(exchange, symbol)
    if mode.get("hedged", False):
        raise RuntimeError("still hedge mode after switch")
    logger.info("已确认单向模式")
    return mode


def amount_norm_str(exchange: ccxt.Exchange, symbol: str, amount: float) -> str:
    return exchange.amount_to_precision(symbol, max(to_float(amount), 0.0))


def _market_for(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    return exchange.market(symbol)


def _extract_min_amount(market: Dict[str, Any]) -> float:
    limits = market.get("limits") or {}
    return to_float((limits.get("amount") or {}).get("min"), 0.0)


def _ticker_price_candidates(ticker: Dict[str, Any], signal: int) -> List[float]:
    info = ticker.get("info") or {}
    ask = to_float(ticker.get("ask"))
    bid = to_float(ticker.get("bid"))
    last = to_float(ticker.get("last"))
    close = to_float(ticker.get("close"))
    mark = to_float(info.get("markPrice"))
    if signal > 0:
        return [ask, mark, last, close, bid]
    if signal < 0:
        return [bid, mark, last, close, ask]
    return [mark, last, close, bid, ask]


def fetch_sizing_price(exchange: ccxt.Exchange, symbol: str, signal: int) -> Tuple[float, Dict[str, Any]]:
    ticker = exchange.fetch_ticker(symbol)
    for px in _ticker_price_candidates(ticker, signal):
        if px > 0:
            return px, ticker
    raise RuntimeError(f"unable to determine sizing price for {symbol}")


def build_open_amount(
    exchange: ccxt.Exchange, symbol: str, signal: int, order_notional_usd: float,
) -> Dict[str, Any]:
    market = _market_for(exchange, symbol)
    min_amount = _extract_min_amount(market)
    target_usd = max(to_float(order_notional_usd), MIN_OPEN_NOTIONAL_USD)
    price, ticker = fetch_sizing_price(exchange, symbol, signal)
    raw_amount = target_usd / price
    if min_amount > 0 and raw_amount < min_amount:
        raw_amount = min_amount
    amount_str = exchange.amount_to_precision(symbol, raw_amount)
    amount = to_float(amount_str, 0.0)
    if amount <= 0:
        raise RuntimeError(f"amount rounded to zero for {symbol}")
    est_notional = amount * price
    if est_notional < MIN_OPEN_NOTIONAL_USD:
        bumped_raw = (MIN_OPEN_NOTIONAL_USD * 1.02) / price
        if min_amount > 0 and bumped_raw < min_amount:
            bumped_raw = min_amount
        amount_str = exchange.amount_to_precision(symbol, bumped_raw)
        amount = to_float(amount_str, 0.0)
        est_notional = amount * price
    if amount <= 0 or est_notional < MIN_OPEN_NOTIONAL_USD:
        raise RuntimeError(f"invalid open amount for {symbol}")
    return {
        "price": price,
        "ticker": ticker,
        "target_usd": target_usd,
        "amount": amount,
        "amount_str": amount_str,
        "est_notional": est_notional,
    }


def safe_set_margin_mode(exchange: ccxt.Exchange, symbol: str, margin_mode: str) -> None:
    try:
        exchange.set_margin_mode(margin_mode, symbol)
    except Exception as exc:
        msg = str(exc)
        if "-4046" in msg or "No need to change margin type" in msg:
            return
        raise


def safe_set_leverage(exchange: ccxt.Exchange, symbol: str, leverage: int) -> None:
    try:
        exchange.set_leverage(leverage, symbol)
    except Exception as exc:
        msg = str(exc)
        if "-4028" in msg or "already" in msg.lower():
            return
        raise


def cancel_symbol_open_orders(exchange: ccxt.Exchange, symbol: str) -> Any:
    try:
        return exchange.cancel_all_orders(symbol)
    except Exception as exc:
        msg = str(exc)
        if "No orders" in msg or "-2011" in msg:
            return []
        raise


def fetch_position(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    positions = exchange.fetch_positions([symbol])
    for pos in positions:
        if pos.get("symbol") != symbol:
            continue
        info = pos.get("info") or {}
        pos_amt = to_float(info.get("positionAmt"), 0.0)
        contracts = to_float(pos.get("contracts"), 0.0)
        if contracts <= 0 and abs(pos_amt) > 0:
            contracts = abs(pos_amt)
        if contracts <= 0:
            continue
        side = (pos.get("side") or "").lower()
        if side not in {"long", "short"}:
            if pos_amt > 0:
                side = "long"
            elif pos_amt < 0:
                side = "short"
        if side not in {"long", "short"}:
            continue
        return {
            "exists": True,
            "symbol": symbol,
            "side": side,
            "contracts": contracts,
            "base_amount": contracts,
            "raw": pos,
        }
    return {
        "exists": False,
        "symbol": symbol,
        "side": "flat",
        "contracts": 0.0,
        "base_amount": 0.0,
        "raw": None,
    }


def position_matches_target(
    exchange: ccxt.Exchange,
    symbol: str,
    pos: Dict[str, Any],
    signal: int,
    expected_amount: Optional[float] = None,
) -> bool:
    if signal == 0:
        return not pos["exists"]
    if not pos["exists"]:
        return False
    want_side = "long" if signal > 0 else "short"
    return pos["side"] == want_side


def wait_until_flat(exchange: ccxt.Exchange, symbol: str) -> Dict[str, Any]:
    last = None
    for _ in range(POSITION_CONFIRM_RETRIES):
        time.sleep(POSITION_CONFIRM_SLEEP)
        last = fetch_position(exchange, symbol)
        if not last["exists"]:
            return last
    raise RuntimeError(f"{symbol} not flat after close")


def wait_until_target(
    exchange: ccxt.Exchange,
    symbol: str,
    signal: int,
    expected_amount: Optional[float],
) -> Dict[str, Any]:
    last = None
    for _ in range(POSITION_CONFIRM_RETRIES):
        time.sleep(POSITION_CONFIRM_SLEEP)
        last = fetch_position(exchange, symbol)
        if position_matches_target(exchange, symbol, last, signal, expected_amount):
            return last
    raise RuntimeError(f"{symbol} not at target after open")


def close_position(exchange: ccxt.Exchange, symbol: str, pos: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not pos["exists"] or pos["contracts"] <= 0:
        return None
    side = "sell" if pos["side"] == "long" else "buy"
    amount_str = exchange.amount_to_precision(symbol, pos["contracts"])
    amount = to_float(amount_str, 0.0)
    if amount <= 0:
        raise RuntimeError("close amount zero")
    logger.info("平仓 %s 数量=%s", symbol, amount_str)
    return exchange.create_order(
        symbol,
        "market",
        side,
        amount,
        None,
        {"reduceOnly": True, "positionSide": "BOTH", "newOrderRespType": "RESULT"},
    )


def open_position(
    exchange: ccxt.Exchange,
    symbol: str,
    signal: int,
    order_notional_usd: float,
    leverage: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    safe_set_margin_mode(exchange, symbol, MARGIN_MODE)
    safe_set_leverage(exchange, symbol, leverage)
    sizing = build_open_amount(exchange, symbol, signal, order_notional_usd)
    side = "buy" if signal > 0 else "sell"
    logger.info(
        "开仓 %s %s 名义≈%.2f 数量=%s",
        symbol,
        side,
        sizing["est_notional"],
        sizing["amount_str"],
    )
    order = exchange.create_order(
        symbol,
        "market",
        side,
        sizing["amount"],
        None,
        {"positionSide": "BOTH", "newOrderRespType": "RESULT"},
    )
    return order, sizing


def _calc_realized_pnl(
    pos_amt: float, entry_price: float, exit_price: float, amount: float,
) -> float:
    if pos_amt > 0:
        raw = (exit_price - entry_price) * amount
    else:
        raw = (entry_price - exit_price) * amount
    fee = (entry_price + exit_price) * amount * TAKER_FEE_RATE
    return raw - fee


def current_expected_amount(job_id: str, signal: int) -> Optional[float]:
    want_side = "long" if signal > 0 else "short" if signal < 0 else "flat"
    with STATE_LOCK:
        st = JOB_STATE.get(job_id, {})
        exp = to_float(st.get("expected_amount"), 0.0)
        exp_side = (st.get("expected_side") or "").lower()
    if signal == 0:
        return None
    if exp > 0 and exp_side == want_side:
        return exp
    return None


def dry_run_reconcile(
    job: Dict[str, Any],
    pos: Dict[str, Any],
    bar: Dict[str, Any],
    expected_amount: Optional[float],
) -> Dict[str, Any]:
    sig = int(bar.get("effective_signal", bar.get("signal", 0)) or 0)
    target_side = {1: "long", -1: "short"}.get(sig, "flat")
    actions: List[str] = []

    if sig == 0:
        if pos["exists"]:
            hma_dir = int(bar.get("hma_direction", bar.get("raw_signal", 0)) or 0)
            pos_dir = 1 if pos["side"] == "long" else -1
            if hma_dir != 0 and hma_dir != pos_dir:
                actions = ["cancel_all_open_orders", "close"]
            else:
                actions = ["noop_manual_hold"]
        else:
            actions = ["noop"]
    else:
        want_side = "long" if sig > 0 else "short"
        if not pos["exists"]:
            actions = ["open_long" if sig > 0 else "open_short"]
        elif pos["side"] != want_side:
            actions = ["cancel_all_open_orders", "close", "open_long" if sig > 0 else "open_short"]
        elif expected_amount is not None and expected_amount > 0 and has_trade_credentials():
            ex = get_trade_exchange()
            same = amount_norm_str(ex, job["trade_symbol"], pos["contracts"]) == amount_norm_str(
                ex, job["trade_symbol"], expected_amount,
            )
            actions = (
                ["noop"]
                if same
                else ["cancel_all_open_orders", "close", "open_long" if sig > 0 else "open_short"]
            )
        else:
            actions = ["noop"]

    return {
        "mode": "dry_run",
        "target_side": target_side,
        "order_margin_usd": to_float(job["order_margin_usd"]),
        "order_notional_usd": to_float(job["order_notional_usd"]),
        "expected_amount": expected_amount,
        "position_before": pos,
        "signal": bar,
        "actions": actions,
    }


def reconcile_job(job: Dict[str, Any], bar: Dict[str, Any]) -> Dict[str, Any]:
    sig = int(bar.get("effective_signal", bar.get("signal", 0)) or 0)
    leverage = int(to_float(job["leverage"]))
    symbol = job["trade_symbol"]
    expected_before = current_expected_amount(job["id"], sig)

    if not ENABLE_TRADING:
        pos = {
            "exists": False,
            "symbol": symbol,
            "side": "flat",
            "contracts": 0.0,
            "base_amount": 0.0,
            "raw": None,
        }
        if has_trade_credentials():
            try:
                pos = fetch_position(get_trade_exchange(), symbol)
            except Exception as exc:
                logger.warning("dry-run 持仓查询失败: %s", exc)
        return dry_run_reconcile(job, pos, bar, expected_before)

    ex = get_trade_exchange()
    pos_before = fetch_position(ex, symbol)

    if position_matches_target(ex, symbol, pos_before, sig, expected_before):
        return {
            "mode": "live",
            "signal": bar,
            "position_before": pos_before,
            "position_after": pos_before,
            "expected_amount_after": expected_before,
            "actions": ["noop"],
        }

    if sig == 0 and pos_before["exists"]:
        hma_dir = int(bar.get("hma_direction", bar.get("raw_signal", 0)) or 0)
        pos_dir = 1 if pos_before["side"] == "long" else -1
        if hma_dir != 0 and hma_dir != pos_dir:
            pass  # proceed to close
        else:
            return {
                "mode": "live",
                "signal": bar,
                "position_before": pos_before,
                "position_after": pos_before,
                "expected_amount_after": pos_before["contracts"],
                "actions": ["noop_manual_hold"],
            }

    cancel_symbol_open_orders(ex, symbol)
    actions: List[str] = ["cancel_all_open_orders"]
    orders: List[Dict[str, Any]] = []
    sizing: Optional[Dict[str, Any]] = None
    close_realized_pnl = 0.0

    if pos_before["exists"]:
        order = close_position(ex, symbol, pos_before)
        if order:
            actions.append("close")
            orders.append(order)
            raw_info = (pos_before.get("raw") or {}).get("info") or {}
            before_amt = to_float(raw_info.get("positionAmt"), 0.0)
            before_entry = to_float(raw_info.get("entryPrice"), 0.0)
            fill_price = to_float(order.get("average"), 0.0)
            close_realized_pnl = _calc_realized_pnl(
                before_amt, before_entry, fill_price, abs(before_amt),
            )
        wait_until_flat(ex, symbol)

    expected_after: Optional[float] = None
    order_margin_usd = to_float(job["order_margin_usd"])
    order_notional_usd = to_float(job["order_notional_usd"])

    if sig != 0:
        job_margin_pct = float(job.get("margin_pct", MARGIN_PCT))
        order_margin_usd, order_notional_usd = compute_dynamic_margin(job_margin_pct)
        order, sizing = open_position(ex, symbol, sig, order_notional_usd, leverage)
        actions.append("open_long" if sig > 0 else "open_short")
        orders.append(order)
        expected_after = sizing["amount"]
        pos_after = wait_until_target(ex, symbol, sig, expected_after)
    else:
        pos_after = fetch_position(ex, symbol)

    if any(a in actions for a in ("close", "open_long", "open_short")):
        send_discord(
            f"[ML CCI] {job['label']}\n"
            f"信号 eff={sig} p_up={bar.get('proba_up')} | 操作 {' -> '.join(actions)}\n"
            f"时间 {_bj_now_str()}"
        )

    return {
        "mode": "live",
        "signal": bar,
        "position_before": pos_before,
        "position_after": pos_after,
        "expected_amount_before": expected_before,
        "expected_amount_after": expected_after,
        "order_margin_usd": order_margin_usd,
        "order_notional_usd": order_notional_usd,
        "close_realized_pnl": close_realized_pnl,
        "actions": actions,
        "orders": orders,
        "sizing": sizing,
    }


def update_job_state(job_id: str, payload: Dict[str, Any]) -> None:
    with STATE_LOCK:
        st = JOB_STATE.get(job_id, {})
        st.update(payload)
        JOB_STATE[job_id] = st


def fetch_ml_signal() -> Dict[str, Any]:
    return get_ml_cci_bar()


def run_job(job: Dict[str, Any], force_reconcile: bool = False) -> Dict[str, Any]:
    bar = fetch_ml_signal()

    with STATE_LOCK:
        prev = dict(JOB_STATE.get(job["id"], {}))

    prev_kline = prev.get("last_kline_time")
    prev_eff = prev.get("last_effective_signal")
    should = force_reconcile
    if prev_kline is not None and (
        prev_kline != bar.get("kline_time")
        or prev_eff != bar.get("effective_signal")
    ):
        should = True

    result: Dict[str, Any] = {
        "job_id": job["id"],
        "label": job.get("label", job["id"]),
        "signal": bar,
        "trade_symbol": job["trade_symbol"],
        "reconciled": False,
    }

    expected_amount_after = prev.get("expected_amount")
    expected_side_after = prev.get("expected_side")

    if should:
        tr = reconcile_job(job, bar)
        result["reconciled"] = True
        result["trade_result"] = tr
        sig = int(bar.get("effective_signal", 0) or 0)
        if tr.get("mode") == "dry_run":
            pass
        elif "noop_manual_hold" in tr.get("actions", []):
            held = tr.get("position_before", {})
            expected_amount_after = held.get("contracts") or held.get("base_amount")
            expected_side_after = held.get("side", "flat")
        else:
            expected_amount_after = tr.get("expected_amount_after")
            expected_side_after = "long" if sig > 0 else "short" if sig < 0 else "flat"
            if sig == 0:
                expected_amount_after = None

    update_job_state(
        job["id"],
        {
            "job_id": job["id"],
            "label": job.get("label", job["id"]),
            "trade_symbol": job["trade_symbol"],
            "last_raw_signal": bar.get("raw_signal"),
            "last_effective_signal": bar.get("effective_signal"),
            "last_kline_time": bar.get("kline_time"),
            "last_close": bar.get("close"),
            "last_run_ts": int(time.time()),
            "expected_amount": expected_amount_after,
            "expected_side": expected_side_after,
            "last_result": result,
        },
    )
    return result


def loop_signal() -> None:
    last_run = 0.0
    tick = 30.0
    job = ML_JOB
    while True:
        now = time.time()
        if not job.get("enabled", True):
            time.sleep(tick)
            continue
        interval = float(job.get("check_seconds", POLL_SECONDS))
        if now - last_run < interval:
            time.sleep(tick)
            continue
        last_run = now
        try:
            run_job(job, force_reconcile=False)
        except Exception as exc:
            logger.exception("run_job 失败: %s", exc)
        time.sleep(tick)


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "ml_cci_trading_server",
            "trading_enabled": ENABLE_TRADING,
            "port": ML_TRADING_PORT,
        }
    )


@app.route("/api/signal")
def api_signal():
    try:
        bar = fetch_ml_signal()
        return jsonify({"ok": True, "bar": bar})
    except Exception as exc:
        logger.exception("api_signal")
        return jsonify({"ok": False, "error": str(exc)}), 500


def main() -> None:
    global ORDER_MARGIN_USD, ORDER_NOTIONAL_USD, ML_TRADING_PORT
    parser = argparse.ArgumentParser(
        description="ML CCI LightGBM 实盘（Binance USDT 永续，ccxt）。环境变量见文件头与 README。",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP 端口，默认环境变量 ML_TRADING_PORT 或 6671",
    )
    args = parser.parse_args()
    if args.port is not None:
        ML_TRADING_PORT = args.port

    ORDER_MARGIN_USD = float(os.getenv("ORDER_MARGIN_USD", str(ORDER_MARGIN_USD)))
    ORDER_NOTIONAL_USD = ORDER_MARGIN_USD * FIXED_LEVERAGE
    ML_JOB["order_margin_usd"] = ORDER_MARGIN_USD
    ML_JOB["order_notional_usd"] = ORDER_NOTIONAL_USD

    logger.info(
        "ML CCI 交易服务 端口=%s 交易=%s 保证金=%.2f 名义≈%.2f 杠杆=%sx",
        ML_TRADING_PORT,
        ENABLE_TRADING,
        ORDER_MARGIN_USD,
        ORDER_NOTIONAL_USD,
        FIXED_LEVERAGE,
    )

    if ENABLE_TRADING and not has_trade_credentials():
        raise RuntimeError("ENABLE_TRADING=1 需要 BINANCE_API_KEY / BINANCE_SECRET")

    if ENABLE_TRADING:
        ex = get_trade_exchange()
        ensure_one_way_mode(ex, ML_JOB["trade_symbol"])

    if RECONCILE_ON_STARTUP:
        try:
            run_job(ML_JOB, force_reconcile=True)
        except Exception as exc:
            logger.exception("启动 reconcile 失败: %s", exc)

    threading.Thread(target=loop_signal, daemon=True, name="ml-signal-loop").start()
    app.run(host="0.0.0.0", port=ML_TRADING_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
