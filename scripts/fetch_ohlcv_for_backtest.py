#!/usr/bin/env python3
"""
从 Binance **现货** 分页下载 1h K 线，保存为 backtest / train 使用的 CSV 格式：

  pd.read_csv(path, parse_dates=["open_time"], index_col="open_time")

列：open, high, low, close, volume（与 cci_lightgbm_signals 一致）

实现：直接请求 GET /api/v3/klines（**不**走 ccxt、**不**请求 exchangeInfo），与 scripts/c3_ml_live.py 一致，
避免部分网络环境下 exchangeInfo 超时。

默认时间范围读取 config/walk_forward.yaml 中的 data_range。

依赖：pip install pandas pyyaml（无需 ccxt）

网络：若访问 api.binance.com 超时/被墙，请开 VPN 或在 PowerShell 中设置代理后重试，例如：
  $env:HTTPS_PROXY="http://127.0.0.1:7890"
  python scripts/fetch_ohlcv_for_backtest.py

可选环境变量：
  BINANCE_SPOT_API_BASE  默认 https://api.binance.com（可换可访问的反向代理域名，需兼容官方路径）
  BINANCE_HTTP_TIMEOUT_S 单次 HTTP 超时秒数，默认 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PAIRS: dict[str, str] = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
}

CHUNK = 1000
BASE_URL = (os.getenv("BINANCE_SPOT_API_BASE", "https://api.binance.com").rstrip("/"))
TIMEOUT_S = float(os.getenv("BINANCE_HTTP_TIMEOUT_S", "60"))


def _load_data_range_from_config(config_path: Path) -> tuple[str, str]:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dr = cfg.get("backtest", {}).get("data_range", {})
    start = dr.get("start", "2024-01-01")
    end = dr.get("end", "2026-03-01")
    return str(start), str(end)


def _parse_utc_date(s: str) -> datetime:
    dt = datetime.strptime(s.strip()[:10], "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def _build_opener() -> urllib.request.OpenerDirector:
    """尊重 HTTPS_PROXY / HTTP_PROXY（与 curl 类似）。"""
    https_p = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
    http_p = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
    if https_p or http_p:
        import urllib.request as ur

        proxies: dict[str, str] = {}
        if http_p:
            proxies["http"] = http_p
        if https_p:
            proxies["https"] = https_p
        proxy_handler = ur.ProxyHandler(proxies)
        return ur.build_opener(proxy_handler)
    return urllib.request.build_opener()


def _http_get_json(url: str) -> list | dict:
    opener = _build_opener()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ml_strat_cci_lightgbm/fetch_ohlcv_for_backtest"},
    )
    try:
        with opener.open(req, timeout=TIMEOUT_S) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"请求失败: {e}\n"
            "若在中国大陆，Binance 域名常被墙或极慢：请开 VPN，或设置 HTTPS_PROXY 后再试。\n"
            "排查: 应 ping 主机名而非 URL，例如: ping api.binance.com"
        ) from e
    return json.loads(raw.decode("utf-8"))


def fetch_klines_rest(
    symbol: str,
    interval: str,
    since_ms: int,
    until_ms: int,
) -> list[list]:
    """
    返回 [[ts_ms, open, high, low, close, volume], ...]
    Binance klines 文档字段 0-5。
    """
    all_rows: list[list] = []
    current = since_ms
    while current < until_ms:
        q = (
            f"{BASE_URL}/api/v3/klines?"
            f"symbol={symbol}&interval={interval}&startTime={current}"
            f"&endTime={until_ms - 1}&limit={CHUNK}"
        )
        data = _http_get_json(q)
        if not data:
            break
        batch: list[list] = []
        for k in data:
            t = int(k[0])
            if t >= until_ms:
                continue
            if t < since_ms:
                continue
            batch.append(
                [
                    t,
                    float(k[1]),
                    float(k[2]),
                    float(k[3]),
                    float(k[4]),
                    float(k[5]),
                ]
            )
        if not batch:
            break
        all_rows.extend(batch)
        last_t = batch[-1][0]
        nxt = last_t + 1
        if nxt <= current:
            nxt = last_t + 3600000
        current = nxt
        if len(data) < CHUNK:
            break
        time.sleep(0.15)

    seen: set[int] = set()
    out: list[list] = []
    for row in sorted(all_rows, key=lambda x: x[0]):
        if row[0] in seen:
            continue
        seen.add(row[0])
        out.append(row)
    return out


def rows_to_dataframe(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["open_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp"])
    df = df.set_index("open_time").sort_index()
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Binance 现货 1h OHLCV 供 backtest/train 使用")
    parser.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "config" / "walk_forward.yaml",
        help="用于读取默认 data_range.start / end",
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--symbols",
        type=str,
        default="btc,eth",
        help="逗号分隔：btc, eth",
    )
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "data")
    parser.add_argument("--timeframe", type=str, default="1h")
    args = parser.parse_args()

    if args.start and args.end:
        start_s, end_s = args.start, args.end
    elif args.config.is_file():
        start_s, end_s = _load_data_range_from_config(args.config)
        print(f"使用 {args.config} 的 data_range: {start_s} .. {end_s}")
    else:
        start_s, end_s = "2024-01-01", "2026-03-01"
        print(f"未找到 config，使用默认区间: {start_s} .. {end_s}")

    since_dt = _parse_utc_date(start_s)
    until_dt = _parse_utc_date(end_s) + timedelta(days=1)
    if until_dt <= since_dt:
        raise SystemExit("end 必须不早于 start")

    since_ms = int(since_dt.timestamp() * 1000)
    until_ms = int(until_dt.timestamp() * 1000)

    keys = [k.strip().lower() for k in args.symbols.split(",") if k.strip()]
    pairs: list[tuple[str, str]] = []
    for k in keys:
        if k not in DEFAULT_PAIRS:
            raise SystemExit(f"未知 symbol 键: {k}，可选: {list(DEFAULT_PAIRS.keys())}")
        pairs.append((k, DEFAULT_PAIRS[k]))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    est_hours = (until_ms - since_ms) / 1000 / 3600
    print(
        f"API={BASE_URL} 超时={TIMEOUT_S}s 预计约 {est_hours:.0f} 根 {args.timeframe} "
        f"（分页 {CHUNK}）"
    )

    for key, symbol in pairs:
        print(f"拉取 {symbol} …")
        rows = fetch_klines_rest(symbol, args.timeframe, since_ms, until_ms)
        if not rows:
            print(f"  无数据: {symbol}")
            continue
        df = rows_to_dataframe(rows)
        out_path = args.out_dir / f"{key}_1h.csv"
        df.index.name = "open_time"
        df.to_csv(out_path)
        print(f"  保存 {out_path}  rows={len(df)}  {df.index[0]} .. {df.index[-1]}")


if __name__ == "__main__":
    main()
