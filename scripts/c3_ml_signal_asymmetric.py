#!/usr/bin/env python3
"""
C3 ML 非对称阈值推理（不修改 c3_ml_live.py）。

- 三分类：p_up > long_thr -> 多；否则 p_down > down_thr -> 空；否则 0。
- 输出供 ml_cci_trading_server 使用的 bar 字典（与 first_trading 语义对齐的字段名）。

环境变量可覆盖默认阈值：
  ML_LONG_THRESHOLD  (默认 0.6)
  ML_SHORT_THRESHOLD (默认 0.4，表示做空需 p_down > 该值)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_ROOT))

from signals.asymmetric_proba_grid import hma_direction_from_proba_up, signal_from_three_class_proba  # noqa: E402

import c3_ml_live as c3  # noqa: E402


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def ensure_c3_model() -> None:
    """若本地无缓存模型，调用一次 get_c3_ml_signal 完成训练与落盘。"""
    if not c3.MODEL_PKL.exists():
        c3.get_c3_ml_signal()


def get_ml_cci_bar(
    long_thr: float | None = None,
    down_thr: float | None = None,
) -> Dict[str, Any]:
    """
    返回与实盘 reconcile 兼容的 bar：
      kline_time, close, raw_signal, effective_signal, effective_signal_source,
      hma_direction, proba_up, proba_down, atr (若可得)
    """
    lo = long_thr if long_thr is not None else _env_float("ML_LONG_THRESHOLD", 0.6)
    sh = down_thr if down_thr is not None else _env_float("ML_SHORT_THRESHOLD", 0.4)

    ensure_c3_model()

    with open(c3.MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    with open(c3.FEAT_JSON) as f:
        feat_cols = json.load(f)

    df = c3.fetch_binance_ohlcv(c3.SYMBOL, c3.INTERVAL, c3.LIMIT)
    feat = c3.compute_features(df)
    for col in feat_cols:
        if col not in feat.columns:
            feat[col] = np.nan
    feat = feat[feat_cols].dropna()
    if len(feat) == 0:
        return {
            "kline_time": 0,
            "close": 0.0,
            "raw_signal": 0,
            "effective_signal": 0,
            "effective_signal_source": "ml_cci",
            "hma_direction": 0,
            "proba_up": 0.5,
            "proba_down": 0.5,
            "atr": 0.0,
            "action": "flat",
            "error": "ERROR_NO_DATA",
        }

    X_live = feat.iloc[[-1]]
    classes = [int(c) for c in model.classes_]
    proba = model.predict_proba(X_live)[0]
    p_up = float(proba[classes.index(1)]) if 1 in classes else 0.5
    p_down = float(proba[classes.index(-1)]) if -1 in classes else 0.5

    sig = signal_from_three_class_proba(p_up, p_down, lo, sh)
    close = float(df["close"].iloc[-1])
    ts = df.index[-1]
    kline_time = int(pd_timestamp_ms(ts))

    atr_val = 0.0
    if "atr" in feat.columns:
        atr_val = float(feat["atr"].iloc[-1])

    hma_dir = hma_direction_from_proba_up(p_up)
    if sig > 0:
        action = "open_long"
    elif sig < 0:
        action = "open_short"
    else:
        action = "flat"

    return {
        "kline_time": kline_time,
        "close": close,
        "raw_signal": sig,
        "effective_signal": sig,
        "effective_signal_source": "ml_cci",
        "hma_direction": hma_dir,
        "proba_up": round(p_up, 4),
        "proba_down": round(p_down, 4),
        "atr": atr_val,
        "action": action,
    }


def pd_timestamp_ms(ts: Any) -> int:
    import pandas as pd

    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC")
    return int(t.timestamp() * 1000)


if __name__ == "__main__":
    import json as _json

    b = get_ml_cci_bar()
    print(_json.dumps(b, indent=2, default=str))
