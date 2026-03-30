#!/usr/bin/env python3
"""
c3_ml_live.py — C3 ML LightGBM Live Inference
=============================================
Downloads BTCUSDT 1h data from Binance, computes 49 features (matching the
walk-forward validation that achieved 18/24 PASS, Sharpe=3.90),
trains a LightGBMClassifier, and generates a live trading signal.

Signal logic:
  proba_up   > 0.55 → LONG  (+1)
  proba_down > 0.55 → SHORT (-1)
  otherwise        → NEUTRAL (0)

Usage:
  python3 c3_ml_live.py          # standalone
  from c3_ml_live import get_c3_ml_signal
  result = get_c3_ml_signal()    # returns dict
"""

import os
import sys
import json
import pickle
import urllib.request
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR   = Path.home() / "Desktop" / "share" / "share_repository" / "data"
MODEL_PKL   = CACHE_DIR / "c3_lgbm.pkl"
FEAT_JSON   = CACHE_DIR / "c3_lgbm_features.json"
CACHE_FILE  = CACHE_DIR / "BTC_USDT_1h.pkl"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
SYMBOL   = "BTCUSDT"
INTERVAL = "1h"
LIMIT    = 2000    # ~83 days of 1h bars

# Feature periods (matching c3_ml_comprehensive_results.md)
PRICE_PERIODS = [5, 10, 20, 60]
RSI_PERIODS   = [7, 14, 21]
STOCH_PERIODS = [7, 14, 21]
CCI_PERIODS   = [7, 14, 21]
ADX_PERIODS   = [7, 14, 21]
BB_PERIODS    = [10, 20]
VOL_PERIODS   = [5, 20]

# ── Indicator helpers ─────────────────────────────────────────────────────────

def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    efast = series.ewm(span=fast, adjust=False).mean()
    eslow = series.ewm(span=slow, adjust=False).mean()
    macd  = efast - eslow
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def calc_bb(series: pd.Series, period: int, std_mult=2.0):
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    pctb  = (series - lower) / (upper - lower + 1e-9)
    width = (upper - lower) / mid
    return upper, mid, lower, pctb, width

def calc_atr(df: pd.DataFrame, period=14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_adx(df: pd.DataFrame, period=14) -> pd.Series:
    high = df["high"]; low = df["low"]; close = df["close"]
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(upper=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr = calc_atr(df, period)
    plus_di  = 100 * plus_dm.rolling(period).mean()  / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(span=period, adjust=False).mean()

def calc_cci(df: pd.DataFrame, period=14) -> pd.Series:
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad + 1e-9)

def calc_stochastic(df: pd.DataFrame, k_period=14, d_period=3):
    low_k  = df["low"].rolling(k_period).min()
    high_k = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_k) / (high_k - low_k + 1e-9)
    return k, k.rolling(d_period).mean()

def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()

# ── Feature engineering ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all ~49 features matching the walk-forward validation."""
    close = df["close"]; high = df["high"]; low = df["low"]; volume = df["volume"]
    feat = pd.DataFrame(index=df.index)
    ret  = close.pct_change()

    # Price features
    for p in PRICE_PERIODS:
        feat[f"return_{p}"]     = ret
        feat[f"volatility_{p}"] = ret.rolling(p).std()
        feat[f"skew_{p}"]       = ret.rolling(p).skew()
        feat[f"log_return_{p}"] = np.log(close / close.shift(p))

    # RSI
    for p in RSI_PERIODS:
        feat[f"rsi_{p}"] = calc_rsi(close, p)

    # MACD
    macd, sig, hist = calc_macd(close)
    feat["macd"] = macd; feat["macd_signal"] = sig; feat["macd_hist"] = hist

    # Bollinger Bands
    for p in BB_PERIODS:
        upper, mid, lower, pctb, width = calc_bb(close, p)
        feat[f"bb_upper_{p}"]  = upper
        feat[f"bb_middle_{p}"] = mid
        feat[f"bb_lower_{p}"]  = lower
        feat[f"bb_pctb_{p}"]   = pctb
        feat[f"bb_width_{p}"]  = width

    # ATR
    feat["atr"] = calc_atr(df)

    # ADX
    for p in ADX_PERIODS:
        feat[f"adx_{p}"] = calc_adx(df, p)

    # CCI
    for p in CCI_PERIODS:
        feat[f"cci_{p}"] = calc_cci(df, p)

    # Stochastic
    for p in STOCH_PERIODS:
        k, d = calc_stochastic(df, p)
        feat[f"stoch_k_{p}"] = k
        feat[f"stoch_d_{p}"] = d

    # Volume
    feat["volume_ratio"] = volume / volume.rolling(20).mean()
    feat["obv"]           = calc_obv(close, volume)
    feat["obv_ma10"]     = feat["obv"].rolling(10).mean()
    for p in VOL_PERIODS:
        feat[f"volume_ma{p}"] = volume.rolling(p).mean()

    return feat

def prepare_X(feat: pd.DataFrame) -> pd.DataFrame:
    X = feat.dropna()
    return X

def make_target(close: pd.Series) -> pd.Series:
    """+1 if next bar up >0.5%, -1 if down >0.5%, else 0."""
    fut = close.shift(-1) / close - 1
    return fut.where(fut >  0.005, -1).where(fut < -0.005, 0).fillna(0).astype(int)

# ── Data fetch ────────────────────────────────────────────────────────────────

def fetch_binance_ohlcv(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    with urllib.request.urlopen(url, timeout=15) as r:
        raw = json.loads(r.read())
    cols = ["timestamp","open","high","low","close","volume",
            "close_time","qv","trades","taker_buy_base","taker_buy_quote","x"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"]), unit="ms", utc=True
    ).dt.tz_convert("Asia/Shanghai")
    return df.set_index("timestamp").sort_index()[["open","high","low","close","volume"]]

# ── Core live signal function ────────────────────────────────────────────────

def get_c3_ml_signal() -> dict:
    """Train (if needed) and run LightGBM inference. Returns signal dict."""
    # 1) Train model if no cached pickle
    if not MODEL_PKL.exists():
        print("[C3 ML] Training model from Binance history …")
        # Build df from as much historical data as possible
        # First: try to load existing 4h 400d data and convert to 1h
        ohlcv_4h_path = CACHE_DIR / "ohlcv_4h" / "BTC_USDT_400d_4h.pkl"
        if ohlcv_4h_path.exists():
            with open(ohlcv_4h_path, "rb") as f:
                raw4h = pickle.load(f)
            cols4h = ["timestamp","open","high","low","close","volume"]
            df4h = pd.DataFrame(raw4h, columns=cols4h)
            df4h["timestamp"] = pd.to_datetime(df4h["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
            df4h = df4h.set_index("timestamp").sort_index()
            # Convert 4h → 1h by exploding each 4h bar into 4 × 1h bars (approximation)
            # More accurate: fetch 1h directly
            print(f"[C3 ML] Loaded 4h data: {len(df4h)} rows from {df4h.index[0]}")
        else:
            df4h = pd.DataFrame()

        df = fetch_binance_ohlcv(SYMBOL, INTERVAL, LIMIT)
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "rb") as f:
                df_cached = pickle.load(f)
            df = pd.concat([df_cached, df]).drop_duplicates().sort_index()
        if not df4h.empty:
            # Prepend 4h data as 1h-equivalent (merge on timestamp)
            df4h_renamed = df4h.rename(columns={"open":"o","high":"h","low":"l","close":"c","volume":"v"})
            df = pd.concat([df4h_renamed[["o","h","l","c","v"]].rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}), df])
            df = df[~df.index.duplicated(keep='first')].sort_index()

        feat = compute_features(df)
        X    = prepare_X(feat)
        y    = make_target(df.loc[X.index, "close"])
        common = X.index.intersection(y.index)
        X = X.loc[common]; y = y.loc[common]

        # Drop neutral class for sharper signals
        mask = (y != 0)
        X_tr = X[mask]; y_tr = y[mask]
        feat_cols = X_tr.columns.tolist()

        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, random_state=42, verbose=-1,
        )
        model.fit(X_tr[feat_cols], y_tr)

        with open(MODEL_PKL, "wb") as f:
            pickle.dump(model, f)
        with open(FEAT_JSON, "w") as f:
            json.dump(feat_cols, f)
        # Save cache
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(df, f)
        print(f"[C3 ML] Trained → {MODEL_PKL} ({len(X_tr)} training rows, {len(feat_cols)} features)")
    else:
        print("[C3 ML] Using cached model")

    # 2) Load model
    with open(MODEL_PKL, "rb") as f:
        model = pickle.load(f)
    with open(FEAT_JSON) as f:
        feat_cols = json.load(f)

    # 3) Fetch latest data & compute features
    df   = fetch_binance_ohlcv(SYMBOL, INTERVAL, LIMIT)
    feat = compute_features(df)

    # Align columns
    for col in feat_cols:
        if col not in feat.columns:
            feat[col] = np.nan
    feat = feat[feat_cols].dropna()
    if len(feat) == 0:
        return {"signal": 0, "proba_up": 0.5, "proba_down": 0.5,
                "price": 0, "direction": "ERROR_NO_DATA", "bars_used": 0}

    X_live = feat.iloc[[-1]]
    classes = [int(c) for c in model.classes_]

    # 4) Predict probabilities
    proba = model.predict_proba(X_live)[0]
    p_up   = float(proba[classes.index(1)])   if  1 in classes else 0.5
    p_down = float(proba[classes.index(-1)]) if -1 in classes else 0.5

    # 5) Signal
    if   p_up   > 0.55: signal = 1; direction = "LONG"
    elif p_down > 0.55: signal = -1; direction = "SHORT"
    else:                 signal = 0; direction = "NEUTRAL"

    price = float(df["close"].iloc[-1])
    ts    = str(df.index[-1])

    print(f"[C3 ML] {ts} | price={price} | p_up={p_up:.4f} p_down={p_down:.4f} "
          f"| signal={signal} ({direction})")

    return {
        "signal":     signal,
        "proba_up":   round(p_up, 4),
        "proba_down": round(p_down, 4),
        "price":      price,
        "direction":  direction,
        "timestamp":  ts,
        "bars_used":  len(feat),
        "model":      "LightGBM 100d d5 lr0.05",
    }

# ── Standalone ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = get_c3_ml_signal()
    print("\n=== C3 ML Live Signal ===")
    print(json.dumps(result, indent=2, default=str))
