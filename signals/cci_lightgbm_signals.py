"""
cci_lightgbm_signals.py — CCI + LightGBM 信号生成器

核心逻辑：
1. 计算因子库（49个因子）
2. IC筛选出 CCI(7/14/21) + stoch_k(7)
3. LightGBM 预测下根K线涨跌概率
4. proba > 0.5 → 做多, proba < 0.5 → 做空
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ─────────────────────────────────────────────
# 因子库
# ─────────────────────────────────────────────

def build_factor_library(df: pd.DataFrame) -> pd.DataFrame:
    """构建49个因子的因子库"""
    factors = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # 价格类因子
    for w in [5, 10, 20, 60]:
        factors[f'log_return_{w}'] = np.log(close / close.shift(w))
        factors[f'volatility_{w}'] = close.pct_change().rolling(w).std()
        factors[f'skew_{w}'] = close.pct_change().rolling(w).skew()
        factors[f'return_{w}'] = close.pct_change(w)

    # RSI
    for p in [7, 14, 21]:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / loss.replace(0, np.nan)
        factors[f'rsi_{p}'] = 100 - (100 / (1 + rs))

    # 随机指标 %K / %D
    for p in [7, 14, 21]:
        low_min = low.rolling(p).min()
        high_max = high.rolling(p).max()
        k = 100 * (close - low_min) / (high_max - low_min + 1e-10)
        factors[f'stoch_k_{p}'] = k
        factors[f'stoch_d_{p}'] = k.rolling(3).mean()

    # CCI
    for p in [7, 14, 21]:
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(p).mean()
        mad = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean())
        factors[f'cci_{p}'] = (tp - sma_tp) / (0.015 * mad + 1e-10)

    # ADX
    for p in [7, 14, 21]:
        deltap = high.diff()
        deltam = -low.diff()
        plus_dm = deltap.where((deltap > deltam) & (deltap > 0), 0)
        minus_dm = deltam.where((deltab > deltap) & (deltam > 0), 0)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()
        plus_di = 100 * plus_dm.rolling(p).mean() / atr
        minus_di = 100 * minus_dm.rolling(p).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        factors[f'adx_{p}'] = dx.rolling(p).mean()

    # 布林带
    for p in [10, 20]:
        sma = close.rolling(p).mean()
        std = close.rolling(p).std()
        factors[f'bb_upper_{p}'] = sma + 2 * std
        factors[f'bb_lower_{p}'] = sma - 2 * std
        factors[f'bb_width_{p}'] = (factors[f'bb_upper_{p}'] - factors[f'bb_lower_{p}']) / sma
        factors[f'bb_pctb_{p}'] = (close - factors[f'bb_lower_{p}']) / (factors[f'bb_upper_{p}'] - factors[f'bb_lower_{p}'] + 1e-10)

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    factors['macd'] = macd
    factors['macd_signal'] = signal
    factors['macd_hist'] = macd - signal

    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    factors['atr'] = tr.rolling(14).mean()

    # 成交量因子
    factors['volume_ma5'] = volume.rolling(5).mean()
    factors['volume_ratio'] = volume / factors['volume_ma5']
    obv = (np.sign(close.diff()) * volume).cumsum()
    factors['obv'] = obv
    factors['obv_ma10'] = obv.rolling(10).mean()

    # 时间因子
    if hasattr(df.index, 'hour'):
        factors['hour'] = df.index.hour
        factors['dayofweek'] = df.index.dayofweek
    else:
        factors['hour'] = pd.to_datetime(df.get('open_time', df.index)).hour
        factors['dayofweek'] = pd.to_datetime(df.get('open_time', df.index)).dayofweek

    return factors


# ─────────────────────────────────────────────
# IC 分析筛选
# ─────────────────────────────────────────────

def ic_filter(factors: pd.DataFrame, returns: pd.Series, threshold: float = 0.05) -> list:
    """IC 筛选，返回 |IC| > threshold 的因子列表"""
    ic_results = {}
    for col in factors.columns:
        valid = factors[col].notna() & returns.notna()
        if valid.sum() > 100:
            ic = factors[col][valid].corr(returns[valid])
            ic_results[col] = ic

    return [k for k, v in ic_results.items() if abs(v) > threshold]


# ─────────────────────────────────────────────
# 信号生成
# ─────────────────────────────────────────────

def generate_signal(df: pd.DataFrame, model, feature_cols: list, threshold: float = 0.5) -> dict:
    """
    生成交易信号

    Returns:
        dict: {
            'signal': 1 (long) / -1 (short) / 0 (flat),
            'position': 'long' / 'short' / 'flat',
            'proba': float,          # 做多概率
            'confidence': float,     # |proba - 0.5| * 2
            'atr': float,
            'stop_loss': float,
            'take_profit': float
        }
    """
    factors = build_factor_library(df)
    latest = factors[feature_cols].iloc[-1:].dropna(axis=1)

    # 补齐缺失因子
    for col in feature_cols:
        if col not in latest.columns:
            latest[col] = 0

    latest = latest[feature_cols]

    proba = model.predict_proba(latest)[:, 1][0]
    atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0

    if proba > threshold:
        signal = 1
        position = 'long'
        stop_loss = df['close'].iloc[-1] - 3 * atr
        take_profit = df['close'].iloc[-1] + 5 * atr
    elif proba < (1 - threshold):
        signal = -1
        position = 'short'
        stop_loss = df['close'].iloc[-1] + 3 * atr
        take_profit = df['close'].iloc[-1] - 5 * atr
    else:
        signal = 0
        position = 'flat'
        stop_loss = 0
        take_profit = 0

    confidence = abs(proba - 0.5) * 2

    return {
        'signal': signal,
        'position': position,
        'proba': proba,
        'confidence': confidence,
        'atr': atr,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'close': df['close'].iloc[-1],
    }
