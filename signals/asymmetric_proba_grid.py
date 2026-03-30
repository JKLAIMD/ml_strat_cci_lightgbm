"""
非对称 proba 阈值（用于网格回测与实盘）。

二元分类（仅 proba_up = P(y=1)）：
  - proba_up > long_thr  -> +1
  - proba_up < short_thr -> -1
  - 否则 -> 0（中性带）

三元分类（c3_ml：p_up, p_down, p_flat）：
  - p_up > long_thr -> +1（优先于空）
  - 否则若 p_down > down_thr -> -1
  - 否则 -> 0

约束：long_thr > short_thr（二元）；long_thr 与 down_thr 由调用方保证合理。
"""

from __future__ import annotations

from typing import Tuple


def validate_binary_thresholds(long_thr: float, short_thr: float) -> Tuple[float, float]:
    """确保 long_thr > short_thr；否则抛 ValueError。"""
    if long_thr <= short_thr:
        raise ValueError(
            f"binary thresholds require long_thr > short_thr, got long={long_thr} short={short_thr}"
        )
    return long_thr, short_thr


def signal_from_binary_proba(proba_up: float, long_thr: float, short_thr: float) -> int:
    """
    单列「涨概率」上的非对称阈值（与 walk-forward 二元 LGBM 一致）。

    proba_up: 模型输出的 P(下一根涨) ∈ [0,1]
    """
    validate_binary_thresholds(long_thr, short_thr)
    if proba_up > long_thr:
        return 1
    if proba_up < short_thr:
        return -1
    return 0


def positions_from_binary_proba_series(proba: "pd.Series", long_thr: float, short_thr: float) -> "pd.Series":
    """向量化仓位 {-1,0,1}，索引与 proba 对齐；涨侧优先于跌侧。"""
    import numpy as np
    import pandas as pd

    validate_binary_thresholds(long_thr, short_thr)
    v = proba.values
    arr = np.where(v > long_thr, 1, np.where(v < short_thr, -1, 0))
    return pd.Series(arr, index=proba.index, dtype=int)


def signal_from_three_class_proba(
    p_up: float,
    p_down: float,
    long_thr: float,
    down_thr: float,
) -> int:
    """
    c3 三分类：优先做多，其次做空。

    down_thr: 做空所需的最小 P(class=-1)，对应实盘 ML_SHORT_THRESHOLD。
    """
    if p_up > long_thr:
        return 1
    if p_down > down_thr:
        return -1
    return 0


def hma_direction_from_proba_up(proba_up: float) -> int:
    """供 reconcile：与 first_trading 中 hma_direction 语义类似（倾向多/空）。"""
    if proba_up > 0.5:
        return 1
    if proba_up < 0.5:
        return -1
    return 0
