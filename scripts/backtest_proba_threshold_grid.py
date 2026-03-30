#!/usr/bin/env python3
"""
Walk-Forward 回测 + 非对称 proba 阈值网格（不改 backtest.py）。

用法:
    python scripts/backtest_proba_threshold_grid.py --config config/walk_forward.yaml --data btc_1h.csv \\
        --long-grid 0.55,0.6,0.65 --short-grid 0.35,0.4,0.45
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from signals.asymmetric_proba_grid import positions_from_binary_proba_series, validate_binary_thresholds
from signals.cci_lightgbm_signals import build_factor_library, ic_filter


def _parse_grid(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def walk_forward_proba_grid(
    df: pd.DataFrame,
    config: dict,
    long_grid: list[float],
    short_grid: list[float],
) -> pd.DataFrame:
    factors = build_factor_library(df)
    returns = df["close"].pct_change().shift(-1)

    train_days = config["backtest"]["windows"]["train_days"]
    test_days = config["backtest"]["windows"]["test_days"]
    step_days = config["backtest"]["windows"]["step_days"]
    ic_thresh = config["backtest"]["ic_filter"]["threshold"]

    pairs: list[tuple[float, float]] = []
    for lo, sh in itertools.product(long_grid, short_grid):
        try:
            validate_binary_thresholds(lo, sh)
            pairs.append((lo, sh))
        except ValueError:
            continue

    if not pairs:
        raise ValueError("no valid (long_thr, short_thr) pairs after filtering long_thr > short_thr")

    train_size = train_days * 24
    test_size = test_days * 24
    step_size = step_days * 24

    n_windows = (len(df) - train_size - test_size) // step_size
    print(f"Walk-Forward: {n_windows} windows, {len(pairs)} threshold pairs")

    rows: list[dict] = []

    for i in range(n_windows):
        train_end = i * step_size + train_size
        test_end = train_end + test_size

        train_df = df.iloc[i * step_size : train_end]
        test_df = df.iloc[train_end:test_end]

        if len(train_df) < train_size * 0.8 or len(test_df) < test_size * 0.8:
            continue

        train_factors = factors.iloc[i * step_size : train_end]
        train_returns = returns.iloc[i * step_size : train_end]
        good_factors = ic_filter(train_factors, train_returns, threshold=ic_thresh)

        if len(good_factors) < 2:
            continue

        X_train = train_factors[good_factors].dropna()
        y_train = train_returns.loc[X_train.index]
        y_train = (y_train > 0).astype(int)

        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        test_factors = factors.iloc[train_end:test_end]
        X_test = test_factors[good_factors].dropna()
        test_close = df["close"].loc[X_test.index]
        proba = model.predict_proba(X_test)[:, 1]
        proba_series = pd.Series(proba, index=X_test.index)

        for long_thr, short_thr in pairs:
            position = positions_from_binary_proba_series(proba_series, long_thr, short_thr)
            strategy_returns = position * test_close.pct_change().fillna(0)
            strategy_returns = strategy_returns[1:]

            sharpe = (
                strategy_returns.mean() / strategy_returns.std() * np.sqrt(24 * 365)
                if strategy_returns.std() > 0
                else 0
            )
            mdd = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
            winrate = (strategy_returns > 0).mean() if len(strategy_returns) else 0

            rows.append(
                {
                    "window": i,
                    "long_thr": long_thr,
                    "short_thr": short_thr,
                    "train_start": train_df.index[0],
                    "test_start": test_df.index[0],
                    "test_end": test_df.index[-1],
                    "n_factors": len(good_factors),
                    "sharpe": sharpe,
                    "mdd": mdd,
                    "winrate": winrate,
                    "n_bars": len(position),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Forward + asymmetric proba threshold grid")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="btc_1h.csv")
    parser.add_argument("--output", default="backtest/results/")
    parser.add_argument(
        "--long-grid",
        default="0.55,0.6,0.65",
        help="comma-separated long thresholds (proba_up > long -> long)",
    )
    parser.add_argument(
        "--short-grid",
        default="0.35,0.4,0.45",
        help="comma-separated short thresholds (proba_up < short -> short)",
    )
    args = parser.parse_args()

    long_grid = _parse_grid(args.long_grid)
    short_grid = _parse_grid(args.short_grid)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Loading data: {args.data}")
    df = pd.read_csv(args.data, parse_dates=["open_time"], index_col="open_time")
    print(f"  Rows: {len(df)}, Range: {df.index[0]} → {df.index[-1]}")

    results = walk_forward_proba_grid(df, config, long_grid, short_grid)
    print(f"\n=== Grid Walk-Forward ===")
    print(f"Total rows (windows × pairs): {len(results)}")
    if len(results) > 0:
        pass_df = results[results["sharpe"] > 1.5]
        print(f"Rows with Sharpe > 1.5: {len(pass_df)}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "proba_grid_walk_forward.csv"
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
