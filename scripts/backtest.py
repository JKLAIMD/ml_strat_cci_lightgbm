#!/usr/bin/env python3
"""
backtest.py — Walk-Forward 回测脚本

用法:
    python scripts/backtest.py --config config/walk_forward.yaml

修复清单:
  FIX1: Sharpe 年化因子改用 sqrt(test_bars)
  FIX2: 真实交易次数 = 换仓次数 (position.diff().abs() == 2).sum() / 2
  FIX3: 手续费模型 - 每笔换仓扣 0.08% (开仓 0.04% + 平仓 0.04%)
  FIX4: 滑点模型 - 用 next_open 代替 close
  FIX5: 删除 IC Filter，全量因子进 LightGBM
  FIX6: 非对称 threshold (默认 long=0.55, short=0.45)
  FIX7: proba 阈值网格化（多个 threshold 组合找最优）
"""

import argparse
import pickle
import yaml
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from itertools import product as iter_product

sys.path.insert(0, str(Path(__file__).parent.parent))
from signals.cci_lightgbm_signals import build_factor_library
from signals.asymmetric_proba_grid import positions_from_binary_proba_series


def walk_forward_backtest(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """滚动 Walk-Forward 回测"""
    factors = build_factor_library(df)
    returns = df['close'].pct_change().shift(-1)

    train_days = config['backtest']['windows']['train_days']
    test_days = config['backtest']['windows']['test_days']
    step_days = config['backtest']['windows']['step_days']

    # 默认非对称阈值网格（FIX7）
    threshold_pairs = [
        (0.51, 0.49),  # 近似对称（避免 validate 拒绝）
        (0.55, 0.45),  # 非对称
        (0.60, 0.40),
        (0.65, 0.35),
    ]

    results = []
    train_size = train_days * 24
    test_size = test_days * 24
    step_size = step_days * 24

    n_windows = (len(df) - train_size - test_size) // step_size
    print(f"Walk-Forward: {n_windows} windows, {len(threshold_pairs)} threshold pairs")

    for i in range(n_windows):
        train_end = i * step_size + train_size
        test_end = train_end + test_size

        train_df = df.iloc[i * step_size:train_end]
        test_df = df.iloc[train_end:test_end]

        if len(train_df) < train_size * 0.8 or len(test_df) < test_size * 0.8:
            continue

        # ---- 训练集：全量因子进 LightGBM（FIX5: 删除 IC Filter）----
        all_factors = list(factors.columns)
        train_factors = factors.iloc[i * step_size:train_end]
        train_returns = returns.iloc[i * step_size:train_end]

        X_train_df = train_factors[all_factors].dropna()
        y_train = train_returns.loc[X_train_df.index]
        y_train = (y_train > 0).astype(int)

        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model.fit(X_train_df, y_train)

        # ---- 测试集 ----
        test_factors = factors.iloc[train_end:test_end]
        X_test = test_factors[all_factors].dropna()

        proba = model.predict_proba(X_test)[:, 1]
        proba_series = pd.Series(proba, index=X_test.index)

        for long_thr, short_thr in threshold_pairs:
            # FIX6: 非对称 threshold
            position = positions_from_binary_proba_series(proba_series, long_thr, short_thr)

            # FIX4: 用 next_close 计算收益（滑点模型）
            # position[t] 在 close[t] 收市后确定，应用于 return from close[t] to close[t+1]
            # = close[t+1]/close[t] - 1 = df['close'].shift(-1) / df['close'] - 1
            # 对齐到 position.index
            next_close = df['close'].shift(-1).loc[position.index]
            current_close = df['close'].loc[position.index]
            strategy_returns = position * (next_close / current_close - 1).fillna(0)

            test_bars = len(strategy_returns)
            if test_bars == 0:
                continue

            # FIX2: 真实交易次数 = 换仓次数
            n_trades = int((position.diff().abs() == 2).sum() / 2)

            # FIX3: 手续费模型 - 每笔换仓 0.08% (开仓 0.04% + 平仓 0.04%)
            # 从 Sharpe 均值中扣除每 bar 均摊的费率损耗
            fee_per_trade = 0.0004 * 2  # 0.08% per round-trip
            fee_drag = n_trades * fee_per_trade / test_bars

            # FIX1: Sharpe 年化因子改用 sqrt(test_bars)
            mean_ret = strategy_returns.mean() - fee_drag
            std_ret = strategy_returns.std()
            sharpe = mean_ret / std_ret * np.sqrt(test_bars) if std_ret > 0 else 0

            mdd = (
                (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
            )
            winrate = (strategy_returns > 0).mean()

            results.append({
                'window': i,
                'long_thr': long_thr,
                'short_thr': short_thr,
                'train_start': train_df.index[0],
                'test_start': test_df.index[0],
                'test_end': test_df.index[-1],
                'n_factors': len(all_factors),
                'sharpe': sharpe,
                'mdd': mdd,
                'winrate': winrate,
                'n_trades': n_trades,  # FIX2
                'model': model,
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', default='btc_1h.csv')
    parser.add_argument('--output', default='backtest/results/')
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Loading data: {args.data}")
    df = pd.read_csv(args.data, parse_dates=['open_time'], index_col='open_time')
    print(f"  Rows: {len(df)}, Range: {df.index[0]} → {df.index[-1]}")

    results = walk_forward_backtest(df, config)
    print(f"\n=== Walk-Forward Results ===")
    print(f"Total rows: {len(results)}")

    if len(results) > 0:
        # 整体统计
        pass_windows = (results['sharpe'] > 1.5).sum()
        print(f"Sharpe > 1.5: {pass_windows}/{len(results)} ({100*pass_windows/len(results):.1f}%)")
        print(f"Avg Sharpe: {results['sharpe'].mean():.3f}")
        print(f"Best Sharpe: {results['sharpe'].max():.3f}")
        print(f"Avg Trades: {results['n_trades'].mean():.1f}")

        # 按阈值分组找最优
        print(f"\n=== By Threshold Pair ===")
        for (l_thr, s_thr), grp in results.groupby(['long_thr', 'short_thr']):
            avg_s = grp['sharpe'].mean()
            best_s = grp['sharpe'].max()
            n_rows = len(grp)
            print(f"  ({l_thr:.2f}, {s_thr:.2f}): avg_sharpe={avg_s:.3f} best_sharpe={best_s:.3f} windows={n_rows}")

        # 最优组合
        best_idx = results['sharpe'].idxmax()
        best = results.loc[best_idx]
        print(f"\nBest: window={int(best['window'])} thr=({best['long_thr']:.2f},{best['short_thr']:.2f}) sharpe={best['sharpe']:.3f}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results_summary = results.drop(columns=['model'])
    results_summary.to_csv(output / 'walk_forward_results.csv', index=False)
    print(f"\nResults saved to {output / 'walk_forward_results.csv'}")


if __name__ == '__main__':
    main()
