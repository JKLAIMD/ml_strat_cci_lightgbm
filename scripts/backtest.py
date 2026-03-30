#!/usr/bin/env python3
"""
backtest.py — Walk-Forward 回测脚本

用法:
    python scripts/backtest.py --config config/walk_forward.yaml
"""

import argparse
import pickle
import yaml
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from signals.cci_lightgbm_signals import build_factor_library, ic_filter


def walk_forward_backtest(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """滚动 Walk-Forward 回测"""
    factors = build_factor_library(df)
    returns = df['close'].pct_change().shift(-1)

    train_days = config['backtest']['windows']['train_days']
    test_days = config['backtest']['windows']['test_days']
    step_days = config['backtest']['windows']['step_days']
    ic_thresh = config['backtest']['ic_filter']['threshold']

    results = []
    train_size = train_days * 24
    test_size = test_days * 24
    step_size = step_days * 24

    n_windows = (len(df) - train_size - test_size) // step_size
    print(f"Walk-Forward: {n_windows} windows")

    for i in range(n_windows):
        train_end = i * step_size + train_size
        test_end = train_end + test_size

        train_df = df.iloc[i * step_size:train_end]
        test_df = df.iloc[train_end:test_end]

        if len(train_df) < train_size * 0.8 or len(test_df) < test_size * 0.8:
            continue

        # IC 筛选
        train_factors = factors.iloc[i * step_size:train_end]
        train_returns = returns.iloc[i * step_size:train_end]
        good_factors = ic_filter(train_factors, train_returns, threshold=ic_thresh)

        if len(good_factors) < 2:
            continue

        # 训练
        X_train = train_factors[good_factors].dropna()
        y_train = train_returns.loc[X_train.index]
        y_train = (y_train > 0).astype(int)

        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        model.fit(X_train, y_train)

        # 测试
        test_factors = factors.iloc[train_end:test_end]
        X_test = test_factors[good_factors].dropna()
        y_test = returns.loc[X_test.index]
        test_close = df['close'].loc[X_test.index]

        proba = model.predict_proba(X_test)[:, 1]

        # 策略收益
        position = pd.Series(proba > 0.5, index=X_test.index).astype(int)
        position[position == 0] = -1  # short when proba < 0.5
        strategy_returns = position * test_close.pct_change().fillna(0)
        strategy_returns = strategy_returns[1:]

        sharpe = (
            strategy_returns.mean() / strategy_returns.std() * np.sqrt(24 * 365)
            if strategy_returns.std() > 0 else 0
        )
        mdd = (
            (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
        )
        winrate = (strategy_returns > 0).mean()

        results.append({
            'window': i,
            'train_start': train_df.index[0],
            'test_start': test_df.index[0],
            'test_end': test_df.index[-1],
            'n_factors': len(good_factors),
            'factors': good_factors,
            'sharpe': sharpe,
            'mdd': mdd,
            'winrate': winrate,
            'n_trades': len(position),
            'model': model,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', default='btc_1h.csv')
    parser.add_argument('--output', default='backtest/results/')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Loading data: {args.data}")
    df = pd.read_csv(args.data, parse_dates=['open_time'], index_col='open_time')
    print(f"  Rows: {len(df)}, Range: {df.index[0]} → {df.index[-1]}")

    results = walk_forward_backtest(df, config)
    print(f"\n=== Walk-Forward Results ===")
    print(f"Total windows: {len(results)}")
    if len(results) > 0:
        pass_windows = (results['sharpe'] > 1.5).sum()
        print(f"Pass (Sharpe > 1.5): {pass_windows}/{len(results)} ({100*pass_windows/len(results):.1f}%)")
        print(f"Avg Sharpe: {results['sharpe'].mean():.2f}")
        print(f"Best Sharpe: {results['sharpe'].max():.2f}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results_summary = results.drop(columns=['model'])
    results_summary.to_csv(output / 'walk_forward_results.csv', index=False)
    print(f"\nResults saved to {output / 'walk_forward_results.csv'}")


if __name__ == '__main__':
    main()
