#!/usr/bin/env python3
"""
train.py — CCI LightGBM 模型训练

用法:
    python scripts/train.py --data btc_1h.csv --output models/
    python scripts/train.py --config config/lightgbm_model.yaml
"""

import argparse
import pickle
import yaml
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.cci_lightgbm_signals import build_factor_library, ic_filter


def train(data_path: str, output_dir: str, config_path: str = None):
    """训练 LightGBM 模型"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=['open_time'], index_col='open_time')

    print("Building factor library (49 factors)...")
    factors = build_factor_library(df)
    print(f"  Total factors: {factors.shape[1]}")

    # 目标变量：下根K线收益 > 0
    returns = df['close'].pct_change().shift(-1)
    target = (returns > 0).astype(int)

    print("Running IC filter (|IC| > 0.05)...")
    good_factors = ic_filter(factors, returns, threshold=0.05)
    print(f"  Selected factors: {len(good_factors)} / {factors.shape[1]}")
    print(f"  Factors: {good_factors}")

    # 对齐数据
    X = factors[good_factors].dropna()
    y = target.loc[X.index].dropna()
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    print(f"  Training samples: {len(X)}")

    # 模型参数
    params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    }

    print(f"Training LightGBM with params: {params}")
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(**params)
    model.fit(X, y)

    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model': model,
        'feature_cols': good_factors,
        'params': params,
        'trained_at': datetime.now().isoformat(),
        'ic_filter_threshold': 0.05,
    }

    model_path = output_path / 'lightgbm_cci_btc_1h.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {model_path}")
    print(f"  Features: {good_factors}")
    print(f"  Training accuracy: {model.score(X, y):.4f}")

    return model_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CCI LightGBM model')
    parser.add_argument('--data', required=True, help='Path to OHLCV CSV file')
    parser.add_argument('--output', default='models/', help='Output directory')
    parser.add_argument('--config', help='Path to model config YAML')
    args = parser.parse_args()

    train(args.data, args.output, args.config)
