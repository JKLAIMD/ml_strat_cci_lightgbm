# ml_strat_cci_lightgbm

CCI + LightGBM 量化交易策略 — BTC 1h 数据专用的因子驱动算法。

## 核心策略

**因子库（49个因子）→ IC筛选 → LightGBM 分类 → Walk-Forward 验证**

| 指标 | 值 |
|------|-----|
| Walk-Forward 通过率 | 18/24 窗口 (75%) |
| 平均 Sharpe | **3.90** |
| 最佳 Sharpe | 12.37 |
| 核心因子 | CCI(7/14/21), stoch_k(7) |

## 目录结构

```
ml_strat_cci_lightgbm/
├── config/
│   ├── signal_server.yaml      # 实时信号服务器配置
│   └── lightgbm_model.yaml    # 模型推理配置
├── backtest/
│   ├── walk_forward.yaml       # Walk-Forward 回测配置
│   └── factor_config.yaml      # 因子库配置
├── signals/
│   └── cci_lightgbm_signals.py  # 信号生成器
└── scripts/
    ├── train.py               # 模型训练脚本
    ├── backtest.py            # Walk-Forward 回测
    └── live_server.py         # 实时信号服务器
```

## 快速开始

### 1. 安装依赖
```bash
pip install lightgbm pandas numpy scikit-learn ta
```

### 2. 训练模型
```bash
python scripts/train.py --data btc_1h.csv --output models/
```

### 3. 跑 Walk-Forward 回测
```bash
python scripts/backtest.py --config config/walk_forward.yaml
```

### 4. 启动实时信号服务器
```bash
python scripts/live_server.py --config config/signal_server.yaml
```

## 策略逻辑

1. **因子计算**：实时计算 CCI(7/14/21) + stoch_k(7) 等因子
2. **模型推理**：LightGBM 分类预测下根 K 线涨跌
3. **信号生成**：proba > 0.5 → 做多；proba < 0.5 → 做空
4. **风控**：ATR 动态止损（3× ATR）

## 验证结果

Walk-Forward: 3mo 训练 / 1mo 测试 / 30天滚动

详见：[Walk-Forward 结果](./backtest/walk_forward_results.md)
