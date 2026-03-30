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
│   ├── signal_server.yaml
│   ├── walk_forward.yaml       # Walk-Forward 回测配置（backtest 脚本使用）
│   └── lightgbm_model.yaml
├── backtest/
│   └── walk_forward_results.md
├── signals/
│   ├── cci_lightgbm_signals.py   # 信号生成器
│   └── asymmetric_proba_grid.py # 非对称 proba 阈值（网格/实盘共用）
└── scripts/
    ├── train.py                        # 模型训练脚本
    ├── backtest.py                     # Walk-Forward 回测（固定 0.5 切分）
    ├── backtest_proba_threshold_grid.py  # Walk-Forward + (long,short) 阈值网格
    ├── c3_ml_live.py                   # C3 实时推理（内置 0.55）
    ├── c3_ml_signal_asymmetric.py      # 非对称阈值推理 + bar 字典
    ├── ml_cci_trading_server.py        # 实盘交易（ccxt，独立进程）
    └── live_server.py                  # 仅信号 + Flask（无下单）
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

### 5. 非对称 Proba 阈值网格回测（Walk-Forward）

在 **二元 proba**（涨概率）上使用多组 `(long_thr, short_thr)`，例如 `long>0.6` 做多、`proba<0.4` 做空，中间为中性。需 `pyyaml`。

```bash
pip install pyyaml
python scripts/backtest_proba_threshold_grid.py --config config/walk_forward.yaml --data btc_1h.csv \
  --long-grid 0.55,0.6,0.65 --short-grid 0.35,0.4,0.45 --output backtest/results/
```

结果：`backtest/results/proba_grid_walk_forward.csv`。

### 6. ML 实盘交易进程（Binance USDT 永续）

独立脚本，**不**依赖 `signal_notifier-main/first_trading_server.py`。信号来自 `c3_ml_signal_asymmetric`（三分类：`p_up` / `p_down` 与非对称阈值）。

**依赖**：`ccxt`, `flask`, `requests`, `lightgbm`, `pandas`, `numpy`（与 `c3_ml_live` 相同）。

**环境变量（节选）**：

| 变量 | 说明 |
|------|------|
| `BINANCE_API_KEY` / `BINANCE_SECRET` | 合约 API |
| `ENABLE_TRADING` | `1` 真实下单，`0` 仅 dry-run |
| `ML_LONG_THRESHOLD` | 做多需 `p_up >` 此值（默认 `0.6`） |
| `ML_SHORT_THRESHOLD` | 做空需 `p_down >` 此值（默认 `0.4`） |
| `ORDER_MARGIN_USD` / `FIXED_LEVERAGE` | 与常见交易脚本类似 |
| `ML_TRADING_PORT` | HTTP 端口（默认 `6671`） |
| `TRADING_DISCORD_WEBHOOK` | 可选，成交通知 |

```bash
# 建议在项目根放置 .env
set ENABLE_TRADING=0
python scripts/ml_cci_trading_server.py
# 健康检查: http://127.0.0.1:6671/health
# 当前 ML bar: http://127.0.0.1:6671/api/signal
```

首次推理若本地无缓存模型，会按 `c3_ml_live` 逻辑从 Binance 拉历史并训练，模型目录为 `~/Desktop/share/share_repository/data/`。

## 策略逻辑

1. **因子计算**：实时计算 CCI(7/14/21) + stoch_k(7) 等因子
2. **模型推理**：LightGBM 分类预测下根 K 线涨跌
3. **信号生成**：proba > 0.5 → 做多；proba < 0.5 → 做空
4. **风控**：ATR 动态止损（3× ATR）

## 验证结果

Walk-Forward: 3mo 训练 / 1mo 测试 / 30天滚动

详见：[Walk-Forward 结果](./backtest/walk_forward_results.md)
