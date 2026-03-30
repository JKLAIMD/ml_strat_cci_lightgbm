# 回测用 OHLCV 数据

将 Binance 现货 1h K 线下载到此目录：

```bash
pip install pandas pyyaml
python scripts/fetch_ohlcv_for_backtest.py
```

若访问 `api.binance.com` 超时，请使用 VPN，或设置 `HTTPS_PROXY` 后再运行（见脚本文件头说明）。

默认生成 `btc_1h.csv`、`eth_1h.csv`，时间范围与 `config/walk_forward.yaml` 的 `data_range` 一致。
