# A股日线 Kronos 离线检验

在本机 CPU 上运行 `NeoQuasar/Kronos-small` 的官方预训练权重，对固定的 12 只证券做滚动预测，并与持平、历史多数方向和简单动量对照。不连接券商，不下单，不改实盘规则。

## 复现

在仓库根目录执行。权重从 Hugging Face 拉取，数据从东方财富公开日线接口拉取。

```bash
python3 -m venv /tmp/kronos-venv
/tmp/kronos-venv/bin/pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
/tmp/kronos-venv/bin/pip install -r experiments/ashare_kronos/requirements-lock.txt
export HF_HOME=/tmp/kronos-hf HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH=.
/tmp/kronos-venv/bin/python -m experiments.ashare_kronos.run_experiment --stage all
```

已有 `experiments/ashare_kronos/data/*_qfq.csv` 和 `*_raw.csv` 时，可以跳过抓取：

```bash
/tmp/kronos-venv/bin/python -m experiments.ashare_kronos.run_experiment --stage forecast
/tmp/kronos-venv/bin/python -m experiments.ashare_kronos.run_experiment --stage evaluate
```

固定参数在 `config.py`：lookback 120，预测 5 个交易日，起点间隔 5 个交易日，seed 20261115，sample_count 5，T 1.0，top_p 0.9，top_k 0。证券池也在该文件中，跑分之前已写死。

## 产物

- `outputs/predictions.csv`：逐条预测
- `outputs/horizon_scores.csv`：1/3/5 日对照
- `outputs/rank_ic_by_date.csv`：截面 Rank IC
- `outputs/trade_summary.csv`：下一开盘成交的成本模拟，不能当成盈利结论
- `outputs/REPORT.md`：中文结论
- `outputs/run_log.txt`：运行日志
- `data/quality_summary.csv` 与 `data/crosscheck.csv`：数据质量和对账
