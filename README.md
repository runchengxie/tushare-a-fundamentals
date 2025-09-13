# TuShare 基本面数据集构建

本项目实现 TuShare 基本面数据的分区化 Parquet 数据集与轻量状态管理。通过“按年分区 + 主键去重 + 最新快照标记”，支持长期增量与历史更正重写。

## 设计蓝图

```
             +------------------+
             |  Scheduler/CLI   |
             +---------+--------+
                       |
                plan datasets
                       v
+-----------+   fetch/transform   +-------------+
| TuShare   +--------------------->|  Ingestion  |
|  API      |   rate limit/retry  |  Workers    |
+-----------+                      +------+------+
                                          |
                                   write partitions
                                          v
                                +---------+----------+
                                | Parquet Dataset(s) |
                                |  data_root/...     |
                                +---------+----------+
                                          |
                               compact / overwrite dirty
                                          v
                                 +--------+--------+
                                 |  State Store    |
                                 | SQLite/DuckDB   |
                                 +--------+--------+
                                          |
                                         read
                                          v
                                   +------+------+
                                   |  Consumers  |
                                   |  DuckDB/BI  |
                                   +-------------+
```

## 快速开始

### 依赖

* Python 3.10+
* `pyarrow`, `pandas`, `duckdb`, `rich`, `typer`
* 环境变量：`TUSHARE_TOKEN="<your token>"`

### 安装

```bash
pip install -r requirements.txt
```

### 配置

编辑 `configs/datasets.yaml`，设定 `root`, `state_store` 以及各数据集的 `partition_by`、`primary_key`、`version_by` 等。

### 常用命令

```bash
# 全量初始化指定数据集（并发受限流保护）
python -m etl.cli init --dataset income

# 每日增量（自动计算缺口）
python -m etl.cli daily --since 2020-01-01

# 回填/重写历史分区
python -m etl.cli rewrite --dataset income --year 2019

# 合并小文件
python -m etl.cli compact --dataset income --year 2022 --target-mb 128

# 导出最新快照为单表
python -m etl.cli materialize --dataset income --view latest --to data_root/materialized/income_latest.parquet
```

### CLI 选项要点

* `--concurrency` 并发抓取数
* `--rate-limit` 每秒请求上限
* `--since/--until` 限定抓取时间窗
* `--dirty` 仅处理被标记为脏的分区
* `--verify-only` 只做校验不写入

## 开发约定

* 写入逻辑统一走 `writers/dataset_writer.py`，屏蔽分区与覆盖细节；
* 去重逻辑集中在 `transforms/deduplicate.py`，输入主键与版本字段，输出 `is_latest`；
* 状态更新由 `meta/state_store.py` 管理；
* 任何 schema 变更需同步更新 `configs/datasets.yaml` 与 `MANIFEST.json`。

## 数据质量校验

* 主键非空率 100%；
* 日期字段可解析且位于合理区间；
* 最新快照内不应存在同主键多行；
* 行数波动超过基线阈值需发出预警。

## 常见问题

* **为什么保留历史版本而不直接覆盖？** 便于追溯与审计，同时快照查询通过 `is_latest=1` 不受影响。
* **小文件太多怎么办？** 调大批量写入条数或定期运行 `compact` 合并；
* **字段漂移导致读失败？** 新增列应设为 nullable，读取时使用统一 schema 合并。

## 故障恢复

* 采集失败的分区不会更新状态表，下一轮会再次计划；
* 标记 `dirty=1` 的分区必将被覆盖重写；
* 若写入中断，遗留的临时文件名带 `.tmp`，下次启动会清理。

## 许可证

遵守 TuShare 使用条款；本仓库不包含任何访问秘钥或私有数据。

## 致谢

* PyArrow 社区
* DuckDB 社区

