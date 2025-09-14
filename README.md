# Tushare 基本面数据批量下载器

本项目旨在实现“按年分区 + 主键去重 + 最新快照标记”的分区化 Parquet 数据集与轻量状态管理；并提供命令行脚本批量/单票抓取 A 股上市公司基本面数据，统一输出年度、单季（非累计）与 TTM （通过滚动四个季度数据求和）三种口径。

当前正在开发的功能：利润表（annual/quarterly/ttm）

待办项：资产负债表、现金流、业绩预告/快报、分红、指标、审计意见、主营构成、披露日期

提示：所有命令行输出与错误信息均为中文；代码实现为英文。

可供参考的Tushare API基本面数据下载API文档：

* [利润表](https://tushare.pro/document/2?doc_id=33)

* [资产负债表](https://tushare.pro/document/2?doc_id=36)

* [现金流量表](https://tushare.pro/document/2?doc_id=44)

* [业绩预告](https://tushare.pro/document/2?doc_id=45)

* [业绩快报](https://tushare.pro/document/2?doc_id=46)

* [分红送股](https://tushare.pro/document/2?doc_id=103)

* [财务指标数据](https://tushare.pro/document/2?doc_id=79)

* [财务审计意见](https://tushare.pro/document/2?doc_id=80)

* [主营业务构成](https://tushare.pro/document/2?doc_id=81)

* [财报披露计划](https://tushare.pro/document/2?doc_id=162)

## 快速开始

### 依赖

* Python 3.10+

* `pandas`, `pyarrow`, `PyYAML`, `python-dotenv`

* 环境变量：`TUSHARE_TOKEN="<your token>"`

### 安装

```bash
# 使用 pip 可编辑安装
pip install -e .

# 或使用 uv 安装项目与开发依赖
uv sync
```

### 运行测试

```bash
pytest
# 如需覆盖率：pytest --cov=src --cov-report=term-missing
```

### 运行 CLI

```bash
# 建议方式（入口脚本，推荐新名称）
funda --help

# 等价方式（模块运行）
python -m tushare_a_fundamentals.cli --help
```

示例：全市场最近 3 年：

```bash
funda --years 3
```

单票下载并输出 TTM：

```bash
funda --ts-code 600000.SH --years 5
```

> 注意：以上顶层参数仅用于兼容旧版流程，推荐改用下文的 `ingest + build` 两步法。

### 两步法：ingest + build（推荐）

目标是把“触网下载”和“离线构建导出”解耦：

1) ingest 下载事实表（建议落 Parquet 数据集）

```bash
funda ingest --since 2018-01-01 --periods quarterly \
  --prefer-single-quarter \
  --dataset-root data_root
```

产物：
- `dataset=fact_income_single/year=YYYY/*.parquet`
- `dataset=fact_income_cum/year=YYYY/*.parquet`
- `dataset=inventory_income/periods.parquet`

2) build 离线构建导出 annual / quarterly / ttm

```bash
funda build --kinds annual,quarterly,ttm \
  --annual-strategy cumulative \
  --out-format csv --out-dir out/csv \
  --dataset-root data_root
```

3) coverage 可视化覆盖情况

```bash
funda coverage --dataset-root data_root --by ts_code
```

默认按 `ts_code` 输出股票×期末日覆盖矩阵，可使用 `--by period` 按期汇总。

说明：
- quarterly 直接来自单季事实表；
- annual 可选 `cumulative`（12-31）或 `sum4`（四季相加）；
- ttm 为最近四季滚动求和；

### 分区化数据集写入（可选）

若希望将“最新快照（或全量历史）”写入按年分区的 Parquet 数据集，可提供数据集配置并指定数据根目录：

```bash
funda --years 3 \
  --datasets-config configs/datasets.yaml \
  --dataset-root data_root
```

写入路径示例：`data_root/dataset=income/year=2023/part-*.parquet`

注：通过 ingest 已接入 `fact_income_single` 与 `fact_income_cum` 的分区化写入；`build` 阶段负责导出 annual/quarterly/ttm。

## 配置

- `config.yml`（根目录）：CLI 行为（模式、时间范围、输出目录、字段选择等）。
  - 初次使用：从模板复制一份并按需修改
    - `cp config.example.yaml config.yml`

- `configs/datasets.yaml`（本地）：数据湖规范（分区、主键、版本字段等）。
  - 初次使用：从模板复制一份并按需修改
    - `cp configs/datasets.example.yaml configs/datasets.yaml`
  - 仓库已忽略 `configs/datasets.yaml`，避免彼此覆盖本地路径等私有配置。

## 开发约定

* 去重与最新快照标记：`src/tushare_a_fundamentals/transforms/deduplicate.py`

* 分区化写入：`src/tushare_a_fundamentals/writers/dataset_writer.py`

* 状态管理（雏形，尚未接入 CLI）：`src/tushare_a_fundamentals/meta/state_store.py`

## 常见问题

* 为什么保留历史版本而不直接覆盖？便于追溯与审计，同时快照查询通过 `is_latest=1` 不受影响。

* 小文件太多怎么办？可后续使用 compactor（计划中）定期合并文件。

* 字段漂移导致读失败？新增列应设为 nullable，读取时使用统一 schema 合并。
