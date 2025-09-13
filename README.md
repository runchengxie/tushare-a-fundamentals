# Tushare 基本面数据批量下载器

本项目旨在实现“按年分区 + 主键去重 + 最新快照标记”的分区化 Parquet 数据集与轻量状态管理；并提供命令行脚本批量/单票抓取 A 股上市公司基本面数据，统一输出年度、单季（非累计）与 TTM 三种口径。

当前已实现并稳定可用：利润表（raw/单季/TTM）；其他数据集（资产负债表、现金流、业绩预告/快报、分红、指标、审计意见、主营构成、披露日期）为待办项。

提示：所有命令行输出与错误信息均为中文；代码实现为英文。

可供参考的Tushare API基本面数据下载API文档：

* [利润表](https://tushare.pro/document/2?doc_id=33)

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

### 运行 CLI

```bash
# 建议方式（入口脚本，推荐新名称）
funda --help

# 等价方式（模块运行）
python -m tushare_a_fundamentals.cli --help

```

示例：全市场批量（VIP）：

```bash
funda --mode quarter --years 3 --vip
```

单票下载并输出 TTM：

```bash
funda --mode ttm --ts-code 600000.SH --years 5
```

### 分区化数据集写入（可选）

若希望将“最新快照（或全量历史）”写入按年分区的 Parquet 数据集，可提供数据集配置并指定数据根目录：

```bash
funda --mode quarter --years 3 --vip \
  --datasets-config configs/datasets.yaml \
  --dataset-root data_root
```

写入路径示例：`data_root/dataset=income/year=2023/part-*.parquet`

注：目前仅对利润表 raw 表接入了分区化写入；`single/ttm` 仍按平铺文件输出。

## 配置

- `config.yml`：CLI 行为（模式、时间范围、输出目录、字段选择等）。
- `configs/datasets.yaml`：数据湖规范（分区、主键、版本字段等）。

## 开发约定

- 去重与最新快照标记：`src/tushare_a_fundamentals/transforms/deduplicate.py`
- 分区化写入：`src/tushare_a_fundamentals/writers/dataset_writer.py`
- 状态管理（雏形，尚未接入 CLI）：`src/tushare_a_fundamentals/meta/state_store.py`

## 常见问题

- 为什么保留历史版本而不直接覆盖？便于追溯与审计，同时快照查询通过 `is_latest=1` 不受影响。
- 小文件太多怎么办？可后续使用 compactor（计划中）定期合并文件。
- 字段漂移导致读失败？新增列应设为 nullable，读取时使用统一 schema 合并。
