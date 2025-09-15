# Tushare 基本面数据批量下载器

本项目旨在提供命令行脚本批量抓取 A 股上市公司基本面数据，并允许输出季度累计（Q1/H1/Q3/FY）或者单季两种口径。

## 批量下载功能开发进度

* [x] 利润表（正在开发）

* [ ] 资产负债表

* [ ] 现金流量表

* [ ] 业绩预告

* [ ] 业绩快报

* [ ] 分红送股

* [ ] 财务指标数据

* [ ] 财务审计意见

* [ ] 主营业务构成

* [ ] 财报披露计划

提示：所有命令行输出与错误信息均为中文；代码实现为英文。

## 使用流程指南

### 配置

* `config.yml`（根目录）：CLI 行为（模式、时间范围、输出目录、字段选择等）。

    * 初次使用：从模板复制一份并按需修改：`cp config.example.yaml config.yml`

* `.env`（本地）：环境变量文件，至少包含 Tushare Token。

    * 初次使用：从模板复制并填入 token：`cp .env.example .env`

    * 说明：程序会通过 python-dotenv 自动读取 `.env`，或从 shell 环境变量读取（变量名：`TUSHARE_TOKEN=<your_tushare_token>`）

* `.envrc`（本地，可选）：开发环境自动化脚本（需安装 direnv）。

    * 初次使用：从模板复制并授权：`cp .envrc.example .envrc && direnv allow`

    * 说明：会自动加载 `.env`，监听关键文件变更，并创建/启用本地虚拟环境 `.venv`（优先使用 uv）。

### 安装

本项目所需python版本和依赖都以pyproject.toml的方式进行管理。

```bash
# 使用 pip 可编辑安装
pip install -e .

# 或使用 uv 安装项目与开发依赖
uv sync
```

### 使用方法

#### 数据下载

```bash
# 下载全市场最近 10 年（默认）
funda download
```

按日期范围下载（按季度粒度自动计算 period）：

```bash
funda download --since 2010-01-01  # 当不注明--until时，默认截止至今天
funda download --since 2010-01-01 --until 2019-12-31
```

下载模式：

* 默认增量补全：若项目数据库已包含所需季度对应的数据则跳过下载

* 强制覆盖：追加 `--force`，无条件重新下载并覆盖输出文件（用于当部分公司进行回溯调整后，用户得以刷新数据库里的旧数据）

参数说明：

* 时间窗口优先级：`--since/--until` > `--quarters` > `--years`（默认 10）。所有下载均按季度粒度（0331/0630/0930/1231）。

* 提供 `--since`（可选 `--until`）时优先使用日期范围；

* `--export-colname ts_code`：导出文件保留旧列名 `ts_code`；默认输出列为 `ticker`；
* `--report-types 1,6`：指定报表 `report_type`（逗号分隔），默认仅下载 `1`（合并报表）；

全量下载（建议）：

* 通过日期范围或年数覆盖全量历史。例如：

    ```bash
    # 从 2000 年至今按季度抓取
    funda download --since 2000-01-01

    # 或按近 30 年抓取
    funda download --years 30
    ```

说明：下载口径固定为“按季度期末日的累计（YTD）值”，当前版本仅导出原始去重表 `raw`。

#### 数据完整性检测/可视化覆盖情况

```bash
funda coverage --dataset-root data_root --by ticker
```

默认按 `ticker` 输出股票×期末日覆盖矩阵，可使用 `--by period` 按期汇总。

说明：

* single 由累计值差分得到单季；

* cumulative 可直接导出季度累计值；

* annual 可选 `cumulative`（12-31）或 `sum4`（四季相加）；

#### 数据导出成csv

若已准备好以下目录结构的数据集，可用 `export` 构建 annual / single / cumulative 导出：

* `dataset=fact_income_cum/year=YYYY/*.parquet`（最新快照）

* `dataset=inventory_income/periods.parquet`（已有数据的季度清单）

随后可用 `export` 构建 annual / single / cumulative 导出：

```bash
funda export --kinds annual,single \
  --annual-strategy cumulative \
  --out-format csv --out-dir out/csv \
  --dataset-root data_root
```

## 开发约定

* 状态管理（雏形，尚未接入 CLI）：`src/tushare_a_fundamentals/meta/state_store.py`

## 常见问题

* CLI运行帮助

    ```bash
    funda download --help
    ```

* 字段漂移导致读失败？新增列应设为 nullable，读取时使用统一 schema 合并。

## 参考Tushare API文档
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

### 运行代码测试

```bash
pytest
# 如需覆盖率：pytest --cov=src --cov-report=term-missing
```
