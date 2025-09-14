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

按日期范围下载（自动按 mode 的粒度计算 period）：

```bash
funda download --since 2010-01-01  # 当不注明--until时，默认截止至今天
funda download --since 2010-01-01 --until 2019-12-31
```

下载模式：

* 默认增量补全：若项目数据库已包含所需季度对应的数据则跳过下载

* 强制覆盖：追加 `--force`，无条件重新下载并覆盖输出文件（用于当部分公司进行回溯调整后，用户得以刷新数据库里的旧数据）

参数说明：

* `--years`/`--year` 或 `--quarters` 与 `--mode` 联合决定时间窗口；

* 提供 `--since`（可选 `--until`）时优先使用日期范围；

* `--export-colname ts_code`：导出文件保留旧列名 `ts_code`；默认输出列为 `ticker`；


#### 数据完整性检测/可视化覆盖情况

```bash
funda coverage --dataset-root data_root --by ticker
```

默认按 `ticker` 输出股票×期末日覆盖矩阵，可使用 `--by period` 按期汇总。

说明：

* quarterly 由累计值差分得到单季；

* annual 可选 `cumulative`（12-31）或 `sum4`（四季相加）；


### 离线构建（build，可选）

若已准备好以下目录结构的数据集，可用 `build` 构建 annual / quarterly 导出：

* `dataset=fact_income_single/year=YYYY/*.parquet`（最新快照）

* `dataset=fact_income_cum/year=YYYY/*.parquet`（最新快照）

* `dataset=inventory_income/periods.parquet`（已有数据的季度清单）

随后可用 `build` 构建 annual / quarterly 导出：

```bash
funda build --kinds annual,quarterly \
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
