# Tushare 基本面数据批量下载器

本项目旨在提供命令行脚本批量抓取 A 股上市公司基本面数据，并允许输出年度，季度累计，或者单季三种口径。

## 批量下载功能开发进度

> 现已提供统一调度器，可按期优先使用 TuShare VIP 接口抓取以下数据集，非 VIP 接口则按月窗口滚动补齐。

* [x] 利润表
* [x] 资产负债表
* [x] 现金流量表
* [x] 业绩预告
* [x] 业绩快报
* [x] 分红送股
* [x] 财务指标数据
* [x] 财务审计意见
* [x] 主营业务构成（默认抓取 type=P/D）
* [x] 财报披露计划

提示：所有命令行输出与错误信息均为中文；代码实现为英文。

## 运行原理/快速运行指南

1. 缓存原始表、构建 parquet 数仓，并同步导出 CSV：`funda download`

### 多数据集批量下载（VIP 优先）

`funda download` 现在支持直接针对多个数据集批量抓取：

```bash
funda download --datasets income balancesheet cashflow forecast express \
  dividend fina_indicator fina_audit fina_mainbz disclosure_date \
  --use-vip --data-dir data --since 2010-01-01
```

要长期保存增量游标，可在配置文件中启用：

```yaml
datasets:
  - name: income
    report_types: [1]
  - balancesheet
  - cashflow
  - forecast
  - express
  - dividend
  - fina_indicator
  - fina_audit
  - name: fina_mainbz
    type: ["P", "D"]
 - disclosure_date
data_dir: data
use_vip: true
max_per_minute: 80
max_retries: 3
recent_quarters: 8  # 按季度回刷窗口
```

运行后输出位于 `data/<dataset>/year=YYYY/part-*.parquet`，增量状态写入 `data/_state/state.json`，默认滚动补齐最近 `recent_quarters` 个季度并继续增量下载。

注意：`--raw-only`、`--build-only`、`--force` 仅适用于旧版利润表流程，若在多数据集模式中使用会被忽略或报错。

## 可选附加功能

1. 检查缓存数据是否完整：`funda coverage`

2. 在已经有缓存数据的情况下，重新导出 CSV 或改写格式：`funda export`

> ### 备注
>
> 上述指令默认下载，检查，并导出近10年数据（即最近40个季度的季度累计合并报表数据，也就是tushare API的默认返回格式），但是必要时可通过指令的附加选项来微调行为，例如只下载近五年的数据（--year 5，代码会自动转化为最近可供下载的20个季度），也可直接指定想要下载的季度数，亦或者近下载某个股票代码的数据（--ticker 600036.SH），这些选择可以通过指令的选项或者更改config.yaml的设置方式来达成

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

* 默认补缺 + 滚动刷新：若检测到缺口会补齐历史数据，并额外重抓最近 8 个季度（可用 `--recent-quarters` 调整）

* 仅补缺：追加 `--skip-existing`，跳过滚动刷新，仅抓取缺失组合

* 强制覆盖：追加 `--force`，无条件重新下载并覆盖输出文件（用于当部分公司进行回溯调整后，用户得以刷新数据库里的旧数据）

参数说明：

* 时间窗口优先级：`--since/--until` > `--quarters` > `--years`（默认 10）。所有下载均按季度粒度（0331/0630/0930/1231）。`--years N` 会自动折算为自“最近一个可披露季度”向前回溯的 `N×4` 个季度。

* 提供 `--since`（可选 `--until`）时优先使用日期范围；

* `--report-types 1,6`：指定报表 `report_type`（逗号分隔），默认仅下载 `1`（合并报表）；

* `--recent-quarters N`：滚动刷新最近 N 个季度（默认 8，设置为 0 可完全关闭刷新窗口）；

* `--max-retries N`：接口异常时最多重试 N 次（默认 3，设为 0 表示只尝试一次）；

* `--skip-existing`：仅补缺，不进行滚动刷新；

* `--raw-only`：只下载 raw，不构建数仓；

* `--build-only`：跳过下载，仅由已有 raw 构建数仓；

* 默认会依据披露截止日裁掉未来季度，如需强制包含可加 `--allow-future`；

* `--no-export` / `--export`：关闭或显式开启派生数据导出（默认开启，失败时仅警告）；

* `--export-format` / `--export-out-dir`：调整导出格式（默认 csv）与输出目录；

* `--export-kinds` / `--export-years` / `--export-annual-strategy`：微调年度/单季/季度累计导出口径与年份窗口；

* 每次下载如遇失败 period/window，会在 `data/_state/failures/` 下生成对应 JSON 清单，方便后续优先补齐。

* `--strict-export`：导出失败时返回非零状态码（默认仅记录警告并继续）。

全量下载（建议）：

* 通过日期范围或年数覆盖全量历史。例如：

    ```bash
    # 从 2000 年至今按季度抓取
    funda download --since 2000-01-01
    
    # 或按近 30 年抓取
    funda download --years 30
    ```

说明：下载口径固定为“按季度期末日的累计（YTD）值”，默认会生成 `dataset=inventory_income` 与 `dataset=fact_income_cum` 两套 parquet 数仓，并自动导出年度/季度累计/单季 CSV；仅需原始去重表时可追加 `--raw-only` 或 `--no-export`。

#### 数据完整性检测/可视化覆盖情况

```bash
funda coverage --by ticker
```

默认会先输出覆盖率摘要，再按 `--by` 生成矩阵（数值约定：`1=覆盖`、`0=缺口`、`-1=豁免`）。追加 `--csv path/to/gaps.csv` 可导出缺口清单。数据集根目录默认为 `out`，可用 `--dataset-root` 指定，`--years` 可调整年份窗口（默认近 10 年）。

说明：

* single 由累计值差分得到单季；

* cumulative 可直接导出季度累计值；

* annual 可选 `cumulative`（12-31）或 `sum4`（四季相加）；

#### 数据导出成csv

若已完成数据下载并且通过数据完整性检验确认后已有想要的数据，可用 `export` 构建按年度数据（指令：annual）/ 季度累计（指令：cumulative）/单季（指令：single）导出：

* `dataset=fact_income_cum/year=YYYY/*.parquet`（最新快照）

* `dataset=inventory_income/periods.parquet`（已有数据的季度清单）

随后可用 `export` 构建 annual / single / cumulative 导出：

```bash
funda export --kinds annual,single,cumulative \\
  --annual-strategy cumulative \\
  --out-format csv --out-dir out/csv
```

同样默认读取 `out` 目录下的数据集，并导出最近 10 年（若在 `download` 阶段未指定 `--export-years` 则默认沿用下载窗口）。若手动指定的窗口大于缓存中已有的季度范围，CLI 会提示并仅导出当前目录下已构建的数据。可通过 `--dataset-root`、`--years` 或 `--out-format` 参数微调。
导出的结果统一使用 `ts_code` 作为证券主键，并按 `ts_code`、`end_date` 排序。

## 开发约定

* 状态管理（雏形，尚未接入 CLI）：`src/tushare_a_fundamentals/meta/state_store.py`

## 常见问题

* CLI运行帮助

    ```bash
    funda download --help
    ```

* 字段漂移导致读失败？新增列应设为 nullable，读取时使用统一 schema 合并。

* 历史 parquet 仍保留 `ticker` 列？运行 `python tools/migrate_ts_code_column.py --dry-run` 预览，确认后去掉 `--dry-run` 批量重命名为 `ts_code`。

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

## 数据格式

1. Tushare的三张报表实际上提供了12种不同的类型，本项目走的是tushare的默认版本，也就是合并报表，但是可以通过修改根目录的config.yaml，或者临时加入指令选项来改变该项目的下载行为，例如--report-types 6，意味着下载母公司报表类型

2. 当系统检测到config.yaml指定的下载类型，和实际在项目缓存的数据类型（在out/parquet的已有数据类型）不符时，将做出提示，如用户确认将按config.yaml指定的下载类型进行下载，可通过附加指令选项来完成

| 代码 | 类型                 | 说明                                             |
| :--- | :------------------- | :----------------------------------------------- |
| 1    | 合并报表             | 上市公司最新报表（默认）                         |
| 2    | 单季合并             | 单一季度的合并报表                               |
| 3    | 调整单季合并表       | 调整后的单季合并报表（如果有）                   |
| 4    | 调整合并报表         | 本年度公布上年同期的财务报表数据，报告期为上年度 |
| 5    | 调整前合并报表       | 数据发生变更，将原数据进行保留，即调整前的原数据 |
| 6    | 母公司报表           | 该公司母公司的财务报表数据                       |
| 7    | 母公司单季表         | 母公司的单季度表                                 |
| 8    | 母公司调整单季表     | 母公司调整后的单季表                             |
| 9    | 母公司调整表         | 该公司母公司的本年度公布上年同期的财务报表数据   |
| 10   | 母公司调整前报表     | 母公司调整之前的原始财务报表数据                 |
| 11   | 母公司调整前合并报表 | 母公司调整之前合并报表原数据                     |
| 12   | 母公司调整前报表     | 母公司报表发生变更前保留的原数据                 |

## 运行代码测试

```bash
pytest
# 如需覆盖率：pytest --cov=src --cov-report=term-missing
```
