# Tushare 基本面数据批量下载器

本项目旨在提供命令行脚本批量抓取 A 股上市公司基本面数据，并允许输出年度，季度累计，或者单季三种口径。

> 本仓库以多数据集批量模式为准，单一利润表旧流程仅保留历史说明。
>
> 由于该项目主要目的是全A市场全量数据下载，强烈建议使用5000积分API账户，非全量数据的下载功能尚未完善，不建议使用（本项目未实现个股枚举 fallback）

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

提示：所有命令行输出与错误信息均为中文；代码实现为英文。

## 最小可行步骤

```bash
cp .env.example .env        # 填好 TUSHARE_TOKEN
funda download              # 批量除了审计意见以外的所有数据下载并缓存（默认不导出 CSV）
funda download --audit-only # 审计意见由于需要逐个个股遍历，运行时间较长，单独运行
```

提示：缺少 `config.yml/config.yaml` 时会自动使用默认参数；日志会提示可以复制模板来自定义。

或者愿意长时间等待也可以`funda download --with-audit`，将进行默认包括审计意见数据在内的全量下载

### 多数据集批量下载

`funda download` 支持直接针对多个数据集批量抓取：

```bash
funda download --datasets income balancesheet cashflow forecast express \
  dividend fina_indicator fina_audit fina_mainbz disclosure_date \
  --vip --data-dir data --since 2010-01-01
```

要长期保存增量游标，可在配置文件中启用：

```yaml
datasets:
  - name: income            # 利润表
    report_types: [1]       # 报表类型：合并报表
  - name: balancesheet      # 资产负债表
    report_types: [1]       # 报表类型：合并报表
  - name: cashflow          # 现金流量表
    report_types: [1]       # 报表类型：合并报表
  - name: forecast          # 业绩预告
  - name: express           # 业绩快报
  - name: dividend          # 分红信息
  - name: fina_indicator    # 财务指标数据
  - name: fina_audit        # 财务审计意见
  - name: fina_mainbz       # 主营业务构成
    type: ["P", "D", "I"]   # 按产品、领域、行业
  - name: disclosure_date   # 财报披露计划
data_dir: "data"
use_vip: true               # 只有 Tushare VIP 版的 token 才支持批量下载
max_per_minute: 90
max_retries: 3
recent_quarters: 4          # 刷新最近的季度数 (建议 2-4 覆盖常见改期)
```

运行后输出位于 `data/<dataset>/year=YYYY/part-*.parquet`，增量状态默认写入 `data/_state/state.json`（JSON 后端），并滚动补齐最近 `recent_quarters` 个季度后继续增量下载。

凡是下载列表中包含 `income` 的任务，会按照配置里的导出选项自动生成 `cumulative/single/annual` CSV；其他数据集保持 parquet 形式，可按需再运行 `funda export`。

注意：若需统一产出目录，可在 `download`、`coverage`、`export` 命令中同时显式指定 `--data-dir`／`--dataset-root` 指向同一路径（如 `data`），避免历史目录造成混淆。

## 可选附加功能

1. 检查缓存数据是否完整：`funda coverage`

   * 默认读取 `data/`。如使用自定义目录，可通过 `--dataset-root` 指定。

2. 在已经有缓存数据的情况下，重新导出 CSV 或改写格式：`funda export`

   * 默认读取 `data/`，可借由 `--dataset-root` 与 `--out-dir` 将输入/输出统一到自定义目录（例如 `data/` 与 `data/export`）。

3. 查看/维护增量状态与失败清单：`funda state show|clear|ls-failures`

### 进度展示

下载和导出都可能需要较长时间，默认会根据终端能力自动选择进度输出：

```bash
# 默认 auto：TTY 下显示 Rich 进度条，重定向/CI 时退化为纯文本节奏提示
funda download --progress auto

# 固定使用 Rich 进度条或强制纯文本/关闭进度
funda download --progress rich
funda download --progress plain
funda download --progress none

# 导出命令同样支持 --progress
funda export --progress plain
```

在进度条模式下，命令会展示任务总数、成功/失败计数和预计剩余时间；纯文本模式则每隔约 1.5 秒输出一次累计进度，兼容日志采集与重定向场景。

### 状态后端与 CLI 限制

* 默认状态后端：

  * JSON：若未显式指定，会将状态写入 `<data_dir>/_state/state.json`。多数据集模式下 `data_dir` 默认为 `data`。

  * SQLite：当仓库根目录存在 `meta/state.db` 时，`--backend auto` 会自动切换为 SQLite；也可手动通过 `--backend sqlite --state-path meta/state.db` 指定。

* 自定义路径：任一后端均可通过 `--state-path` 覆盖默认位置。指定 `.db` 后缀时会强制使用 SQLite，其余文件视为 JSON。

* 操作限制：`funda state` 支持 `show`、`clear`、`set`、`ls-failures` 四种操作。其中 `set` 仅支持 JSON 状态文件；对 SQLite 后端执行会返回错误提示。失败清单始终位于 `<data_dir>/_state/failures/`。

> ### 备注
>
> 上述指令默认下载，检查，并导出近10年数据（即最近40个季度的季度累计合并报表数据，也就是tushare API的默认返回格式），但是必要时可通过指令的附加选项来微调行为，例如只下载近五年的数据（--year 5，代码会自动转化为最近可供下载的20个季度），也可直接指定想要下载的季度数，这些选择可以通过指令的选项或者更改config.yaml的设置方式来达成

## 使用流程指南

### 配置

* `config.yml` 或 `config.yaml`（根目录）：CLI 行为（模式、时间范围、输出目录、字段选择等）。

  * 初次使用：从模板复制一份并按需修改：`cp config.example.yaml config.yml`

  * 若两者都存在，程序会拒绝继续并要求只保留一个；若均不存在，会打印提示并使用内建默认值。

* `.env`（本地）：环境变量文件，至少包含 Tushare Token。

  * 初次使用：从模板复制并填入 token：`cp .env.example .env`

  * 说明：支持以下变量：

    * `TUSHARE_TOKEN`：主凭证，必填。

    * `TUSHARE_TOKEN_2`：如需配置第二个 TuShare 凭证以实现轮询调用，可选填。

    * `TUSHARE_VIP_TOKENS`：可选。逗号分隔的 token 列表，用于显式指定哪些凭证用于 VIP 接口轮询。未设置时将根据积分（≥5000）自动识别。

    * `TUSHARE_DETECT_VIP`：可选。设为 `false`/`0` 可禁用自动识别 VIP 凭证；禁用后若未指定 `TUSHARE_VIP_TOKENS`，默认仅第一个 token 用于 VIP 调用。

    * `TUSHARE_API_KEY`：旧变量名。若设置且 `TUSHARE_TOKEN` 留空，会自动映射为主凭证。

  * 程序会通过 python-dotenv 自动读取 `.env`，或从 shell 环境变量读取（变量名同上）。

  * 若配置了多个 token，程序会在启动时打印积分检测结果，仅对满足门槛的 token 使用 VIP 接口。未检测到合格凭证且 `--use-vip` 为真时会直接报错退出，避免运行中途触发权限异常。

* `.envrc`（本地，可选）：开发环境自动化脚本（需安装 direnv）。

  * 初次使用：从模板复制并授权：`cp .envrc.example .envrc && direnv allow`

  * 说明：会自动加载 `.env`，监听关键文件变更，并创建/启用本地虚拟环境 `.venv`（优先使用 uv）。

### 安装

本项目基于 Python 3.10—3.12 运行（`>=3.10,<3.13`，详见 `pyproject.toml`），其余依赖亦在其中声明。

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

* 仅补缺：将 `--recent-quarters` 设为 0（或配置 `recent_quarters: 0`），跳过滚动刷新，仅抓取缺失组合

参数说明：

* 时间窗口优先级：`--since/--until` > `--quarters` > `--years`（默认 10）。所有下载均按季度粒度（0331/0630/0930/1231）。`--years N` 会自动折算为自“最近一个可披露季度”向前回溯的 `N×4` 个季度。

* 提供 `--since`（可选 `--until`）时优先使用日期范围；

* `--datasets income balancesheet ...`：直接在命令行启用多数据集模式；与配置项 `datasets` 互补。

* `--data-dir DIR`：多数据集输出目录（默认 `data`）。状态与失败列表亦存储在该目录下的 `_state/`。

* `--use-vip`：显式启用 VIP 接口（默认启用）；`--max-per-minute N` 控制限速窗口（默认 90）。如需禁用 VIP，可在配置文件中设置 `use_vip: false`。

* `--report-types 1,6`：指定报表 `report_type`（逗号或空格分隔），默认仅下载 `1`（合并报表）。

* `--recent-quarters N`：滚动刷新最近 N 个季度（默认 8，设为 0 表示纯补缺）。配置文件中的同名字段共享该默认值。

* `--max-retries N`：接口异常时最多重试 N 次（默认 3，设为 0 表示只尝试一次）。

* `--state-path PATH`：覆盖增量状态文件位置；JSON 默认 `<data_dir>/_state/state.json`，SQLite 后端默认 `meta/state.db`。

* 默认会依据披露截止日裁掉未来季度，如需强制包含可加 `--allow-future`；

* `--no-export` / `--export`：显式关闭或开启下载结束后的导出流程（默认仅缓存 parquet，不触发导出）；`--export-format`、`--export-out-dir`、`--export-kinds`、`--export-years`、`--export-annual-strategy` 用于自定义导出行为。

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

说明：下载口径固定为“按季度期末日的累计（YTD）值”，默认会生成按年份分区的 parquet 数据集（如 `income`、`balancesheet` 等），不会自动导出 CSV；需要年度/季度/单季 CSV 或其它数据集的扁平化表格时，请在下载完成后运行 `funda export`。

#### 数据完整性检测/可视化覆盖情况

```bash
funda coverage
```

默认会先输出覆盖率摘要，再按 `--by` 生成矩阵（数值约定：`1=覆盖`、`0=缺口`、`-1=豁免`）。追加 `--csv path/to/gaps.csv` 可导出缺口清单。数据集根目录默认为 `data`，可用 `--dataset-root` 指定。`--years` 可调整年份窗口（默认近 10 年）。

说明：

* single 由累计值差分得到单季；

* cumulative 可直接导出季度累计值；

* annual 可选 `cumulative`（12-31）或 `sum4`（四季相加）；

#### 数据导出成csv

一般来说，默认情况下，`funda download` 只负责抓取数据并写入 parquet 缓存。如需生成 CSV，可在缓存就绪后执行 `funda export`，构建年度（`annual`）/季度累计（`cumulative`）/单季（`single`）口径，并将其与其余数据集一并导出为平面 CSV。

* `dataset=fact_income_cum/year=YYYY/*.parquet`（最新快照）

* `dataset=inventory_income/periods.parquet`（已有数据的季度清单）

随后可用 `export` 构建 annual / single / cumulative 导出：

```bash
funda export --kinds annual,single,cumulative \\
  --annual-strategy cumulative \\
  --dataset-root data --out-format csv --out-dir data/export
```

同样默认读取 `data` 目录下的数据集，并导出最近 10 年（若在 `download` 阶段未指定 `--export-years` 则默认沿用下载窗口）。若手动指定的窗口大于缓存中已有的季度范围，CLI 会提示并仅导出当前目录下已构建的数据。可通过 `--dataset-root`（示例中切换到 `data/`）、`--years` 或 `--out-format` 参数微调。
导出的结果统一使用 `ts_code` 作为证券主键，并按 `ts_code`、`end_date` 排序。

### 兼容旧版单数据集流程

* 历史流程沿用 `out/` 目录作为默认输入/输出；如需复刻旧目录结构，可在命令中显式设置 `--dataset-root out` 或 `--out-dir out`。
* `--raw-only`：只下载原始数据，不构建数仓派生层。
* `--build-only`：跳过下载，仅由现有原始数据构建数仓派生层。
* 旧版存在 `--skip-existing`、`--force` 等覆盖策略开关，现已被统一的 `recent_quarters` 配置取代。

## 常见问题

* CLI运行帮助

    ```bash
    funda download --help
    ```

* 字段漂移导致读失败？新增列应设为 nullable，读取时使用统一 schema 合并。

## 数据格式

### 利润表 / 资产负债表 / 现金流量表

1. Tushare的三张报表实际上提供了12种不同的类型，本项目走的是tushare的默认版本，也就是合并报表，但是可以通过修改根目录的config.yaml，或者临时加入指令选项来改变该项目的下载行为，例如--report-types 6，意味着下载母公司报表类型

2. 当系统检测到config.yaml指定的下载类型，和实际在项目缓存的数据类型（在 `data/parquet` 或自定义 `--dataset-root` 下的已有数据类型）不符时，将做出提示，如用户确认将按config.yaml指定的下载类型进行下载，可通过附加指令选项来完成

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

### 主营业务构成

主营业务构成有三种模式，按产品，按地区，按行业，该项目默认三种模式全量下载，也可以通过修改config.yaml文件来调整仅按其中一种或者两种形式下载数据

## 运行代码测试

```bash
pytest
# 如需覆盖率：pytest --cov=src --cov-report=term-missing
```

## 贡献指南

欢迎通过 Issue 或 Pull Request 参与协作。提交前请参考仓库根目录的 [AGENTS.md](./AGENTS.md) 获取代码风格、测试与提交流程约定。

## 许可协议

本项目使用 [MIT License](./LICENSE)。
