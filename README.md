# A股利润表批量下载工具

本项目提供一个命令行脚本，基于 TuShare 批量/单票抓取中国 A 股上市公司利润表数据，并统一输出三种口径：原始（raw）、单季（single）、以及 TTM（ttm）。

提示：所有命令行输出与错误信息均为中文；代码实现为英文。

## 依赖与环境

* Python 3.10+

* 依赖：tushare、pandas、pyyaml、numpy、python-dotenv、pyarrow（parquet 支持）
  - 若环境中未自动安装，可执行 `pip install pyarrow`

* TuShare Token：优先读取环境变量 `TUSHARE_TOKEN`，也可通过 `--token` 传入
  - 不使用 direnv 时，可复制并编辑 `.env`：`cp .env.example .env`

### 可选安装

* 安装 direnv并配置 shell自动加载环境：

    * 第一步：安装direnv并配置 shell：

        * bash: `echo 'eval "$(direnv hook bash)"' >> ~/.bashrc`

        * zsh: `echo 'eval "$(direnv hook zsh)"'  >> ~/.zshrc`

        * fish: `echo 'direnv hook fish | source' >> ~/.config/fish/config.fish`

    * 第二步，启动`.envrc`脚本

    ```bash
    cp .envrc.example .envrc
    direnv allow
    ```
    `.envrc` 会自动加载同目录下的 `.env`（若存在）。

* 安装 uv：

    * 可用包管理器或 `pipx install uv`（项目要求 `uv >= 0.8.0`）


### 安装（可编辑模式）

```bash
# 使用pip
pip install -e .

# 使用uv
uv sync
```

### 快速开始（Quickstart）

```bash
# 查看帮助
income-downloader --help

# 全市场季度，最近 40 季，优先单季口径（默认 vip: true，批量路径调用 income_vip）
income-downloader --mode quarter --quarters 40 --vip --prefer-single-quarter

# 再次运行时若文件已存在并完整，可跳过下载
income-downloader --mode quarter --quarters 40 --vip --skip-existing

# 单票 TTM，最近 24 季
income-downloader --mode ttm --ts-code 600000.SH --quarters 24

# 年度，最近 12 年（默认 parquet，需已安装 pyarrow）
income-downloader --mode annual --years 12 --vip
```

或直接从源码运行：

```bash
python3 -m tushare_a_fundamentals.cli --help
```

## 配置文件

参考 `config.yml` 示例，CLI 参数可覆盖配置值。

关键配置项：

* mode: annual | quarter | ttm（`quarterly` 为兼容别名，会打印弃用警告）
* years 或 quarters：抓取范围（二选一）
* ts_code：为空则全市场 VIP 路径（示例：`600000.SH`、`000001.SZ`）
* vip: true|false：是否走 income_vip（默认 true，批量路径调用 `income_vip`）
* prefer_single_quarter: true|false：优先请求单季报表
* skip_existing: true|false：若输出文件已包含所需 end_date，则跳过下载
* fields：字段列表（包含标识列与流量列）
* outdir / prefix / format（默认 parquet，示例配置中已设为 parquet）
* token：可选，优先使用环境变量

## 输出产物

文件会写到 `out/csv` 或 `out/parquet`，并按 `{prefix}_vip_{mode}_{kind}.{ext}`（全市场）或 `{prefix}_{ts_code}_{mode}_{kind}.{ext}`（单票）命名。

* 原始：`*_raw.(csv|parquet)`（经去重、规范化 end_date）
* 单季：`*_single.(csv|parquet)`（quarter/ttm）
* TTM：`*_ttm.(csv|parquet)`（ttm）

## 口径与处理规则（摘要）

* 去重：同 (ts_code, end_date) 优先 report_type=1；其次 f_ann_date、ann_date 降序；update_flag=Y 优先
* 单季：若单季不可得，对流量字段在同年内相邻期差分；Q1 保留原值
* TTM：在单季口径上滚动 4 季求和；不足 4 季为空
* 流量字段白名单：total_revenue, revenue, total_cogs, operate_profit, total_profit, income_tax, n_income, n_income_attr_p, ebit, ebitda, rd_exp

## 异常与重试

* 指数退避重试：max_tries=5, base_sleep=0.8s
* 典型错误：缺少 token、接口空返回、权限不足、网络错误。均会打印中文错误并以非零码退出。

## 开发与测试

* 代码风格：black、ruff（见 pyproject.toml）
* 运行 lints：

```bash
ruff check .
black --check .
```

* 运行测试：

```bash
pytest
```

* 覆盖率配置见 pyproject.toml 中 [tool.pytest.ini_options]
