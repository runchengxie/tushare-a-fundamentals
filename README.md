# A股利润表批量下载工具

本项目提供一个命令行脚本，基于 TuShare 批量/单票抓取中国 A 股上市公司利润表数据，并统一输出三种口径：原始（raw）、单季（single）、以及 TTM（ttm）。

提示：所有命令行输出与错误信息均为中文；代码实现为英文。

## 依赖与环境
- Python 3.9+
- 依赖：tushare、pandas、pyyaml、numpy
- TuShare Token：优先读取环境变量 `TUSHARE_TOKEN`，也可通过 `--token` 传入

安装依赖（推荐使用 uv 或 pip）：
```
pip install -e .
```

## 配置文件
参考 `config.yml` 示例，CLI 参数可覆盖配置值。

关键配置项：
- mode: annual | semiannual | quarterly | ttm
- years 或 quarters：抓取范围（二选一）
- ts_code：为空则全市场 VIP 路径
- vip: true|false：是否走 income_vip（全市场）
- prefer_single_quarter: true|false：优先请求单季报表
- fields：字段列表（包含标识列与流量列）
- outdir / prefix / format(csv|parquet)
- token：可选，优先使用环境变量

## 使用示例
- 全市场，季度，近 40 季：
```
python app.py --mode quarterly --quarters 40 --vip --prefer-single-quarter
```
- 全市场，年度，近 12 年，存 parquet：
```
python app.py --mode annual --years 12 --vip --format parquet
```
- 单票，TTM，近 24 季：
```
python app.py --mode ttm --ts-code 600000.SH --quarters 24
```

## 输出产物
- 原始：`*_raw.(csv|parquet)`（经去重、规范化 end_date）
- 单季：`*_single.(csv|parquet)`（quarterly/ttm）
- TTM：`*_ttm.(csv|parquet)`（ttm）

命名：
- 全市场：`{prefix}_vip_{mode}_{kind}.{ext}`
- 单票：`{prefix}_{ts_code}_{mode}_{kind}.{ext}`

## 口径与处理规则（摘要）
- 去重：同 (ts_code, end_date) 优先 report_type=1；其次 f_ann_date、ann_date 降序；update_flag=Y 优先
- 单季：若单季不可得，对流量字段在同年内相邻期差分；Q1 保留原值
- TTM：在单季口径上滚动 4 季求和；不足 4 季为空
- 流量字段白名单：total_revenue, revenue, total_cogs, operate_profit, total_profit, income_tax, n_income, n_income_attr_p, ebit, ebitda, rd_exp

## 异常与重试
- 指数退避重试：max_tries=5, base_sleep=0.8s
- 典型错误：缺少 token、接口空返回、权限不足、网络错误。均会打印中文错误并以非零码退出。

## 许可证
MIT
