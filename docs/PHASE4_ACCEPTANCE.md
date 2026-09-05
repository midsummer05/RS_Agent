# Phase 4 重建验收记录 — 2026-09-05

## 结论与纠错

此前将“评测框架和合成测试完成”说成“Phase 4 完成”不成立：当时正式案例库为空，没有 24 例真实数据及相应真实模型评测。本次已补齐数据、执行对照、生成并校验报告。

**本次完成的是 Phase 4 的数据和评测交付，不是证明解译达到生产精度。LLM 本轮指标低于规则基线，不能宣称模型带来精度提升。**

## 可直接核验的文件

- [HTML 报告](../evaluation-output/phase4-/20260905T084252.230423Z/evaluation.html)
- [JSON 明细](../evaluation-output/phase4-/20260905T084252.230423Z/evaluation.json) / [CSV 明细](../evaluation-output/phase4-/20260905T084252.230423Z/evaluation.csv)
- [完整审计结果](../evaluation-output/phase4-/20260905T084252.230423Z/audit.json)
- [运行环境、源码与清单哈希](../evaluation-output/phase4-/20260905T084252.230423Z/run_metadata.json)
- [独立烟测报告](../evaluation-output/smoke-/20260905T084240.061842Z/evaluation.html)
- [固定选样清单](../data/phase4-selection.json) / [案例清单目录](../data/manifests) / [固定规则基准](../data/baselines)

这些产物在本机真实落盘，但运行报告和 TIFF 被 gitignore 排除，不能把仅有代码的远程仓库说成已经携带完整数据和报告。

## 对照结果

24 个正式用例在两条路径各执行一次；宏平均包括有掩膜的 QA 拦截案例，没有删除低分或零分案例。

| 指标 | 固定规则 | 真实 LLM |
| --- | ---: | ---: |
| 评测案例数 | 24 | 24 |
| 工作流完成 | 20/24（83.33%） | 18/24（75.00%） |
| 等待人工质检处理 | 4 | 6 |
| 平均 IoU | 0.300853 | 0.217172 |
| 平均 F1 | 0.368467 | 0.275257 |
| 光学平均 IoU | 0.365657 | 0.355784 |
| SAR 平均 IoU | 0.236049 | 0.078560 |
| 工具路由 / 计划合法率 | 100% / 100% | 100% / 100% |
| 已完成步骤重复执行数 | 0 | 0 |
| Trace 事件完整率 | 100% | 100% |
| 单例平均耗时（秒） | 0.723 | 51.034 |
| 模型请求 / HTTP 200 | 0 / 0 | 96 / 96 |
| 规则回退阶段 | 0 | 0 |
| 服务商返回 total_tokens 合计 | 0 | 138,160 |

模型实际返回标识为 `deepseek-v4-pro`，不是测试替身。24 例各有 4 个模型规划阶段；固定报告阶段不计调用。保存了 96 个互不相同的响应 ID、HTTP 状态、usage 和时延。未读取服务商账单，不报告货币费用。

每个传感器/阶段当前只有一个合法工具，100% 工具路由率不能证明复杂路由能力。LLM 差异主要是参数选择。例如 Spain SAR 的 LLM 将 percentile 设为 5，该次输出为空且被 QA 拦截；规则同例 IoU 为 0.798339。没有针对这个正式案例手工调整参数掩盖问题。

## 开发文档逐项核对

| Phase 4 要求 | 本次落地 |
| --- | --- |
| 光学 12 + SAR 12，另 2–4 烟测 | 24 正式 + 4 独立烟测；12 个正式切片对、2 个烟测切片对，正式覆盖 10 个事件区域 |
| 2–3 档分辨率 | UTM 10/20/30m 派生网格，每档 8 例；不冒称原生多分辨率传感器 |
| 免训练算法或明确许可预训练模型 | deterministic-water-v2：NDWI / SAR 分位数 / 形态学，无训练和权重；LLM 只规划，不接收像素或标签 |
| 清单、影像、标签、来源许可、任务、预期路由、基准 | 28 份清单、源/派生 TIFF、28 份规则基准，含日期、波段、范围、CRS、SHA-256 和变换来源 |
| 固定版本、烟测隔离、不逐例调参 | 先固定官方 test/valid 划分和 ID 再下载；独立 SQLite；未手工调整正式案例参数 |
| 规则/LLM 对照 | 48 次任务执行，真实 API 96 次成功，无规则冒充 LLM |
| 自动报告和任务证据包 | CSV/JSON/HTML、48 个证据目录、326 个通过哈希校验的产物、checkpoint、trace、请求记录 |
| 一条命令 | 下方命令获取缺失数据并执行全量对照 |

## 验证和复现

```powershell
.venv/Scripts/python.exe scripts/run_phase4.py --prepare
```

离线规则验证加 `--rule-only`；独立烟测加 `--smoke --rule-only`。读取项目 `.env`；未配置客户端时禁止宣称 LLM 对照。新运行写入新时间目录，不覆盖本次证据。

```powershell
.venv/Scripts/python.exe scripts/audit_phase4.py evaluation-output/phase4-/20260905T084252.230423Z
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src tests scripts
```

审计通过：24+4 数量/划分、源/派生文件和坐标网格、源码/清单哈希、48 份结果及产物、96 个独立成功响应、凭据未写入证据文件。单元测试 15 项通过，Ruff 通过。Docker api/worker 已重建更新；API `/health` 正常，容器内 4 个独立烟测全部完成且指标与本地一致。

## 限制

来源为发布方 [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) v1.1 公共研究下载入口；影像/标签 STAC 均标记 proprietary，不能当成自由再分发许可。TIFF 已排除出版本控制，对外分发或商业使用须另核实授权。10/20/30m 为派生网格，光学/SAR 成对，不能扩写为 24 个独立原生多分辨率场景。

未证明生产精度、跨数据集泛化、矢量边界独立精度、QA 召回率、真实故障恢复率或 LLM 优于规则。完整方法、许可链接和边界见 [评测说明](PHASE4_EVALUATION.md)。
