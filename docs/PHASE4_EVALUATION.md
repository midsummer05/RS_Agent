# Phase 4 重建与评测说明

## 纠正此前状态

此前只有 Harness 框架和合成测试，缺少正式数据、24 个案例以及真实 LLM 对照结果，不能认定 Phase 4 已完成。此次按开发文档重建，运行结果另见带日期的验收报告；单元测试通过不等于模型调用成功。

## 数据和许可

- 来源：[发布方 Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)，版本 v1.1；使用人工标注，不使用弱标签冒充人工真值。
- [影像许可元数据](https://storage.googleapis.com/sen1floods11/v1.1/catalog/sen1floods11_hand_labeled_source/collection.json) 和 [标签许可元数据](https://storage.googleapis.com/sen1floods11/v1.1/catalog/sen1floods11_hand_labeled_label/collection.json) 均写明 `proprietary`。发布方提供公开研究访问，但不据此推断商业使用或再分发授权。原始与派生 TIFF 均加入 gitignore；对外发布须另核实授权。
- 选样：官方 test CSV 按事件名排序、事件内 chip ID 字典序，轮转取前 12 个，不读取标签挑选“高分案例”。每个切片光学/SAR 各一例，共 24 例；它们是 12 个成对场景，不是 24 个独立地理场景。
- 烟测：官方 valid CSV 独立取 2 个切片 × 两种传感器，共 4 例；不计入正式指标。
- `data/phase4-selection.json` 固定选样规则和 ID；manifest 记录源文件和派生文件 SHA-256、日期、波段、CRS、范围、裁切方式、预期工具、基准指标引用。
- 保留完整源切片并投影到当地 UTM；影像双线性、标签最近邻。10/20/30 米为派生评测网格，不宣称三种传感器原生分辨率。光学 13 波段中 green=2、NIR=7（零基索引），SAR VV=0。
- 标签 -1 为忽略区，0 非水体，1 水体；计分前强制检查 CRS、仿射变换和尺寸完全对齐。

## 算法与比较边界

算法版本 `deterministic-water-v2`：光学使用原始反射率波段 NDWI，默认阈值 0；SAR 使用 VV 中值滤波与第 35 百分位阈值；后处理统一开闭运算、填洞、最小连通域 9 像素。无训练、无模型权重或预训练视觉网络。RGB 回退不用于此正式库。算法实现见 `src/rs_agent/tools/remote_sensing.py`，每次报告记录源代码哈希。

LLM 只在存在两个以上合法工具分支时做工具规划，不接收影像像素或人工真值；它不再控制连续阈值。当前每个传感器的解译阶段都只有一个合法工具，因此规划器会审计性地记录 `rule_single_eligible_route`，保留验证过的规则配置，也不会发起无意义的模型请求。报告阶段固定规则。每个案例运行使用隔离 SQLite，避免前一案例经验记忆污染。

当前每个传感器/阶段只有一个合法工具，因此 100% 工具路由准确率不能证明复杂路由能力；应同时看分割指标、非法输出/回退、成本与时延。引入第二个已实现且经离线验证的工具分支后，才应重新做真实 LLM 路由对照。SAR 暗目标、光学云影等仍会导致明显错误。

## 一条命令和产物

```powershell
.venv/Scripts/python.exe scripts/run_phase4.py --prepare
```

需先安装项目依赖，并在项目 `.env` 中配置 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL` 和可选 `DEEPSEEK_BASE_URL`。不打印凭据；最多并发两例。已有数据不重复下载。`--rule-only` 不调用模型；`--smoke` 将烟测报告单独输出。

- CSV / JSON / HTML：逐例状态、IoU、F1、面积相对误差、对象数绝对误差、路由/计划合法性、重复步骤、trace 完整性、请求次数、HTTP 成功次数、token 数、耗时。
- 面积误差基于等面积像元计数之比；对象数基于有效区域四连通域。无真值水体时面积相对误差为 null，不伪造为零。
- QA 拦截保留 `waiting_human`，不自动批准来美化成功率；有掩膜的拦截案例仍计分。报告中的平均 IoU/F1 是有有效掩膜案例的宏平均，须同时查看成功率。
- `data/baselines/<case_id>.json` 保存首次固定规则基准；重复运行不会覆盖它。
- evidence：`case.json`、`job.json`、`traces.json`、`llm_calls.json`、`result.json`、SQLite checkpoint、内容哈希产物（影像掩膜、矢量、统计和完成任务的 Markdown 报告）。任务没完成不伪造最终报告。
- 模型调用证据：服务商响应 ID、实际返回模型、HTTP 状态、token usage 和时延；不是 API key 存在与否。模型失败/修复/规则回退分别计数。费用金额未接入账单，不伪造货币费用。

## 当前范围之外

尚无矢量边界精度独立评测、带错误真值的 QA 召回率或独立恢复压力评测；不得从此批量运行推导这些能力达标。没有训练、没有证明跨数据集泛化，也没有将公开可下载等同于自由再分发授权。
