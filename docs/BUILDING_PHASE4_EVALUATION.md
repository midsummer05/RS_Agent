# 建筑物 Phase 4 评测库

## 固定范围

本库使用 [WHU Building Dataset 的公开 Hugging Face 镜像](https://huggingface.co/datasets/giswqs/WHU-Building-Dataset) 中 `test/Image` 和 `test/Mask` 的前 12 个字典序切片（`test_0001` 至 `test_0012`）。每张为 512×512 RGB 航片及其像素级二元建筑物掩膜；数据卡声明 0.3m GSD、0/255 掩膜编码和 CC-BY-4.0 许可。选样规则在下载与读取标签之前冻结于 `data/building-phase4-selection.json`，不按分数、覆盖率或视觉效果筛选。

影像与标签预计约 10--20MB，低于 100MB 限制；它们位于被 Git 忽略的 `data/sources/` 与 `data/samples/`，仓库只提交下载脚本、固定选择、manifest 和基线摘要。重新取得数据时运行：

```powershell
.venv/Scripts/python.exe scripts/run_building_phase4.py --prepare
```

## 数据契约和地理参考边界

每个 manifest 记录数据源、许可、任务、光学 RGB 波段、分辨率、标签类别、源与派生文件 SHA-256、固定工具、基准引用与转换来源。源 PNG 不提供采集日期、CRS 或地理范围，故 `acquisition_date` 保持空值；为让 Agent 的 GeoTIFF、面积和栅格对齐契约可以运行，转换时采用 0.3m 像元的本地度量格网（EPSG:3857、合成原点）。这是工程测试坐标，**不是** WHU 原始影像的真实位置或 CRS，不能用于地图显示或跨区域面积结论。

## 基线与报告

`deterministic-building-rgb-v1` 为不训练的 RGB 亮度/低饱和度指数，阈值只能是 Tool Registry 登记的 -0.1、0、0.1 档案；有 NIR/SWIR 的光学输入仍使用 NDBI-like 分支。统一形态学后处理与 QA 门禁随后生成掩膜、矢量、统计、报告、checkpoint 与 trace。脚本只执行规则路由，避免没有实际模型 HTTP 成功请求时声称完成 LLM 评测。

评测按完整栅格对齐后计算 IoU、F1、面积误差和连通对象数误差。固定规则首次运行把逐案例指标写入 `data/baselines/`；每次运行还在 `evaluation-output/building-phase4-/` 写 CSV、JSON、HTML 和证据包。该小型单一来源库用于验证工程闭环，不代表跨城市、跨传感器、跨季节或生产精度。

## 首次已核验运行（2026-09-05）

首次 12 例规则评测已真实运行：端到端完成率 12/12、计划合法率 12/12、trace 完整率 12/12、平均 IoU **0.196136**、平均 F1 **0.302137**，LLM 请求为 0。结果刻意保留为 RGB 免训练颜色基线的实际水平；不得将工程完成率或该单一数据源结果写成建筑物生产精度，亦不得写成 LLM 效果。

## 真实模型规划对照（2026-09-07）

使用项目 `.env` 中配置的 `deepseek-v4-pro` 对同一 12 个固定案例运行规则/LLM 对照。LLM 在预处理、解译、后处理、QA 四阶段受工具契约和有限参数档案约束，报告阶段保持确定性；模型不接收像素或标签。

| 指标 | 规则 | 真实 LLM 规划 |
| --- | ---: | ---: |
| 完成率 | 12/12 | 12/12 |
| 平均 IoU / F1 | 0.196136 / 0.302137 | 0.196136 / 0.302137 |
| 计划合法率 / trace 完整率 | 12/12 / 12/12 | 12/12 / 12/12 |
| 真实 HTTP 200 / 请求数 | 0 / 0 | 47 / 49 |
| LLM 选中阶段 / 规则回退 | 0 / 0 | 47 / 1 |
| 总 token / 平均端到端时延 | 0 / 1.23 秒 | 66,035 / 42.48 秒 |

第 7 个案例的预处理请求发生一次 `RemoteProtocolError` 和一次 `ConnectError`，因此按既定容错策略回退规则预处理；其余三个可变阶段获得 HTTP 200 并由 LLM 给出合法计划。结果仍完成且没有重复成功步骤。运行证据位于本机被忽略的 `evaluation-output/building-phase4-/20260907T091909.891287Z/`；检查了 evidence，未发现凭据文本。

两个路径指标相同并非模型未接入，而是本轮每个阶段只有一个合法工具、且模型选择了相同的已登记参数档案。该结果证明受约束规划与规则基线能得到一致且可恢复的执行，但不证明 LLM 改善 RGB 建筑物分割精度。
