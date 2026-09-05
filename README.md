# Remote Sensing Interpretation Agent

面向多源、多分辨率遥感影像的可恢复、可审计、可人工介入解译工作流 Agent。

项目以 Agent 工程能力为重点：固定生命周期、阶段内受约束的动态路由、数据流与控制流分离、可观测性、断点恢复、人工审批与评测。

开发路线见 [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md)。

## Phase 0 quick start

启动 Docker Desktop 后运行：

```powershell
docker compose up --build
```

- Web 骨架：<http://localhost:8080>
- API 文档：<http://localhost:8001/docs>

如需使用其他端口，可在启动前设定 `$env:RS_AGENT_API_PORT = "8002"`，再访问对应端口的 `/docs`。

创建任务后，worker 会执行五阶段 mock 流程。`fail_once_stage` 仅用于演示可恢复错误，例如传入 `"interpret"` 会在核心解译阶段失败一次；随后调用 `POST /jobs/{job_id}/resume`（或 `rs-agent resume {job_id}`）即可继续，已完成阶段不会重复执行。

## Phase 1 imagery

支持 PNG/JPEG、光学 GeoTIFF 和 SAR GeoTIFF 的确定性水体提取。GeoTIFF 使用 Rasterio 的 wheel API 读取、写出和矢量化；项目不调用 GDAL CLI。Docker 运行时将仓库 `data/` 以只读方式挂载到 worker 的 `/data/input`，因此提交任务时可使用例如 `/data/input/samples/optical.tif` 的 `image_uri`。

光学四波段默认把 band 2（零基索引 1）视为绿光、band 4（索引 3）视为 NIR；可通过 `band_indices` 覆盖，例如：

```json
{"sensor_type":"optical", "image_uri":"/data/input/samples/optical.tif", "band_indices":{"green":2,"nir":7}}
```

流程产出内容哈希化的预处理数组、掩膜 GeoTIFF、GeoJSON、水体统计 JSON 和 Markdown 报告。对缺少 CRS 或使用地理坐标单位的输入，面积统计保留像素/原生单位，并将平方米面积标为 `0`，避免将角度误报为面积。

## Phase 2 planning

Phase 2 在预处理、核心解译、后处理和质检阶段引入受约束 Planner。它只接收影像元数据、产物 URI/哈希、失败摘要和允许调用的工具契约，绝不接收原始影像或像素数组。Planner 输出必须符合 `ExecutionPlan` JSON Schema，并在执行前由工具注册表检查阶段、传感器、任务和参数范围。

配置 `DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL`（使用当前 DeepSeek V4 Pro 标识）后，服务端会启用 DeepSeek 的兼容 Chat Completions 调用；未配置时，或模型输出连续两次无效时，自动回退到确定性规则路由。密钥不会写入任务、checkpoint、artifact、日志或前端容器。

将 `.env.example` 复制为未提交的 `.env`，填写密钥和官方当前模型标识，然后重建 worker：

```powershell
Copy-Item .env.example .env
# 在 .env 中填写 DEEPSEEK_API_KEY 和 DEEPSEEK_MODEL；不要提交或粘贴密钥到聊天。
docker compose up --build -d worker
```

凭据仅注入 worker（Planner 所在容器），不会注入 `web`。未配置时 UI/任务 trace 中的 `deterministic rule route` 是明确的规则回退标记，并不代表 LLM 已被调用。

## Phase 3 control and audit

质检未通过时，job 会进入 `waiting_human`，不会继续生成最终报告。使用 `POST /jobs/{job_id}/approval` 或 `rs-agent approval` 提交：`approve` 接受当前结果并继续、`edit` 指定重启阶段和参数后重跑、`reject` 清除该阶段及后续计划并重新规划。`GET /jobs/{job_id}/traces` 可回放状态迁移、规划上下文摘要、计划、工具参数/耗时/错误和审批决定。

`failure_injections` 支持 `timeout`、`file_missing`、`qa_failure`、`invalid_planner_output`，配合 `injection_stage` 用于演示恢复与审计。经验记忆只记录传感器/任务层级的有效工具和参数摘要，可通过 `GET /memory/{sensor_type}/{task_type}` 查询；不存影像、像素、完整 trace 或密钥。

## Phase 4 evaluation

2026-09-05 重建后的实际结果与证据见 [Phase 4 验收记录](docs/PHASE4_ACCEPTANCE.md)：24 个正式用例、4 个独立烟测、96 次真实模型成功请求。LLM 本轮精度低于规则基线，详见报告。

针对该结果，当前实现已调整为开发文档要求的四个受约束规划点：LLM 在预处理、解译、后处理和 QA 阶段规划工具与已登记的参数档案；报告阶段固定规则执行。详见 [受约束 LLM 规划策略](docs/LLM_ROUTING_POLICY.md)。

Phase 4 的框架测试不等于数据集和真实模型评测已经完成。固定数据准备与批量评测入口如下（读取项目 `.env` 中的模型配置）：

```powershell
.venv/Scripts/python.exe scripts/run_phase4.py --prepare
```

该命令下载固定子集，并运行规则与受约束的 LLM 规划策略，输出 `evaluation-output/phase4-/<UTC时间>/evaluation.csv`、`evaluation.json`、`evaluation.html` 及逐任务 evidence。无客户端时禁止宣称 LLM 规划已启用。LLM 参数必须匹配工具注册表中有限、可审计的档案；非法计划修复一次后回退规则路由。`--rule-only` 可离线执行规则对照，`--smoke` 仅运行独立的 4 个烟测案例。模型调用会产生服务商费用。

数据采用 Sen1Floods11 v1.1 的 12 个测试切片 × 两种传感器；10/20/30m 为派生网格。STAC 标记 proprietary，不将其宣称为开放许可或随仓库分发。详见 [Phase 4 评测说明](docs/PHASE4_EVALUATION.md)。实际完成情况以有日期的报告为准。

## Phase 5 web demo

Web UI 是正式演示入口：<http://localhost:8080>。它支持上传/选择影像、实时阶段状态、审批、artifact 下载与文本预览，以及 trace 时间线。API 使用 SSE 推送 job 状态；浏览器仅经 API 访问产物，不读取容器中的原始文件路径或服务端密钥。

简历表述与面试讲解建议见 [docs/RESUME_SUGGESTIONS.md](docs/RESUME_SUGGESTIONS.md)。

本地开发（Python 3.11+）：

```powershell
py -3.11 -m pip install -e ".[dev]"
py -3.11 -m pytest -q
```
