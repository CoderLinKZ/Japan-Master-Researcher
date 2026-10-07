# Japan Master Researcher LangGraph

Japan Master Researcher 是面向日本修士申请的可恢复研究助手。它围绕目标大学、研究科与教授检索论文和 KAKEN 课题，核验证据，生成研究方向，并在用户确认后产出有证据链的套磁研究段落。

项目只有一套生产工作流和一组公开构图 API：`build_graph`、`compile_graph` 与 `create_initial_state`。图状态 schema 当前为 3；更早工作流的 checkpoint、状态或导入路径不在支持范围内，需要新建 Case。

## 已实现能力

- 八阶段 LangGraph 闭环：申请信息校验、并行检索、证据核验、方向生成与选择、套磁草拟、审查返修与完成。
- 论文与 KAKEN Worker 均为最多 6 轮的有界 ReAct；只能调用当前 Node Manifest 允许的 MCP 工具，最终结果必须来自已观测的工具返回。
- `interrupt()` / `Command(resume=...)` 的人在回路：补全申请信息、确认教授身份、补充证据来源、确认 Memory、选择方向和提交返修信息。
- PostgreSQL 三层持久化：`PostgresSaver` 保存 Case 内恢复点，`PostgresStore` 保存跨 Case 的用户 Memory，`jmr` schema 保存业务事实与版本链。
- 文件对象存储保存较大内容，Graph State 只保留身份、阶段、有界消息和业务记录引用。
- `user_id` 所有权隔离、`case_id == thread_id` 绑定、幂等键、乐观锁、工具白名单和 SSRF/敏感字段边界。
- 最多 3 次的可重试瞬时故障策略，结构化且脱敏的 JSONL 运行事件。
- 版本化数据库迁移、存活/就绪检查、PostgreSQL + 对象存储联合备份与校验恢复、保留策略判定。
- FastAPI 对外服务：Bearer Token 身份映射、Case 执行/恢复/查询、Memory CRUD、Case 删除预览/确认和健康检查。

详细需求见 [需求文档.md](./需求文档.md)，节点提示词见 [研究室调查提示词.md](./研究室调查提示词.md)。

## 运行架构

```text
调用方 (user_id + case_id/thread_id)
  └─ Compiled StateGraph
       ├─ 确定性 Node：校验、路由、持久化、核验、最终化
       ├─ 模型 Node：结构化提取、有界 ReAct、Generator–Critic、审查返修
       ├─ interrupt / Command(resume=...)
       └─ NodeScopedMCPGateway
            ├─ scholar：官方来源发现、论文检索
            └─ kaken：官方科研课题检索

PostgreSQL
  ├─ LangGraph checkpoints
  ├─ LangGraph Store / 用户 Memory
  └─ jmr 业务 schema

FileObjectStore
  └─ 大对象与正文引用
```

Runtime Context 注入模型、MCP、Repository、Memory Store、对象存储、重试、安全与事件输出，这些资源不进入 checkpoint。

## 环境与启动

需求 Python `>=3.12,<3.13`、PostgreSQL、Anthropic 或兼容 Anthropic Messages API 的服务，以及 Scholar/KAKEN 数据源的网络访问。

```bash
uv sync --frozen --group dev
cp .env.example .env
```

至少配置：

```dotenv
ANTHROPIC_API_KEY=replace-with-your-api-key
MODEL_ID=replace-with-your-model-id
DATABASE_URL=postgresql:///jmr_langgraph?host=/tmp&user=jmr_app
LANGGRAPH_STRICT_MSGPACK=true

# 可选：兼容 Anthropic API 的自定义端点
# ANTHROPIC_BASE_URL=https://api.anthropic.com
# 可选：OpenAlex 额度
# OPENALEX_API_KEY=replace-with-your-openalex-api-key
# 可选：未提供官网 URL 时，自动发现教授/研究室/论文页面
# SERPAPI_API_KEY=replace-with-your-serpapi-api-key
# 可选回退：BRAVE_API_KEY=replace-with-your-brave-api-key
# 真实 KAKEN 检索必需
# KAKEN_APP_ID=replace-with-your-cinii-application-id
```

首次启动用 `--setup` 应用 Checkpointer、Store 和业务 schema 迁移；后续启动不要反复传入。`case-id` 是稳定的恢复标识，继续同一 Case 时必须复用。

```bash
uv run --frozen python src/main.py \
  --user-id user-001 \
  --case-id case-001 \
  --setup
```

当终端显示 `Resume JSON >` 时，输入当前中断载荷要求的 JSON 对象。普通消息不能绕过待处理中断。

官网发现优先使用 `SERPAPI_API_KEY` 对应的 SerpApi Google 搜索（日本地区、日语结果）；未配置时可回退到 `BRAVE_API_KEY`。两者都未配置时，用户仍可在中断处提供官网 URL。搜索结果只作为候选，遇到多个接近的研究室官网仍需人工确认。使用公网检索还需要相应服务的凭据和网络权限。

官网确认卡会直接展示候选链接，也可手动粘贴真实官网、点击“重新检索官网”，或明确选择“我确认没有研究室官网”。报告没有官网后，工作流继续核验现有论文等证据；如果所有来源都缺少可引用证据，则要求补充来源，不会编造研究结论。

## FastAPI 服务

HTTP 服务复用同一个 `run_turn` 与生产图，不在路由中复制业务逻辑。调用者不能提交 `user_id`；服务只从已配置的 Bearer Token 得到可信主体。先在未跟踪的 `.env` 中配置至少 16 字符的随机 Token：

```dotenv
JMR_API_TOKENS_JSON='{"replace-with-a-long-random-token":"root"}'
```

启动单 Worker API：

```bash
PYTHONPATH=src uv run --frozen --env-file .env python -m jmr.api
```

交互式 OpenAPI 文档位于 `http://127.0.0.1:8000/docs`。创建 Case 的示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/cases \
  -H 'Authorization: Bearer replace-with-a-long-random-token' \
  -H 'Idempotency-Key: create-case-001' \
  -H 'Content-Type: application/json' \
  -d '{"message":"请调查东京大学目标教授"}'
```

### 浏览器工作台

前端源文件位于 [`frontend/`](frontend/)，由同一个 FastAPI 进程直接提供，无需安装 Node.js 或单独启动前端服务。启动 API 后访问 `http://127.0.0.1:8000/frontend`。登录框预选 `root`，输入 `.env` 中映射给 `root` 的 Bearer Token；前端不会通过手填 `user_id` 绕过鉴权。需要切换用户时，先在 `JMR_API_TOKENS_JSON` 为不同 `user_id` 配置不同 Token，重启 API，再从左下角切换；Token 仅保存在当前浏览器会话中。

工作台提供 Case 列表、对话式启动/续写、人工中断确认、研究资料分类浏览（论文、KAKEN、生成与已选 Idea、每版套磁信草稿和审阅意见）、长期记忆查看/手动添加/确认删除及文档上传下载。上传支持 UTF-8 的 TXT/MD/CSV/JSON 与可提取文本的 PDF，单文件不超过 2 MB；原文件保存在对象存储中，最多提取前 32 KiB 文本写入长期记忆。扫描版 PDF 暂不支持 OCR。删除长期记忆会将其标记为 `DELETED`，不再出现在活动列表或供后续 Agent 选择；这不是物理清除，已生成的 Case 资料、审计记录和对象存储中的原文件不会随之删除。提交 Case 后欢迎选项会收起，对话区从用户提问开始，按时间展示 Agent 阶段、实际 MCP 工具调用、研究方向和草稿；执行期间前端轮询 `GET /api/v1/cases/{case_id}/progress` 展示已落库的进度，模型输出仍非逐字流式传输。新提交的完整用户提问保存在 PostgreSQL 的 `jmr.case_messages`，不再依赖有长度上限的 checkpoint；阶段、工具审计、研究产物和 Case 关联的长期记忆也保存在数据库中。右侧资料面板保留按类型查看完整研究产物的入口。

程序化删除长期记忆可调用 `DELETE /api/v1/memories/{memory_id}`，使用所属用户的 Bearer Token，并提交 `{"expected_version":1,"confirmed_by_user":true,"idempotency_key":"delete-memory-001"}`。`expected_version` 应取自当前记忆的 `version`；若版本已变化，先重新读取再确认删除。

服务提供 Case 创建、普通消息、HITL 恢复、状态/结果查询，用户 Memory CRUD、Case 删除预览/确认，以及 liveness/readiness。响应只返回允许公开的阶段、状态、中断和结果引用，不返回完整 checkpoint 或消息历史。

创建 Case 时必须传 `Idempotency-Key`；如果网络超时，使用相同 Key 和请求体重试，服务会返回同一 Case。当 `pending_interrupt` 不为空时，恢复请求必须回传响应中的 `interrupt_token`，格式为 `{"interrupt_token":"...","payload":{...}}`。如果 Case 响应的 `retry_available` 为 `true`，说明某个阶段执行失败但检查点仍在；可调用 `POST /api/v1/cases/{case_id}/retry` 从该阶段继续，不会增加一条用户消息。模型服务持续超时时，先检查模型连接再重试。删除前先调用 `GET /api/v1/cases/{case_id}/deletion-plan`，确认时提交 `{"confirmed_by_user":true,"plan_token":"..."}`；若 Case 或对象清单变化，需重新预览。请求体上限为 128 KiB。

若目标身份确认没有官网候选，前端会要求手动填写官网 URL；提交时由用户明确确认该链接，Agent 不会擅自选定同名教授。后端控制台会打印脱敏的阶段、MCP 调用、耗时和失败原因码；详细 JSONL 事件仍保存在 `JMR_EVENT_LOG_PATH`。

当前执行是同步的，同一 Case 通过进程内锁和 PostgreSQL advisory lock 互斥。标准启动命令仍使用一个 Uvicorn Worker；多副本大规模执行前，还需持久任务队列、工作者恢复和压测。不要用 FastAPI `BackgroundTasks` 承载长时间 Agent 执行。

## 程序化调用

```python
from langgraph.types import Command

from jmr.graph import compile_graph, create_initial_state

graph = compile_graph(checkpointer)
config = {
    "configurable": {
        "user_id": "user-001",
        "thread_id": "case-001",
    }
}

result = graph.invoke(
    create_initial_state(
        "user-001",
        "case-001",
        messages=[{"role": "user", "content": "请调查目标教授"}],
    ),
    config=config,
    context=runtime,
)

# 只有存在 pending interrupt 时才可恢复；config 必须完全复用。
result = graph.invoke(
    Command(resume={"fields": {...}, "idempotency_key": "input-001"}),
    config=config,
    context=runtime,
)
```

`user_id` 是跨 Case 稳定的已认证主体；`case_id` 是单次研究任务，并必须与 `thread_id` 相同。生产程序必须为 Runtime Context 提供上述全部依赖。

## 验证

统一本地门禁：

```bash
./scripts/check.sh
```

启用真实 PostgreSQL 集成测试：

```bash
JMR_RUN_POSTGRES_TESTS=1 \
DATABASE_URL='postgresql://jmr_app:password@localhost:5432/jmr_test' \
./scripts/check.sh
```

CI 使用 PostgreSQL service 执行 Ruff、格式检查、编译和全量测试。真实模型、KAKEN/OpenAlex/官网公网、FastAPI 压力与并发测试、备份恢复演练不使用仓库密钥自动运行，仍是发布前必须留存记录的外部门禁。

## 本地数据操作

首次初始化 Checkpointer、Store 和业务 schema 后，可用 `migrations status` 检查业务版本。跨 Case Memory 的本地运维入口支持列出、读取、创建、更新和删除；所有变更须在 JSON 载荷中提供 `confirmed_by_user: true` 与稳定的 `idempotency_key`：

```bash
PYTHONPATH=src uv run --frozen python -m jmr.operations.memory --user-id user-001 list </dev/null
PYTHONPATH=src uv run --frozen python -m jmr.operations.memory --user-id user-001 --payload-file /secure/path/memory.json create
```

单个 Case 删除默认仅报告精确目标；确认前必须停止该 Case 的写入。审计事件按独立保留策略留存，删除失败可用同一命令重试：

```bash
PYTHONPATH=src uv run --frozen python -m jmr.operations.cases --user-id user-001 --case-id case-001
PYTHONPATH=src uv run --frozen python -m jmr.operations.cases --user-id user-001 --case-id case-001 --confirm
```

这些是可信本机操作员入口，`--user-id` 不是身份认证；对外提供多用户服务时必须在调用层验证主体并注入 `user_id`。

## 运维与文档

- [src/Readme.md](./src/Readme.md)：源码边界和依赖方向。
- [实现规划.md](./实现规划.md)：P0–P7 实现与放行状态。
- [docs/operations.md](./docs/operations.md)：迁移、健康检查、备份恢复、保留策略、回滚与发布门禁。
- [.env.example](./.env.example)：环境变量模板。

密钥只能位于未跟踪的 `.env` 或密钥管理系统，不得提交到仓库、写入 checkpoint 或输出到日志。
