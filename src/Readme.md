# `src` 源码架构

`src` 只承载当前 Japan Master Researcher 生产架构。`jmr.graph.compile_graph` 编译唯一的 LangGraph，`jmr.cli` 与 `jmr.api` 提供 CLI/HTTP 传输边界，`mcp_servers` 封装可被 Node 白名单授权的外部检索能力。

## 目录职责

```text
src/
├── main.py                         # CLI 入口
├── jmr/
│   ├── cli.py                      # 生产资源组装与交互式恢复
│   ├── api/                        # FastAPI、认证、HTTP schema 与应用服务
│   ├── domain/                     # 阶段、状态、证据与确定性核验
│   ├── graph/
│   │   ├── full.py                 # 唯一 StateGraph、边与构图 API
│   │   ├── state.py                # schema 3 状态与 reducer
│   │   ├── identity.py             # user/case/thread 身份绑定
│   │   ├── manifest.py             # 34 个 Node 及 MCP 白名单
│   │   └── nodes/                  # 申请、检索、方向、草拟审查
│   ├── persistence/
│   │   ├── repository.py           # jmr 业务表 Repository
│   │   ├── store.py / memory.py    # PostgresStore 与用户 Memory
│   │   ├── object_store.py         # 文件/内存对象存储
│   │   └── schema.py               # 版本化业务迁移
│   ├── runtime/
│   │   ├── context.py / ports.py   # 非序列化依赖端口
│   │   ├── models.py               # Anthropic 结构化/工具调用适配
│   │   ├── mcp.py                  # Server Registry 与 Node 级 Gateway
│   │   ├── checkpointer.py         # PostgresSaver 构造
│   │   ├── reliability.py          # 有界、分类重试
│   │   ├── observability.py        # 脱敏事件与 Node 仪表化
│   │   └── security.py             # 运行主体安全策略
│   └── operations/                  # 迁移、健康、备份恢复、保留策略
└── mcp_servers/
    ├── scholar/                    # 官网发现、OpenAlex 与论文聚合
    ├── kaken/                      # NII KAKEN OpenSearch
    └── memory/                     # 用户 Memory 工具契约
```

## 依赖方向

```text
graph nodes
  → runtime ports + domain models
      → persistence / model / MCP adapters
          → PostgreSQL / object store / external providers
```

- Domain 不得依赖 CLI、数据库连接或外部 Provider。
- Graph Node 只通过 `JMRRuntimeContext` 端口使用副作用；Runtime 资源不得进入 Graph State。
- Repository 不得调用模型；MCP Server 不得修改 Graph State 或进行阶段跳转。
- CLI 与 API 只调用应用服务/生产图，不复制 Node 业务规则；HTTP 响应不得返回完整 State。
- Provider 结果先经领域模型验证和持久化，State 中只保留引用。
- 每个 Agent 只获得 `NODE_MANIFEST` 声明的 `agent_tools`；业务写入始终是宿主确定性操作。

## 图契约

`JMRGraphState` 保存身份、阶段、运行状态、业务记录 ID、待处理中断、有界消息、警告和错误。正文、完整证据、工具响应、数据库连接与模型 Client 都不得进入 State。

普通调用必须传入：

```python
config = {
    "configurable": {
        "user_id": user_id,
        "thread_id": case_id,
    }
}
```

`user_id`、`case_id` 和 `thread_id` 由调用身份校验绑定。恢复中断时必须复用相同配置并传入 `Command(resume=payload)`；有未处理中断时不允许用普通消息绕过。

## Agent 模式与上限

- 申请信息：Structured Extraction。
- 论文/KAKEN 检索：Bounded ReAct，每个 Worker 最多 6 次模型调用，禁止未观测结果落盘。
- 方向：Generator–Critic / Reflection，最多 2 轮修订。
- 套磁：Plan-to-Execute / Prompt Chaining。
- 审查：Evaluator–Optimizer，最多 2 轮自动返修，需要时转人工中断。

路由、循环上限、去重、权限、持久化和最终化由确定性代码掌控，不接受自由文本决定。

## 持久化与运行边界

- `PostgresSaver`：Case 内 checkpoint、interrupt 和重启恢复。
- `PostgresStore`：按 `user_id` namespace 隔离的跨 Case Memory。
- `jmr` schema：Case、目标、计划、检索批次、证据、方向、选择、草稿、审查和审计事件。
- `FileObjectStore`：按内容哈希和引用保存大对象。

所有外部写入都需要幂等键、所有权检查和明确的失败语义。仅用户主动操作或明确确认的信息才能写入长期 Memory。

## 验证要求

`./scripts/check.sh` 是唯一本地自动门禁，覆盖 Ruff、格式、编译和全量 unittest。设置 `JMR_RUN_POSTGRES_TESTS=1` 时还必须提供可用的 `DATABASE_URL`，以验证 Checkpointer、Store、Repository、HTTP 服务层、用户隔离与重启恢复。

运维过程与发布门禁见 [`docs/operations.md`](../docs/operations.md)。
