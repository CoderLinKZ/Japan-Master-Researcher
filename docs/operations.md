# Japan Master Researcher 运维手册

本手册面向当前单一生产工作流。所有命令默认在项目根目录执行，示例中的路径和 DSN 都必须替换为目标环境的实际值。

## 1. 数据与责任边界

| 数据层 | 内容 | 管理方 |
| --- | --- | --- |
| LangGraph Checkpointer | Case 内状态、interrupt、恢复点 | `langgraph-checkpoint-postgres` |
| LangGraph Store | 按 `user_id` 隔离的跨 Case Memory | `langgraph-checkpoint-postgres` |
| PostgreSQL `jmr` schema | Case、证据、方向、草稿、审查、审计与幂等事实 | `jmr.persistence.schema` |
| FileObjectStore | 大对象和正文 | 应用挂载目录 |
| JSONL 事件 | 脱敏的 Node 运行事件 | 应用或日志采集器 |

启动 FastAPI 的控制台同时输出 `[JMR] case=... stage=... node=... status=...`；调用 MCP 时附带 `mcp=...` 和耗时，失败时附带 `reason=...` 与 `failure_id=...`。控制台与 JSONL 都不记录 Token、模型正文或 MCP 参数。Case 响应中的 `failure` 提供可公开的失败节点和简短原因，`retry_available=true` 时可重试该节点；完整运行事件继续保留在 `JMR_EVENT_LOG_PATH`。

检索来源对象当前保存完整 MCP 工具响应 JSON，并记录 SHA-256、MIME、大小和时间；它不等于上游 HTML/PDF 原件。需要原件归档的部署应由数据源适配器另行采集并取得许可。

数据库与对象存储必须作为同一数据集合协调备份、恢复、保留与删除；仅恢复其中一层会破坏引用一致性。JSONL 事件不属于事务数据集，应由日志采集平台独立归档并按审计策略保留。

## 2. 环境变量与密钥

生产环境至少需要：

```dotenv
ANTHROPIC_API_KEY=...
MODEL_ID=...
DATABASE_URL=postgresql://...
LANGGRAPH_STRICT_MSGPACK=true
JMR_OBJECT_STORE_DIR=/srv/jmr/objects
JMR_EVENT_LOG_PATH=/var/log/jmr/events.jsonl
JMR_API_TOKENS_JSON='{"replace-with-a-long-random-token":"user-001"}'
```

健康检查命令使用 `JMR_OBJECT_STORE_ROOT`；它应与应用的 `JMR_OBJECT_STORE_DIR` 指向同一目录。真实 KAKEN 检索还需要 `KAKEN_APP_ID`，OpenAlex 的 `OPENALEX_API_KEY` 可选。配置 `SERPAPI_API_KEY` 后，未提供官网 URL 的 Case 可通过 SerpApi Google 搜索发现官网候选；未配置时可回退到 `BRAVE_API_KEY`。搜索结果仍需身份核验，歧义候选须由用户确认。

- 凭据只从密钥管理系统或权限为最小化的未跟踪环境文件注入。
- 不在命令行参数、CI 配置、工单、日志、checkpoint 或备份 manifest 中写入密钥和 DSN。
- 数据库用户只授予目标数据库所需权限，备份/恢复用户与应用用户分离。
- 生产数据库连接使用 TLS；对象存储与备份目录使用平台级静态加密和访问审计。
- 模型、KAKEN 和数据库凭据应定期轮换；轮换后执行 readiness 与最小 smoke test。
- API Token 必须由密钥管理系统生成和注入，不得使用示例值；轮换时允许短期同时配置新旧 Token，客户端切换完成后删除旧 Token。

## 3. 迁移

### 3.1 检查业务 schema

```bash
PYTHONPATH=src uv run --frozen python -m jmr.operations.migrations status
```

输出的 `current` 必须等于 `latest`。数据库版本高于当前应用支持版本时不得启动旧应用。

### 3.2 应用业务 schema 迁移

执行前先创建并校验完整备份，再在单一发布操作中运行：

```bash
PYTHONPATH=src uv run --frozen python -m jmr.operations.migrations migrate
PYTHONPATH=src uv run --frozen python -m jmr.operations.migrations status
```

`migrations` CLI 只管理 `jmr` 业务 schema。LangGraph Checkpointer 和 Store 的官方迁移由它们各自的 `setup()` 管理。首次部署可用应用入口一次性完成三者：

```bash
uv run --frozen python src/main.py \
  --user-id deployment-smoke-user \
  --case-id deployment-smoke-case \
  --setup
```

`--setup` 只用于明确的初始化/发布步骤，平时启动不要传入。

本地开发可创建单独的 `jmr_test` 数据库，并用 `JMR_RUN_POSTGRES_TESTS=1`、指向该库的 `DATABASE_URL` 运行集成测试；测试会重建其中的 `jmr` schema，严禁指向生产库。当前业务 schema 版本为 3。

## 4. 健康检查

### 4.1 存活

存活检查不访问外部依赖，适用于进程重启判定：

```bash
PYTHONPATH=src uv run --frozen python -m jmr.operations.health liveness
```

### 4.2 就绪

运维 CLI 的就绪检查验证数据库连通性、`jmr` schema 版本，以及对象存储目录的读/写/进入权限：

```bash
export JMR_OBJECT_STORE_ROOT=/srv/jmr/objects
PYTHONPATH=src uv run --frozen python -m jmr.operations.health readiness
```

返回为 JSON，任一检查不是 `ok` 时进程退出码为 1。就绪失败的实例不得接收流量。模型和公网数据源的深度健康检查应作为低频 smoke test，不应放在高频 readiness 路径上。

### 4.3 FastAPI 服务

部署前完成三类数据库迁移并配置 Bearer Token，然后启动当前受支持的单 Worker 服务：

```bash
PYTHONPATH=src uv run --frozen python -m jmr.api --host 0.0.0.0 --port 8000
```

- `GET /health/live` 不访问外部依赖。
- `GET /health/ready` 额外检查 Checkpointer/Store 表、模型配置和认证配置；失败时返回 HTTP 503。
- `/api/v1/*` 必须携带 `Authorization: Bearer <token>`，`user_id` 只由 Token 映射产生，不能从请求参数接受。
- API 返回阶段、运行状态、中断载荷和显式结果，不返回完整消息、checkpoint、模型 prompt 或数据库异常正文。
- 同一 Case 的消息、恢复和删除由进程内互斥锁与 PostgreSQL advisory lock 共同保护。当前标准入口固定一个 Uvicorn Worker；多副本高可用部署前还需要持久任务队列和工作者恢复验证。本机 CLI 操作员必须在停止相关 API 写入后操作同一 Case。
- POST 创建 Case 必须提供 `Idempotency-Key`。Resume 必须回传最新 `interrupt_token`；删除确认必须回传预览产生的 `plan_token`（有效期 15 分钟）。当前状态变化时服务返回 HTTP 409，客户端应重新读取。HTTP 请求体限制为 128 KiB。
- 工作流目前在同步请求中运行。反向代理超时必须覆盖目标模型调用上限；生产异步化时使用外部队列，禁止把长任务放入进程内 `BackgroundTasks`。

## 5. 备份

备份命令需要 `pg_dump`，目标目录必须不存在，对象存储根目录必须存在且不包含符号链接。工具按顺序导出数据库和对象文件，本身不能冻结应用写入；要获得一致恢复点，执行前必须进入维护模式并停止所有数据库与对象存储写入，直到两个产物及 manifest 写完。

```bash
PYTHONPATH=src uv run --frozen python -m jmr.operations.backup backup \
  /secure-backups/jmr-2026-09-21T120000Z \
  --object-store-root /srv/jmr/objects \
  --confirm-quiesced
```

产物包含：

- `postgres.dump`：完整目标数据库，包括 Checkpointer、Store 和 `jmr` schema；
- `objects.tar.gz`：同一时点的对象存储快照；
- `manifest.json`：格式版本、业务 schema 版本、数据范围和 SHA-256 checksum。

备份完成后必须验证：

```bash
PYTHONPATH=src uv run --frozen python -m jmr.operations.backup verify \
  /secure-backups/jmr-2026-09-21T120000Z
```

应由调度器在维护窗口执行备份，并在独立故障域保留至少一份已验证副本。当前工具校验完整性，不自行加密；加密必须由备份介质或密钥管理的存储层提供。

## 6. 恢复演练与灾难恢复

`restore` 会执行 `pg_restore --clean --if-exists`，是破坏性操作。仅能在已停止所有应用写入的目标环境执行，且对象存储恢复目标必须为空。

1. 下线目标实例并阻断后台 Worker。
2. 记录目标数据库和对象存储路径，确认操作对象。
3. 用 `verify` 再次校验备份。
4. 使用与备份数据库主版本兼容的 `pg_restore`。
5. 执行恢复：

```bash
PYTHONPATH=src uv run --frozen python -m jmr.operations.backup restore \
  /secure-backups/jmr-2026-09-21T120000Z \
  --object-store-root /srv/jmr/objects-restored \
  --confirm-destructive-restore
```

6. 运行 `migrations status` 和 readiness。
7. 抽样验证 Case 所有权、业务记录数量、object URI 可读性，以及一个已中断 Case 的同 `thread_id` 恢复。
8. 仅在上述验证完成后恢复流量。

每个发布周期至少在隔离环境执行一次备份恢复演练，记录 RPO、RTO、备份大小、checksum、抽样结果和执行人。

## 7. 保留策略

`RetentionPolicy` 默认值为：

- 已完成 Case checkpoint：30 天；
- 备份：30 天；
- 审计数据：365 天。

保留模块默认只做判定/报告，不自动删除备份或业务记录：

- `checkpoint_cleanup_due(...)` 判定已完成 Case 是否到期。
- `cleanup_checkpoint(..., dry_run=True)` 默认只报告；对确认完成的精确 `thread_id` 同时传入 `dry_run=False, confirm=True` 才删除 checkpoint。
- `expired_backup_directories(...)` 只列出到期且完整校验通过的备份目录；实际删除前再次校验。
- `cleanup_audit_events(...)` 先列出超过 `audit_days` 的精确审计 ID，再在确认后事务内删除。
- `referenced_object_uris(connection, checkpointer)` 汇总 `source_objects` 与所有保留 checkpoint 中的对象引用；必须在停止相关写入后生成完整清单，再传给 `cleanup_orphan_objects(...)` 求差集并复核 dry-run 报告。
- `cleanup_expired_backups(...)` 只处理配置根目录下有有效 manifest 的到期子目录，执行前需 `dry_run=False, confirm=True`。

调度的保留任务必须先以 dry-run 生成审核列表，再由有权限的独立步骤删除精确目标。不得对对象存储根目录、备份根目录或数据库 schema 执行递归式模糊删除。法律保留、用户删除请求与审计保留期优先于默认值。

## 8. Memory 与单 Case 删除

Memory 管理只面向可信本机操作员。先用 `list`/`get` 查看，`create`/`update`/`delete` 的 JSON 载荷必须含 `confirmed_by_user: true` 和稳定的 `idempotency_key`；重复提交同一键会返回先前结果。变更及当前 Case 的 Memory 选择写入 `jmr.audit_events`。例如：

```bash
PYTHONPATH=src uv run --frozen python -m jmr.operations.memory --user-id user-001 list </dev/null
PYTHONPATH=src uv run --frozen python -m jmr.operations.memory --user-id user-001 --payload-file /secure/path/memory.json create
```

用户请求删除 Case 时，先停止该 Case 的所有前台与后台写入，再用不带 `--confirm` 的命令核对 `case_id`、所有权和精确对象 URI 清单。确认后操作会记录持久删除任务，依次清理 checkpoint、业务记录与对象；如果中途失败，重复同一命令即可继续。审计记录不会被此命令删除，仍由审计保留策略处理。

```bash
PYTHONPATH=src uv run --frozen python -m jmr.operations.cases --user-id user-001 --case-id case-001
PYTHONPATH=src uv run --frozen python -m jmr.operations.cases --user-id user-001 --case-id case-001 --confirm
```

上述 CLI 中的 `--user-id` 是操作员明确选择的已认证主体，不提供登录或授权机制；对外服务必须由上层认证并传递主体，不能让请求方自行指定任意用户 ID。

## 9. 回滚

### 9.1 应用回滚

只有当前一版应用明确支持已迁移的 schema 时，才可仅回滚应用镜像。如果 schema 不向后兼容，必须先停止流量并按迁移回滚或完整备份恢复流程操作。

### 9.2 业务 schema 回滚

当前应用只提供已明确实现的有限回滚，不会从应用代码删除基础 schema。回滚前必须有已验证备份，并且已下线应用写入：

```bash
PYTHONPATH=src uv run --frozen python -m jmr.operations.migrations rollback \
  --target-version 1 \
  --confirm
```

回滚后运行 `status`、readiness 和与目标应用版本匹配的 smoke test。需要恢复到 schema 1 以前或撤销不可逆数据变更时，不得手工 `DROP` 表；必须恢复完整备份。

### 9.3 Graph State 版本

当前图只读取 schema 3 状态。不对更早 checkpoint 做在线转换；发布切换前应让正在运行的 Case 在原版本完成，或完整保留审计数据后新建 Case。

## 10. 发布步骤与门禁

### 自动门禁

1. CI 的 Ruff、format check、compileall 和全量 unittest 全部绿色。
2. CI PostgreSQL service 中的 Checkpointer、Store、Repository、所有权和重启恢复测试通过。
3. 无未审查迁移、无明文密钥、无高危依赖或超范围网络权限。

### 外部门禁

1. 目标模型上的结构化输出、ReAct 工具调用和完整 Case E2E。
2. 真实 KAKEN/OpenAlex/官方站点的成功、部分结果、限流、超时和阻断语义。
3. 带校验的完整备份，以及在隔离环境完成的恢复演练。
4. 多用户并发、进程重启、依赖故障注入、重试上限和告警路由验证。
5. FastAPI 认证、跨用户枚举、同 Case 并发冲突、代理超时、请求体上限和 Token 轮换验证。
6. 日志、事件与错误样本的密钥、DSN、prompt、正文和个人信息脱敏复核。

### 发布顺序

1. 记录发布版本、迁移目标、变更窗口、执行人和回滚条件。
2. 下线写入或进入维护模式。
3. 创建并验证联合备份。
4. 部署应用并执行迁移。
5. 执行 liveness、readiness、最小 Case smoke 和中断恢复 smoke。
6. 逐步恢复流量，监控失败率、重试、中断、数据库冲突、MCP 部分结果和时延。
7. 达到回滚条件时立即停止流量，按第 9 节操作；不在故障现场即席修改 schema 或强行跳过中断。

每项外部门禁都必须在发布单中留存时间、环境、版本、执行人、证据位置和结论；没有记录就不视为已完成。
