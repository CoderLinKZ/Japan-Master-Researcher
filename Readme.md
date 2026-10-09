# Japan Master Researcher

**面向日本修士申请的 AI 研究助手。**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](./pyproject.toml)
[![LangGraph](https://img.shields.io/badge/Powered%20by-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://github.com/fastapi/fastapi)

Japan Master Researcher 帮助你调查目标教授与研究室，整理论文和科研课题，探索与个人背景匹配的研究方向，并准备套磁邮件中的研究段落。从收集资料到修改草稿，每项研究都可以在同一个工作台中完成。

[功能介绍](#功能介绍) · [快速开始](#快速开始) · [使用示例](#使用示例) · [配置说明](#配置说明) · [开发与贡献](#开发与贡献)

## 为什么使用 Japan Master Researcher？

申请日本修士时，了解教授近期的研究、找到自己的切入点、写出具体的套磁内容，往往需要在研究室官网、论文页面和科研项目之间反复查找。

Japan Master Researcher 将这些步骤串联起来：以公开研究资料为依据，把教授的研究方向与你的经历联系起来，让套磁内容有具体的研究问题、方法和来源支撑。

## 功能介绍

- **教授与研究室调查**：查找官网与研究资料，在遇到同名教授或多个候选时由你确认目标。
- **论文与科研课题检索**：结合 OpenAlex、KAKEN 和官方页面，整理可追溯的研究证据。
- **研究方向探索**：结合教授的研究与申请者背景提出方向，支持比较和选择。
- **套磁内容草拟**：围绕选定方向生成研究段落，并根据审阅意见与个人反馈修改。
- **浏览器工作台**：通过对话启动研究，查看论文、科研课题、研究方向和各版草稿。
- **个人资料与持续研究**：保存研究经历和申请偏好，上传背景资料，在需要确认时暂停并继续任务。

## 工作流程

```text
描述申请目标 → 确认教授与研究室 → 检索与核验资料 → 选择研究方向 → 草拟与修改套磁内容
```

你可以在流程中补充个人背景、确认资料来源、选择研究方向或提出修改意见。研究任务和资料会被保存，方便后续继续。

## 快速开始

### 环境准备

- Python 3.12
- [uv](https://github.com/astral-sh/uv)
- PostgreSQL：准备一个数据库，并确保连接用户具有建表权限
- 支持工具调用的 Anthropic 或兼容 Anthropic Messages API 的模型服务

### 1. 下载与安装

```bash
git clone https://github.com/CoderLinKZ/Japan-Master-Researcher.git
cd Japan-Master-Researcher
uv sync --frozen --group dev
cp .env.example .env
```

### 2. 配置环境变量

编辑 `.env`，填写模型与数据库连接信息。以下值需要替换为自己的配置：

```dotenv
ANTHROPIC_API_KEY=YOUR_API_KEY
MODEL_ID=YOUR_MODEL_ID
DATABASE_URL=postgresql://YOUR_DB_USER:YOUR_DB_PASSWORD@localhost:5432/jmr_langgraph
LANGGRAPH_STRICT_MSGPACK=true

# 浏览器工作台与 API 的访问 Token，替换为至少 16 字符的随机字符串
JMR_API_TOKENS_JSON='{"YOUR_RANDOM_ACCESS_TOKEN":"root"}'
```

完整配置模板见 [`.env.example`](./.env.example)。论文检索、官网搜索和 KAKEN 的配置见下方[配置说明](#配置说明)。

### 3. 初始化数据库

首次运行时执行：

```bash
uv run --frozen python src/main.py \
  --user-id root \
  --case-id setup \
  --setup
```

看到 `JMR >` 后输入 `exit`，即可退出并启动工作台。`--setup` 用于数据库初始化；日常使用无需重复执行。

### 4. 启动工作台

```bash
PYTHONPATH=src uv run --frozen --env-file .env python -m jmr.api
```

打开 [http://127.0.0.1:8000/frontend](http://127.0.0.1:8000/frontend)，选择 `root` 用户并输入上面配置的访问 Token。

前端由同一个服务提供，无需额外安装 Node.js 或启动前端进程。交互式 API 文档位于 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。

## 使用示例

### 在工作台中开始研究

创建一个研究任务，输入目标大学、研究科、教授姓名和自己的背景，例如：

```text
我计划申请日本修士，请调查【大学】的【研究科】中【教授姓名】教授的研究。

我的背景：计算机专业，熟悉 Python 和机器学习，做过医学图像分类项目。
希望了解教授近期的论文和科研课题，找到与我的经历相关的研究方向，
并为套磁邮件准备一段具体的研究提案。

研究室官网：【如果已知，请填写 URL】
```

根据工作台提示确认教授身份与来源，补充必要资料，选择研究方向，再查看和修改草稿。论文、科研课题和草稿版本可以在研究资料面板中分别查看。

在长期记忆页面中，可以保存研究经历、技能和申请偏好，也可以上传 PDF、TXT、Markdown、CSV 或 JSON 资料。单文件上限为 2 MB，PDF 需包含可提取的文本。

### 使用命令行

```bash
uv run --frozen python src/main.py \
  --user-id root \
  --case-id my-first-research
```

在 `JMR >` 输入研究需求。当程序显示 `Resume JSON >` 时，根据提示提交确认信息。继续同一项研究时，复用相同的 `--user-id` 和 `--case-id`。

## 配置说明

| 环境变量 | 用途 | 是否必需 |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | 模型服务的 API Key | 是 |
| `MODEL_ID` | 使用的模型 ID | 是 |
| `DATABASE_URL` | PostgreSQL 连接地址 | 是 |
| `JMR_API_TOKENS_JSON` | 工作台与 API 的用户访问 Token | 工作台 / API 必需 |
| `ANTHROPIC_BASE_URL` | 兼容 Anthropic API 的自定义地址 | 可选 |
| `OPENALEX_API_KEY` | OpenAlex 论文检索凭据 | 可选 |
| `SERPAPI_API_KEY` | 自动发现教授和研究室官网 | 可选 |
| `BRAVE_API_KEY` | 官网搜索的备用服务 | 可选 |
| `KAKEN_APP_ID` | KAKEN 科研课题检索的 CiNii Application ID | KAKEN 检索必需 |

官网搜索优先使用 SerpApi，未配置时可使用 Brave。没有配置搜索服务时，也可以手动提供官网链接。

`.env` 已加入 Git 忽略规则，请将个人凭据保存在该文件中。

## 项目结构

```text
Japan-Master-Researcher/
├── frontend/          # 浏览器工作台
├── src/
│   ├── main.py        # 命令行入口
│   ├── jmr/           # 研究工作流、API 与数据管理
│   └── mcp_servers/   # 论文、官网与科研课题检索工具
├── tests/             # 测试
├── docs/              # 部署与运维文档
├── scripts/           # 开发检查脚本
├── .env.example       # 环境变量模板
└── pyproject.toml     # 项目与依赖配置
```

## 开发与贡献

欢迎通过 [Issues](https://github.com/CoderLinKZ/Japan-Master-Researcher/issues) 报告问题、提出建议，或提交 Pull Request 改进检索、交互与文档。

提交代码前运行：

```bash
./scripts/check.sh
```

如需运行 PostgreSQL 集成测试，请使用独立的测试数据库：

```bash
JMR_RUN_POSTGRES_TESTS=1 \
DATABASE_URL='postgresql://YOUR_DB_USER:YOUR_DB_PASSWORD@localhost:5432/jmr_test' \
./scripts/check.sh
```

## 文档

- [部署与运维](./docs/operations.md)：数据库迁移、健康检查、备份与恢复。
- [源码结构](./src/Readme.md)：模块职责与扩展入口。
- [研究提示词](./研究室调查提示词.md)：教授调查与研究方向生成的提示词。
- [环境变量模板](./.env.example)：完整配置项与示例。
