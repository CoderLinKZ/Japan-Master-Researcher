### src\main.py

main.py的主要任务是：

**1.完成初始化**，包括：检查环境，模型ID，创建Anthropic客户端，创建AgentLoop，载入系统提示词，创建MainAgent

**2.执行用户输入循环**，包括：等待用户输入，判断输入类型，调用MainAgent处理用户的输入。不断重复上述过程

```C++
main()
  → build_agent()
      → create_anthropic_caller()
          → 加载 .env
          → 检查 MODEL_ID
          → 创建 Anthropic 客户端
          → 返回 call_llm 函数
      → 创建 AgentLoop
      → load_system_prompt()
      → 创建 MainAgent //主Agent
  → 打印 Japan Master Researcher
  → While(True)//进入外层用户输入循环：
        等待用户输入
          ↓
        判断是否为 q / exit
          ↓
        调用 MainAgent.run(user_input)
          ↓
        打印最终回答
          ↓
        继续等待下一条用户输入
```



### src\main\agent.py

MainAgent负责的任务是：**处理用户输入的消息**，整体的流程如下：

```C++
MainAgent.run(user_input)
   检查输入是否为空
    ↓
   记录当前历史长度
    ↓
   将用户输入加入 messages
    ↓
   调用 AgentLoop.run(messages, system_prompt) 
    ↓
   返回最终文本
```



### src\harness\agent_loop\loop.py

建议把这部分修改为下面的表述。需要强调：`AgentLoop.run()` 处理的是“一条用户消息触发的完整 Agent 回合”，其中可能包含多次模型调用和工具调用。

`AgentLoop.run()` 负责处理当前用户消息对应的完整 Agent 回合。它接收已有的 `messages` 和系统提示词，在内层循环中反复调用模型和执行工具，直到模型不再请求工具并返回最终文本。

```
进入当前用户回合
  ↓
模型调用计数器加一并检查安全上限
  ↓
重新组装当前可用工具池
  ↓
将 messages、system_prompt 和工具定义发送给模型
  ↓
将 assistant 响应加入 messages
  ↓
检查响应内容中是否存在 tool_use
  ├─ 不存在
  │    ↓
  │  提取最终文本
  │    ↓
  │  结束当前 Agent 回合
  │
  └─ 存在
       ↓
     根据工具名称查找对应 Handler
       ↓
     执行工具
       ↓
     将执行结果或错误构造成 tool_result
       ↓
     将 tool_result 加入 messages
       ↓
     返回内层循环顶部，再次调用模型
```



**1. 记录当前回合的模型调用次数**

`model_calls` 用于记录当前用户回合内已经进行的模型调用次数，防止模型与工具之间出现无法自行结束的循环：

```
模型调用工具
  ↓
工具返回错误
  ↓
模型再次调用相同工具
  ↓
工具继续返回错误
  ↓
模型继续调用相同工具
  ↓
循环无法结束
```



**2. 重新组装当前可用工具池**

本项目计划采用按需连接的动态 MCP 工具池机制。因此，在每次模型调用前，**Harness 都会重新组装当前可用的工具定义和执行入口**：

Harness 内置工具 + 当前已经连接的 MCP Server 所提供的工具

其中：

- `tools` 是提供给模型的工具定义，包括工具名称、用途和输入参数结构；
- `handlers` 是保存在 Harness 内部的工具执行函数，模型无法直接看到。

动态连接的完整过程如下：

```
第一次组装工具池
  ↓
模型看到内置的 connect_mcp
  ↓
模型调用 connect_mcp("scholar")
  ↓
Harness 建立 MCP 连接并调用 tools/list
  ↓
保存 MCP Client 和发现的工具定义
  ↓
返回 Agent Loop 顶部
  ↓
第二次组装工具池
  ↓
新工具 mcp__scholar__* 被加入工具池
  ↓
模型在下一次调用中看到并选择这些工具
```

因此，“**每轮重新组装**”的作用是：

> 让刚刚通过 `connect_mcp` 连接并发现的 MCP 工具，可以在紧接着的下一次模型调用中立即提供给模型。
