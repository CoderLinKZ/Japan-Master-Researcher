# File: agent.py
# Author: L1nzhk0
# Purpose: This file defines the main agent configuration, prompts, state, and runtime coordination.

from __future__ import annotations

from enum import Enum
from typing import Any

from harness.agent_loop import AgentLoop

# 定义业务流程状态枚举
class WorkflowState(str, Enum):
    # 校验大学、研究室、教授、相关链接、检索时间范围和用户经历；
    VALIDATING_APPLICATION_INPUTS = "VALIDATING_APPLICATION_INPUTS"
    # 最低必填信息齐全后进入研究证据检索。

    # 并行检索并整理教授论文与 KAKEN 课题；
    RETRIEVING_RESEARCH_EVIDENCE = "RETRIEVING_RESEARCH_EVIDENCE"
    # 两条检索链均完成，或失败项已有明确降级说明后进入证据整合。

    # 对论文与 KAKEN 结果进行去重、身份消歧、冲突核查和方向映射；
    MERGING_AND_VERIFYING_EVIDENCE = "MERGING_AND_VERIFYING_EVIDENCE"
    # 关键记录达到可引用状态后进入志望研究方向生成。

    # 结合已核验的研究证据和用户背景生成 3–6 个可落地的志望方向；
    GENERATING_RESEARCH_DIRECTIONS = "GENERATING_RESEARCH_DIRECTIONS"
    # 每个方向通过来源与硕士阶段可行性检查后等待用户选择。

    # 输出志望方向调研报告并等待用户选择 1–2 个方向；
    AWAITING_DIRECTION_SELECTION = "AWAITING_DIRECTION_SELECTION"
    # 收到明确选择或用户自带的明确研究构想后进入研究计划草拟。

    # 根据用户选定方向及其证据，按照四步逻辑和 3W 原则草拟中文研究计划段落；
    DRAFTING_OUTREACH_RESEARCH_PLAN = "DRAFTING_OUTREACH_RESEARCH_PLAN"
    # 草稿完成后进入内容与语言审查。

    # 审查研究计划的事实、逻辑和可行性，并在必要时检查目标语言；
    REVIEWING_OUTREACH_RESEARCH_PLAN = "REVIEWING_OUTREACH_RESEARCH_PLAN"
    # 不存在阻断问题或问题已明确反馈后进入完成状态。

    # 主 Agent 已输出符合用户当前请求的最终内容，当前任务结束。
    COMPLETED = "COMPLETED"


class MainAgent:
    """Own the conversation and delegate one user turn to the agent loop."""

    # 初始化主 Agent
    def __init__(self, agent_loop: AgentLoop, system_prompt: str) -> None:
        if not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")

        self._agent_loop = agent_loop
        self.system_prompt = system_prompt
        self.state = WorkflowState.VALIDATING_APPLICATION_INPUTS
        self.messages: list[dict[str, Any]] = []

    # 处理单轮用户输入
    def run(self, user_input: str) -> str:
        """Add a user message and run the model-tool loop to completion."""
        if not user_input.strip():
            raise ValueError("user_input cannot be empty")

        history_checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        try:
            return self._agent_loop.run(self.messages, self.system_prompt)
        except Exception:
            # A failed turn must not leave a partial user/tool exchange in the
            # conversation supplied to the next model call.
            del self.messages[history_checkpoint:]
            raise

    # 重置对话历史和业务状态
    def reset(self) -> None:
        """Start a new conversation and reset the business workflow state."""
        self.messages.clear()
        self.state = WorkflowState.VALIDATING_APPLICATION_INPUTS
