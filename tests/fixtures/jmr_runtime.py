"""Deterministic runtime doubles for the single production JMR graph."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from jmr.runtime import ModelToolCall, ToolModelTurn

PUBLICATION_TOOL = "mcp__scholar__search_publications"
KAKEN_TOOL = "mcp__kaken__search_projects"
DISCOVERY_TOOL = "mcp__scholar__discover_official_sources"


class DeterministicNodeModel:
    """Exercise structured and ReAct ports without network/model access."""

    def __init__(self, node_name: str) -> None:
        self.node_name = node_name
        self._tool_round = 0

    def invoke_structured(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system_prompt: str,
        output_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del system_prompt, output_schema
        if self.node_name == "extract_application_inputs":
            return {
                "extracted": {
                    "university_name": "东京大学",
                    "graduate_school_name": "工学系研究科",
                    "professor_name": "山田太郎",
                },
                "explicit_corrections": [],
                "ambiguous_fragments": [],
            }
        if self.node_name == "critique_direction_candidates":
            return {"status": "PASS", "issues": []}
        if self.node_name == "generate_direction_candidates":
            payload = json.loads(str(messages[-1]["content"]))
            papers = payload["evidence_bundle"]["records"]
            paper = papers[0]
            paper_id = paper["evidence_id"]
            title = paper["title"]
            directions = []
            for idea_title, idea_summary, question, method in (
                (
                    "多轮对话检索中的意图漂移检测",
                    "识别多轮提问中研究目标变化的关键转折，并衡量检测误报与漏报。",
                    "能否从连续提问中识别用户目标发生变化的时刻？",
                    "构建标注样本，比较检索日志特征与轻量语言模型的检测效果。",
                ),
                (
                    "澄清问题触发时机的自适应策略",
                    "在回答不确定时决定是否追问，以任务成功率和交互成本衡量收益。",
                    "何时提出澄清问题才能提升后续检索质量而不过度打断用户？",
                    "设计不同触发策略，并以任务成功率和交互轮数评估。",
                ),
                (
                    "对话搜索回答的证据可追溯性评估",
                    "建立回答片段与检索证据的对照标准，识别引用遗漏与证据不一致。",
                    "如何衡量生成回答与原始检索证据之间的一致性？",
                    "建立小规模评价集，比较引用完整性与人工核验结果。",
                ),
            ):
                directions.append(
                    {
                        "title": idea_title,
                        "summary": idea_summary,
                        "application_scenario": "所选论文的研究场景",
                        "research_question": question,
                        "method": method,
                        "masters_deliverable": "可复现的硕士研究报告",
                        "evaluation": "复现与对照实验",
                        "lab_fit": f"直接延伸《{title}》的公开成果",
                        "assumptions": ["数据可获得性需要核实"],
                        "applicant_connection": (
                            "" if payload["allow_unpersonalized"] else "结合已确认背景"
                        ),
                        "evidence_ids": [paper_id],
                    }
                )
            return {"directions": directions}
        if self.node_name == "revise_direction_candidates":
            return {
                "directions": json.loads(str(messages[-1]["content"]))["directions"]
            }
        if self.node_name in {
            "review_outreach_content",
            "review_outreach_language",
        }:
            return {"status": "PASS", "issues": [], "feedback": ""}
        if self.node_name == "draft_outreach_paragraph":
            payload = json.loads(str(messages[-1]["content"]))
            citations = _direction_evidence_ids(payload)
            content = (
                "我关注可靠人工智能在实际环境中的可解释性与安全性问题。"
                "基于您已公开的研究成果，我理解到模型的性能不仅取决于预测准确率，"
                "还需要在数据偏移、不确定性和用户信任之间建立可验证的联系。"
                "如果有机会进入硕士阶段，我希望从一个边界清晰的任务出发，"
                "构建包含基准模型、反事实解释与稳健性评估的实验流程，"
                "并用可复现的定量指标比较不同方法。"
                "我会先复现相关论文的设定，再逐步收窄研究问题，"
                "将方法创新限定在硕士期间可完成、可证伪的范围内，"
                "期待在您的指导下形成兼顾学术严谨性与实际价值的研究成果。"
            )
            return {"content": content, "citation_ids": citations[:3]}
        return {}

    def invoke_tool_step(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system_prompt: str,
        tools: Sequence[Mapping[str, Any]],
        output_schema: Mapping[str, Any],
    ) -> ToolModelTurn:
        del system_prompt, output_schema
        self._tool_round += 1
        last = messages[-1]
        if last.get("role") == "user" and isinstance(last.get("content"), list):
            blocks = last["content"]
            if blocks and blocks[0].get("type") == "tool_result":
                observation = json.loads(blocks[0]["content"])
                return ToolModelTurn(
                    final_result={"request_id": observation["request_id"]}
                )
        names = {str(tool["name"]) for tool in tools}
        desired = KAKEN_TOOL if KAKEN_TOOL in names else PUBLICATION_TOOL
        return ToolModelTurn(
            tool_calls=(
                ModelToolCall(
                    call_id=f"{self.node_name}-{self._tool_round}",
                    name=desired,
                    arguments={},
                ),
            )
        )


class DeterministicModelRegistry:
    def __init__(self) -> None:
        self.models: dict[str, DeterministicNodeModel] = {}

    def for_node(self, node_name: str) -> DeterministicNodeModel:
        return self.models.setdefault(node_name, DeterministicNodeModel(node_name))


class FakeRetrievalGateway:
    def __init__(self, *, with_records: bool = True) -> None:
        self.with_records = with_records
        self.calls: list[str] = []

    def tools_for_node(self, node_name: str) -> list[dict[str, Any]]:
        names = (
            [KAKEN_TOOL]
            if node_name == "kaken_research_agent"
            else [DISCOVERY_TOOL, PUBLICATION_TOOL]
        )
        return [
            {
                "name": name,
                "description": f"test double for {name}",
                "input_schema": {"type": "object", "additionalProperties": True},
            }
            for name in names
        ]

    def call_tool(
        self,
        *,
        node_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        trusted_context: Mapping[str, str],
    ) -> Mapping[str, Any]:
        del node_name, arguments, trusted_context
        self.calls.append(tool_name)
        is_publication = tool_name == PUBLICATION_TOOL
        evidence_type = "publication" if is_publication else "kaken_project"
        worker = "publication" if is_publication else "kaken"
        status = "SUCCESS" if self.with_records else "NO_RESULT"
        records = []
        if self.with_records:
            records.append(
                {
                    "evidence_id": f"{worker}-evidence",
                    "evidence_type": evidence_type,
                    "title": (
                        "Evaluating Conversational Search with Large Language Models"
                        if is_publication
                        else "Grounded kaken result"
                    ),
                    "publication_date": "2026-06-01" if is_publication else None,
                    "source_name": "fake-provider",
                    "source_url": f"https://example.edu/{worker}/evidence",
                    "retrieved_at": "2026-09-20T00:00:00+00:00",
                    "authors": ["山田太郎"],
                    "verification_status": "VERIFIED",
                }
            )
        return {
            "status": status,
            "request_id": f"{worker}-request",
            "evidence_type": evidence_type,
            "query": "bounded target query",
            "source_name": "fake-provider",
            "retrieved_at": "2026-09-20T00:00:00+00:00",
            "records": records,
            "discovered_sources": [],
            "warnings": [],
            "errors": [],
            "artifact_ref": None,
        }


def _direction_evidence_ids(plan: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for direction in plan.get("selected_directions", []):
        if not isinstance(direction, Mapping):
            continue
        for evidence_id in direction.get("evidence_ids", []):
            if isinstance(evidence_id, str) and evidence_id not in result:
                result.append(evidence_id)
    return result
