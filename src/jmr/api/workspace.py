"""User-scoped read models for the browser workspace.

These queries expose business artifacts, never raw checkpoints or object URIs.
"""

from __future__ import annotations

from typing import Any

from jmr.persistence import open_postgres_repository


def case_timeline(connection: Any, user_id: str, case_id: str) -> list[dict[str, Any]]:
    """Combine durable user turns and business/audit facts for the chat UI."""

    events: list[dict[str, Any]] = []
    messages = connection.execute(
        """
        SELECT message_id, content, created_at FROM jmr.case_messages
        WHERE case_id=%s AND user_id=%s
        ORDER BY created_at DESC, message_id DESC LIMIT 250
        """,
        (case_id, user_id),
    ).fetchall()
    first = connection.execute(
        """
        SELECT message_id, content, created_at FROM jmr.case_messages
        WHERE case_id=%s AND user_id=%s
        ORDER BY created_at, message_id LIMIT 1
        """,
        (case_id, user_id),
    ).fetchone()
    if first is not None and all(row[0] != first[0] for row in messages):
        messages.append(first)
    events.extend(
        {
            "id": f"message:{row[0]}",
            "kind": "user_message",
            "content": row[1],
            "created_at": row[2].isoformat(),
        }
        for row in messages
    )
    history = connection.execute(
        """
        SELECT history_id, to_stage, run_status, node_name, created_at
        FROM jmr.case_status_history WHERE case_id=%s AND user_id=%s
        ORDER BY created_at DESC, history_id DESC LIMIT 250
        """,
        (case_id, user_id),
    ).fetchall()
    events.extend(
        {
            "id": f"stage:{row[0]}",
            "kind": "stage",
            "stage": row[1],
            "status": row[2],
            "node": row[3],
            "created_at": row[4].isoformat(),
        }
        for row in history
    )
    tool_calls = connection.execute(
        """
        SELECT audit_event_id, payload, created_at FROM jmr.audit_events
        WHERE case_id=%s AND user_id=%s AND event_type='mcp.tool_called'
        ORDER BY created_at DESC, audit_event_id DESC LIMIT 300
        """,
        (case_id, user_id),
    ).fetchall()
    events.extend(
        {
            "id": f"tool:{row[0]}",
            "kind": "mcp_tool",
            "tool": str(row[1].get("tool", "")),
            "node": str(row[1].get("node", "")),
            "status": str(row[1].get("status", "")),
            "created_at": row[2].isoformat(),
        }
        for row in tool_calls
    )
    directions = connection.execute(
        """
        SELECT d.direction_id, d.direction, d.ordinal, b.revision_round,
               b.created_at FROM jmr.research_directions d
        JOIN jmr.direction_batches b ON b.direction_batch_id=d.direction_batch_id
        WHERE d.case_id=%s AND d.user_id=%s
        ORDER BY b.created_at DESC, d.ordinal LIMIT 100
        """,
        (case_id, user_id),
    ).fetchall()
    events.extend(
        {
            "id": f"direction:{row[0]}",
            "kind": "direction",
            "content": dict(row[1]),
            "ordinal": row[2],
            "revision_round": row[3],
            "created_at": row[4].isoformat(),
        }
        for row in directions
    )
    drafts = connection.execute(
        """
        SELECT iteration_id, iteration_no, content, created_at
        FROM jmr.outreach_iterations WHERE case_id=%s AND user_id=%s
        ORDER BY created_at DESC, iteration_no DESC LIMIT 100
        """,
        (case_id, user_id),
    ).fetchall()
    events.extend(
        {
            "id": f"draft:{row[0]}",
            "kind": "draft",
            "version": row[1],
            "content": row[2],
            "created_at": row[3].isoformat(),
        }
        for row in drafts
    )
    events.sort(
        key=lambda event: (
            event["created_at"],
            0 if event["kind"] == "user_message" else 1,
            event["id"],
        )
    )
    return events


def list_cases(environment: dict[str, str] | Any, user_id: str) -> dict[str, Any]:
    with open_postgres_repository(environment=environment) as repository:
        rows = repository.connection.execute(
            """
            SELECT c.case_id, c.workflow_stage, c.run_status, c.created_at,
                   c.updated_at, t.target
            FROM jmr.research_cases c
            LEFT JOIN jmr.case_targets t
              ON t.case_id=c.case_id AND t.user_id=c.user_id
            WHERE c.user_id=%s AND NOT EXISTS (
                SELECT 1 FROM jmr.case_deletions d WHERE d.case_id=c.case_id
            )
            ORDER BY c.updated_at DESC, c.case_id DESC LIMIT 100
            """,
            (user_id,),
        ).fetchall()
        return {
            "items": [
                {
                    "case_id": row[0],
                    "workflow_stage": row[1],
                    "run_status": row[2],
                    "created_at": row[3].isoformat(),
                    "updated_at": row[4].isoformat(),
                    "target": dict(row[5]) if row[5] else None,
                }
                for row in rows
            ]
        }


def case_workspace(
    environment: dict[str, str] | Any, user_id: str, case_id: str
) -> dict[str, Any] | None:
    with open_postgres_repository(environment=environment) as repository:
        case = repository.get_case(user_id=user_id, case_id=case_id)
        if case is None:
            return None
        connection = repository.connection

        def rows(query: str) -> list[tuple[Any, ...]]:
            return connection.execute(query, (case_id, user_id)).fetchall()

        evidence = rows(
            """
            SELECT evidence_id, evidence_kind, external_id, title, record,
                   verification_status, created_at
            FROM jmr.evidence_records WHERE case_id=%s AND user_id=%s
            ORDER BY created_at DESC, evidence_id DESC LIMIT 200
            """
        )
        directions = rows(
            """
            SELECT d.direction_id, d.direction, d.ordinal,
                   d.direction_batch_id, b.revision_round, b.created_at
            FROM jmr.research_directions d
            JOIN jmr.direction_batches b
              ON b.direction_batch_id=d.direction_batch_id
            WHERE d.case_id=%s AND d.user_id=%s
            ORDER BY b.created_at DESC, d.ordinal LIMIT 100
            """
        )
        selections = rows(
            """
            SELECT selection_id, direction_batch_id, selected_direction_ids,
                   custom_direction, created_at
            FROM jmr.direction_selections WHERE case_id=%s AND user_id=%s
            ORDER BY created_at DESC LIMIT 50
            """
        )
        drafts = rows(
            """
            SELECT iteration_id, iteration_no, parent_iteration_id, content,
                   language, citation_ids, created_at
            FROM jmr.outreach_iterations WHERE case_id=%s AND user_id=%s
            ORDER BY iteration_no DESC LIMIT 100
            """
        )
        reviews = rows(
            """
            SELECT review_id, iteration_id, reviewer_kind, status, result,
                   created_at
            FROM jmr.outreach_reviews WHERE case_id=%s AND user_id=%s
            ORDER BY created_at DESC LIMIT 200
            """
        )
        runs = rows(
            """
            SELECT retrieval_run_id, worker_kind, status, result_summary,
                   created_at FROM jmr.retrieval_runs
            WHERE case_id=%s AND user_id=%s
            ORDER BY created_at DESC LIMIT 100
            """
        )
        plans = rows(
            """
            SELECT research_plan_id, plan, created_at
            FROM jmr.research_plans WHERE case_id=%s AND user_id=%s
            ORDER BY created_at DESC LIMIT 20
            """
        )
        history = rows(
            """
            SELECT to_stage, run_status, node_name, created_at
            FROM jmr.case_status_history WHERE case_id=%s AND user_id=%s
            ORDER BY created_at, history_id LIMIT 200
            """
        )
        memory_links = rows(
            """
            SELECT memory_id, created_at FROM jmr.case_memory_links
            WHERE case_id=%s AND user_id=%s ORDER BY created_at DESC
            """
        )
        return {
            "case_id": case_id,
            "user_id": user_id,
            "target": case.get("target"),
            "plans": [
                {"id": str(r[0]), "content": r[1], "created_at": r[2].isoformat()}
                for r in plans
            ],
            "retrieval_runs": [
                {
                    "id": str(r[0]),
                    "worker_kind": r[1],
                    "status": r[2],
                    "summary": r[3],
                    "created_at": r[4].isoformat(),
                }
                for r in runs
            ],
            "evidence": [
                {
                    "id": str(r[0]),
                    "kind": r[1],
                    "external_id": r[2],
                    "title": r[3],
                    "record": r[4],
                    "verification_status": r[5],
                    "created_at": r[6].isoformat(),
                }
                for r in evidence
            ],
            "directions": [
                {
                    "id": str(r[0]),
                    "content": r[1],
                    "ordinal": r[2],
                    "batch_id": str(r[3]),
                    "revision_round": r[4],
                    "created_at": r[5].isoformat(),
                }
                for r in directions
            ],
            "selections": [
                {
                    "id": str(r[0]),
                    "batch_id": str(r[1]),
                    "selected_direction_ids": r[2],
                    "custom_direction": r[3],
                    "created_at": r[4].isoformat(),
                }
                for r in selections
            ],
            "drafts": [
                {
                    "id": str(r[0]),
                    "version": r[1],
                    "parent_id": str(r[2]) if r[2] else None,
                    "content": r[3],
                    "language": r[4],
                    "citation_ids": r[5],
                    "created_at": r[6].isoformat(),
                }
                for r in drafts
            ],
            "reviews": [
                {
                    "id": str(r[0]),
                    "draft_id": str(r[1]),
                    "reviewer_kind": r[2],
                    "status": r[3],
                    "result": r[4],
                    "created_at": r[5].isoformat(),
                }
                for r in reviews
            ],
            "history": [
                {
                    "stage": r[0],
                    "status": r[1],
                    "node": r[2],
                    "created_at": r[3].isoformat(),
                }
                for r in history
            ],
            "selected_memory_ids": [r[0] for r in memory_links],
        }
