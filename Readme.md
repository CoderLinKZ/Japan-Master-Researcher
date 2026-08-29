```C++
Japan Master Researcher/
├── src/
│   ├── main.py
│   ├── agent/
│   │   ├── main/
│   │   └── subagents/
│   │       ├── publication_research/
│   │       ├── kaken_research/
│   │       ├── evidence_merge/
│   │       ├── direction_generation/
│   │       ├── plan_drafting/
│   │       ├── content_review/
│   │       └── japanese_review/
│   ├── harness/
│   │   ├── agent_loop/
│   │   ├── context_compaction/
│   │   ├── hooks/
│   │   ├── mcp_gateway/
│   │   ├── permissions/
│   │   └── background_manager/
│   └── mcp_servers/
│       ├── scholar/
│       │   ├── tools/
│       │   ├── adapters/
│       │   └── services/
│       ├── kaken/
│       │   ├── tools/
│       │   ├── adapters/
│       │   └── services/
│       ├── artifact/
│       │   ├── tools/
│       │   └── storage/
│       ├── jmr_workflow/
│       │   ├── tools/
│       │   └── workflows/
│       │       ├── collect_publications/
│       │       ├── collect_kaken/
│       │       ├── recommend_directions/
│       │       ├── narrow_idea/
│       │       ├── draft_plan/
│       │       ├── review_plan/
│       │       └── review_japanese/
│       ├── memory/
│       │   ├── tools/
│       │   └── storage/
│       ├── outreach/
│       │   ├── tools/
│       │   └── storage/
│       ├── task/
│       │   ├── tools/
│       │   └── storage/
│       └── jobs/
│           ├── tools/
│           ├── workers/
│           └── storage/
├── config/
│   ├── mcp/
│   ├── permissions/
│   └── runtime/
├── prompts/
│   ├── system/
│   ├── subagents/
│   └── reviewers/
├── schemas/
│   ├── mcp/
│   ├── artifacts/
│   ├── memory/
│   ├── outreach/
│   ├── tasks/
│   ├── jobs/
│   └── reviews/
├── tests/
│   ├── unit/
│   │   ├── harness/
│   │   └── mcp_servers/
│   ├── integration/
│   │   ├── mcp/
│   │   └── workflows/
│   └── fixtures/
├── artifacts/
├── .transcripts/
├── .task_outputs/
├── .memory/
├── .tasks/
└── .jobs/
```

