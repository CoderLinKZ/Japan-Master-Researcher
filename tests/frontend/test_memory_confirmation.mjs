import assert from "node:assert/strict";
import test from "node:test";

import {
  memoryChoices,
  memoryConfirmationValid,
  paperChoices,
  selectedPaperPayload,
  selectedMemoryPayload,
  unpersonalizedPayload,
} from "../../frontend/memory-confirmation.js";

test("empty paused memory list still offers a valid unpersonalized path", () => {
  assert.deepEqual(memoryChoices({ options: [] }), []);
  assert.deepEqual(unpersonalizedPayload(), {
    selected_memory_ids: [],
    allow_unpersonalized: true,
  });
});

test("current active memories replace stale paused options after an upload", () => {
  const paused = { options: [{ memory_id: "old", label: "旧内容", summary: "已删除" }] };
  const current = [
    { memory_id: "new", status: "ACTIVE", value: { content: { filename: "profile.txt", text: "我的研究经历" } } },
    { memory_id: "deleted", status: "DELETED", value: { content: { text: "不可用" } } },
  ];
  assert.deepEqual(memoryChoices(paused, current), [
    { id: "new", title: "profile.txt", description: "我的研究经历" },
  ]);
  assert.deepEqual(selectedMemoryPayload(["new", "new"]), {
    selected_memory_ids: ["new"],
    allow_unpersonalized: false,
  });
});

test("paused options remain usable if refreshing memories fails", () => {
  assert.deepEqual(memoryChoices({ options: [{ memory_id: "m-1", label: "个人经历", summary: { text: "机器学习" } }] }), [
    { id: "m-1", title: "个人经历", description: '{"text":"机器学习"}' },
  ]);
});

test("paper selection is required before confirming memory mode", () => {
  const interrupt = { paper_options: [
    { evidence_id: "paper-1", title: "Recent paper", recent: true },
    { evidence_id: "paper-1", title: "Duplicate" },
  ] };
  assert.deepEqual(paperChoices(interrupt), [interrupt.paper_options[0]]);
  assert.deepEqual(selectedPaperPayload(["paper-1", "paper-1"]), {
    selected_paper_evidence_ids: ["paper-1"],
  });
  assert.equal(memoryConfirmationValid({ ...unpersonalizedPayload(), selected_paper_evidence_ids: [] }), false);
  assert.equal(memoryConfirmationValid({ ...unpersonalizedPayload(), selected_paper_evidence_ids: ["paper-1"] }), true);
  assert.equal(memoryConfirmationValid({ ...unpersonalizedPayload(), selected_paper_evidence_ids: ["a", "b", "c", "d"] }), false);
});
