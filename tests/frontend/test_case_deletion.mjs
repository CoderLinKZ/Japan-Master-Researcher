import assert from "node:assert/strict";
import test from "node:test";

import { JmrApi } from "../../frontend/api.js";

test("case deletion previews before sending a confirmed DELETE", async () => {
  const calls = [];
  globalThis.sessionStorage = { getItem: () => "root-token", setItem: () => {}, removeItem: () => {} };
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return { ok: true, json: async () => ({ case_id: "case/1", plan_token: "a".repeat(32) }) };
  };

  const api = new JmrApi();
  const preview = await api.caseDeletionPlan("case/1");
  await api.deleteCase("case/1", preview.plan_token);

  assert.equal(calls[0].url, "/api/v1/cases/case%2F1/deletion-plan");
  assert.equal(calls[0].options.method, "GET");
  assert.equal(calls[1].url, "/api/v1/cases/case%2F1");
  assert.equal(calls[1].options.method, "DELETE");
  assert.equal(calls[1].options.headers.Authorization, "Bearer root-token");
  assert.deepEqual(JSON.parse(calls[1].options.body), {
    confirmed_by_user: true,
    plan_token: "a".repeat(32),
  });
});
