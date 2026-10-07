import assert from "node:assert/strict";
import test from "node:test";

import { JmrApi } from "../../frontend/api.js";

test("failed case retry posts to the owned case endpoint", async () => {
  const calls = [];
  globalThis.sessionStorage = {
    getItem: () => "root-token",
    setItem: () => {},
    removeItem: () => {},
  };
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return { ok: true, json: async () => ({ retry_available: false }) };
  };

  const result = await new JmrApi().retry("case/1");

  assert.equal(result.retry_available, false);
  assert.equal(calls[0].url, "/api/v1/cases/case%2F1/retry");
  assert.equal(calls[0].options.method, "POST");
  assert.equal(calls[0].options.headers.Authorization, "Bearer root-token");
});
