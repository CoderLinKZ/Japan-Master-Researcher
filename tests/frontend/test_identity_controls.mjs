import assert from "node:assert/strict";
import test from "node:test";

import {
  identityOptions,
  isHttpUrl,
  noSitePayload,
  selectedSitePayload,
} from "../../frontend/identity.js";

test("missing or empty option lists still permit the manual and no-site path", () => {
  assert.deepEqual(identityOptions({ kind: "TARGET_IDENTITY_CONFIRMATION_REQUIRED" }), []);
  assert.deepEqual(identityOptions({ options: [] }), []);
  assert.deepEqual(noSitePayload(), {
    action: "no_official_site",
    selected_candidate_id: null,
    manual_url: false,
    confirmed_by_user: true,
  });
});

test("candidate and manual URL choices keep explicit user confirmation", () => {
  assert.deepEqual(
    identityOptions({ options: [{ candidate_id: "https://sakailab.com/", title: "酒井研究室" }] })[0].url,
    "https://sakailab.com/",
  );
  assert.deepEqual(selectedSitePayload("https://sakailab.com/"), {
    action: "select_candidate",
    selected_candidate_id: "https://sakailab.com/",
    manual_url: false,
    confirmed_by_user: true,
  });
  assert.equal(selectedSitePayload("https://other.example/lab", { manual: true }).manual_url, true);
  assert.equal(isHttpUrl("https://sakailab.com/"), true);
  assert.equal(isHttpUrl("not-a-url"), false);
  assert.equal(isHttpUrl("https://user:secret@example.com/"), false);
});
