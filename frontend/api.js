/** Same-origin, bearer-authenticated HTTP client. Tokens stay in session storage. */
export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

export class JmrApi {
  constructor() { this.token = sessionStorage.getItem("jmr_token") || ""; }
  setToken(token) { this.token = token; sessionStorage.setItem("jmr_token", token); }
  clearToken() { this.token = ""; sessionStorage.removeItem("jmr_token"); }
  async request(path, { method = "GET", body, idempotencyKey } = {}) {
    const headers = { Authorization: `Bearer ${this.token}` };
    if (body !== undefined && !(body instanceof FormData)) headers["Content-Type"] = "application/json";
    if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
    let response;
    try {
      response = await fetch(`/api/v1${path}`, { method, headers, body: body instanceof FormData ? body : body === undefined ? undefined : JSON.stringify(body) });
    } catch { throw new ApiError("无法连接服务，请检查 FastAPI 是否正在运行。", 0); }
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      const detail = data.error?.message || (typeof data.detail === "string" ? data.detail : null);
      throw new ApiError(detail || `请求失败（HTTP ${response.status}）`, response.status);
    }
    return response.json();
  }
  session() { return this.request("/session"); }
  cases() { return this.request("/cases"); }
  createCase(message, key, caseId) { return this.request("/cases", { method: "POST", body: { message, case_id: caseId }, idempotencyKey: key }); }
  case(caseId) { return this.request(`/cases/${encodeURIComponent(caseId)}`); }
  caseDeletionPlan(caseId) { return this.request(`/cases/${encodeURIComponent(caseId)}/deletion-plan`); }
  deleteCase(caseId, planToken) {
    return this.request(`/cases/${encodeURIComponent(caseId)}`, {
      method: "DELETE",
      body: { confirmed_by_user: true, plan_token: planToken },
    });
  }
  progress(caseId) { return this.request(`/cases/${encodeURIComponent(caseId)}/progress`); }
  conversation(caseId) { return this.request(`/cases/${encodeURIComponent(caseId)}/conversation`); }
  workspace(caseId) { return this.request(`/cases/${encodeURIComponent(caseId)}/workspace`); }
  result(caseId) { return this.request(`/cases/${encodeURIComponent(caseId)}/result`); }
  message(caseId, message) { return this.request(`/cases/${encodeURIComponent(caseId)}/messages`, { method: "POST", body: { message } }); }
  resume(caseId, interruptToken, payload) { return this.request(`/cases/${encodeURIComponent(caseId)}/resume`, { method: "POST", body: { interrupt_token: interruptToken, payload } }); }
  retry(caseId) { return this.request(`/cases/${encodeURIComponent(caseId)}/retry`, { method: "POST" }); }
  memories() { return this.request("/memories?limit=100"); }
  createMemory(kind, content) { return this.request("/memories", { method: "POST", body: { kind, content: { text: content }, tags: [], confirmed_by_user: true, idempotency_key: `ui-${crypto.randomUUID()}` } }); }
  deleteMemory(memoryId, expectedVersion, idempotencyKey) {
    return this.request(`/memories/${encodeURIComponent(memoryId)}`, {
      method: "DELETE",
      body: { expected_version: expectedVersion, confirmed_by_user: true, idempotency_key: idempotencyKey },
    });
  }
  upload(file) { const body = new FormData(); body.append("file", file); return this.request("/memories/upload", { method: "POST", body }); }
  async download(memoryId, filename) {
    const response = await fetch(`/api/v1/memories/${encodeURIComponent(memoryId)}/download`, { headers: { Authorization: `Bearer ${this.token}` } });
    if (!response.ok) throw new ApiError("文件下载失败", response.status);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a"); link.href = url; link.download = filename || "document"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}
