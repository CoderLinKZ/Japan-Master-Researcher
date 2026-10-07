import { JmrApi } from "./api.js";
import { identityOptions, isHttpUrl, noSitePayload, selectedSitePayload } from "./identity.js";
import { memoryChoices, memoryConfirmationValid, paperChoices, selectedMemoryPayload, selectedPaperPayload, unpersonalizedPayload } from "./memory-confirmation.js";

const api = new JmrApi();
const $ = (id) => document.getElementById(id);
const state = { user: "root", cases: [], caseId: null, case: null, workspace: null, conversation: [], timeline: [], pendingMessage: null, pendingBaseUserCount: 0, memories: [], memoriesLoaded: false, deletingMemory: null, deletingCase: null, tab: "overview", busy: false, page: "cases", requestId: 0 };
const labels = {
  VALIDATING_APPLICATION_INPUTS: "确认申请信息", RETRIEVING_RESEARCH_EVIDENCE: "检索研究资料",
  MERGING_AND_VERIFYING_EVIDENCE: "核验资料", GENERATING_RESEARCH_DIRECTIONS: "生成研究 Idea",
  AWAITING_DIRECTION_SELECTION: "等待选择 Idea", DRAFTING_OUTREACH_RESEARCH_PLAN: "起草套磁信",
  REVIEWING_OUTREACH_RESEARCH_PLAN: "审阅草稿", COMPLETED: "已完成",
  READY: "准备就绪", WAITING_FOR_USER: "等待你的确认", RUNNING: "研究进行中", FAILED: "执行失败", BLOCKED: "受阻",
  SUCCESS: "已完成", PARTIAL: "部分完成", NO_RESULT: "暂无结果", NEEDS_USER_CONFIRMATION: "等待确认",
  TARGET_IDENTITY_CONFIRMATION_REQUIRED: "确认研究室官网",
  EVIDENCE_SOURCE_REQUIRED: "补充研究资料",
  APPLICANT_MEMORY_CONFIRMATION_REQUIRED: "选择论文与背景",
  DIRECTION_SELECTION_REQUIRED: "选择研究方向",
  REVISION_INPUT_REQUIRED: "提供草稿修改意见",
  plan_research: "规划研究资料检索", generate_direction_candidates: "生成研究方向",
};
function el(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = String(text); return node; }
function clear(node) { node.replaceChildren(); }
function nice(value) { return labels[value] || value || "—"; }
function short(value, size = 90) { const text = String(value ?? ""); return text.length > size ? `${text.slice(0, size)}…` : text; }
function summary(value) {
  if (value == null) return "—";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(summary).join(" · ");
  if (typeof value === "object") return value.title || value.name || value.summary || value.topic || value.text || JSON.stringify(value, null, 2);
  return String(value);
}
function toast(message) { const node = $("toast"); node.textContent = message; node.classList.remove("hidden"); clearTimeout(toast.timer); toast.timer = setTimeout(() => node.classList.add("hidden"), 4200); }
function showError(error) { const node = $("error-banner"); node.textContent = error.message || String(error); node.classList.remove("hidden"); }
function hideError() { $("error-banner").classList.add("hidden"); }
function setBusy(value) { state.busy = value; $("send-button").disabled = value; $("chat-input").disabled = value; $("send-button").textContent = value ? "…" : "↑"; $("composer-hint").textContent = value ? "Agent 正在处理，请稍候…" : "Enter 发送 · Shift+Enter 换行"; }
function openDetail() { $("detail-pane").classList.add("mobile-visible"); document.body.classList.add("detail-open"); }
function closeDetail() { $("detail-pane").classList.remove("mobile-visible"); document.body.classList.remove("detail-open"); }
function focusCurrentStep() {
  if (state.case?.pending_interrupt) $("interrupt-panel").scrollIntoView({ block: "start" });
  else $("chat-scroll").scrollTop = $("chat-scroll").scrollHeight;
}

function caseTitle(item) {
  const target = item?.target;
  const title = target && (target.professor_name || target.professor || target.lab_name || target.laboratory || target.name || target.university);
  return title ? short(title, 24) : `研究 ${short(item?.case_id || "", 8)}`;
}
function renderCases() {
  const list = $("case-list"); clear(list); $("case-count").textContent = state.cases.length;
  if (!state.cases.length) { list.append(el("p", "muted small", "还没有研究记录")); return; }
  for (const item of state.cases) {
    const row = el("div", "case-row");
    const button = el("button", `case-item${item.case_id === state.caseId ? " selected" : ""}`);
    button.type = "button";
    const icon = el("span", "case-icon", "✳");
    const copy = el("span", "case-copy"); copy.append(el("strong", "", caseTitle(item)), el("small", "", nice(item.workflow_stage)));
    button.append(icon, copy); button.title = item.case_id; button.addEventListener("click", () => openCase(item.case_id));
    const deleteButton = el("button", "case-delete", "删除"); deleteButton.type = "button";
    deleteButton.setAttribute("aria-label", `删除研究：${caseTitle(item)}`);
    deleteButton.addEventListener("click", () => prepareCaseDeletion(item, deleteButton));
    row.append(button, deleteButton); list.append(row);
  }
}
async function prepareCaseDeletion(item, button) {
  if (state.busy) { toast("请等待当前请求完成后再删除研究"); return; }
  if (state.deletingCase) return;
  const selected = { id: item.case_id, title: caseTitle(item), plan: null, submitting: false };
  state.deletingCase = selected;
  button.disabled = true;
  try {
    const plan = await api.caseDeletionPlan(selected.id);
    if (state.deletingCase !== selected) return;
    if (!plan.dry_run || !plan.plan_token || plan.case_id !== selected.id) throw new Error("删除预览无效，请稍后重试");
    selected.plan = plan;
    $("delete-case-name").textContent = selected.title;
    $("delete-case-id").textContent = selected.id;
    $("delete-case-file-count").textContent = String(plan.object_count);
    $("delete-case-error").classList.add("hidden");
    $("delete-case-dialog").showModal();
  } catch (error) {
    if (state.deletingCase === selected) state.deletingCase = null;
    toast(error.message || "无法预览删除范围");
  } finally { button.disabled = false; }
}
function chatRow(kind, content) {
  const row = el("div", `message-row ${kind === "user" ? "user-row" : "agent-row"}`);
  const avatar = el("span", `avatar ${kind}`, kind === "user" ? state.user[0]?.toUpperCase() || "U" : "✳");
  if (kind === "user") row.append(content, avatar);
  else row.append(avatar, content);
  return row;
}
function agentBubble(title, body, meta = "研究 Agent") {
  const wrap = el("div", "agent-body agent-bubble");
  wrap.append(el("div", "agent-name", `✳ ${meta}`), el("h3", "agent-heading", title));
  if (body) wrap.append(el("p", "agent-copy", body));
  return wrap;
}
function processSteps(events) {
  const steps = el("ol", "process-steps");
  for (const event of events) {
    const step = el("li", `process-step ${event.kind === "mcp_tool" ? "tool-step" : "stage-step"}`);
    if (event.kind === "stage") {
      step.append(el("span", "process-mark", "●"), el("span", "process-label", nice(event.stage)));
      step.append(el("small", "process-meta", nice(event.status)));
    } else {
      step.append(el("span", "process-mark", "⌘"), el("span", "process-label", event.tool || "MCP Tool"));
      step.append(el("small", "process-meta", nice(event.status)));
    }
    steps.append(step);
  }
  return steps;
}
function renderProcess(events) {
  if (!events.length) return null;
  const body = agentBubble("研究进程", "每一步都来自已保存的阶段记录和 MCP 调用审计。");
  if (events.length > 8) {
    const older = el("details", "older-steps");
    older.append(el("summary", "", `查看更早的 ${events.length - 8} 个步骤`), processSteps(events.slice(0, -8)));
    body.append(older);
  }
  body.append(processSteps(events.slice(-8)));
  return chatRow("agent", body);
}
function renderDirection(event) {
  const content = event.content || {};
  const title = content.title || content.topic || `研究方向 ${event.ordinal || ""}`;
  const description = content.summary || content.description || content.research_question || "已生成一个可供选择的研究方向。";
  const body = agentBubble(title, description, `研究 Agent · 第 ${(event.revision_round || 0) + 1} 轮方向`);
  const details = el("details", "artifact-details");
  details.append(el("summary", "", "查看完整方向"), el("pre", "", JSON.stringify(content, null, 2)));
  body.append(details);
  return chatRow("agent", body);
}
function renderChat() {
  const list = $("message-list"); clear(list);
  const active = !!state.caseId || !!state.pendingMessage;
  $("welcome").classList.toggle("hidden", active);
  const knownCase = state.cases.find((x) => x.case_id === state.caseId);
  $("page-title").textContent = state.pendingMessage && !knownCase ? "正在创建研究" : state.caseId ? caseTitle(knownCase || { case_id: state.caseId, target: state.workspace?.target }) : "新的研究";
  const status = state.case?.retry_available ? "FAILED" : state.case?.run_status || (state.pendingMessage ? "RUNNING" : "READY");
  $("status-badge").textContent = nice(status);
  $("status-badge").dataset.status = status;
  const events = state.timeline.length ? [...state.timeline] : state.conversation.map((message, index) => ({ id: `fallback-${index}`, kind: "user_message", content: message.content }));
  if (state.pendingMessage && events.filter((event) => event.kind === "user_message").length <= state.pendingBaseUserCount) {
    const pending = { id: "pending-message", kind: "user_message", content: state.pendingMessage };
    if (state.pendingBaseUserCount === 0) events.unshift(pending);
    else events.push(pending);
  }
  let progress = [];
  const flushProgress = () => { const row = renderProcess(progress); if (row) list.append(row); progress = []; };
  for (const event of events) {
    if (event.kind === "stage" || event.kind === "mcp_tool") { progress.push(event); continue; }
    flushProgress();
    if (event.kind === "user_message") list.append(chatRow("user", el("div", "message-bubble", event.content)));
    if (event.kind === "direction") list.append(renderDirection(event));
    if (event.kind === "draft") list.append(chatRow("agent", agentBubble(`套磁信草稿 · V${event.version}`, event.content, "研究 Agent · 草稿")));
  }
  flushProgress();
  if (active) {
    const body = agentBubble(state.pendingMessage && !state.caseId ? "正在启动研究" : `当前阶段：${nice(state.case?.workflow_stage)}`, state.pendingMessage && !state.caseId ? "提问已提交，正在等待 Agent 返回。" : `状态：${nice(status)}`);
    if (state.case?.warnings?.length) body.append(el("p", "agent-warning", `提醒：${state.case.warnings.join("；")}`));
    if (state.case?.errors?.length) body.append(el("p", "agent-error", state.case.errors.join("；")));
    if (state.case?.retry_available) {
      const failure = state.case.failure;
      body.append(el("p", "agent-error", `${nice(failure?.node || state.case.workflow_stage)}：${failure?.message || "当前阶段执行失败；可从保存的进度重试。"}`));
      const retryButton = el("button", "inline-action", "重试当前阶段 →");
      retryButton.disabled = state.busy;
      retryButton.addEventListener("click", async () => {
        if (state.busy) return;
        setBusy(true); hideError();
        try { state.case = await api.retry(state.caseId); await refreshCurrent(); toast("已从保存的进度继续"); }
        catch (error) { await refreshCurrent().catch(() => {}); showError(error); }
        finally { setBusy(false); renderChat(); }
      });
      body.append(retryButton);
    }
    if (state.case?.result_available) {
      const resultButton = el("button", "inline-action", "查看最终套磁信 →"); resultButton.addEventListener("click", () => { state.tab = "drafts"; renderDetail(); openDetail(); }); body.append(resultButton);
    }
    list.append(chatRow("agent", body));
  }
  renderInterrupt();
  $("chat-input").placeholder = state.case?.retry_available ? "请先重试失败的阶段…" : state.case?.pending_interrupt ? "请先完成上方的人工确认…" : state.caseId ? "继续补充说明或提出修改意见…" : "描述你想研究的教授、研究室或问题…";
  $("chat-input").disabled = state.busy || !!state.case?.pending_interrupt || !!state.case?.retry_available || ["COMPLETED", "CANCELLED"].includes(status);
}
function renderInterrupt() {
  const panel = $("interrupt-panel"); clear(panel);
  const interrupt = state.case?.pending_interrupt; panel.classList.toggle("hidden", !interrupt); if (!interrupt) return;
  const isIdentity = interrupt.kind === "TARGET_IDENTITY_CONFIRMATION_REQUIRED";
  const isMemory = interrupt.kind === "APPLICANT_MEMORY_CONFIRMATION_REQUIRED";
  const isDirection = interrupt.kind === "DIRECTION_SELECTION_REQUIRED";
  panel.append(el("span", "interrupt-tag", "需要你确认"), el("h3", "", nice(interrupt.kind)), el("p", "", isMemory ? "先选择 1–3 篇教授论文，Agent 会以这些论文为依据提出可继续研究的 Idea。个人背景可选；不选背景也不影响论文选择。" : isIdentity ? "请从下方选择：确认检索到的网址、自己填写网址，或说明没有研究室官网。" : interrupt.prompt || "Agent 已暂停等待你的决定。完成下方内容后继续。"));
  const details = el("details", "interrupt-details"); details.append(el("summary", "", "查看 Agent 的完整请求"), el("pre", "", JSON.stringify(interrupt, null, 2))); if (!isMemory && !isDirection) panel.append(details);
  const textarea = el("textarea", "interrupt-input"); textarea.rows = 7; textarea.setAttribute("aria-label", "人工确认 JSON");
  const defaultPayload = { idempotency_key: `ui-${crypto.randomUUID()}` };
  if (interrupt.kind === "APPLICATION_INPUT_REQUIRED") defaultPayload.fields = interrupt.current_values || {};
  if (interrupt.kind === "DIRECTION_SELECTION_REQUIRED") { defaultPayload.direction_batch_id = interrupt.direction_batch_id; defaultPayload.selected_direction_ids = []; }
  if (isMemory) { defaultPayload.selected_paper_evidence_ids = []; defaultPayload.selected_memory_ids = []; defaultPayload.allow_unpersonalized = false; }
  if (interrupt.kind === "REVISION_INPUT_REQUIRED") { defaultPayload.action = "feedback"; defaultPayload.feedback = ""; }
  if (interrupt.kind === "EVIDENCE_SOURCE_REQUIRED") { defaultPayload.action = "provide_sources"; defaultPayload.source_urls = []; }
  const updatePayload = (change) => { let current; try { current = JSON.parse(textarea.value); } catch { current = defaultPayload; } textarea.value = JSON.stringify({ ...current, ...change }, null, 2); };
  textarea.value = JSON.stringify(defaultPayload, null, 2);
  if (interrupt.kind === "APPLICATION_INPUT_REQUIRED") {
    const required = [...new Set([...(interrupt.required_fields || []), ...(interrupt.invalid_fields || [])])];
    for (const name of required) {
      const label = el("label", "interrupt-field", name);
      const input = el("input", ""); input.value = String(interrupt.current_values?.[name] ?? ""); input.placeholder = `请填写 ${name}`;
      input.addEventListener("input", () => { let current; try { current = JSON.parse(textarea.value); } catch { current = defaultPayload; } updatePayload({ fields: { ...(current.fields || {}), [name]: input.value } }); });
      label.append(input); panel.append(label);
    }
  }
  if (interrupt.kind === "REVISION_INPUT_REQUIRED") {
    const label = el("label", "interrupt-field", "你的修改意见"); const input = el("textarea", ""); input.rows = 4; input.placeholder = "例如：突出我在机器学习项目中的经历，并让语气更自然。";
    input.addEventListener("input", () => updatePayload({ action: "feedback", feedback: input.value })); label.append(input); panel.append(label);
  }
  if (interrupt.kind === "EVIDENCE_SOURCE_REQUIRED") {
    const label = el("label", "interrupt-field", "可引用的来源网址（每行一个）"); const input = el("textarea", ""); input.rows = 3; input.placeholder = "https://...";
    input.addEventListener("input", () => updatePayload({ action: "provide_sources", source_urls: input.value.split(/\r?\n/).map((x) => x.trim()).filter(Boolean) })); label.append(input); panel.append(label);
  }
  let directionSubmit = null;
  if (isDirection) {
    const section = el("section", "direction-confirmation");
    section.append(el("p", "memory-guidance", "选择 1–2 个生成的方向，或在下方填写自己的方向。"));
    const choices = el("div", "interrupt-choices direction-candidates");
    const selected = new Set();
    const label = el("label", "interrupt-field direction-custom", "我想研究其他方向");
    const customInput = el("textarea", ""); customInput.rows = 3; customInput.placeholder = "写明你想研究的具体问题或方向…";
    for (const option of Array.isArray(interrupt.options) ? interrupt.options : []) {
      if (!option.direction_id) continue;
      const choice = el("button", "choice-button", `${option.title || "研究方向"} · ${short(option.summary || "", 100)}`); choice.type = "button"; choice.setAttribute("aria-pressed", "false");
      choice.addEventListener("click", () => {
        if (selected.has(option.direction_id)) selected.delete(option.direction_id); else if (selected.size < 2) selected.add(option.direction_id); else { toast("最多选择两个方向"); return; }
        customInput.value = "";
        updatePayload({ selected_direction_ids: [...selected], custom_direction: null });
        choice.classList.toggle("selected", selected.has(option.direction_id)); choice.setAttribute("aria-pressed", String(selected.has(option.direction_id)));
        directionSubmit.disabled = selected.size === 0;
        directionSubmit.textContent = selected.size ? `使用 ${selected.size} 个方向并继续 →` : "请先选择或填写方向";
      }); choices.append(choice);
    }
    customInput.addEventListener("input", () => {
      selected.clear();
      for (const choice of choices.children) { choice.classList.remove("selected"); choice.setAttribute("aria-pressed", "false"); }
      updatePayload({ selected_direction_ids: [], custom_direction: customInput.value.trim() || null });
      directionSubmit.disabled = !customInput.value.trim();
      directionSubmit.textContent = customInput.value.trim() ? "使用自定义方向并继续 →" : "请先选择或填写方向";
    });
    label.append(customInput); section.append(choices, label); panel.append(section);
  }
  let identitySubmit = null;
  if (isIdentity) {
    const options = identityOptions(interrupt);
    const section = el("section", "identity-section");
    if (options.length) {
      section.append(el("p", "identity-guidance", "检索到的候选网址（选择前请核对教授与学校）："));
      const choices = el("div", "interrupt-choices identity-candidates");
      for (const option of options) {
        const url = option.url;
        const choice = el("button", "choice-button identity-candidate"); choice.type = "button";
        choice.append(el("strong", "", "官网首页候选 · 请核对"), el("span", "identity-url", url));
        if (option.title) choice.append(el("small", "identity-source", `搜索依据：${option.title}`));
        choice.addEventListener("click", () => {
          for (const button of choices.children) button.classList.remove("selected");
          noSite.classList.remove("selected");
          choice.classList.add("selected");
          input.value = url;
          updatePayload(selectedSitePayload(url));
          identitySubmit.disabled = false;
          identitySubmit.textContent = "确认这个网址并继续 →";
        });
        choices.append(choice);
      }
      section.append(choices);
    } else {
      section.append(el("p", "identity-guidance", "暂时没有可信的官网候选。你可以重新检索；知道网址就填在下面；确实没有官网则选择继续。"));
    }
    const label = el("label", "interrupt-field identity-url-label", "我知道官网网址，粘贴在这里");
    const input = el("input", ""); input.id = "identity-url-input"; input.type = "url"; input.inputMode = "url"; input.placeholder = "粘贴以 https:// 开头的真实官网链接";
    const urlFeedback = el("p", "identity-url-feedback hidden", "请输入完整、有效的 http:// 或 https:// 网址。");
    input.addEventListener("input", () => {
      for (const choice of section.querySelectorAll(".identity-candidate")) choice.classList.remove("selected");
      noSite.classList.remove("selected");
      const url = input.value.trim();
      updatePayload(selectedSitePayload(url, { manual: true }));
      identitySubmit.disabled = !isHttpUrl(url);
      identitySubmit.textContent = "使用填写的网址继续 →";
      urlFeedback.classList.toggle("hidden", !url || isHttpUrl(url));
    });
    label.append(input, urlFeedback); section.append(label);
    const actions = el("div", "identity-actions");
    const noSite = el("button", "choice-button identity-no-site", "我确认没有研究室官网"); noSite.type = "button";
    noSite.addEventListener("click", () => {
      input.value = "";
      for (const choice of section.querySelectorAll(".identity-candidate")) choice.classList.remove("selected");
      noSite.classList.add("selected");
      updatePayload(noSitePayload());
      identitySubmit.disabled = false;
      identitySubmit.textContent = "确认没有官网并继续 →";
    });
    const retry = el("button", "secondary-button identity-retry", "重新检索官网"); retry.type = "button";
    retry.addEventListener("click", () => submitInterrupt({ action: "retry_discovery", idempotency_key: `ui-${crypto.randomUUID()}` }, retry));
    actions.append(noSite, retry); section.append(actions);
    section.append(el("p", "identity-note", "没有官网不会阻止继续核验论文等资料；若所有来源都缺少可引用证据，Agent 会请你补充来源。"));
    panel.append(section);
  }
  let memorySubmit = null;
  if (isMemory) {
    const papers = paperChoices(interrupt);
    const options = memoryChoices(interrupt, state.memoriesLoaded ? state.memories : null);
    const section = el("section", "memory-confirmation");
    section.append(el("h4", "memory-subheading", "第一步 · 选择教授论文"));
    section.append(el("p", "memory-guidance", papers.length ? "这里只显示过去 12 个月内的论文；最多选择 3 篇。请核对标题、日期与来源。" : "过去 12 个月内没有可选择的已核验论文。请先补充论文来源。"));
    const paperList = el("div", "interrupt-choices paper-candidates");
    const selectedPapers = new Set();
    const updateMemoryReady = () => {
      let payload; try { payload = JSON.parse(textarea.value); } catch { payload = null; }
      memorySubmit.disabled = !memoryConfirmationValid(payload);
      memorySubmit.textContent = memorySubmit.disabled ? "请先选论文和背景方式" : `基于 ${selectedPapers.size || payload.selected_paper_evidence_ids.length} 篇论文生成 Idea →`;
    };
    for (const paper of papers) {
      const card = el("div", "paper-choice-card");
      const choice = el("button", "choice-button paper-candidate"); choice.type = "button"; choice.setAttribute("aria-pressed", "false");
      choice.append(el("strong", "", paper.title), el("small", "", `${paper.publication_date || "日期待核实"} · ${paper.venue || "来源待核实"} · ${paper.verification_status === "VERIFIED" ? "已核验" : "部分核验"} · 近一年`));
      choice.addEventListener("click", () => {
        if (selectedPapers.has(paper.evidence_id)) selectedPapers.delete(paper.evidence_id);
        else if (selectedPapers.size < 3) selectedPapers.add(paper.evidence_id);
        else { toast("最多选择三篇论文"); return; }
        choice.classList.toggle("selected", selectedPapers.has(paper.evidence_id)); choice.setAttribute("aria-pressed", String(selectedPapers.has(paper.evidence_id)));
        updatePayload(selectedPaperPayload([...selectedPapers])); updateMemoryReady();
      });
      card.append(choice);
      if (isHttpUrl(paper.source_url || "")) { const source = el("a", "paper-source", "打开论文来源 ↗"); source.href = paper.source_url; source.target = "_blank"; source.rel = "noopener noreferrer"; card.append(source); }
      paperList.append(card);
    }
    section.append(paperList, el("h4", "memory-subheading", "第二步 · 选择个人背景"));
    section.append(el("p", "memory-guidance", options.length ? "可选择一条或多条已保存的背景记忆；也可以不使用背景。" : "当前没有长期记忆。你可以添加，或者选择不使用背景。"));
    const choices = el("div", "interrupt-choices memory-candidates");
    const selected = new Set();
    const skip = el("button", "choice-button memory-skip", "不使用个人背景，只依据所选论文"); skip.type = "button"; skip.setAttribute("aria-pressed", "false");
    for (const option of options) {
      const choice = el("button", "choice-button memory-candidate"); choice.type = "button"; choice.setAttribute("aria-pressed", "false");
      choice.append(el("strong", "", option.title), el("span", "", short(option.description, 180)));
      choice.addEventListener("click", () => {
        if (selected.has(option.id)) selected.delete(option.id); else selected.add(option.id);
        choice.classList.toggle("selected", selected.has(option.id)); choice.setAttribute("aria-pressed", String(selected.has(option.id)));
        skip.classList.remove("selected"); skip.setAttribute("aria-pressed", "false");
        updatePayload(selectedMemoryPayload([...selected]));
        updateMemoryReady();
      });
      choices.append(choice);
    }
    section.append(choices);
    skip.addEventListener("click", () => {
      selected.clear();
      for (const choice of choices.children) { choice.classList.remove("selected"); choice.setAttribute("aria-pressed", "false"); }
      skip.classList.add("selected"); skip.setAttribute("aria-pressed", "true");
      updatePayload(unpersonalizedPayload());
      updateMemoryReady();
    });
    section.append(skip);
    const actions = el("div", "memory-add-actions");
    const add = el("button", "secondary-button", "＋ 手动添加背景"); add.type = "button"; add.addEventListener("click", () => $("memory-dialog").showModal());
    const upload = el("button", "secondary-button", "↑ 上传背景文件"); upload.type = "button"; upload.addEventListener("click", () => $("file-input").click());
    actions.append(add, upload); section.append(actions);
    panel.append(section);
  }
  const button = el("button", "primary-button", isIdentity ? "请先选择上方的一项" : isMemory ? "请先选论文和背景方式" : isDirection ? "请先选择或填写方向" : "确认并继续 →"); button.type = "button";
  if (isIdentity) { button.disabled = true; identitySubmit = button; }
  if (isMemory) { button.disabled = true; memorySubmit = button; }
  if (isDirection) { button.disabled = true; directionSubmit = button; }
  async function submitInterrupt(payload, trigger) {
    if (!state.case?.interrupt_token) { toast("缺少中断 Token，请刷新 Case"); return; }
    trigger.disabled = true; setBusy(true); hideError();
    try { state.case = await api.resume(state.caseId, state.case.interrupt_token, payload); await refreshCurrent(); focusCurrentStep(); toast("已提交确认"); }
    catch (error) { await refreshCurrent().catch(() => {}); showError(error); trigger.disabled = false; }
    finally { setBusy(false); }
  }
  button.addEventListener("click", () => {
    let payload; try { payload = JSON.parse(textarea.value); if (!payload || Array.isArray(payload) || typeof payload !== "object") throw new Error(); } catch { toast("请输入有效的 JSON 对象"); return; }
    if (isMemory) {
      if (!memoryConfirmationValid(payload)) { toast("请选择 1–3 篇论文，并确认是否使用背景记忆"); return; }
    }
    if (isDirection) {
      const ids = payload.selected_direction_ids;
      if (!Array.isArray(ids) || ids.length > 2 || !ids.every((id) => typeof id === "string" && id) || Boolean(ids.length) === Boolean(payload.custom_direction)) { toast("请选择 1–2 个方向，或填写一个自定义方向"); return; }
    }
    submitInterrupt(payload, button);
  }); panel.append(button);
  if (isMemory || isDirection) {
    panel.append(details);
    const advanced = el("details", "interrupt-details advanced"); advanced.append(el("summary", "", "高级：编辑提交 JSON"), textarea); panel.append(advanced);
    textarea.addEventListener("input", () => {
      let payload; try { payload = JSON.parse(textarea.value); } catch { button.disabled = true; return; }
      if (isMemory) {
        button.disabled = !memoryConfirmationValid(payload);
      } else {
        const ids = payload?.selected_direction_ids;
        button.disabled = !Array.isArray(ids) || ids.length > 2 || !ids.every((id) => typeof id === "string" && id) || Boolean(ids.length) === Boolean(payload.custom_direction);
      }
      button.textContent = button.disabled ? isMemory ? "请先选论文和背景方式" : "请先选择或填写方向" : "确认 JSON 选择并继续 →";
    });
  } else {
    const advanced = el("details", "interrupt-details advanced"); advanced.append(el("summary", "", "高级：编辑提交 JSON"), textarea); panel.insertBefore(advanced, button);
  }
}
function card(title, subtitle, body, meta) {
  const node = el("article", "data-card"); node.append(el("div", "data-meta", meta || "研究记录"), el("h3", "", title || "未命名"));
  if (subtitle) node.append(el("p", "data-subtitle", subtitle));
  if (body) { const pre = el("pre", "data-body", typeof body === "string" ? body : JSON.stringify(body, null, 2)); node.append(pre); }
  return node;
}
function renderDetail() {
  const content = $("detail-content"); clear(content);
  for (const button of $("detail-tabs").querySelectorAll("button")) button.classList.toggle("active", button.dataset.tab === state.tab);
  const workspace = state.workspace;
  $("detail-intro").textContent = workspace ? `${caseTitle({ case_id: state.caseId, target: workspace.target })} · ${state.caseId}` : "选择一个 Case 后，可在这里查看论文、KAKEN、Idea 和套磁信版本。";
  if (!workspace) { content.append(el("div", "detail-empty", "✳\n研究资料会在这里逐步汇集")); return; }
  const evidence = workspace.evidence || [];
  const papers = evidence.filter((x) => !/kaken|grant|project|科研费/i.test(x.kind || ""));
  const kaken = evidence.filter((x) => /kaken|grant|project|科研费/i.test(x.kind || ""));
  if (state.tab === "overview") {
    const stats = el("div", "stats-grid");
    for (const [label, value] of [["论文与资料", papers.length], ["KAKEN", kaken.length], ["研究 Idea", workspace.directions.length], ["草稿版本", workspace.drafts.length]]) { const item = el("div", "stat"); item.append(el("strong", "", value), el("span", "", label)); stats.append(item); }
    content.append(stats);
    if (workspace.target) content.append(card("目标研究室", "当前 Case 的目标信息", workspace.target, "TARGET"));
    if (state.case?.selected_papers?.length) {
      const chosen = state.case.selected_papers.map((paper) => `${paper.title}${paper.publication_date ? ` (${paper.publication_date})` : ""}`);
      content.append(card("本 Case 选择的论文", `${chosen.length} 篇 · Idea 以这些论文为依据`, chosen, "PAPERS"));
    }
    for (const plan of workspace.plans.slice(0, 2)) content.append(card("研究计划", plan.created_at?.slice(0, 10), plan.content, "PLAN"));
    if (workspace.selected_memory_ids?.length) content.append(card("本 Case 使用的长期记忆", `${workspace.selected_memory_ids.length} 条`, workspace.selected_memory_ids, "MEMORY"));
    for (const run of workspace.retrieval_runs.slice(0, 6)) content.append(card(nice(run.worker_kind), nice(run.status), run.summary, "RETRIEVAL"));
  } else if (state.tab === "papers" || state.tab === "kaken") {
    const items = state.tab === "papers" ? papers : kaken;
    for (const item of items) content.append(card(item.title || item.external_id || item.kind, `${nice(item.verification_status)} · ${item.external_id || ""}`, item.record, item.kind));
    if (state.tab === "kaken") for (const run of workspace.retrieval_runs.filter((x) => /kaken/i.test(x.worker_kind))) content.append(card("KAKEN 检索", nice(run.status), run.summary, "RETRIEVAL"));
    if (!items.length && !(state.tab === "kaken" && workspace.retrieval_runs.some((x) => /kaken/i.test(x.worker_kind)))) content.append(el("div", "detail-empty", "暂无对应研究资料"));
  } else if (state.tab === "ideas") {
    for (const item of workspace.directions) content.append(card(`Idea ${item.ordinal} · 第 ${item.revision_round + 1} 轮`, item.id, item.content, "RESEARCH IDEA"));
  } else if (state.tab === "selected") {
    for (const item of workspace.selections) {
      const chosen = item.selected_direction_ids.map((id) => workspace.directions.find((x) => x.id === id)?.content || id);
      content.append(card("已选研究方向", item.created_at?.slice(0, 10), item.custom_direction || chosen, "SELECTED"));
    }
  } else if (state.tab === "drafts") {
    for (const item of workspace.drafts) content.append(card(`套磁信草稿 · V${item.version}`, `${item.language} · ${item.created_at?.slice(0, 10)}`, item.content, "DRAFT"));
  } else if (state.tab === "reviews") {
    for (const item of workspace.reviews) content.append(card(`${nice(item.reviewer_kind)} · ${nice(item.status)}`, `关联草稿 ${short(item.draft_id, 8)}`, item.result, "REVIEW"));
  }
  if (!content.children.length) content.append(el("div", "detail-empty", "当前阶段暂无记录。Agent 运行后会自动出现在这里。"));
}
function renderMemories() {
  const list = $("memory-list"); clear(list);
  const query = $("memory-search").value.trim().toLowerCase();
  const items = state.memories.filter((x) => JSON.stringify(x.value || {}).toLowerCase().includes(query));
  if (!items.length) { list.append(el("div", "memory-empty", "这里还没有匹配的长期记忆。上传文件或手动添加一条吧。")); return; }
  for (const item of items) {
    const value = item.value || {}; const content = value.content || {}; const node = el("article", "memory-card");
    node.append(el("span", "memory-type", value.kind === "uploaded_document" ? "↥ 上传文件" : `◈ ${value.kind || "记忆"}`));
    node.append(el("h3", "", content.filename || short(content.text || summary(content), 45)));
    node.append(el("p", "", short(content.text || summary(content), 180)));
    const foot = el("div", "memory-foot"); foot.append(el("small", "", `V${item.version} · ${(item.created_at || "").slice(0, 10)}`));
    const actions = el("div", "memory-actions");
    if (value.kind === "uploaded_document") { const button = el("button", "text-button", "下载原文件"); button.type = "button"; button.addEventListener("click", () => api.download(item.memory_id, content.filename).catch((error) => toast(error.message))); actions.append(button); }
    const deleteButton = el("button", "delete-button", "删除"); deleteButton.type = "button";
    deleteButton.setAttribute("aria-label", `删除记忆：${content.filename || short(content.text || summary(content), 30)}`);
    deleteButton.addEventListener("click", () => {
      state.deletingMemory = { id: item.memory_id, version: item.version, key: `ui-delete-${crypto.randomUUID()}` };
      $("delete-memory-name").textContent = content.filename || short(content.text || summary(content), 55);
      $("delete-memory-error").classList.add("hidden");
      $("delete-memory-dialog").showModal();
    });
    actions.append(deleteButton); foot.append(actions);
    node.append(foot); list.append(node);
  }
}
function setPage(page) {
  state.page = page; $("chat-page").classList.toggle("hidden", page !== "cases"); $("memory-page").classList.toggle("hidden", page !== "memory");
  $("nav-cases").classList.toggle("active", page === "cases"); $("nav-memory").classList.toggle("active", page === "memory");
  $("detail-pane").classList.toggle("hidden-for-memory", page === "memory");
  document.querySelector(".app-shell").classList.toggle("memory-mode", page === "memory");
  $("open-detail").classList.toggle("hidden", page === "memory");
  if (page === "memory") closeDetail();
  if (page === "memory") { $("page-title").textContent = "长期记忆"; loadMemories(); } else renderChat();
  document.body.classList.remove("menu-open");
}
async function loadCases() { const data = await api.cases(); state.cases = data.items || []; renderCases(); }
async function loadMemories() { try { state.memories = (await api.memories()).items || []; state.memoriesLoaded = true; renderMemories(); if (state.page === "cases" && state.case?.pending_interrupt?.kind === "APPLICANT_MEMORY_CONFIRMATION_REQUIRED") renderInterrupt(); } catch (error) { toast(error.message); } }
async function openCase(caseId) {
  if (state.caseId !== caseId) state.case = null;
  state.caseId = caseId; state.tab = "overview"; setPage("cases"); state.workspace = null; state.conversation = []; state.timeline = []; renderCases(); renderChat(); renderDetail();
  const requestId = ++state.requestId;
  try {
    const [caseResult, conversationResult, workspaceResult] = await Promise.allSettled([api.case(caseId), api.conversation(caseId), api.workspace(caseId)]);
    if (requestId !== state.requestId) return;
    if (caseResult.status === "rejected") throw caseResult.reason;
    state.case = caseResult.value;
    if (state.case?.pending_interrupt?.kind === "APPLICANT_MEMORY_CONFIRMATION_REQUIRED") {
      try { state.memories = (await api.memories()).items || []; state.memoriesLoaded = true; } catch { state.memoriesLoaded = false; }
      if (requestId !== state.requestId) return;
    }
    if (conversationResult.status === "fulfilled") { state.conversation = conversationResult.value.messages || []; state.timeline = conversationResult.value.events || []; }
    if (workspaceResult.status === "fulfilled") state.workspace = workspaceResult.value;
    renderChat(); renderDetail();
    if (state.case?.pending_interrupt) focusCurrentStep();
    else $("chat-scroll").scrollTop = 0;
  } catch (error) { if (requestId === state.requestId) showError(error); }
  document.body.classList.remove("menu-open");
}
async function refreshCurrent() {
  await loadCases();
  if (state.caseId) await openCase(state.caseId);
  if (state.page === "memory") await loadMemories();
}
function startProgressPolling(caseId) {
  let active = true; let timer;
  const poll = async () => {
    if (!active) return;
    try {
      const progress = await api.progress(caseId);
      if (!active) return;
      if (state.caseId === caseId) {
        state.case = { ...(state.case || {}), workflow_stage: progress.workflow_stage, run_status: progress.run_status };
        state.timeline = progress.events || [];
        renderChat();
        $("chat-scroll").scrollTop = $("chat-scroll").scrollHeight;
      }
    } catch { /* The case may not exist yet; the final request reports errors. */ }
    if (active) timer = setTimeout(poll, 1500);
  };
  poll();
  return () => { active = false; clearTimeout(timer); };
}
function newCase() { ++state.requestId; state.caseId = null; state.case = null; state.workspace = null; state.conversation = []; state.timeline = []; state.pendingMessage = null; hideError(); setPage("cases"); renderCases(); renderDetail(); $("chat-input").focus(); }
async function connect() {
  const token = $("login-token").value.trim(); if (!token) return;
  const oldToken = api.token; api.setToken(token);
  try {
    const session = await api.session();
    const expected = $("login-user").value.trim();
    if (session.user_id !== expected) throw new Error(`此 Token 属于 ${session.user_id}，不是 ${expected}。`);
    if ($("delete-case-dialog").open) $("delete-case-dialog").close();
    state.deletingCase = null;
    state.user = session.user_id; state.memories = []; state.memoriesLoaded = false; $("active-user").textContent = state.user; $("connection-label").textContent = "已连接服务"; $("connection-dot").classList.add("online");
    $("login-error").classList.add("hidden"); $("login-dialog").close(); $("login-token").value = ""; newCase(); await loadCases();
  } catch (error) {
    api.setToken(oldToken); $("login-error").textContent = error.message; $("login-error").classList.remove("hidden");
  }
}
$("login-form").addEventListener("submit", (event) => { event.preventDefault(); connect(); });
$("login-cancel").addEventListener("click", () => $("login-dialog").close());
$("user-switch").addEventListener("click", () => { $("login-user").value = state.user; $("login-dialog").showModal(); });
$("new-case").addEventListener("click", newCase);
$("nav-cases").addEventListener("click", () => setPage("cases"));
$("nav-memory").addEventListener("click", () => setPage("memory"));
$("refresh").addEventListener("click", () => refreshCurrent().catch(showError));
$("mobile-menu").addEventListener("click", () => document.body.classList.toggle("menu-open"));
$("close-detail").addEventListener("click", closeDetail);
$("detail-backdrop").addEventListener("click", closeDetail);
$("open-detail").addEventListener("click", openDetail);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDetail(); });
$("detail-tabs").addEventListener("click", (event) => { const tab = event.target.closest("button[data-tab]"); if (tab) { state.tab = tab.dataset.tab; renderDetail(); } });
for (const button of document.querySelectorAll("[data-prompt]")) button.addEventListener("click", () => { $("chat-input").value = button.dataset.prompt; $("chat-input").focus(); });
$("chat-input").addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("chat-form").requestSubmit(); } });
$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const message = $("chat-input").value.trim(); if (!message || state.busy) return;
  if (state.case?.pending_interrupt) { toast("请先完成当前人工确认"); return; }
  const previousCaseId = state.caseId;
  const requestedCaseId = previousCaseId || crypto.randomUUID();
  const key = `ui-${crypto.randomUUID()}`;
  state.pendingBaseUserCount = state.timeline.filter((item) => item.kind === "user_message").length;
  state.pendingMessage = message;
  state.caseId = requestedCaseId;
  setBusy(true); hideError(); renderChat();
  if (previousCaseId) $("chat-scroll").scrollTop = $("chat-scroll").scrollHeight;
  const stopPolling = startProgressPolling(requestedCaseId);
  let accepted = false;
  try {
    const view = previousCaseId ? await api.message(requestedCaseId, message) : await api.createCase(message, key, requestedCaseId);
    accepted = true;
    state.caseId = view.case_id; state.case = view; $("chat-input").value = "";
    await refreshCurrent();
    focusCurrentStep();
  } catch (error) {
    if (!previousCaseId && !accepted) {
      try {
        const progress = await api.progress(requestedCaseId);
        state.case = { workflow_stage: progress.workflow_stage, run_status: progress.run_status };
        state.timeline = progress.events || [];
        await loadCases();
      } catch {
        state.caseId = null; state.case = null; state.timeline = []; $("chat-input").value = message;
      }
    }
    if (state.caseId) await refreshCurrent().catch(() => {});
    showError(error);
  } finally { stopPolling(); state.pendingMessage = null; setBusy(false); renderChat(); if (state.case?.pending_interrupt) focusCurrentStep(); }
});
$("file-input").addEventListener("change", async (event) => {
  const file = event.target.files?.[0]; if (!file) return;
  if (file.size > 2 * 1024 * 1024) { toast("文件不能超过 2 MB"); event.target.value = ""; return; }
  toast("正在提取文件并保存长期记忆…");
  try { await api.upload(file); await loadMemories(); toast("文件已存入长期记忆"); }
  catch (error) { toast(error.message); }
  finally { event.target.value = ""; }
});
$("memory-search").addEventListener("input", renderMemories);
$("add-memory").addEventListener("click", () => $("memory-dialog").showModal());
$("memory-cancel").addEventListener("click", () => $("memory-dialog").close());
$("memory-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try { await api.createMemory($("memory-kind").value.trim(), $("memory-content").value.trim()); $("memory-dialog").close(); $("memory-content").value = ""; await loadMemories(); toast("记忆已保存"); }
  catch (error) { $("memory-error").textContent = error.message; $("memory-error").classList.remove("hidden"); }
});
$("delete-case-cancel").addEventListener("click", () => $("delete-case-dialog").close());
$("delete-case-dialog").addEventListener("cancel", (event) => { if (state.deletingCase?.submitting) event.preventDefault(); });
$("delete-case-dialog").addEventListener("close", () => { state.deletingCase = null; });
$("delete-case-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const selected = state.deletingCase;
  if (!selected?.plan || selected.submitting) return;
  if (state.busy) { $("delete-case-error").textContent = "请等待当前请求完成后再删除研究"; $("delete-case-error").classList.remove("hidden"); return; }
  selected.submitting = true;
  const confirmButton = $("delete-case-confirm"); const cancelButton = $("delete-case-cancel");
  confirmButton.disabled = true; cancelButton.disabled = true;
  $("delete-case-error").classList.add("hidden");
  try {
    const result = await api.deleteCase(selected.id, selected.plan.plan_token);
    if (result.dry_run || result.status !== "COMPLETED" || result.case_id !== selected.id) throw new Error("服务端未确认删除完成，请刷新后核对");
    $("delete-case-dialog").close();
    state.cases = state.cases.filter((item) => item.case_id !== selected.id);
    if (state.caseId === selected.id) newCase(); else renderCases();
    try { await loadCases(); toast("研究记录已永久删除"); }
    catch { toast("研究已删除，但列表刷新失败；请稍后手动刷新"); }
  } catch (error) {
    if (error.status === 409) {
      try {
        const plan = await api.caseDeletionPlan(selected.id);
        selected.plan = plan;
        $("delete-case-file-count").textContent = String(plan.object_count);
        $("delete-case-error").textContent = "研究内容已变化，删除范围已重新预览。请再次确认。";
      } catch { $("delete-case-error").textContent = "删除预览已失效，请取消后重试。"; selected.plan = null; }
    } else $("delete-case-error").textContent = error.message || "删除失败，请稍后重试";
    $("delete-case-error").classList.remove("hidden");
  } finally {
    selected.submitting = false;
    confirmButton.disabled = !selected.plan;
    cancelButton.disabled = false;
  }
});
$("delete-memory-cancel").addEventListener("click", () => { $("delete-memory-dialog").close(); state.deletingMemory = null; });
$("delete-memory-dialog").addEventListener("close", () => { state.deletingMemory = null; });
$("delete-memory-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const selected = state.deletingMemory;
  if (!selected) return;
  const confirmButton = $("delete-memory-confirm");
  confirmButton.disabled = true;
  $("delete-memory-error").classList.add("hidden");
  try {
    await api.deleteMemory(selected.id, selected.version, selected.key);
    $("delete-memory-dialog").close();
    state.deletingMemory = null;
    await loadMemories();
    toast("记忆已从活动列表删除");
  } catch (error) {
    if (error.status === 409 || error.status === 404) {
      await loadMemories();
      $("delete-memory-dialog").close();
      state.deletingMemory = null;
      toast("记忆状态已变化，请检查刷新后的列表");
    } else {
      $("delete-memory-error").textContent = error.message;
      $("delete-memory-error").classList.remove("hidden");
    }
  } finally {
    confirmButton.disabled = false;
  }
});

async function boot() {
  renderChat(); renderDetail();
  if (!api.token) { $("login-dialog").showModal(); return; }
  try { const session = await api.session(); state.user = session.user_id; $("active-user").textContent = state.user; $("connection-label").textContent = "已连接服务"; $("connection-dot").classList.add("online"); await loadCases(); }
  catch { api.clearToken(); $("login-dialog").showModal(); }
}
boot();
