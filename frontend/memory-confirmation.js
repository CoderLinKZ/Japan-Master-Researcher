/** Normalize the paused workflow's snapshot and the user's current active memories. */
export function memoryChoices(interrupt, currentMemories = null) {
  const items = Array.isArray(currentMemories)
    ? currentMemories.filter((item) => item?.status === undefined || item.status === "ACTIVE")
    : Array.isArray(interrupt?.options) ? interrupt.options : [];
  const seen = new Set();
  return items.flatMap((item) => {
    const id = item?.memory_id;
    if (typeof id !== "string" || !id || seen.has(id)) return [];
    seen.add(id);
    const value = item.value || {};
    const content = value.content || {};
    const title = content.filename || value.label || item.label || "背景记忆";
    const raw = content.text || value.summary || item.summary || content;
    const description = typeof raw === "string" ? raw : JSON.stringify(raw);
    return [{ id, title, description: description || "可用于个性化研究方向" }];
  });
}

export function selectedMemoryPayload(ids) {
  return { selected_memory_ids: [...new Set(ids)], allow_unpersonalized: false };
}

export function unpersonalizedPayload() {
  return { selected_memory_ids: [], allow_unpersonalized: true };
}

export function paperChoices(interrupt) {
  const items = Array.isArray(interrupt?.paper_options) ? interrupt.paper_options : [];
  const seen = new Set();
  return items.filter((item) => {
    if (typeof item?.evidence_id !== "string" || !item.evidence_id || seen.has(item.evidence_id)) return false;
    seen.add(item.evidence_id);
    return typeof item.title === "string" && !!item.title.trim();
  });
}

export function selectedPaperPayload(ids) {
  return { selected_paper_evidence_ids: [...new Set(ids)] };
}

export function memoryConfirmationValid(payload) {
  if (!payload || typeof payload !== "object") return false;
  const papers = payload.selected_paper_evidence_ids;
  const memories = payload.selected_memory_ids;
  return Array.isArray(papers)
    && papers.length >= 1 && papers.length <= 3
    && papers.every((id) => typeof id === "string" && !!id)
    && new Set(papers).size === papers.length
    && Array.isArray(memories)
    && memories.every((id) => typeof id === "string" && !!id)
    && Boolean(memories.length) !== (payload.allow_unpersonalized === true);
}
