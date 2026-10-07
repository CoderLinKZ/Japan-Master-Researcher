/** Small, testable helpers for the human official-site confirmation form. */

export function identityOptions(interrupt) {
  const raw = Array.isArray(interrupt?.options) ? interrupt.options : [];
  return raw
    .filter((item) => item && typeof item === "object")
    .map((item) => ({ ...item, url: item.url || item.candidate_id }))
    .filter((item) => typeof item.url === "string" && isHttpUrl(item.url));
}

export function isHttpUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol)
      && !!url.hostname && !url.username && !url.password;
  } catch {
    return false;
  }
}

export function selectedSitePayload(url, { manual = false } = {}) {
  return {
    action: "select_candidate",
    selected_candidate_id: url,
    manual_url: manual,
    confirmed_by_user: true,
  };
}

export function noSitePayload() {
  return {
    action: "no_official_site",
    selected_candidate_id: null,
    manual_url: false,
    confirmed_by_user: true,
  };
}
