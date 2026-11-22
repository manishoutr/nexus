// Tiny API client. Tokens live in localStorage (simple; readable by any script on the page,
// which is why the app serves no third-party scripts). A 401 triggers one refresh attempt.
const KEY = "nexus.tokens";

export const tokens = {
  get() { try { return JSON.parse(localStorage.getItem(KEY)) || null; } catch { return null; } },
  set(t) { localStorage.setItem(KEY, JSON.stringify(t)); },
  clear() { localStorage.removeItem(KEY); },
};

async function raw(path, options, access) {
  const headers = { ...(options.headers || {}) };
  if (access) headers.Authorization = `Bearer ${access}`;
  return fetch(path, { ...options, headers });
}

async function tryRefresh() {
  const t = tokens.get();
  if (!t?.refresh_token) return false;
  const res = await fetch("/api/auth/refresh", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: t.refresh_token }),
  });
  if (!res.ok) { tokens.clear(); return false; }
  tokens.set(await res.json());
  return true;
}

export async function api(path, options = {}) {
  let res = await raw(path, options, tokens.get()?.access_token);
  if (res.status === 401 && tokens.get() && !path.startsWith("/api/auth/") && await tryRefresh()) {
    res = await raw(path, options, tokens.get()?.access_token);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = typeof j.detail === "string" ? j.detail : "Request failed"; } catch { /* keep statusText */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

export const post = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export async function signIn(mode, email, password) {
  const t = await post(`/api/auth/${mode}`, { email, password });
  tokens.set(t);
  return api("/api/auth/me");
}
