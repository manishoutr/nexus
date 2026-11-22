/* Nexus frontend - Reading room UI. Plain fetch against /api/*. No build step. */
const $ = (sel) => document.querySelector(sel);
let currentWorkspace = null;
let workspaces = [];
let lastQuestion = "";

function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

// --- health badge ---
api("/api/health").then((h) => {
  const badge = $("#mode-badge");
  badge.textContent = h.llm_mode === "claude" ? "Claude connected" : "Offline mode";
  badge.classList.toggle("claude", h.llm_mode === "claude");
});

// --- workspaces ---
async function loadWorkspaces() {
  workspaces = await api("/api/workspaces");
  $("#workspace-select").innerHTML =
    workspaces.map((w) => `<option value="${w.id}">${esc(w.name)}</option>`).join("") ||
    "<option value=''>No workspaces yet</option>";
  currentWorkspace = workspaces.length ? workspaces[workspaces.length - 1].id : null;
  if (workspaces.length) $("#workspace-select").value = currentWorkspace;
  const w = workspaces.find((x) => x.id === currentWorkspace);
  $("#ws-label").textContent = w ? w.name : "No workspace";
  loadDocuments();
}

$("#new-ws-btn").addEventListener("click", () => $("#new-ws-row").classList.toggle("hidden"));

$("#create-workspace").addEventListener("click", async () => {
  const name = $("#new-workspace").value.trim();
  if (!name) return;
  await api("/api/workspaces", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  $("#new-workspace").value = "";
  $("#new-ws-row").classList.add("hidden");
  loadWorkspaces();
});

$("#workspace-select").addEventListener("change", (e) => {
  currentWorkspace = Number(e.target.value) || null;
  const w = workspaces.find((x) => x.id === currentWorkspace);
  $("#ws-label").textContent = w ? w.name : "No workspace";
  loadDocuments();
});

// --- library tree ---
async function loadDocuments() {
  if (!currentWorkspace) { $("#doc-list").innerHTML = ""; return; }
  const docs = await api(`/api/workspaces/${currentWorkspace}/documents`);
  $("#doc-list").innerHTML = docs.map((d) => `
    <div class="doc-item" data-name="${esc(d.filename)}" data-meta='${esc(JSON.stringify({ size: d.size_bytes, pages: d.page_count, status: d.status, error: d.error }))}'>
      ${esc(d.filename)}<span class="status ${d.status}">${d.status}</span>
    </div>`).join("") || "<p class='muted'>No documents yet.</p>";
  document.querySelectorAll(".doc-item").forEach((el) => {
    el.addEventListener("click", () => {
      document.querySelectorAll(".doc-item").forEach((x) => x.classList.remove("selected"));
      el.classList.add("selected");
      const meta = JSON.parse(el.dataset.meta);
      $("#reader-content").innerHTML = `
        <small class="eyebrow">Document</small>
        <h2 class="serif">${esc(el.dataset.name)}</h2>
        <p class="doc">Status: ${esc(meta.status)} · ${(meta.size / 1024).toFixed(1)} KB${meta.pages ? " · " + meta.pages + " pages" : ""}</p>
        ${meta.error ? `<p class="doc">Error: ${esc(meta.error)}</p>` : ""}
        <p class="muted">Search above to read passages from this document, or ask a cited question on the right.</p>`;
    });
  });
  if (docs.some((d) => d.status === "pending" || d.status === "processing")) {
    setTimeout(loadDocuments, 1500);  // poll while ingestion runs
  }
}

// --- upload ---
$("#upload-btn").addEventListener("click", () => {
  if (!currentWorkspace) { $("#upload-status").textContent = "Create a workspace first."; return; }
  $("#doc-file").click();
});
$("#doc-file").addEventListener("change", async () => {
  const file = $("#doc-file").files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  $("#upload-status").textContent = "Uploading...";
  try {
    await api(`/api/workspaces/${currentWorkspace}/documents`, { method: "POST", body: fd });
    $("#upload-status").textContent = "Uploaded. Processing...";
    $("#doc-file").value = "";
    loadDocuments();
  } catch (e) {
    $("#upload-status").textContent = "Error: " + e.message;
  }
});

// --- passage reader helpers ---
function showPassage(eyebrow, title, text) {
  $("#reader-content").innerHTML = `
    <small class="eyebrow">${esc(eyebrow)}</small>
    <h2 class="serif">${esc(title)}</h2>
    <p class="doc passage-body">${esc(text)}</p>`;
}

// --- search ---
async function runSearch() {
  const q = $("#search-input").value.trim();
  if (!q || !currentWorkspace) return;
  $("#reader-content").innerHTML = "<p class='muted'>Searching...</p>";
  try {
    const data = await api(`/api/workspaces/${currentWorkspace}/search`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, k: 5 }),
    });
    if (!data.results.length) {
      $("#reader-content").innerHTML = `
        <small class="eyebrow">Search</small>
        <h2 class="serif">${esc(q)}</h2>
        <p class="doc">No passages found.</p>`;
      return;
    }
    $("#reader-content").innerHTML = `
      <small class="eyebrow">Search results</small>
      <h2 class="serif">${esc(q)}</h2>` +
      data.results.map((r, i) => `
        <div class="result-item" data-i="${i}">
          <span class="score">score ${r.score}</span>
          <span class="loc">${String(i + 1).padStart(2, "0")} / ${esc(r.document_name)}${r.page ? " · page " + r.page : ""}${r.section ? " · " + esc(r.section) : ""}</span>
          <div class="excerpt doc">${esc(r.text.slice(0, 220))}${r.text.length > 220 ? "..." : ""}</div>
        </div>`).join("");
    $("#reader-content").querySelectorAll(".result-item").forEach((el) => {
      el.addEventListener("click", () => {
        const r = data.results[Number(el.dataset.i)];
        showPassage(
          "Retrieved passage" + (r.page ? " / page " + r.page : ""),
          r.document_name + (r.section ? " - " + r.section : ""),
          r.text);
      });
    });
  } catch (e) {
    $("#reader-content").innerHTML = "<p class='muted'>Error: " + esc(e.message) + "</p>";
  }
}

$("#search-btn").addEventListener("click", runSearch);
$("#search-input").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });

// --- ask ---
async function runAsk() {
  const q = $("#ask-input").value.trim();
  if (!q || !currentWorkspace) return;
  lastQuestion = q;
  $("#ask-input").value = "";
  $("#ask-output").innerHTML = "<p class='muted'>Thinking...</p>";
  try {
    const res = await api(`/api/workspaces/${currentWorkspace}/ask`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, k: 5 }),
    });
    // inline citation chips: [n] markers, or [filename, page N] markers from offline mode
    let chipCount = 0;
    let answerHtml = esc(res.answer).replace(/\[([^\]]+)\]/g, (m, inner) => {
      let idx = -1;
      if (/^\d+$/.test(inner.trim())) idx = Number(inner.trim()) - 1;
      else {
        const name = inner.split(",")[0].trim().toLowerCase();
        idx = res.citations.findIndex((c) => c.document_name.toLowerCase() === name);
      }
      if (idx < 0 || !res.citations[idx]) return m;
      chipCount += 1;
      const c = res.citations[idx];
      return `<span class="cite-chip" data-c="${idx}">${idx + 1}${c.page ? " · p" + c.page : ""}</span>`;
    });
    $("#ask-output").innerHTML = `
      <div class="ask-q">${esc(q)}</div>
      <div class="ask-a ${res.abstained ? "abstained" : ""}">${answerHtml}
        <span class="mode-tag">mode: ${res.mode}${res.abstained ? " · abstained (insufficient evidence)" : ""}</span>
      </div>
      ${res.citations.length ? `<span class="sources-pill">${res.citations.length} source${res.citations.length > 1 ? "s" : ""} shown</span>` +
        res.citations.map((c, i) => `
          <div class="citation" data-c="${i}">
            <span class="src">${i + 1}. ${esc(c.document_name)}${c.page ? " · page " + c.page : ""}${c.section ? " · " + esc(c.section) : ""}</span>
            <span class="excerpt">${esc(c.excerpt)}...</span>
          </div>`).join("") : ""}`;
    $("#ask-output").querySelectorAll(".cite-chip, .citation").forEach((el) => {
      el.addEventListener("click", () => {
        const c = res.citations[Number(el.dataset.c)];
        if (c) showPassage("Cited passage" + (c.page ? " / page " + c.page : ""), c.document_name, c.excerpt);
      });
    });
  } catch (e) {
    $("#ask-output").innerHTML = "<p class='muted'>Error: " + esc(e.message) + "</p>";
  }
}

$("#ask-btn").addEventListener("click", runAsk);
$("#ask-input").addEventListener("keydown", (e) => { if (e.key === "Enter") runAsk(); });

loadWorkspaces();
