/* Nexus frontend - plain fetch against /api/*. No build step. */
const $ = (sel) => document.querySelector(sel);
let currentWorkspace = null;

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

// --- tabs ---
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("#tab-" + btn.dataset.tab).classList.add("active");
  });
});

// --- health badge ---
api("/api/health").then((h) => {
  const badge = $("#mode-badge");
  badge.textContent = h.llm_mode === "claude" ? "Claude connected" : "Offline mode";
  badge.classList.toggle("claude", h.llm_mode === "claude");
});

// --- workspaces ---
async function loadWorkspaces() {
  const list = await api("/api/workspaces");
  $("#workspace-select").innerHTML =
    list.map((w) => `<option value="${w.id}">${esc(w.name)}</option>`).join("") ||
    "<option value=''>No workspaces yet</option>";
  currentWorkspace = list.length ? list[list.length - 1].id : null;
  if (list.length) $("#workspace-select").value = currentWorkspace;
  loadDocuments();
}

$("#create-workspace").addEventListener("click", async () => {
  const name = $("#new-workspace").value.trim();
  if (!name) return;
  await api("/api/workspaces", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  $("#new-workspace").value = "";
  loadWorkspaces();
});

$("#workspace-select").addEventListener("change", (e) => {
  currentWorkspace = Number(e.target.value) || null;
  loadDocuments();
});

// --- documents ---
async function loadDocuments() {
  if (!currentWorkspace) { $("#doc-list").innerHTML = ""; return; }
  const docs = await api(`/api/workspaces/${currentWorkspace}/documents`);
  $("#doc-list").innerHTML = docs.map((d) => `
    <div class="card">
      <h3>${esc(d.filename)} <span class="status ${d.status}">${d.status}</span></h3>
      <div class="meta">${(d.size_bytes / 1024).toFixed(1)} KB${d.page_count ? " · " + d.page_count + " pages" : ""}${d.error ? " · " + esc(d.error) : ""}</div>
    </div>`).join("") || "<p class='hint'>No documents yet.</p>";
  if (docs.some((d) => d.status === "pending" || d.status === "processing")) {
    setTimeout(loadDocuments, 1500);  // poll while ingestion runs
  }
}

$("#upload-btn").addEventListener("click", async () => {
  if (!currentWorkspace) { $("#upload-status").textContent = "Create a workspace first."; return; }
  const file = $("#doc-file").files[0];
  if (!file) { $("#upload-status").textContent = "Choose a file first."; return; }
  const fd = new FormData();
  fd.append("file", file);
  $("#upload-status").textContent = "Uploading...";
  try {
    await api(`/api/workspaces/${currentWorkspace}/documents`, { method: "POST", body: fd });
    $("#upload-status").textContent = "Uploaded. Processing...";
    loadDocuments();
  } catch (e) {
    $("#upload-status").textContent = "Error: " + e.message;
  }
});

// --- search ---
$("#search-btn").addEventListener("click", async () => {
  const q = $("#search-input").value.trim();
  if (!q || !currentWorkspace) return;
  $("#search-results").innerHTML = "<p class='hint'>Searching...</p>";
  try {
    const data = await api(`/api/workspaces/${currentWorkspace}/search`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, k: 5 }),
    });
    $("#search-results").innerHTML = data.results.map((r) => `
      <div class="card">
        <span class="score">score ${r.score}</span>
        <h3>${esc(r.document_name)}${r.page ? " · page " + r.page : ""}${r.section ? " · " + esc(r.section) : ""}</h3>
        <div class="passage">${esc(r.text)}</div>
      </div>`).join("") || "<p class='hint'>No passages found.</p>";
  } catch (e) {
    $("#search-results").innerHTML = "<p class='hint'>Error: " + esc(e.message) + "</p>";
  }
});

// --- ask ---
$("#ask-btn").addEventListener("click", async () => {
  const q = $("#ask-input").value.trim();
  if (!q || !currentWorkspace) return;
  $("#ask-output").innerHTML = "<p class='hint'>Thinking...</p>";
  try {
    const res = await api(`/api/workspaces/${currentWorkspace}/ask`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, k: 5 }),
    });
    $("#ask-output").innerHTML = `
      <div class="answer ${res.abstained ? "abstained" : ""}">${esc(res.answer)}
        <span class="mode-tag">mode: ${res.mode}${res.abstained ? " · abstained (insufficient evidence)" : ""}</span>
      </div>
      ${res.citations.length ? `<div class="citations"><h3 style="margin:12px 0 8px">Sources</h3>` +
        res.citations.map((c) => `
          <div class="citation">
            <span class="src">${esc(c.document_name)}${c.page ? " · page " + c.page : ""}${c.section ? " · " + esc(c.section) : ""}</span>
            <span class="excerpt">${esc(c.excerpt)}...</span>
          </div>`).join("") + "</div>" : ""}`;
  } catch (e) {
    $("#ask-output").innerHTML = "<p class='hint'>Error: " + esc(e.message) + "</p>";
  }
});

$("#search-input").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#search-btn").click(); });
$("#ask-input").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#ask-btn").click(); });

loadWorkspaces();
