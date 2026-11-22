import { useCallback, useEffect, useState } from "react";
import { api, post, tokens } from "./api";
import AuthPanel from "./AuthPanel";
import { splitAnswer } from "./citations";

const canWrite = (ws) => ws && (ws.role === "owner" || ws.role === "editor");

export default function App() {
  const [user, setUser] = useState(null);
  const [showAuth, setShowAuth] = useState(false);
  const [health, setHealth] = useState(null);
  const [workspaces, setWorkspaces] = useState([]);
  const [wsId, setWsId] = useState(null);
  const [docs, setDocs] = useState([]);
  const [reader, setReader] = useState(null); // {eyebrow,title,body,results}
  const [searchText, setSearchText] = useState("");
  const [askText, setAskText] = useState("");
  const [ask, setAsk] = useState(null);
  const [status, setStatus] = useState("");
  const [newName, setNewName] = useState("");
  const [showNew, setShowNew] = useState(false);
  const [docsTick, setDocsTick] = useState(0); // bump to reload the list and restart polling

  const ws = workspaces.find((w) => w.id === wsId);

  const loadWorkspaces = useCallback(async () => {
    const list = await api("/api/workspaces");
    setWorkspaces(list);
    setWsId((cur) => (list.some((w) => w.id === cur) ? cur : list.length ? list[list.length - 1].id : null));
  }, []);

  useEffect(() => {
    api("/api/health").then(setHealth).catch(() => {});
    if (tokens.get()) api("/api/auth/me").then(setUser).catch(() => tokens.clear());
  }, []);
  useEffect(() => { loadWorkspaces().catch(() => {}); }, [user, loadWorkspaces]);

  useEffect(() => {
    if (!wsId) { setDocs([]); return undefined; }
    let stop = false;
    let timer;
    async function load() {
      try {
        const d = await api(`/api/workspaces/${wsId}/documents`);
        if (stop) return;
        setDocs(d);
        if (d.some((x) => x.status === "pending" || x.status === "processing")) timer = setTimeout(load, 1500);
      } catch { /* shown elsewhere */ }
    }
    load();
    return () => { stop = true; clearTimeout(timer); };
  }, [wsId, docsTick]);

  function signOut() {
    tokens.clear();
    setUser(null);
    setAsk(null);
    setReader(null);
  }

  async function createWorkspace() {
    if (!newName.trim()) return;
    try {
      const w = await post("/api/workspaces", { name: newName.trim() });
      setNewName(""); setShowNew(false);
      await loadWorkspaces();
      setWsId(w.id);
    } catch (e) { setStatus("Error: " + e.message); }
  }

  async function upload(e) {
    const file = e.target.files[0];
    if (!file || !ws) return;
    const fd = new FormData();
    fd.append("file", file);
    setStatus("Uploading...");
    try {
      await api(`/api/workspaces/${ws.id}/documents`, { method: "POST", body: fd });
      setStatus("Uploaded. Processing...");
      setDocsTick((t) => t + 1);
    } catch (err) { setStatus("Error: " + err.message); }
    e.target.value = "";
  }

  async function removeDoc(d) {
    try {
      await api(`/api/documents/${d.id}`, { method: "DELETE" });
      setDocs((x) => x.filter((y) => y.id !== d.id));
    } catch (err) { setStatus("Error: " + err.message); }
  }

  async function runSearch() {
    if (!searchText.trim() || !ws) return;
    setReader({ eyebrow: "Search", title: searchText, body: "Searching..." });
    try {
      const data = await post(`/api/workspaces/${ws.id}/search`, { query: searchText.trim(), k: 5 });
      setReader({ eyebrow: "Search results", title: searchText, results: data.results });
    } catch (e) { setReader({ eyebrow: "Search", title: searchText, body: "Error: " + e.message }); }
  }

  async function runAsk() {
    const q = askText.trim();
    if (!q || !ws) return;
    setAskText("");
    setAsk({ q, loading: true });
    try {
      setAsk({ q, res: await post(`/api/workspaces/${ws.id}/ask`, { question: q, k: 5 }) });
    } catch (e) { setAsk({ q, error: e.message }); }
  }

  const showCitation = (c) =>
    setReader({ eyebrow: "Cited passage" + (c.page ? ` / page ${c.page}` : ""), title: c.document_name, body: c.excerpt });

  const claude = health?.llm_mode === "claude";

  return (
    <>
      <header className="top">
        <span className="brand">Nexus / Reading room</span>
        <span className="push">
          <span className="ws-label">{ws ? ws.name : "No workspace"}</span>
          <span className={"mode" + (claude ? " claude" : "")}>{health ? (claude ? "Claude connected" : "Offline mode") : "..."}</span>
          {user ? (
            <>
              <span className="ws-label" data-testid="who">{user.email}</span>
              <button className="mini" onClick={signOut}>Sign out</button>
            </>
          ) : (
            <button className="mini" onClick={() => setShowAuth(true)}>Sign in</button>
          )}
        </span>
      </header>

      <div className="shell">
        <aside className="rail">
          <small className="rail-label">Research library</small>
          {showAuth && !user && <AuthPanel onDone={(u) => { setUser(u); setShowAuth(false); }} onCancel={() => setShowAuth(false)} />}
          <div className="ws-row">
            <select aria-label="Workspace" value={wsId ?? ""} onChange={(e) => setWsId(Number(e.target.value) || null)}>
              {workspaces.length === 0 && <option value="">No workspaces yet</option>}
              {workspaces.map((w) => <option key={w.id} value={w.id}>{w.name}{w.is_public ? " (public)" : ""}</option>)}
            </select>
            {user && <button className="mini" onClick={() => setShowNew(!showNew)}>+ New</button>}
          </div>
          {showNew && (
            <div className="ws-row">
              <input placeholder="Workspace name" value={newName} onChange={(e) => setNewName(e.target.value)} />
              <button className="mini" onClick={createWorkspace}>Create</button>
            </div>
          )}
          {ws && <p className="muted">Your role: {ws.role}</p>}
          <div>
            {docs.map((d) => (
              <div key={d.id} className="doc-item"
                   onClick={() => setReader({ eyebrow: "Document", title: d.filename,
                     body: `Status: ${d.status} · ${(d.size_bytes / 1024).toFixed(1)} KB${d.page_count ? ` · ${d.page_count} pages` : ""}${d.error ? ` · Error: ${d.error}` : ""}` })}>
                {d.filename}<span className={"status " + d.status}>{d.status}</span>
                {canWrite(ws) && <button className="mini del" aria-label={`Delete ${d.filename}`}
                  onClick={(e) => { e.stopPropagation(); removeDoc(d); }}>x</button>}
              </div>
            ))}
            {docs.length === 0 && <p className="muted">No documents yet.</p>}
          </div>
          <div className="rail-bottom">
            {canWrite(ws) ? (
              <label className="upload">+ Upload files
                <input type="file" accept=".pdf,.docx,.txt,.md" className="hidden" onChange={upload} />
              </label>
            ) : (
              <p className="muted">{user ? "Read-only workspace." : "Sign in to create a workspace and upload files."}</p>
            )}
            <p className="muted" role="status">{status}</p>
          </div>
        </aside>

        <main className="reader">
          <div className="searchbar">
            <input placeholder="Search the library by keyword or meaning" value={searchText}
                   onChange={(e) => setSearchText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && runSearch()} />
            <button className="btn" onClick={runSearch}>Search</button>
          </div>
          {!reader && (
            <>
              <small className="eyebrow">Welcome</small>
              <h2 className="serif">A quiet library with a question beside the text.</h2>
              <p className="doc">Upload documents, search them here, and ask cited questions on the right. Nexus answers only from retrieved passages and says when the evidence is not enough.{!claude && " Right now it runs in offline mode: answers are extracted from the passages, not written by a language model."}</p>
            </>
          )}
          {reader && (
            <>
              <small className="eyebrow">{reader.eyebrow}</small>
              <h2 className="serif">{reader.title}</h2>
              {reader.body && <p className="doc passage-body">{reader.body}</p>}
              {reader.results && reader.results.length === 0 && <p className="doc">No passages found.</p>}
              {reader.results && reader.results.map((r, i) => (
                <div key={r.chunk_id} className="result-item"
                     onClick={() => setReader({ eyebrow: "Retrieved passage" + (r.page ? ` / page ${r.page}` : ""),
                       title: r.document_name + (r.section ? ` - ${r.section}` : ""), body: r.text })}>
                  <span className="score">score {r.score}</span>
                  <span className="loc">{String(i + 1).padStart(2, "0")} / {r.document_name}{r.page ? ` · page ${r.page}` : ""}{r.section ? ` · ${r.section}` : ""}</span>
                  <div className="excerpt doc">{r.text.slice(0, 220)}{r.text.length > 220 ? "..." : ""}</div>
                </div>
              ))}
            </>
          )}
        </main>

        <aside className="ask-panel">
          <small className="rail-label">Ask the library</small>
          <div id="ask-output">
            {!ask && <p className="muted">Ask a question about the current workspace. Every answer links back to its sources.</p>}
            {ask && <div className="ask-q">{ask.q}</div>}
            {ask?.loading && <p className="muted">Thinking...</p>}
            {ask?.error && <p className="error" role="alert">Error: {ask.error}</p>}
            {ask?.res && (
              <>
                <div className={"ask-a" + (ask.res.abstained ? " abstained" : "")}>
                  {splitAnswer(ask.res.answer, ask.res.citations).map((p, i) =>
                    p.chip !== undefined ? (
                      <span key={i} className="cite-chip" onClick={() => showCitation(ask.res.citations[p.chip])}>
                        {p.chip + 1}{ask.res.citations[p.chip].page ? ` · p${ask.res.citations[p.chip].page}` : ""}
                      </span>
                    ) : <span key={i}>{p.text}</span>)}
                  <span className="mode-tag">mode: {ask.res.mode}{ask.res.abstained ? " · abstained (insufficient evidence)" : ""}</span>
                </div>
                {ask.res.warnings.map((w) => <p key={w} className="muted">{w}</p>)}
                {ask.res.citations.length > 0 && <span className="sources-pill">{ask.res.citations.length} source{ask.res.citations.length > 1 ? "s" : ""} shown</span>}
                {ask.res.citations.map((c, i) => (
                  <div key={c.chunk_id} className="citation" onClick={() => showCitation(c)}>
                    <span className="src">{i + 1}. {c.document_name}{c.page ? ` · page ${c.page}` : ""}{c.section ? ` · ${c.section}` : ""}</span>
                    <span className="excerpt">{c.excerpt}...</span>
                  </div>
                ))}
              </>
            )}
          </div>
          <div className="ask-input-row">
            <input placeholder="Ask a question" value={askText} onChange={(e) => setAskText(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && runAsk()} />
            <button className="mini" aria-label="Ask" onClick={runAsk}>&#8593;</button>
          </div>
        </aside>
      </div>
    </>
  );
}
