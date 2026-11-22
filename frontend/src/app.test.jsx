import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api, tokens } from "./api";
import { splitAnswer } from "./citations";

// A Response body can be read once, so json() returns a thunk that builds a fresh Response per call.
const json = (body, status = 200) => () => Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
const WS = { id: 1, name: "Demo", role: "viewer", is_public: true };

// Route table keyed by "METHOD path". Each call is also recorded for assertions.
function mockApi(routes) {
  const calls = [];
  global.fetch = vi.fn((path, opts = {}) => {
    const key = `${opts.method || "GET"} ${path}`;
    calls.push({ key, headers: opts.headers || {}, body: opts.body });
    const h = routes[key];
    if (!h) return json({ detail: "unmocked " + key }, 404);
    let out = h(opts);
    if (typeof out === "function") out = out();
    return out;
  });
  return calls;
}
const base = {
  "GET /api/health": json({ llm_mode: "offline" }),
  "GET /api/workspaces": json([WS]),
  "GET /api/workspaces/1/documents": json([{ id: 1, workspace_id: 1, filename: "a.txt", size_bytes: 10, page_count: 0, status: "ready", error: "" }]),
};

beforeEach(() => { localStorage.clear(); });

describe("splitAnswer", () => {
  const cites = [{ document_name: "Policy.pdf" }, { document_name: "b.txt" }];
  it("turns numeric and filename markers into chips and keeps text", () => {
    const parts = splitAnswer("Yes [1] and also [b.txt, page 2]. Unknown [9] stays.", cites);
    expect(parts.filter((p) => p.chip !== undefined).map((p) => p.chip)).toEqual([0, 1]);
    expect(parts.map((p) => p.text ?? "").join("")).toContain("[9]");
  });
});

describe("App", () => {
  it("anonymous visitors see the demo, the offline label, and no upload control", async () => {
    mockApi(base);
    render(<App />);
    expect(await screen.findByText("Offline mode")).toBeInTheDocument();
    expect(await screen.findByText("a.txt")).toBeInTheDocument();
    expect(screen.queryByText(/Upload files/)).toBeNull();
    expect(screen.getByText(/Sign in to create a workspace/)).toBeInTheDocument();
  });

  it("signs in, stores tokens, sends the bearer token and shows upload for editors", async () => {
    const calls = mockApi({
      ...base,
      "POST /api/auth/login": json({ access_token: "A1", refresh_token: "R1", token_type: "bearer" }),
      "GET /api/auth/me": json({ id: 5, email: "me@example.com" }),
      "GET /api/workspaces": (o) => json([{ ...WS, role: o.headers?.Authorization ? "owner" : "viewer" }]),
    });
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await userEvent.type(screen.getByPlaceholderText("Email"), "me@example.com");
    await userEvent.type(screen.getByPlaceholderText(/Password/), "hunter2hunter2");
    await userEvent.click(screen.getAllByRole("button", { name: "Sign in" }).pop());
    expect(await screen.findByTestId("who")).toHaveTextContent("me@example.com");
    expect(tokens.get().access_token).toBe("A1");
    expect(await screen.findByText(/Upload files/)).toBeInTheDocument();
    expect(calls.some((c) => c.key === "GET /api/workspaces" && c.headers.Authorization === "Bearer A1")).toBe(true);
  });

  it("shows the server error on a failed login and stores nothing", async () => {
    mockApi({ ...base, "POST /api/auth/login": json({ detail: "Invalid email or password" }, 401) });
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    await userEvent.type(screen.getByPlaceholderText("Email"), "x@y.co");
    await userEvent.type(screen.getByPlaceholderText(/Password/), "wrong");
    await userEvent.click(screen.getAllByRole("button", { name: "Sign in" }).pop());
    expect(await screen.findByRole("alert")).toHaveTextContent("Invalid email or password");
    expect(tokens.get()).toBeNull();
  });

  it("asks a question, renders citation chips and the abstained state", async () => {
    mockApi({
      ...base,
      "POST /api/workspaces/1/ask": json({
        answer: "Refunds take 14 days [1].", mode: "offline", abstained: false, warnings: [], flagged_passages: 0,
        citations: [{ chunk_id: 7, document_id: 1, document_name: "a.txt", page: 2, section: "", excerpt: "Refunds are paid in 14 days" }],
      }),
    });
    render(<App />);
    await screen.findByText("a.txt");
    await userEvent.type(screen.getByPlaceholderText("Ask a question"), "refund time?{Enter}");
    const chip = await screen.findByText(/^1 · p2$/);
    await userEvent.click(chip);
    expect(await screen.findByText("Cited passage / page 2")).toBeInTheDocument();
    expect(screen.getByText(/mode: offline/)).toBeInTheDocument();
  });

  it("shows an abstained answer without citations", async () => {
    mockApi({ ...base, "POST /api/workspaces/1/ask": json({ answer: "I can't answer that from these documents.", mode: "offline", abstained: true, warnings: [], citations: [] }) });
    render(<App />);
    await screen.findByText("a.txt");
    await userEvent.type(screen.getByPlaceholderText("Ask a question"), "capital of mars?{Enter}");
    expect(await screen.findByText(/abstained \(insufficient evidence\)/)).toBeInTheDocument();
    expect(screen.queryByText(/sources? shown/)).toBeNull();
  });
});

describe("api client", () => {
  it("refreshes once on 401 and retries with the new token", async () => {
    tokens.set({ access_token: "old", refresh_token: "R" });
    let n = 0;
    const calls = mockApi({
      "GET /api/things": (o) => (o.headers.Authorization === "Bearer new" ? json({ ok: true }) : (n++, json({ detail: "expired" }, 401))),
      "POST /api/auth/refresh": json({ access_token: "new", refresh_token: "R2", token_type: "bearer" }),
    });
    expect(await api("/api/things")).toEqual({ ok: true });
    expect(n).toBe(1);
    expect(tokens.get().refresh_token).toBe("R2");
    expect(calls.filter((c) => c.key === "POST /api/auth/refresh")).toHaveLength(1);
  });

  it("clears tokens when refresh fails", async () => {
    tokens.set({ access_token: "old", refresh_token: "bad" });
    mockApi({ "GET /api/things": json({ detail: "expired" }, 401), "POST /api/auth/refresh": json({ detail: "no" }, 401) });
    await expect(api("/api/things")).rejects.toThrow("expired");
    await waitFor(() => expect(tokens.get()).toBeNull());
  });
});

describe("upload polling", () => {
  it("keeps polling after an upload until the document is no longer pending", async () => {
    let listCalls = 0;
    mockApi({
      ...base,
      "GET /api/workspaces": json([{ ...WS, role: "owner" }]),
      "GET /api/workspaces/1/documents": () => {
        listCalls += 1;
        return json([{ id: 1, workspace_id: 1, filename: "n.txt", size_bytes: 5, page_count: 0, status: listCalls < 3 ? "pending" : "done", error: "" }]);
      },
      "POST /api/workspaces/1/documents": json({ id: 1 }, 201),
    });
    const { container } = render(<App />);
    await screen.findByText("n.txt");
    const file = new File(["hello"], "n.txt", { type: "text/plain" });
    await userEvent.upload(container.querySelector("input[type=file]"), file);
    await waitFor(() => expect(container.querySelector(".status.done")).not.toBeNull(), { timeout: 6000 });
  }, 10000);
});
