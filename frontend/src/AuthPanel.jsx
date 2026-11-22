import { useState } from "react";
import { signIn } from "./api";

export default function AuthPanel({ onDone, onCancel }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      onDone(await signIn(mode, email.trim(), password));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="auth-card" onSubmit={submit} aria-label="Sign in">
      <small className="rail-label">{mode === "login" ? "Sign in" : "Create account"}</small>
      <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} required />
      <input type="password" placeholder="Password (8+ characters)" value={password}
             onChange={(e) => setPassword(e.target.value)} required minLength={mode === "register" ? 8 : 1} />
      {error && <p className="error" role="alert">{error}</p>}
      <button className="btn" disabled={busy}>{mode === "login" ? "Sign in" : "Register"}</button>
      <button type="button" className="mini" onClick={() => setMode(mode === "login" ? "register" : "login")}>
        {mode === "login" ? "New here? Create an account" : "Have an account? Sign in"}
      </button>
      <button type="button" className="mini" onClick={onCancel}>Keep browsing the demo</button>
    </form>
  );
}
