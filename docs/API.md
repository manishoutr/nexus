# API reference

Base URL: `http://localhost:8000`. Interactive docs from FastAPI: `/docs` (Swagger) and `/redoc`.
JSON in, JSON out. Protected routes need `Authorization: Bearer <access token>`.

Errors use `{"detail": "..."}`. Common codes: 401 not signed in or token expired, 403 role too low, 404 not found
**or** no access (private workspaces answer 404 so ids cannot be probed), 413 file too large, 415 file type not allowed,
422 invalid input, 429 rate limit hit.

## Roles
`viewer` (search, ask, read) < `editor` (upload, delete documents) < `owner` (add members). A seeded demo workspace is
public: anyone can read it without signing in.

## Health
| Method + path | Auth | Purpose |
|---|---|---|
| `GET /api/health` | none | Liveness. Returns `status`, `llm_mode` (`offline` or `claude`), and workspace and chunk counts. |
| `GET /api/ready` | none | Readiness: database, migrations at head, storage, and Redis when `QUEUE_BACKEND=celery`. 200 when all pass, 503 otherwise, with the failing check named. |
| `GET /api/metrics` | user | Counters: documents, chunks, jobs by status, questions, abstentions. |

## Auth
| Method + path | Body | Result |
|---|---|---|
| `POST /api/auth/register` | `{email, password}` (password 8-128 chars) | 201 `{access_token, refresh_token, token_type}`. 409 if the email exists. |
| `POST /api/auth/login` | `{email, password}` | 200 tokens. Same 401 for unknown email and wrong password. |
| `POST /api/auth/refresh` | `{refresh_token}` | 200 new tokens. A refresh token is not accepted as an access token. |
| `GET /api/auth/me` | | `{id, email}` |

Access tokens last 30 minutes, refresh tokens 7 days. Auth routes are limited to 10 requests per minute.
There is no password reset, email verification or refresh-token revocation (see known limitations).

## Workspaces
| Method + path | Auth | Purpose |
|---|---|---|
| `GET /api/workspaces` | optional | Public workspaces plus the ones you own or belong to, each with your `role`. |
| `POST /api/workspaces` | user | `{name}` creates a private workspace you own. |
| `POST /api/workspaces/{id}/members` | owner | `{email, role: viewer|editor}` adds or changes a member. 204. |

## Documents
| Method + path | Auth | Purpose |
|---|---|---|
| `POST /api/workspaces/{id}/documents` | editor | Multipart field `file` (.pdf, .docx, .txt, .md, up to `MAX_UPLOAD_MB`). Checked by content, not just extension. 201 with the new document (status `pending`). The same bytes uploaded again return the existing document with 200. |
| `GET /api/workspaces/{id}/documents` | viewer | List with `status`: `pending`, `processing`, `done`, `failed`, and an `error` text when failed. |
| `GET /api/documents/{id}` | viewer | One document. |
| `DELETE /api/documents/{id}` | editor | Removes chunks, jobs, the row and the stored file. 204. |

Ingestion runs in the background (database poller, or Celery when `QUEUE_BACKEND=celery`). Poll the list until the
status is `done`.

## Search and ask
| Method + path | Auth | Body |
|---|---|---|
| `POST /api/workspaces/{id}/search` | viewer | `{query (<=1000 chars), k (1-20, optional)}` -> `{query, count, results:[{chunk_id, document_id, document_name, page, section, text, score}]}` |
| `POST /api/workspaces/{id}/ask` | viewer | `{question (<=2000 chars), k}` -> `{answer, citations:[{chunk_id, document_id, document_name, page, section, excerpt}], mode, abstained, warnings, flagged_passages}` |

`mode` is `offline` (answer built from retrieved sentences) or `claude`. `abstained: true` means the best evidence was
too weak and the answer says so. `flagged_passages` counts retrieved passages that looked like instructions aimed at the
model; they are removed before generation. Search and ask are limited to 20 per minute.

Example:
```bash
TOKEN=$(curl -s localhost:8000/api/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"me@example.com","password":"correct-horse-9"}' | jq -r .access_token)
curl -s localhost:8000/api/workspaces/1/ask -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"question":"Why combine keyword and vector search?"}'
```
