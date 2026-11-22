"""Baseline schema (everything up to the Phase 4 upgrade).

Idempotent on purpose: databases created before Alembic existed (by
create_all plus the old ALTER shim) already have some of these tables and
columns. Missing tables are created; missing columns are added; nothing that
exists is touched. Fresh databases get the full schema from scratch.

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _tables():
    return [
        ("users", [
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("email", sa.String, nullable=False),
            sa.Column("password_hash", sa.String, nullable=False),
            sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        ]),
        ("workspaces", [
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("name", sa.String, nullable=False),
            sa.Column("owner_id", sa.Integer, nullable=True),
            sa.Column("is_public", sa.Integer, server_default="0"),
            sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        ]),
        ("memberships", [
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("workspace_id", sa.Integer, nullable=False),
            sa.Column("user_id", sa.Integer, nullable=False),
            sa.Column("role", sa.String, nullable=False),
            sa.UniqueConstraint("workspace_id", "user_id", name="uq_member"),
        ]),
        ("documents", [
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("workspace_id", sa.Integer),
            sa.Column("filename", sa.String, nullable=False),
            sa.Column("storage_key", sa.String, server_default=""),
            sa.Column("content_hash", sa.String, nullable=True),
            sa.Column("content_type", sa.String, server_default=""),
            sa.Column("size_bytes", sa.Integer, server_default="0"),
            sa.Column("page_count", sa.Integer, server_default="0"),
            sa.Column("status", sa.String, server_default="pending"),
            sa.Column("error", sa.Text, server_default=""),
            sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
            sa.UniqueConstraint("workspace_id", "content_hash", name="uq_doc_ws_hash"),
        ]),
        ("chunks", [
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("document_id", sa.Integer),
            sa.Column("workspace_id", sa.Integer),
            sa.Column("seq", sa.Integer, server_default="0"),
            sa.Column("text", sa.Text, server_default=""),
            sa.Column("page", sa.Integer, server_default="0"),
            sa.Column("section", sa.String, server_default=""),
            sa.Column("embedding", sa.Text, server_default="[]"),
            sa.Column("embedding_model", sa.String, server_default=""),
        ]),
        ("ingestion_jobs", [
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("document_id", sa.Integer),
            sa.Column("status", sa.String, server_default="pending"),
            sa.Column("attempts", sa.Integer, server_default="0"),
            sa.Column("max_attempts", sa.Integer, server_default="3"),
            sa.Column("next_run_at", sa.DateTime, nullable=True),
            sa.Column("locked_at", sa.DateTime, nullable=True),
            sa.Column("worker_id", sa.String, server_default=""),
            sa.Column("error", sa.Text, server_default=""),
            sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
        ]),
        ("qa_logs", [
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("workspace_id", sa.Integer),
            sa.Column("question", sa.Text),
            sa.Column("answer", sa.Text),
            sa.Column("citations", sa.Text, server_default="[]"),
            sa.Column("mode", sa.String, server_default="offline"),
            sa.Column("abstained", sa.Integer, server_default="0"),
            sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        ]),
    ]


INDEXES = [
    ("ix_users_email", "users", ["email"], True),
    ("ix_workspaces_owner_id", "workspaces", ["owner_id"], False),
    ("ix_memberships_workspace_id", "memberships", ["workspace_id"], False),
    ("ix_memberships_user_id", "memberships", ["user_id"], False),
    ("ix_documents_workspace_id", "documents", ["workspace_id"], False),
    ("ix_documents_status", "documents", ["status"], False),
    ("ix_chunks_document_id", "chunks", ["document_id"], False),
    ("ix_chunks_workspace_id", "chunks", ["workspace_id"], False),
    ("ix_ingestion_jobs_document_id", "ingestion_jobs", ["document_id"], False),
    ("ix_ingestion_jobs_status", "ingestion_jobs", ["status"], False),
    ("ix_qa_logs_workspace_id", "qa_logs", ["workspace_id"], False),
]


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = set(insp.get_table_names())
    for name, cols in _tables():
        if name not in existing:
            op.create_table(name, *cols)
            continue
        have = {c["name"] for c in insp.get_columns(name)}
        for c in cols:
            if isinstance(c, sa.Column) and c.name not in have:
                with op.batch_alter_table(name) as b:
                    b.add_column(sa.Column(c.name, c.type, nullable=True,
                                           server_default=c.server_default))
    insp = sa.inspect(bind)
    for ix, table, cols, unique in INDEXES:
        have_ix = {i["name"] for i in insp.get_indexes(table)}
        if ix not in have_ix:
            op.create_index(ix, table, cols, unique=unique)
    insp = sa.inspect(bind)
    have_ix = {i["name"] for i in insp.get_indexes("documents")}
    uniques = {u["name"] for u in insp.get_unique_constraints("documents")}
    if "uq_doc_ws_hash" not in have_ix | uniques:
        op.create_index("uq_doc_ws_hash", "documents", ["workspace_id", "content_hash"], unique=True)


def downgrade() -> None:
    for name, _ in reversed(_tables()):
        op.drop_table(name)
