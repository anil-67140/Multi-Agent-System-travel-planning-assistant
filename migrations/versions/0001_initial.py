"""initial schema: users, preferences, plans, itinerary versions, approvals, sources, run events

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("failed_logins", sa.Integer, nullable=False, server_default="0"),
        sa.Column("locked_until", TS, nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "user_preferences",
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("home_city", sa.String(80)),
        sa.Column("currency", sa.String(3)),
        sa.Column("style", sa.String(16)),
        sa.Column("pace", sa.String(16)),
        sa.Column("dietary", sa.JSON, nullable=False),
        sa.Column("interests", sa.JSON, nullable=False),
        sa.Column("bundle_affinity", sa.JSON, nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
    )

    op.create_table(
        "plans",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("run_no", sa.Integer, nullable=False, server_default="0"),
        sa.Column("destination", sa.String(120)),
        sa.Column("start_date", sa.Date),
        sa.Column("end_date", sa.Date),
        sa.Column("travelers", sa.Integer),
        sa.Column("currency", sa.String(3)),
        sa.Column("budget_amount", sa.Numeric(18, 4)),
        sa.Column("trip", sa.JSON),
        sa.Column("budget", sa.JSON),
        sa.Column("bundles", sa.JSON),
        sa.Column("selected_label", sa.String(24)),
        sa.Column("notes", sa.JSON, nullable=False),
        sa.Column("tool_errors", sa.JSON, nullable=False),
        sa.Column("issues", sa.JSON, nullable=False),
        sa.Column("clarification", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column("heartbeat_at", TS),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
    )
    op.create_index("ix_plans_user_id", "plans", ["user_id"])
    op.create_index("ix_plans_status", "plans", ["status"])
    op.create_index("ix_plans_user_created", "plans", ["user_id", "created_at"])

    op.create_table(
        "itinerary_versions",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "plan_id", sa.String(32), sa.ForeignKey("plans.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("itinerary", sa.JSON, nullable=False),
        sa.Column("feedback", sa.Text),
        sa.Column("created_at", TS, nullable=False),
        sa.UniqueConstraint("plan_id", "version", name="uq_plan_version"),
    )
    op.create_index("ix_itinerary_versions_plan_id", "itinerary_versions", ["plan_id"])

    op.create_table(
        "approvals",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "plan_id", sa.String(32), sa.ForeignKey("plans.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("itinerary_version", sa.Integer, nullable=False),
        sa.Column("approved", sa.Boolean, nullable=False),
        sa.Column("feedback", sa.Text),
        sa.Column("idempotency_key", sa.String(80), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.UniqueConstraint("plan_id", "idempotency_key", name="uq_approval_idem"),
    )
    op.create_index("ix_approvals_plan_id", "approvals", ["plan_id"])

    op.create_table(
        "source_records",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "plan_id", sa.String(32), sa.ForeignKey("plans.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("provider", sa.String(60), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("origin", sa.String(16), nullable=False),
        sa.Column("url", sa.String(2000)),
        sa.Column("title", sa.String(300)),
        sa.Column("retrieved_at", TS, nullable=False),
    )
    op.create_index("ix_source_records_plan_id", "source_records", ["plan_id"])

    op.create_table(
        "run_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "plan_id", sa.String(32), sa.ForeignKey("plans.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("run_no", sa.Integer, nullable=False, server_default="0"),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("node", sa.String(32)),
        sa.Column("message", sa.String(400), nullable=False),
        sa.Column("data", sa.JSON, nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.UniqueConstraint("plan_id", "seq", name="uq_event_seq"),
    )


def downgrade() -> None:
    for table in (
        "run_events",
        "source_records",
        "approvals",
        "itinerary_versions",
        "plans",
        "user_preferences",
        "users",
    ):
        op.drop_table(table)
