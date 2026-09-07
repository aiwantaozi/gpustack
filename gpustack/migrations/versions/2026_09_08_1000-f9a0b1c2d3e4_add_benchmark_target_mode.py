"""add benchmark target mode

A run is now aimed either at one instance (an engine, straight at its own
port) or at a route (the deployment, through the entrance clients call). The
distinction did not exist before, and every run written until now was the
former, so existing rows are backfilled to `instance` — which is also the
default for a create body that says nothing.

Revision ID: f9a0b1c2d3e4
Revises: e8f9a0b1c2d3
Create Date: 2026-09-08 10:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

from gpustack.migrations.utils import column_exists, table_exists

# revision identifiers, used by Alembic.
revision: str = 'f9a0b1c2d3e4'
down_revision: Union[str, None] = 'e8f9a0b1c2d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not table_exists("benchmarks"):
        return
    if column_exists("benchmarks", "target_mode"):
        return

    # Added nullable, backfilled, then left nullable: SQLite cannot add a NOT
    # NULL column with a server default in one step, and the model supplies the
    # default on every write anyway. A NULL that somehow survives reads as
    # `instance` on the way out, which is what such a row was.
    op.add_column(
        "benchmarks",
        sa.Column("target_mode", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.execute(
        "UPDATE benchmarks SET target_mode = 'instance' WHERE target_mode IS NULL"
    )


def downgrade() -> None:
    if not table_exists("benchmarks"):
        return
    if not column_exists("benchmarks", "target_mode"):
        return
    op.drop_column("benchmarks", "target_mode")
