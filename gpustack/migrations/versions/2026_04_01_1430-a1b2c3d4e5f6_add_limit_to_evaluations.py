"""add limit to evaluations

Revision ID: a1b2c3d4e5f6
Revises: 4f0d0f4e1d2a
Create Date: 2026-04-01 14:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "4f0d0f4e1d2a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("evaluations", sa.Column("limit", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("evaluations", "limit")
