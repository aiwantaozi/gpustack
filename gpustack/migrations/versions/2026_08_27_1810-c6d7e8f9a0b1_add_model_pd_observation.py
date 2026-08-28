"""add model pd_observation

⚠️ Superseded immediately by the drop below. Kept only so the revision chain
resolves for anyone whose database was already stamped with it — the column
never shipped, and neither migration has any effect on a fresh install beyond
adding a column and taking it away again.

Revision ID: c6d7e8f9a0b1
Revises: b5c6d7e8f9a0
Create Date: 2026-08-27 18:10:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'c6d7e8f9a0b1'
down_revision: Union[str, None] = 'b5c6d7e8f9a0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('pd_observation', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('pd_observation')
