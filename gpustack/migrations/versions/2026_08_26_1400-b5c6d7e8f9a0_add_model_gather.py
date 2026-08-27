"""add model gather

The per-deployment override of the cluster's default gather policy: how tightly
this group's members must sit together, and whether "must" is meant literally.

The column is nullable and stays NULL on existing rows, which is the correct
value rather than a missing one — NULL means "inherit the cluster's default",
and inheriting is what every model did before the field existed. So there is
nothing to backfill, and a cluster that has declared no default keeps the
solver's own behaviour (place into the tightest domain that fits, widen to the
root rather than refuse).

Deliberately NOT part of `model_spec_digest`: gather is a preference for the
*next* scheduling decision, and nothing re-places a running group. A digest
bump would restart every member to relocate none of them.

Revision ID: b5c6d7e8f9a0
Revises: a4b5c6d7e8f9
Create Date: 2026-08-26 14:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists


# revision identifiers, used by Alembic.
revision: str = 'b5c6d7e8f9a0'
down_revision: Union[str, None] = 'a4b5c6d7e8f9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLE = 'models'
_COLUMN = 'gather'


def upgrade() -> None:
    if not table_exists(_TABLE) or column_exists(_TABLE, _COLUMN):
        return
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.add_column(sa.Column(_COLUMN, sa.JSON(), nullable=True))


def downgrade() -> None:
    if not table_exists(_TABLE) or not column_exists(_TABLE, _COLUMN):
        return
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.drop_column(_COLUMN)
