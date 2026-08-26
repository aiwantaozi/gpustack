"""add cluster topology

Records, per cluster, how far apart its workers are: a chain of operator-named
layers between the cluster root and the worker, plus the default gather policy
models inherit from.

The column is nullable and stays NULL on existing rows. NULL is not a degraded
state here — it means "no layers declared", which the tree already handles by
giving every worker a leaf of its own under the root. Such a cluster still
schedules, still offers the tightest gather choice (the leaf layer is built in
and takes the worker's name, not a label), and simply cannot tell two workers
apart above that. So there is nothing to backfill: an upgraded cluster is in
exactly the state a freshly created one is in until someone declares a layer.

Revision ID: a4b5c6d7e8f9
Revises: f2a3b4c5d6e7
Create Date: 2026-08-26 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists


# revision identifiers, used by Alembic.
revision: str = 'a4b5c6d7e8f9'
down_revision: Union[str, None] = 'f2a3b4c5d6e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLE = 'clusters'
_COLUMN = 'topology'


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
