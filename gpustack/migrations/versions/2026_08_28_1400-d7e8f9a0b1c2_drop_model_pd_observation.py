"""drop model pd_observation

The column held the latest verdict of an in-process observer that scraped the
engines from the server. That whole path is gone: the worker's aggregator now
normalizes the same counters onto `gpustack:pd_*`, Prometheus scrapes the
worker over a path that handles tunnelled hosts, and
`GET /models/{id}/pd-metrics` answers the question with PromQL at request
time. A column would be a second, staler copy of that answer -- and one that
could only ever hold the present, where the endpoint can show history.

Dropped rather than left in place: an unread status column is indistinguishable
from a stale one to the next reader.

Revision ID: d7e8f9a0b1c2
Revises: c6d7e8f9a0b1
Create Date: 2026-08-28 14:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from gpustack.migrations.utils import column_exists

revision: str = 'd7e8f9a0b1c2'
down_revision: Union[str, None] = 'c6d7e8f9a0b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Guarded: an installation that never ran the add (it was unreleased) must
    # not fail here on a column it never had.
    if column_exists('models', 'pd_observation'):
        with op.batch_alter_table('models', schema=None) as batch_op:
            batch_op.drop_column('pd_observation')


def downgrade() -> None:
    if not column_exists('models', 'pd_observation'):
        with op.batch_alter_table('models', schema=None) as batch_op:
            batch_op.add_column(
                sa.Column('pd_observation', sa.JSON(), nullable=True)
            )
