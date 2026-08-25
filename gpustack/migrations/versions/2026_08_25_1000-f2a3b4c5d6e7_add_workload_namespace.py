"""add workload namespace

Records, per workload, the Kubernetes namespace it is deployed into.

The namespace is per-tenant (``gpustack-<org>``, the same family the GPU
instance path already uses), and a worker DaemonSet is per-cluster, so it
cannot be "in" every tenant's namespace at once. The namespace therefore has
to travel on the workload row rather than in a worker-wide setting: the
server resolves it from the row's owner at creation time, and the worker
reads it back to create, read, delete and log-stream the workload.

All three columns are nullable and stay NULL on existing rows **on purpose**:
NULL means "wherever a workload declaring no namespace goes", which is the
runtime's configured default namespace — the only namespace those workloads
were ever created in. That is what keeps already-running Pods and containers
readable and, above all, *deletable* across the upgrade; backfilling a
computed namespace here would point every operation at a namespace their Pods
are not in, and a Pod that cannot be deleted holds its accelerators forever.

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-08-25 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

from gpustack.migrations.utils import column_exists, table_exists


# revision identifiers, used by Alembic.
revision: str = 'f2a3b4c5d6e7'
down_revision: Union[str, None] = 'e1f2a3b4c5d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLES = ('model_instances', 'cache_service_instances', 'benchmarks')


def _column() -> sa.Column:
    return sa.Column('namespace', sqlmodel.sql.sqltypes.AutoString(), nullable=True)


def upgrade() -> None:
    for table in _TABLES:
        if not table_exists(table) or column_exists(table, 'namespace'):
            continue
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.add_column(_column())


def downgrade() -> None:
    for table in reversed(_TABLES):
        if not table_exists(table) or not column_exists(table, 'namespace'):
            continue
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_column('namespace')
