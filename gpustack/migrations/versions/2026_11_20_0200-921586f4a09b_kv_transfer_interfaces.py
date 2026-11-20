"""Persist KV transfer interfaces for workers and deployed instances.

Revision ID: 921586f4a09b
Revises: f8c4a291bd65
Create Date: 2026-11-20 02:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from gpustack.migrations.utils import column_exists, table_exists

revision: str = '921586f4a09b'
down_revision: Union[str, None] = 'f8c4a291bd65'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if table_exists('workers') and not column_exists('workers', 'kv_transfer_ifname'):
        op.add_column('workers', sa.Column('kv_transfer_ifname', sa.Text(), nullable=True))
    if table_exists('model_instances') and not column_exists(
        'model_instances', 'kv_transfer_ifnames'
    ):
        op.add_column(
            'model_instances',
            sa.Column('kv_transfer_ifnames', sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    if table_exists('model_instances') and column_exists(
        'model_instances', 'kv_transfer_ifnames'
    ):
        op.drop_column('model_instances', 'kv_transfer_ifnames')
    if table_exists('workers') and column_exists('workers', 'kv_transfer_ifname'):
        op.drop_column('workers', 'kv_transfer_ifname')
