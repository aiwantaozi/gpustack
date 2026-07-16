"""custom benchmark datasets (Dataset resource)

Revision ID: a1b2c3d4e5f6
Revises: f0e1d2c3b4a5
Create Date: 2026-07-15 12:00:00.000000

Adds the ``datasets`` table (a first-class, worker-scoped downloadable dataset
resource mirroring ``model_files``) and a ``benchmarks.dataset_id`` FK so a
benchmark can point at a custom dataset (dataset_name == 'Dataset'). Random /
ShareGPT benchmarks are unaffected.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401
import gpustack  # noqa: F401
from gpustack.schemas.common import JSON, UTCDateTime

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f0e1d2c3b4a5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'datasets',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        # DatasetSource
        sa.Column('source', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            'huggingface_repo_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True
        ),
        sa.Column(
            'huggingface_filename', sqlmodel.sql.sqltypes.AutoString(), nullable=True
        ),
        sa.Column(
            'model_scope_model_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True
        ),
        sa.Column(
            'model_scope_file_path', sqlmodel.sql.sqltypes.AutoString(), nullable=True
        ),
        sa.Column('local_path', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        # DatasetBase
        sa.Column('local_dir', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('worker_id', sa.Integer(), nullable=True),
        sa.Column('cleanup_on_delete', sa.Boolean(), nullable=True),
        sa.Column('size', sa.BigInteger(), nullable=True),
        sa.Column('download_progress', sa.Float(), nullable=True),
        sa.Column('resolved_paths', JSON(), nullable=True),
        sa.Column('columns', JSON(), nullable=True),
        sa.Column('sample_rows', JSON(), nullable=True),
        sa.Column('inspect_error', sa.Text(), nullable=True),
        sa.Column('column_mapping', JSON(), nullable=True),
        sa.Column(
            'state',
            sqlmodel.sql.sqltypes.AutoString(),
            nullable=False,
            server_default='downloading',
        ),
        sa.Column('state_message', sa.Text(), nullable=True),
        # Dataset (table)
        sa.Column('source_index', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('cluster_id', sa.Integer(), nullable=True),
        sa.Column('owner_principal_id', sa.Integer(), nullable=True),
        sa.Column('created_at', UTCDateTime(), nullable=False),
        sa.Column('updated_at', UTCDateTime(), nullable=False),
        sa.Column('deleted_at', UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(['owner_principal_id'], ['principals.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_datasets_source_index', 'datasets', ['source_index'], unique=False
    )

    with op.batch_alter_table('benchmarks') as batch_op:
        batch_op.add_column(sa.Column('dataset_id', sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column(
                'dataset_seed_increment',
                sa.Boolean(),
                nullable=True,
                server_default=sa.true(),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table('benchmarks') as batch_op:
        batch_op.drop_column('dataset_seed_increment')
        batch_op.drop_column('dataset_id')
    op.drop_index('ix_datasets_source_index', table_name='datasets')
    op.drop_table('datasets')
