"""add evaluations tables

Revision ID: 4f0d0f4e1d2a
Revises: 8ad0f94c92e8
Create Date: 2026-03-31 20:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4f0d0f4e1d2a'
down_revision: Union[str, None] = '8ad0f94c92e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'evaluations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('suite_id', sa.String(), nullable=False),
        sa.Column('suite_name', sa.String(), nullable=False),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('model_id', sa.Integer(), nullable=True),
        sa.Column('model_name', sa.String(), nullable=True),
        sa.Column('model_instance_name', sa.String(), nullable=True),
        sa.Column('cluster_id', sa.Integer(), nullable=True),
        sa.Column('worker_id', sa.Integer(), nullable=True),
        sa.Column('pid', sa.Integer(), nullable=True),
        sa.Column('state', sa.String(), nullable=False),
        sa.Column('state_message', sa.Text(), nullable=True),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('task_count', sa.Integer(), nullable=True),
        sa.Column('sample_count', sa.Integer(), nullable=True),
        sa.Column('snapshot', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=False), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=False), nullable=True),
        sa.Column('deleted_at', sa.TIMESTAMP(timezone=False), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_evaluations_name'), 'evaluations', ['name'], unique=True)
    op.create_index(op.f('ix_evaluations_state'), 'evaluations', ['state'], unique=False)

    op.create_table(
        'evaluation_tasks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('evaluation_id', sa.Integer(), nullable=False),
        sa.Column('task_name', sa.String(), nullable=False),
        sa.Column('task_alias', sa.String(), nullable=True),
        sa.Column('task_group', sa.String(), nullable=True),
        sa.Column('display_name', sa.String(), nullable=True),
        sa.Column('dataset_name', sa.String(), nullable=True),
        sa.Column('version', sa.String(), nullable=True),
        sa.Column('sample_count', sa.Integer(), nullable=True),
        sa.Column('n_shot', sa.Integer(), nullable=True),
        sa.Column('output_type', sa.String(), nullable=True),
        sa.Column('primary_metric_key', sa.String(), nullable=True),
        sa.Column('primary_metric_value', sa.Float(), nullable=True),
        sa.Column('primary_stderr', sa.String(), nullable=True),
        sa.Column('primary_higher_is_better', sa.Boolean(), nullable=True),
        sa.Column('raw_metrics', sa.JSON(), nullable=True),
        sa.Column('config_snapshot', sa.JSON(), nullable=True),
        sa.Column('task_metadata', sa.JSON(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=False), nullable=True),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=False), nullable=True),
        sa.Column('deleted_at', sa.TIMESTAMP(timezone=False), nullable=True),
        sa.ForeignKeyConstraint(['evaluation_id'], ['evaluations.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_evaluation_tasks_evaluation_id'), 'evaluation_tasks', ['evaluation_id'], unique=False)
    op.create_index(op.f('ix_evaluation_tasks_task_name'), 'evaluation_tasks', ['task_name'], unique=False)
    op.create_index(op.f('ix_evaluation_tasks_task_group'), 'evaluation_tasks', ['task_group'], unique=False)
    op.create_index(op.f('ix_evaluation_tasks_primary_metric_key'), 'evaluation_tasks', ['primary_metric_key'], unique=False)
    op.create_index(op.f('ix_evaluation_tasks_primary_metric_value'), 'evaluation_tasks', ['primary_metric_value'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_evaluation_tasks_primary_metric_value'), table_name='evaluation_tasks')
    op.drop_index(op.f('ix_evaluation_tasks_primary_metric_key'), table_name='evaluation_tasks')
    op.drop_index(op.f('ix_evaluation_tasks_task_group'), table_name='evaluation_tasks')
    op.drop_index(op.f('ix_evaluation_tasks_task_name'), table_name='evaluation_tasks')
    op.drop_index(op.f('ix_evaluation_tasks_evaluation_id'), table_name='evaluation_tasks')
    op.drop_table('evaluation_tasks')
    op.drop_index(op.f('ix_evaluations_state'), table_name='evaluations')
    op.drop_index(op.f('ix_evaluations_name'), table_name='evaluations')
    op.drop_table('evaluations')
