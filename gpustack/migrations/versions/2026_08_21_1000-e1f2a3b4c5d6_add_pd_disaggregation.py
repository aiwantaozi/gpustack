"""add pd disaggregation

Freezes the API/DB surface for prefill-decode disaggregation. Everything here
is additive and nullable, so a deployment with no ``roles`` set behaves
byte-for-byte as before.

1. ``models.roles`` / ``models.disaggregation`` — user intent. A Model with
   ``roles`` set is a *group*: one pool, one router, one generation at a time.
   ``roles`` alone is plain multi-role orchestration; both together is PD.

2. ``models.state`` / ``state_message`` / ``role_status`` / ``stale`` /
   ``degradations`` — server-owned status, all written by a single owner
   (``sync_model_status``) from one scan of the model's instances.

   ``ready_replicas`` is deliberately left alone: it stays a plain count of
   RUNNING instances. Under PD a count no longer implies servability (3P1D
   with the router still down is four RUNNING instances and zero service),
   so servability moved to ``state`` and per-role detail to ``role_status``
   instead of overloading the counter.

   ``role_status`` is persisted rather than computed per request because the
   list endpoint returns models without their instances, and the UI needs
   per-role detail on rows it hasn't expanded.

3. ``model_instances.role`` / ``group_id`` / ``spec_digest`` /
   ``named_ports`` — group membership and generation. ``group_id`` is a
   generation, not a replica index; pairing binds to it rather than to peer
   addresses because serving ports were measured to change on every rebuild.
   It is indexed because the reconcilers group by it.


Revision ID: e1f2a3b4c5d6
Revises: d5e8f0a1b2c3
Create Date: 2026-08-21 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

from gpustack.migrations.utils import column_exists


# revision identifiers, used by Alembic.
revision: str = 'e1f2a3b4c5d6'
down_revision: Union[str, None] = 'd5e8f0a1b2c3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_MODEL_COLUMNS = [
    lambda: sa.Column('roles', sa.JSON(), nullable=True),
    lambda: sa.Column('disaggregation', sa.JSON(), nullable=True),
    lambda: sa.Column('state', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    lambda: sa.Column('state_message', sa.Text(), nullable=True),
    lambda: sa.Column('role_status', sa.JSON(), nullable=True),
    lambda: sa.Column('stale', sa.Boolean(), nullable=True),
    lambda: sa.Column('degradations', sa.JSON(), nullable=True),
]

_MODEL_INSTANCE_COLUMNS = [
    lambda: sa.Column('role', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    lambda: sa.Column('group_id', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    lambda: sa.Column('spec_digest', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    lambda: sa.Column('named_ports', sa.JSON(), nullable=True),
]


def _add_missing(table: str, factories) -> None:
    """Add whichever of these columns aren't there yet, in one batch.

    One batch per table: on SQLite, batch mode recreates and copies the whole
    table per block, so a block per column would rewrite it seven times.
    """
    missing = [f() for f in factories if not column_exists(table, f().name)]
    if not missing:
        return
    with op.batch_alter_table(table, schema=None) as batch_op:
        for column in missing:
            batch_op.add_column(column)


def _backfill_state() -> None:
    """Seed ``state`` from the counters it is derived from.

    Every reader already falls back to those counters while the column is NULL,
    so this is not what keeps a fleet routable across the upgrade. It exists to
    keep the row self-consistent between the migration and the first reconcile,
    so that nothing has to hold two answers for one model.

    **The mapping must be exactly what the writer produces**, and only two
    values are reachable for a role-less model — which is every model that
    exists before this revision. ``state`` answers "can this serve", so one
    ready replica is ``running``; being short of the requested count is a
    ``ratio_unmet`` degradation beside it, not a lifecycle value.
    ``partial`` means "members up and still unservable", which a role-less
    model cannot be, so writing it here would invent a value the writer never
    emits *and* one the servability gate reads as unroutable — the upgrade
    itself would take every partially-scaled model out of service. A model at
    ``replicas = 0`` lands on ``pending``, which is what the writer returns for
    it too: "stopped" is an intent, not a state, and the enum has no value for
    it.
    """
    models = sa.table(
        'models',
        sa.column('state', sa.String()),
        sa.column('replicas', sa.Integer()),
        sa.column('ready_replicas', sa.Integer()),
    )
    op.execute(
        models.update()
        .where(models.c.state.is_(None))
        .values(
            state=sa.case(
                (models.c.ready_replicas <= 0, 'pending'),
                else_='running',
            )
        )
    )


def upgrade() -> None:
    _add_missing('models', _MODEL_COLUMNS)
    _add_missing('model_instances', _MODEL_INSTANCE_COLUMNS)
    _backfill_state()

    if column_exists('model_instances', 'group_id') and not _index_exists(
        'model_instances', 'ix_model_instances_group_id'
    ):
        op.create_index(
            'ix_model_instances_group_id',
            'model_instances',
            ['group_id'],
            unique=False,
        )


def downgrade() -> None:
    if _index_exists('model_instances', 'ix_model_instances_group_id'):
        op.drop_index('ix_model_instances_group_id', table_name='model_instances')

    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        for factory in reversed(_MODEL_INSTANCE_COLUMNS):
            name = factory().name
            if column_exists('model_instances', name):
                batch_op.drop_column(name)

    with op.batch_alter_table('models', schema=None) as batch_op:
        for factory in reversed(_MODEL_COLUMNS):
            name = factory().name
            if column_exists('models', name):
                batch_op.drop_column(name)


def _index_exists(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return any(
        index["name"] == index_name for index in inspector.get_indexes(table_name)
    )
