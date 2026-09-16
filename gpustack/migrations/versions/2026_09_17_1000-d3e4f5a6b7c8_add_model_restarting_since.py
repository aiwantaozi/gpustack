"""restart guard: remember when a deployment was last torn down

`POST /models/{id}/restart` refuses a second teardown while the first is still
rebuilding, because that teardown would delete the replacements the first one
just created. The refusal used to be inferred from the live members spanning
more than one `spec_digest`, which cannot happen: the teardown is synchronous
and the reconcile rebuilds from the same target digest, so the old and new
generations are never in the table together. The guard was dead and a double
click cost the group a second full startup.

The fact is not derivable from the rows, so it is recorded: set when the
teardown runs, cleared by the status pass on reaching RUNNING, and lapsing on
its own so a group that never converges can still be restarted.

Nullable with no default and no backfill: NULL means "no restart in flight",
which is true of every existing deployment.

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-09-17 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'd3e4f5a6b7c8'
down_revision: Union[str, None] = 'c2d3e4f5a6b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('restarting_since', sa.DateTime(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('restarting_since')
