"""soft scale-down: remember when a member was taken out of rotation

Scaling a prefill down cannot be "stop and exit once your blocks have been
fetched" — vLLM has no such shutdown. So the orchestration layer does it: the
member leaves the router's registry at once and its container keeps running
for a window, long enough for the decodes mid-request to finish pulling from
it.

The timestamp is on the row rather than in the server's memory for one
reason: a restart mid-window would otherwise leave a member no router knows
about and nothing will ever delete, holding its accelerators indefinitely. On
the row, the reaper finds it again, and a window that elapsed during the
downtime just reaps on the next pass.

Nullable with no default and no backfill: NULL means "not draining", which is
what every existing member is.

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-09-15 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'c2d3e4f5a6b7'
down_revision: Union[str, None] = 'b1c2d3e4f5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.add_column(sa.Column('draining_since', sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('model_instances', schema=None) as batch_op:
        batch_op.drop_column('draining_since')
