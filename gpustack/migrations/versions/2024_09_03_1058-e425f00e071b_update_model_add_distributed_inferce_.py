"""update model add distributed_inferce_across_workers

Revision ID: e425f00e071b
Revises: 7aea786e3acf
Create Date: 2024-09-03 10:58:32.019003

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = 'e425f00e071b'
down_revision: Union[str, None] = '7aea786e3acf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('models', sa.Column('distributed_inference_across_workers', sa.Boolean(), nullable=False))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('models', 'distributed_inference_across_workers')
    # ### end Alembic commands ###
