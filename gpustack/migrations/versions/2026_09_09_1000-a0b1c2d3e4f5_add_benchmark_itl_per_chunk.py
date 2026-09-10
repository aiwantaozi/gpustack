"""add measured per-interval ITL columns to benchmarks

The existing per-token columns (``inter_token_latency_*``) hold one value per
REQUEST — guidellm's field of that name, which is the industry's TPOT. A single
decode stall inside a request is divided away by that request's other gaps, so
those columns cannot show it at any percentile.

These four columns hold the other metric: the MEASURED gaps between consecutive
streamed outputs, one sample per gap, pooled across requests. That is what
vLLM / SGLang / evalscope report as ITL, and it is what a stall shows up in.
``_max`` is carried alongside the percentiles because the worst single gap is
the finding for a stall hunt.

Not backfilled, and left nullable: the gaps are recorded by benchmark-runner in
its streaming loop, so no point measured before that landed has them, and a
non-streaming run never will. NULL therefore means "not measured" — which must
stay distinguishable from "measured, and the gaps were 0 ms".

Both tables get the columns because both mirror BenchmarkMetricsLite:
``benchmarks`` (the representative point) and ``benchmark_results`` (the grid).

Revision ID: a0b1c2d3e4f5
Revises: f9a0b1c2d3e4
Create Date: 2026-09-09 10:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from gpustack.migrations.utils import column_exists, table_exists

# revision identifiers, used by Alembic.
revision: str = 'a0b1c2d3e4f5'
down_revision: Union[str, None] = 'f9a0b1c2d3e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = ('benchmarks', 'benchmark_results')
_COLUMNS = (
    'itl_per_chunk_mean',
    'itl_per_chunk_p95',
    'itl_per_chunk_p99',
    'itl_per_chunk_max',
)


def upgrade() -> None:
    for table in _TABLES:
        if not table_exists(table):
            continue
        for column in _COLUMNS:
            # Checked per column, not per table: a run interrupted midway
            # through would otherwise skip the whole table on retry.
            if column_exists(table, column):
                continue
            op.add_column(table, sa.Column(column, sa.Float(), nullable=True))


def downgrade() -> None:
    for table in _TABLES:
        if not table_exists(table):
            continue
        for column in _COLUMNS:
            if not column_exists(table, column):
                continue
            op.drop_column(table, column)
