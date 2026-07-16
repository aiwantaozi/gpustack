"""benchmark multi-point (stages / adaptive auto-tune) schema

Revision ID: f0e1d2c3b4a5
Revises: c4d7e8f9a0b1
Create Date: 2026-06-25 17:00:00.000000

Adds the benchmark load-curve schema:

* ``benchmark_results`` — one row per measured (input_tokens, rate) grid point,
  carrying the flat metric columns plus this point's raw ``benchmarks[i]`` dump.
* ``benchmarks`` gains the stages / auto-tune / data-distribution / SLA /
  constraint / best-point config columns (all nullable). The parent's flat
  ``*_mean`` columns stay as the representative (throughput-peak) point.

Load type (fixed_rate / concurrency) is the load axis; ``auto_tune`` toggles the
adaptive ramp.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel  # noqa: F401
import gpustack  # noqa: F401
from gpustack.schemas.common import JSON, UTCDateTime

# revision identifiers, used by Alembic.
revision: str = 'f0e1d2c3b4a5'
down_revision: Union[str, None] = 'c4d7e8f9a0b1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Flat metric columns shared by benchmarks and benchmark_results
# (BenchmarkMetricsLite).
_METRIC_FLOAT_COLS = [
    'requests_per_second_mean',
    'request_latency_mean',
    'time_per_output_token_mean',
    'inter_token_latency_mean',
    'time_to_first_token_mean',
    'tokens_per_second_mean',
    'output_tokens_per_second_mean',
    'input_tokens_per_second_mean',
    'request_concurrency_mean',
    'request_concurrency_max',
]
_METRIC_INT_COLS = [
    'request_total',
    'request_successful',
    'request_errored',
    'request_incomplete',
]
# P99 percentile metric columns (added to BenchmarkMetricsLite). Present on both
# benchmarks and benchmark_results, but NOT backfilled (old rows have no p99).
_METRIC_P99_COLS = [
    'time_to_first_token_p99',
    'time_per_output_token_p99',
    'request_latency_p99',
]

# All nullable columns added to ``benchmarks`` (name, sqlalchemy type).
_BENCHMARK_COLUMNS = [
    # load type
    ('load_type', sqlmodel.sql.sqltypes.AutoString()),
    # latency-SLA targets (avg + p99 TTFT/TPOT + e2e latency; all "<=" ms)
    ('sla_avg_ttft_ms', sa.Float()),
    ('sla_avg_tpot_ms', sa.Float()),
    ('sla_p99_ttft_ms', sa.Float()),
    ('sla_p99_tpot_ms', sa.Float()),
    ('sla_avg_latency_ms', sa.Float()),
    ('sla_p99_latency_ms', sa.Float()),
    # p99 metric columns (also added to benchmark_results in create_table)
    *[(c, sa.Float()) for c in _METRIC_P99_COLS],
    # SLA / saturation results + constraints + multi-turn
    ('sla_met_rate', sa.Float()),
    ('recommended_rate', sa.Float()),
    ('turns', sa.Integer()),
    ('warmup', sa.Float()),
    ('cooldown', sa.Float()),
    ('max_errors', sa.Integer()),
    ('max_error_rate', sa.Float()),
    ('stop_on_saturation', sa.Boolean()),
    # token-length distribution (inference-perf style)
    ('dataset_input_stdev', sa.Integer()),
    ('dataset_input_min', sa.Integer()),
    ('dataset_input_max', sa.Integer()),
    ('dataset_output_stdev', sa.Integer()),
    ('dataset_output_min', sa.Integer()),
    ('dataset_output_max', sa.Integer()),
    # stages (manual mode) / shared prefix / best operating points
    ('stages', JSON()),
    ('peak_rate', sa.Float()),
    ('knee_rate', sa.Float()),
    ('validity', JSON()),
    ('prefix_buckets', JSON()),
    # global duration cap (non-stage runs)
    ('max_seconds', sa.Float()),
    # auto-tune (adaptive ramp) flag + budget / bounds
    ('auto_tune', sa.Boolean()),
    ('lower_bound', sa.Float()),
    ('upper_bound', sa.Float()),
    ('max_points', sa.Integer()),
    ('max_total_seconds', sa.Float()),
]


def upgrade() -> None:
    # 1. benchmark_results sub-table (one row per (input_tokens, rate) point).
    op.create_table(
        'benchmark_results',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('benchmark_id', sa.Integer(), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('rate', sa.Float(), nullable=True),
        sa.Column('strategy_type', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('sequence', sa.Integer(), nullable=False, server_default='0'),
        *[sa.Column(c, sa.Float(), nullable=True) for c in _METRIC_FLOAT_COLS],
        *[sa.Column(c, sa.Float(), nullable=True) for c in _METRIC_P99_COLS],
        *[sa.Column(c, sa.Integer(), nullable=True) for c in _METRIC_INT_COLS],
        sa.Column('raw_metrics', JSON(), nullable=True),
        sa.Column('created_at', UTCDateTime(), nullable=False),
        sa.Column('updated_at', UTCDateTime(), nullable=False),
        sa.Column('deleted_at', UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ['benchmark_id'], ['benchmarks.id'], ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_benchmark_results_benchmark_id',
        'benchmark_results',
        ['benchmark_id'],
        unique=False,
    )

    # 2. benchmarks: all config columns (nullable additions).
    with op.batch_alter_table('benchmarks') as batch_op:
        for name, col_type in _BENCHMARK_COLUMNS:
            batch_op.add_column(sa.Column(name, col_type, nullable=True))

    # 3. Backfill: each completed benchmark -> one benchmark_results row.
    metric_cols = ", ".join(_METRIC_FLOAT_COLS + _METRIC_INT_COLS)
    op.execute(
        f"""
        INSERT INTO benchmark_results (
            benchmark_id, rate, sequence,
            {metric_cols},
            raw_metrics, created_at, updated_at
        )
        SELECT
            id, request_rate, 0,
            {metric_cols},
            raw_metrics, created_at, updated_at
        FROM benchmarks
        WHERE request_total IS NOT NULL
        """
    )


def downgrade() -> None:
    op.drop_index('ix_benchmark_results_benchmark_id', table_name='benchmark_results')
    op.drop_table('benchmark_results')
    with op.batch_alter_table('benchmarks') as batch_op:
        for name, _ in reversed(_BENCHMARK_COLUMNS):
            batch_op.drop_column(name)
