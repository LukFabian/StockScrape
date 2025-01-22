"""add chart.adx and chart.rsi

Revision ID: 53578ae7facf
Revises: d02b4880524a
Create Date: 2025-01-11 22:41:45.356176

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '53578ae7facf'
down_revision: Union[str, None] = 'd02b4880524a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('chart', sa.Column('adx_14', sa.DOUBLE_PRECISION(), nullable=True))
    op.add_column('chart', sa.Column('adx_120', sa.DOUBLE_PRECISION(), nullable=True))
    op.add_column('chart', sa.Column('rsi_14', sa.DOUBLE_PRECISION(), nullable=True))
    op.add_column('chart', sa.Column('rsi_120', sa.DOUBLE_PRECISION(), nullable=True))


def downgrade() -> None:
    op.drop_column('chart', 'adx_14')
    op.drop_column('chart', 'rsi_14')
    op.drop_column('chart', 'adx_120')
    op.drop_column('chart', 'rsi_120')
