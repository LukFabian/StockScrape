"""create chart schema

Revision ID: d02b4880524a
Revises: 999665486309
Create Date: 2024-12-18 19:23:17.620245

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'd02b4880524a'
down_revision: Union[str, None] = '999665486309'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Modify stocks table to include new columns
    op.add_column('stock', sa.Column('isNasdaq100', sa.Boolean(), nullable=True))
    op.add_column('stock', sa.Column('deltaIndicator', sa.VARCHAR(), nullable=True))

    # Create chart table
    op.create_table(
        'chart',
        sa.Column('high', sa.BigInteger(), nullable=False),
        sa.Column('low', sa.BigInteger(), nullable=False),
        sa.Column('open', sa.BigInteger(), nullable=False),
        sa.Column('close', sa.BigInteger(), nullable=False),
        sa.Column('volume', sa.BigInteger(), nullable=False),
        sa.Column('date', sa.Date(), primary_key=True, nullable=False),
        sa.Column('symbol', sa.VARCHAR(), sa.ForeignKey('stock.symbol'), primary_key=True, nullable=False),
    )


def downgrade() -> None:
    # Drop chart table
    op.drop_table('chart')

    # Remove added columns from stock table
    op.drop_column('stock', 'isNasdaq100')
    op.drop_column('stock', 'deltaIndicator')
