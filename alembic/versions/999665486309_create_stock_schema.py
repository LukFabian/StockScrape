"""create stock schema

Revision ID: 999665486309
Revises: 
Create Date: 2024-12-17 16:03:37.860075

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '999665486309'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'stock',
        sa.Column('symbol', sa.VARCHAR(), primary_key=True, nullable=False, unique=True),
        sa.Column('name', sa.VARCHAR(), nullable=False),
        sa.Column('industry', sa.VARCHAR()),
        sa.Column('marketcap', sa.BigInteger(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('stock')
