"""create balance sheet

Revision ID: 3fe69ff03f96
Revises: bc0c0b00466a
Create Date: 2025-06-27 15:54:18.877677

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import BigInteger

# revision identifiers, used by Alembic.
revision: str = '3fe69ff03f96'
down_revision: Union[str, None] = 'bc0c0b00466a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        'balance_sheet',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('symbol', sa.String(length=16), sa.ForeignKey('stock.symbol', ondelete='CASCADE'), nullable=False),
        sa.Column('period_ending', sa.Date, nullable=False),
        sa.Column('cash_and_cash_equivalents', BigInteger, nullable=True),
        sa.Column('short_term_investments', BigInteger, nullable=True),
        sa.Column('net_receivables', BigInteger, nullable=True),
        sa.Column('inventory', BigInteger, nullable=True),
        sa.Column('other_current_assets', BigInteger, nullable=True),
        sa.Column('total_current_assets', BigInteger, nullable=True),
        sa.Column('fixed_assets', BigInteger, nullable=True),
        sa.Column('goodwill', BigInteger, nullable=True),
        sa.Column('intangible_assets', BigInteger, nullable=True),
        sa.Column('other_assets', BigInteger, nullable=True),
        sa.Column('deferred_asset_charges', BigInteger, nullable=True),
        sa.Column('total_assets', BigInteger, nullable=True),
        sa.Column('accounts_payable', BigInteger, nullable=True),
        sa.Column('short_term_debt', BigInteger, nullable=True),
        sa.Column('other_current_liabilities', BigInteger, nullable=True),
        sa.Column('total_current_liabilities', BigInteger, nullable=True),
        sa.Column('long_term_debt', BigInteger, nullable=True),
        sa.Column('other_liabilities', BigInteger, nullable=True),
        sa.Column('deferred_liability_charges', BigInteger, nullable=True),
        sa.Column('misc_stocks', BigInteger, nullable=True),
        sa.Column('minority_interest', BigInteger, nullable=True),
        sa.Column('total_liabilities', BigInteger, nullable=True),
        sa.Column('common_stocks', BigInteger, nullable=True),
        sa.Column('capital_surplus', BigInteger, nullable=True),
        sa.Column('treasury_stock', BigInteger, nullable=True),
        sa.Column('other_equity', BigInteger, nullable=True),
        sa.Column('total_equity', BigInteger, nullable=True),
        sa.Column('total_liabilities_and_equity', BigInteger, nullable=True),
    )


def downgrade():
    op.drop_table('balance_sheet')
