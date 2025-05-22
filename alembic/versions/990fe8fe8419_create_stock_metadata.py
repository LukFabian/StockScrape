"""create stock metadata

Revision ID: 990fe8fe8419
Revises: ff5e537fb660
Create Date: 2025-05-18 13:02:01.557083
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '990fe8fe8419'
down_revision: Union[str, None] = 'ff5e537fb660'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'stock_profile',
        sa.Column('symbol', sa.String(), sa.ForeignKey('stock.symbol'), primary_key=True),
        sa.Column('asset_type', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('cik', sa.String(), nullable=False),
        sa.Column('exchange', sa.String(), nullable=False),
        sa.Column('currency', sa.String(), nullable=False),
        sa.Column('country', sa.String(), nullable=False),
        sa.Column('sector', sa.String(), nullable=False),
        sa.Column('industry', sa.String(), nullable=False),
        sa.Column('address', sa.String(), nullable=False),
        sa.Column('official_site', sa.String(), nullable=False),
        sa.Column('fiscal_year_end', sa.String(), nullable=False),
    )

    op.create_table(
        'stock_metric',
        sa.Column('symbol', sa.String(), sa.ForeignKey('stock.symbol'), primary_key=True),
        sa.Column('date', sa.DateTime(), primary_key=True),
        sa.Column('latest_quarter', sa.DateTime(), nullable=True),
        sa.Column('market_capitalization', sa.BIGINT(), nullable=True),
        sa.Column('ebitda', sa.BIGINT(), nullable=True),
        sa.Column('pe_ratio', sa.Float(), nullable=True),
        sa.Column('peg_ratio', sa.Float(), nullable=True),
        sa.Column('book_value', sa.Float(), nullable=True),
        sa.Column('dividend_per_share', sa.Float(), nullable=True),
        sa.Column('dividend_yield', sa.Float(), nullable=True),
        sa.Column('eps', sa.Float(), nullable=True),
        sa.Column('revenue_per_share_ttm', sa.Float(), nullable=True),
        sa.Column('profit_margin', sa.Float(), nullable=True),
        sa.Column('operating_margin_ttm', sa.Float(), nullable=True),
        sa.Column('return_on_assets_ttm', sa.Float(), nullable=True),
        sa.Column('return_on_equity_ttm', sa.Float(), nullable=True),
        sa.Column('revenue_ttm', sa.BIGINT(), nullable=True),
        sa.Column('gross_profit_ttm', sa.BIGINT(), nullable=True),
        sa.Column('diluted_eps_ttm', sa.Float(), nullable=True),
        sa.Column('quarterly_earnings_growth_yoy', sa.Float(), nullable=True),
        sa.Column('quarterly_revenue_growth_yoy', sa.Float(), nullable=True),
        sa.Column('analyst_target_price', sa.Float(), nullable=True),
        sa.Column('analyst_rating_strong_buy', sa.Integer(), nullable=True),
        sa.Column('analyst_rating_buy', sa.Integer(), nullable=True),
        sa.Column('analyst_rating_hold', sa.Integer(), nullable=True),
        sa.Column('analyst_rating_sell', sa.Integer(), nullable=True),
        sa.Column('analyst_rating_strong_sell', sa.Integer(), nullable=True),
        sa.Column('trailing_pe', sa.Float(), nullable=True),
        sa.Column('forward_pe', sa.Float(), nullable=True),
        sa.Column('price_to_sales_ratio_ttm', sa.Float(), nullable=True),
        sa.Column('price_to_book_ratio', sa.Float(), nullable=True),
        sa.Column('ev_to_revenue', sa.Float(), nullable=True),
        sa.Column('ev_to_ebitda', sa.Float(), nullable=True),
        sa.Column('beta', sa.Float(), nullable=True),
        sa.Column('fifty_two_week_high', sa.Float(), nullable=True),
        sa.Column('fifty_two_week_low', sa.Float(), nullable=True),
        sa.Column('fifty_day_moving_average', sa.Float(), nullable=True),
        sa.Column('two_hundred_day_moving_average', sa.Float(), nullable=True),
        sa.Column('shares_outstanding', sa.BIGINT(), nullable=True),
        sa.Column('dividend_date', sa.DateTime(), nullable=True),
        sa.Column('ex_dividend_date', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('stock_metric')
    op.drop_table('stock_profile')
