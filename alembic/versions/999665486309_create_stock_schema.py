"""create stock schema

Revision ID: 999665486309
Revises: 
Create Date: 2024-12-17 16:03:37.860075

"""
import uuid
from typing import Sequence, Union
from sqlalchemy.dialects.postgresql import UUID
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '999665486309'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'stocks',
        sa.Column('uuid', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False),
        sa.Column('symbol', sa.VARCHAR(), nullable=False, unique=True),
        sa.Column('name', sa.VARCHAR(), nullable=False),
        sa.Column('industry', sa.VARCHAR()),
        sa.Column('marketcap', sa.DOUBLE_PRECISION(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('stocks')
