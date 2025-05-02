"""add last_modified stock column

Revision ID: ff5e537fb660
Revises: 53578ae7facf
Create Date: 2025-05-02 15:34:18.500943

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ff5e537fb660'
down_revision: Union[str, None] = '53578ae7facf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('stock', sa.Column('last_modified', sa.Date(), nullable=False))


def downgrade() -> None:
    op.drop_column('stock', 'last_modified')
