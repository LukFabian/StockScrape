"""Add DMI plus and minus to chart

Revision ID: bc0c0b00466a
Revises: 990fe8fe8419
Create Date: 2025-06-23 18:41:44.681008

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bc0c0b00466a'
down_revision: Union[str, None] = '990fe8fe8419'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    with op.batch_alter_table('chart') as batch_op:
        batch_op.add_column(sa.Column('dmi_plus_14', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('dmi_minus_14', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('dmi_plus_120', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('dmi_minus_120', sa.Float(), nullable=True))
        batch_op.drop_column('dmi_14')
        batch_op.drop_column('dmi_120')


def downgrade():
    with op.batch_alter_table('chart') as batch_op:
        batch_op.add_column(sa.Column('dmi_14', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('dmi_120', sa.Float(), nullable=True))

        batch_op.drop_column('dmi_plus_14')
        batch_op.drop_column('dmi_minus_14')
        batch_op.drop_column('dmi_plus_120')
        batch_op.drop_column('dmi_minus_120')