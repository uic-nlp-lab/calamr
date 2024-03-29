"""Generate Latex tables in a .sty file from CSV files.  The paths to the CSV
files to create tables from and their metadata is given as a YAML configuration
file.

Example::
    latextablenamehere:
        type: slack
        slack_col: 0
        path: ../config/table-name.csv
        caption: Some Caption
        placement: t!
        size: small
        single_column: true
        percent_column_names: ['Proportion']


"""

from .domain import *
from .table import *
from .mng import *
from .desc import *
from .app import *
from .cli import *
