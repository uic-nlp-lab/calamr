"""This module contains classes that generate tables.

"""
__author__ = 'Paul Landes'

from typing import (
    Dict, List, Sequence, Tuple, Any, ClassVar, Optional, Callable, Union
)
from dataclasses import dataclass, field
import sys
import re
import string
import itertools as it
from io import TextIOWrapper, StringIO
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from zensols.persist import persisted, PersistedWork, PersistableContainer
from zensols.config import Dictable
from . import VariableParam


@dataclass
class Table(PersistableContainer, Dictable):
    """Generates a Zensols styled Latex table from a CSV file.

    """
    _VARIABLE: ClassVar[str] = 'VAR'
    _VARIABLE_ATTRIBUTES: ClassVar[Tuple[VariableParam]] = (
        VariableParam('placement', value_format='{val}'),
        VariableParam('size'))

    path: Union[Path, str] = field()
    """The path to the CSV file to make a latex table."""

    name: str = field()
    """The name of the table, also used as the label."""

    caption: str = field()
    """The human readable string used to the caption in the table."""

    head: str = field(default=None)
    """The header to use for the table, which is used as the text in the list of
    tables and made bold in the table."""

    placement: str = field(default=None)
    """The placement of the table."""

    size: str = field(default='normalsize')
    """The size of the table, and one of:

      * Huge
      * huge
      * LARGE
      * Large
      * large
      * normalsize (default)
      * small
      * footnotesize
      * scriptsize
      * tiny

    """
    uses: Sequence[str] = field(default=('zentable',))
    """Comma separated list of packages to use."""

    single_column: bool = field(default=True)
    """Makes the table one column wide in a two column.  Setting this to false
    generates a ``table*`` two column table, which won't work in beamer
    (slides) document types.

    """
    hlines: Sequence[int] = field(default_factory=set)
    """Indexes of rows to put horizontal line breaks."""

    double_hlines: Sequence[int] = field(default_factory=set)
    """Indexes of rows to put double horizontal line breaks."""

    column_keeps: Optional[List[str]] = field(default=None)
    """If provided, only keep the columns in the list"""

    column_removes: List[str] = field(default_factory=list)
    """The name of the columns to remove from the table, if any."""

    column_renames: Dict[str, str] = field(default_factory=dict)
    """Columns to rename, if any."""

    column_value_replaces: Dict[str, Dict[Any, Any]] = \
        field(default_factory=dict)
    """Data values to replace in the dataframe.  It is keyed by the column name
    and values are the replacements.  Each value is a ``dict`` with orignal
    value keys and the replacements as values.

    """
    column_aligns: str = field(default=None)
    """The alignment/justification (i.e. ``|l|l|`` for two columns).  If not
    provided, they are automatically generated based on the columns of the
    table.

    """
    percent_column_names: Sequence[str] = field(default=())
    """Column names that have a percent sign to be escaped."""

    make_percent_column_names: Dict[str, int] = field(default_factory=dict)
    """Each columnn in the map will get rounded to the value * 100 of the name.
    For example, ``{'ann_per': 3}`` will round column ``ann_per`` to 3 decimal
    places.

    """
    format_thousands_column_names: Dict[str, Optional[Dict[str, Any]]] = \
        field(default_factory=dict)
    """Columns to format using thousands.  The keys are the column names of the
    table and the values are either ``None`` or the keyword arguments to
    :meth:`format_thousand`.

    """
    column_evals: Dict[str, str] = field(default_factory=dict)
    """Keys are column names with values as functions (i.e. lambda expressions)
    evaluated with a single column value parameter.  The return value replaces
    the column identified by the key.

    """
    read_kwargs: Dict[str, str] = field(default_factory=dict)
    """Keyword arguments used in the :meth:`~pandas.read_csv` call when reading the
    CSV file.

    """
    write_kwargs: Dict[str, str] = field(
        default_factory=lambda: {'disable_numparse': True})
    """Keyword arguments used in the :meth:`~tabulate.tabulate` call when
    writing the table.  The default tells :mod:`tabulate` to not parse/format
    numerical data.

    """
    replace_nan: str = field(default=None)
    """Replace NaN values with a the value of this field as :meth:`tabulate` is
    not using the missing value due to some bug I assume.

    """
    blank_columns: List[int] = field(default_factory=list)
    """A list of column indexes to set to the empty string (i.e. 0th to fixed the
    ``Unnamed: 0`` issues).

    """
    bold_cells: List[Tuple[int, int]] = field(default_factory=list)
    """A list of row/column cells to bold."""

    bold_max_columns: List[str] = field(default_factory=list)
    """A list of column names that will have its max value bolded."""

    capitalize_columns: Dict[str, bool] = field(default_factory=dict)
    """Capitalize either sentences (``False`` values) or every word (``True``
    values).  The keys are column names.

    """
    index_col_name: str = field(default=None)
    """If set, add an index column with the given name."""

    df_code: str = field(default=None)
    """Python code executed that manipulates the table's dataframe.  The code
    has a local ``df`` variable and the returned value is used as the
    replacement.  This is usually a one-liner used to subset the data etc.  The
    code is evaluated with :func:`eval`.

    """
    df_code_pre: str = field(default=None)
    """Like :obj:`df_code` but right after the source data is read and before
    any modifications.  The code is evaluated with :func:`eval`.

    """
    df_code_exec: str = field(default=None)
    """Like :obj:`df_code` but invoke with :func:`exec` instead of :func:`eval`.

    """
    df_code_exec_pre: str = field(default=None)
    """Like :obj:`df_code_pre` but invoke with :func:`exec` instead of
    :func:`eval`.

    """
    def __post_init__(self):
        super().__init__()
        if isinstance(self.uses, str):
            self.uses = re.split(r'\s*,\s*', self.uses)
        if isinstance(self.hlines, (tuple, list)):
            self.hlines = set(self.hlines)
        if isinstance(self.double_hlines, (tuple, list)):
            self.double_hlines = set(self.double_hlines)
        self._formatted_dataframe = PersistedWork(
            '_formatted_dataframe', self, transient=True)

    @property
    def latex_environment(self) -> str:
        """Return the latex environment for the table.

        """
        tab: str
        if self.single_column:
            tab = 'zztable'
        else:
            if self.placement is None:
                tab = 'zztabletcol'
            else:
                tab = 'zztabletcolplace'
        if self.head is not None:
            tab += 'head'
        return tab

    @property
    def columns(self) -> str:
        """Return the columns field in the Latex environment header.

        """
        cols: str = self.column_aligns
        if cols is None:
            df = self.formatted_dataframe
            cols = 'l' * df.shape[1]
            cols = '|' + '|'.join(cols) + '|'
        return cols

    def get_cmd_args(self, add_brackets: bool) -> Dict[str, str]:
        args = {}
        var: VariableParam
        for i, var in enumerate(self._VARIABLE_ATTRIBUTES):
            attr: str = var.name
            val = getattr(self, attr)
            if val is None:
                val = ''
            elif val == self._VARIABLE:
                val = var.index_format.format(index=(i + 1), val=val, var=var)
            else:
                val = var.value_format.format(index=(i + 1), val=val, var=var)
            if add_brackets and len(val) > 0:
                val = f'[{val}]'
            args[attr] = val
        return args

    @property
    @persisted('_var_args')
    def var_args(self) -> Tuple[str]:
        var = tuple(map(lambda a: (a, getattr(self, a.name)),
                        self._VARIABLE_ATTRIBUTES))
        return tuple(map(lambda x: x[0],
                         filter(lambda x: x[1] == self._VARIABLE, var)))

    def get_params(self, add_brackets: bool) -> Dict[str, str]:
        """Return the parameters used for creating the table.

        """
        params = {'tabname': self.name,
                  'latex_environment': self.latex_environment,
                  'caption': self.caption,
                  'columns': self.columns}
        params.update(self.get_cmd_args(add_brackets))
        return params

    @staticmethod
    def format_thousand(x: int, apply_k: bool = True,
                        add_comma: bool = True) -> str:
        """Format a number as a string with comma separating thousands.

        :param x: the number to format

        :param apply_k: add a ``K`` to the end of large numbers

        :param add_comma: whether to add a comma

        """
        add_k = False
        if x > 10000:
            if apply_k:
                x = round(x / 1000)
                add_k = True
        if add_comma:
            x = f'{x:,}'
        else:
            x = str(x)
        if add_k:
            x += 'K'
        return x

    @property
    def header(self) -> str:
        """The Latex environment header.

        """
        head: str = self._get_header()
        if self.head is not None:
            head += f'{{{self.head}}}'
        return head

    def _get_header(self) -> str:
        """Return the Latex environment header.

        """
        params = self.get_params(False)
        if len(params['placement']) == 0:
            params['placement'] = 'h!'
        return """\\begin{%(latex_environment)s}[%(placement)s]{%(tabname)s}%%
{%(caption)s}{%(size)s}{%(columns)s}""" % params

    def _apply_df_eval_pre(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.df_code_exec_pre is not None:
            _locs = locals()
            exec(self.df_code_exec_pre)
            df = _locs['df']
        if self.df_code_pre is not None:
            df = eval(self.df_code_pre)
        return df

    def _apply_df_number_format(self, df: pd.DataFrame) -> pd.DataFrame:
        col: str
        for col in self.percent_column_names:
            df[col] = df[col].apply(lambda s: s.replace('%', '\\%'))
        kwargs: Optional[Dict[str, Any]]
        for col, kwargs in self.format_thousands_column_names.items():
            kwargs = {} if kwargs is None else kwargs
            df[col] = df[col].apply(lambda x: self.format_thousand(x, **kwargs))
        for col, rnd in self.make_percent_column_names.items():
            fmt = f'{{v:.{rnd}f}}\\%'
            df[col] = df[col].apply(
                lambda v: fmt.format(v=round(v * 100, rnd), rnd=rnd))
        return df

    def _apply_df_eval_post(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.df_code_exec is not None:
            exec(self.df_code_exec)
        for col, code, in self.column_evals.items():
            func = eval(code)
            df[col] = df[col].apply(func)
        if self.df_code is not None:
            df = eval(self.df_code)
        return df

    def _apply_df_add_indexes(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.index_col_name is not None:
            df[self.index_col_name] = range(1, len(df) + 1)
            cols = df.columns.to_list()
            cols = [cols[-1]] + cols[:-1]
            df = df[cols]
        return df

    def _apply_df_column_modifies(self, df: pd.DataFrame) -> pd.DataFrame:
        col: str
        repl: Dict[Any, Any]
        for col, repl in self.column_value_replaces.items():
            df[col] = df[col].apply(lambda v: repl.get(v, v))
        df = df.drop(columns=self.column_removes)
        if self.column_keeps is not None:
            df = df[self.column_keeps]
        df = df.rename(columns=self.column_renames)
        return df

    def _apply_df_font_format(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.replace_nan is not None:
            df = df.fillna(self.replace_nan)
        if len(self.blank_columns) > 0:
            cols = df.columns.to_list()
            for i in self.blank_columns:
                cols[i] = ''
            df.columns = cols
        if len(self.bold_cells) > 0:
            df = self._apply_df_bold_cells(df, self.bold_cells)
        return df

    def _apply_df_bold_cells(self, df: pd.DataFrame,
                             cells: Sequence[Tuple[int, int]]):
        str_cols: bool = len(cells) > 0 and isinstance(cells[0][1], str)
        cixs: Dict[str, int] = dict(zip(df.columns, it.count()))
        r: int
        c: int
        for r, c in cells:
            val: Any = df[c].iloc[r] if str_cols else df.iloc[r, c]
            fmt: str = '\\textbf{' + str(val) + '}'
            if str_cols:
                c = cixs[c]
            df.iloc[r, c] = fmt
        return df

    def _apply_df_capitalize(self, df: pd.DataFrame):
        for col, capwords in self.capitalize_columns.items():
            fn: Callable = string.capwords if capwords else str.capitalize
            df[col] = df[col].apply(fn)
        return df

    def _get_bold_columns(self, df: pd.DataFrame) -> Tuple[Tuple[int, int]]:
        if len(self.bold_max_columns) > 0:
            cixs: List[str] = self.bold_max_columns
            return tuple(zip(df[cixs].idxmax(), cixs))
        else:
            return ()

    @property
    def dataframe(self) -> pd.DataFrame:
        """The Pandas dataframe that holds the CSV data."""
        if not hasattr(self, '_dataframe_val'):
            self._dataframe_val = pd.read_csv(self.path, **self.read_kwargs)
        return self._dataframe_val

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame):
        """The Pandas dataframe that holds the CSV data."""
        self._dataframe_val = dataframe
        self._formatted_dataframe.clear()

    @property
    @persisted('_formatted_dataframe')
    def formatted_dataframe(self) -> pd.DataFrame:
        """The :obj:`dataframe` with the formatting applied to it used to create
        the Latex table.  Modifications such as string replacements for adding
        percents is done.

        """
        df: pd.DataFrame = self.dataframe
        df = self._apply_df_eval_pre(df)
        bold_cols: Tuple[Tuple[int, int]] = self._get_bold_columns(df)
        df = self._apply_df_number_format(df)
        df = self._apply_df_eval_post(df)
        df = self._apply_df_bold_cells(df, bold_cols)
        df = self._apply_df_capitalize(df)
        df = self._apply_df_add_indexes(df)
        df = self._apply_df_column_modifies(df)
        df = self._apply_df_font_format(df)
        return df

    def _get_header_rows(self, df: pd.DataFrame) -> List[List[Any]]:
        cols = [tuple(map(lambda c: f'\\textbf{{{c}}}', df.columns))]
        return it.chain(cols, map(lambda x: x[1].tolist(), df.iterrows()))

    def _get_tabulate_params(self) -> Dict[str, Any]:
        params = dict(tablefmt='latex_raw', headers='firstrow')
        params.update(self.write_kwargs)
        return params

    def write(self, depth: int = 0, writer: TextIOWrapper = sys.stdout):
        df: pd.DataFrame = self.formatted_dataframe
        data = self._get_header_rows(df)
        params: Dict[str, Any] = self._get_tabulate_params()
        lines = tabulate(data, **params).split('\n')
        params = dict(self.get_params(True))
        params['cvars'] = ''
        n_var_args = len(self.var_args)
        if n_var_args > 0:
            params['cvars'] = f'[{n_var_args}]'
        writer.write('\n\\newcommand{\\%(tabname)s}%(cvars)s{%%\n' % params)
        writer.write(self.header)
        writer.write('\n')
        for lix, ln in enumerate(lines[1:-1]):
            writer.write(ln + '\n')
            if (lix - 2) in self.hlines:
                writer.write('\\hline  \n')
            if (lix - 2) in self.double_hlines:
                writer.write('\\hline \\hline \n')
        writer.write('\\end{%s}}\n' % self.latex_environment)

    def _from_dictable(self, *args, **kwargs) -> Dict[str, Any]:
        dct = super()._from_dictable(*args, **kwargs)
        dct['type'] = re.sub(r'Table$', '', self.__class__.__name__).lower()
        def_inst = self.__class__(name=None, path=None, caption=None)
        dels: List[str] = []
        for k, v in dct.items():
            if (not hasattr(def_inst, k) or v == getattr(def_inst, k)) or \
               (isinstance(v, (list, set, tuple, dict)) and len(v) == 0):
                dels.append(k)
        for k in dels:
            del dct[k]
        return dct

    def serialize(self) -> Dict[str, Any]:
        """Return a data structure usable for YAML or JSON output by flattening
        Python objects.

        """
        tab_name: str = self.name
        # using json to recursively convert OrderedDict to dicts
        tab_def: Dict[str, Any] = self.asflatdict()
        del tab_def['name']
        return {tab_name: tab_def}

    def __str__(self):
        return f'{self.name}: env={self.latex_environment}, size={self.size}'


@dataclass
class SlackTable(Table):
    """An instance of the table that fills up space based on the widest column.

    """
    slack_col: int = field(default=0)
    """Which column elastically grows or shrinks to make the table fit."""

    @property
    def latex_environment(self):
        return 'zzvarcoltable' if self.single_column else 'zzvarcoltabletcol'

    def _get_header(self) -> str:
        params = self.get_params(False)
        width = '\\columnwidth' if self.single_column else '\\textwidth'
        params['width'] = width
        return """\\begin{%(latex_environment)s}[%(width)s]{%(placement)s}{%(tabname)s}{%(caption)s}%%
{%(size)s}{%(columns)s}""" % params

    @property
    def columns(self) -> str:
        cols: str = self.column_aligns
        if cols is None:
            df = self.formatted_dataframe
            i = self.slack_col
            cols = ('l' * (df.shape[1] - 1))
            cols = cols[:i] + 'X' + cols[i:]
            cols = '|' + '|'.join(cols) + '|'
        return cols


@dataclass
class LongTable(SlackTable):
    @property
    def latex_environment(self):
        return 'zzvarcoltabletcollong'

    def _get_header(self) -> str:
        df = self.formatted_dataframe
        hcols = ' & '.join(map(lambda c: f'\\textbf{{{c}}}', df.columns))
        return f'{super()._get_header()}{{{hcols}}}{{{df.shape[1]}}}'

    def _get_header_rows(self, df: pd.DataFrame) -> List[List[Any]]:
        return map(lambda x: x[1].tolist(), df.iterrows())

    def _get_tabulate_params(self) -> Dict[str, Any]:
        params = super()._get_tabulate_params()
        del params['headers']
        return params

    def write(self, depth: int = 0, writer: TextIOWrapper = sys.stdout):
        sio = StringIO()
        super().write(depth, writer)
        sio.seek(0)
        hlremove = 1
        for line in map(str.strip, sio.readlines()):
            if line == '\\hline' and hlremove > 0:
                hlremove += 1
                continue
            writer.write(line + '\n')
