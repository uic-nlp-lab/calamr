#!/usr/bin/env python

from typing import Tuple
from dataclasses import dataclass, field
from typing import List, Any
import logging
from pathlib import Path
import plac
import pandas as pd
from zensols.persist import persisted
from zensols.datdesc import DataFrameDescriber, DataDescriber
from zensols.rend import ApplicationFactory, BrowserManager

logger = logging.getLogger(__name__)


@dataclass
class ReentrancyCalculator(object):
    """Computes reentracy statistics by document.

    """
    align_df: pd.DataFrame = field()
    """The dataframe to analyze."""

    group_columns: Tuple[Tuple[str, str]] = field()
    """The columns to group by in the form:

      ``((<column name>, <description of column>), ...)``

    """
    @property
    def stats_describer(self) -> DataFrameDescriber:
        rows: List[Any] = []
        group_cols: List[str] = list(map(lambda t: t[0], self.group_columns))
        cols = group_cols + 'failed n_reentracy n_reentracy_flow'.split()
        col_descs = (
            *tuple(map(lambda t: t[1], self.group_columns)),
            'whether a critical failure occured resulting in no flow',
            'number of total reentrancies',
            'number of reentrancies with flow',
            'portion of reentrant edges with non-zero flow')
        for gcols, df in self.align_df.groupby(group_cols):
            rows.append((
                *gcols,
                df['flow'].sum() == 0,
                len(df[df['reentrancy']]),
                len(df[df['reentrancy'] & (df['flow'] > 0)])))
        df = pd.DataFrame(rows, columns=cols)
        df['reentrancy_fixed'] = df['n_reentracy_flow'] / df['n_reentracy']
        return DataFrameDescriber(
            name='reentrancies-doc',
            df=df,
            desc='Reentrancy statistics by Proxy Report document',
            meta=tuple(zip((*cols, 'reentrancy_fixed'), col_descs)))


@dataclass
class AlignCompiler(object):
    """Create statistics on the proxy report and mismatch corpora alignments.

    """
    results_dir: Path = field()
    """The directory where the alignmnt stats output was written."""

    @property
    def base_results_dir(self) -> Path:
        return self.results_dir.parent

    @property
    @persisted('_browser_manager')
    def browser_manager(self) -> BrowserManager:
        return ApplicationFactory.get_browser_manager()

    def _compile_results(self, group_columns: Tuple[Tuple[str, str]]) -> \
            pd.DataFrame:
        if not self.results_dir.is_dir():
            raise ValueError(f'Not a directory: {self.results_dir}')
        logger.info(f'compiling results in {self.results_dir}')
        dfs: List[pd.DataFrame] = []
        cols: Tuple[str] = tuple(map(lambda t: t[0], group_columns))
        path: Path
        for path in self.results_dir.glob('**/*.csv'):
            vals = (path.parent.parent.name,
                    path.parent.name,
                    path.stem)
            df = pd.read_csv(path)
            for i, (col, val) in enumerate(zip(cols, vals)):
                df.insert(i, col, val)
            dfs.append(df)
        if len(dfs) == 0:
            raise ValueError(f'No results found: {self.results_dir}')
        logger.info(f'compiled results from {len(dfs)} files')
        return pd.concat(dfs)

    @property
    def reentrancy_describer(self) -> DataFrameDescriber:
        group_columns = (
            ('reentrancy_type',
             'whether the document graph reetrancies were fixed'),
            ('corpus', 'the original corpus'),
            ('doc_id', 'the proxy report document identifier'))
        df: pd.DataFrame = self._compile_results(group_columns)
        calc = ReentrancyCalculator(
            group_columns=group_columns,
            align_df=df)
        dd: DataFrameDescriber = calc.stats_describer
        return dd

    def _create_align_row(self, df: pd.DataFrame) -> Tuple[int]:
        """Create a row in the describer :obj:`.alignment_doc_describer`.

        """
        def get_rels(df: pd.DataFrame) -> Tuple[int, int]:
            """Create coreference relations

            :return: a tuple of ``(<# component refs>, <# bipartite refs>)``

            """
            rels_df = df[~df['rel_id'].isnull()]
            bipartite_df = rels_df[rels_df['is_bipartite']]
            n_bipartite_rels = len(bipartite_df)
            n_comp_rels = len(rels_df) - n_bipartite_rels
            return (n_comp_rels, n_bipartite_rels)

        def count_edges(name: str, non_zero_flow: bool) -> Tuple[int, int]:
            """Count role and align edges.

            :param name: ``source`` or ``summary``

            :param non_zero_flow: whether to filter 0 flow edges

            :return: a tupe of ``(<# aligned edges>, <# role edges>)``

            """
            dfn: pd.DataFrame = df[df['name'] == name]
            if non_zero_flow:
                dfn = dfn[dfn['flow'] > 0]
            dfr: pd.DataFrame = dfn[dfn['edge_type'] == 'role']
            dfa: pd.DataFrame = dfn[dfn['edge_type'] == 'align']
            role: int = len(dfr)
            align: int = len(dfa['t_id'].drop_duplicates())
            return (align, role)

        return (*count_edges('source', False),
                *count_edges('summary', False),
                *count_edges('source', True),
                *count_edges('summary', True),
                *get_rels(df[df['name'] == 'source']),
                *get_rels(df[df['name'] == 'summary']))

    @property
    def alignment_doc_describer(self) -> DataFrameDescriber:
        """Create a data frame describer of the alignment graph flow results by
        document.

        """
        group_columns = (
            ('reentrancy_type',
             'whether the document graph reetrancies were fixed'),
            ('split', 'the dataset split'),
            ('doc_id', 'the proxy report document identifier'))
        cols = (
            ('split', 'the dataset split'),
            ('doc_id', 'the proxy report document identifier'),

            ('src_align_tot', 'number of bipartite alignment edges originating from the source'),
            ('src_role_tot', 'number of source graph role edges'),
            ('smy_align_tot', 'number of bipartite alignment edges originating from the summary'),
            ('smy_role_tot', 'number of summary graph role edges'),

            ('src_align_nzf', 'number of non-zero flow bipartite alignment edges originating from the source'),
            ('src_role_nzf', 'number of non-zero flow source graph role edges'),
            ('smy_align_nzf', 'number of non-zero flow bipartite alignment edges originating from the summary'),
            ('smy_role_nzf', 'number of non-zero flow summary graph role edges'),

            ('src_comp_rel', 'number of source component coreference relations'),
            ('src_bipartite_rel', 'number of source bipartite coreference relations'),
            ('smy_comp_rel', 'number of summary component coreference relations'),
            ('smy_bipartite_rel', 'number of summarysource bipartite coreference relations'))
        df: pd.DataFrame = self._compile_results(group_columns)
        df = df[df['reentrancy_type'] == 'fix']
        df = df[(df['s_descr'] != 'S') & (df['t_descr'] != 'T')]
        rows: List[Tuple] = []
        for (split, doc_id), dfg in df.groupby('split doc_id'.split()):
            rows.append((split, doc_id, *self._create_align_row(dfg)))
        return DataFrameDescriber(
            name='alignment-graph-flows',
            df=pd.DataFrame(
                rows,
                columns=tuple(map(lambda t: t[0], cols))),
            desc='Alignment Graph Flow Results by Document',
            meta=cols)

    @property
    def alignment_describer(self) -> DataFrameDescriber:
        """Like :obj:`alignment_doc_describer` but summarize by summing over the
        dataset splits and add porportions of edge counts.

        """
        dfd: DataFrameDescriber = self.alignment_doc_describer
        dfs = dfd.df.groupby('split').sum()
        dfs.insert(0, 'split', dfs.index)
        dfs = dfs.reset_index(drop=True)
        # proportions of role and alignment edges
        dfs.insert(1, 'align_src_por', dfs['src_align_nzf'] / dfs['src_role_tot'])
        dfs.insert(2, 'role_src_por', dfs['src_role_nzf'] / dfs['src_role_tot'])
        dfs.insert(3, 'align_smy_por', dfs['smy_align_nzf'] / dfs['smy_role_tot'])
        dfs.insert(4, 'role_smy_por', dfs['smy_role_nzf'] / dfs['smy_role_tot'])
        meta = (
            ('align_src_por', 'portion of source positive flow alignment to total role edges'),
            ('role_src_por', 'portion of source positive flow role to total role edges'),
            ('align_smy_por', 'portion of summary positive flow alignment to total role edges'),
            ('role_smy_por', 'portion of summary~ positive flow role to total role edges'))
        return dfd.derive(
            df=dfs,
            meta=meta)

    @property
    def liu_edge_coverage(self) -> DataFrameDescriber:
        """The previous Liu et al. results."""
        path: Path = self.base_results_dir / 'prev/liu-edge-coverage.csv'
        df: pd.DataFrame = pd.read_csv(path)
        meta: pd.DataFrame = df.iloc[0:1].T
        meta.columns = ['description']
        df = df.drop(index=0)
        # paper uses percentages, so put it in the same format as our quotients
        df.iloc[:, 1:] = df.iloc[:, 1:].astype(float) / 100.
        return DataFrameDescriber(
            name=path.stem,
            desc='Liu et al. (2015) Edge Coverage',
            meta=meta,
            df=df)

    @property
    def combined_describer(self) -> DataFrameDescriber:
        """Combined aligned Calamr with Liu et al results."""
        dfd_a: DataFrameDescriber = self.alignment_describer
        dfd_l: DataFrameDescriber = self.liu_edge_coverage
        df = dfd_a.df.merge(dfd_l.df, left_on='split', right_on='split')
        return dfd_a.derive(
            df=df,
            meta=dfd_l.meta)

    def __call__(self):
        dfd: DataFrameDescriber = self.combined_describer
        dd = DataDescriber(describers=(dfd,))
        dd.save(output_dir=self.base_results_dir / 'condensed')


@plac.annotations(
    results=('the directory with the align results', 'option',
             None, Path))
def main(results: Path = Path('results/aligns-1_0')):
    """Create statistics on the proxy report and mismatch corpora alignments.

    """
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.DEBUG)
    logging.getLogger('zensols.datdesc').setLevel(logging.INFO)
    comp = AlignCompiler(results_dir=results)
    comp()


if (__name__ == '__main__'):
    plac.call(main)
