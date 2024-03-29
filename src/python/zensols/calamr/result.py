"""Results and analysis.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Any, ClassVar, Dict
from dataclasses import dataclass, field
import logging
import sys
import itertools as it
from pathlib import Path
import pandas as pd
from zensols.util import time
from zensols.persist import persisted, PersistedWork, Stash
from zensols.datdesc import DataDescriber, DataFrameDescriber
from zensols.amr import AmrFeatureDocument
from . import (
    DocumentGraph, DocumentGraphFactory, DocumentGraphAligner, FlowDocumentGraph
)

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResultGenerator(object):
    """Create alignment tabular data and write to CSV files.  Each file is named
    by its unique identifier and the data generated from
    :meth:`.FlowDocumentGraph.create_align_df`.

    """
    anon_doc_stash: Stash = field()
    """Contains human annotated AMRs from a small toy corpus or from the AMR 3.0
    Proxy Report corpus.

    """
    doc_graph_aligner: DocumentGraphAligner = field()
    """Create document graphs."""

    doc_graph_factory: DocumentGraphFactory = field()
    """Create document graphs."""

    results_dir: Path = field()
    """Where the results to be read are located (has ``docs`` and ``parse``
    subdirs).

    """
    limit: int = field(default=sys.maxsize)
    """Max number of document to align."""

    def _align(self, out_dir: Path, key: str):
        logger.info(f'aligning {key}')
        out_file: Path = out_dir / f'{key}.csv'
        doc: AmrFeatureDocument = self.anon_doc_stash[key]
        doc_graph: DocumentGraph = self.doc_graph_factory(doc)
        fdg: FlowDocumentGraph = self.doc_graph_aligner.align(doc_graph)
        df: pd.DataFrame = fdg.create_align_df(True)
        df.to_csv(out_file, index=False)
        logger.info(f'wrote: {out_file}')

    def __call__(self, name: str, split_name: str, keys: Tuple[str]):
        logger.info(f'creating alignments in {self.results_dir}')
        out_dir: Path = self.results_dir / name / split_name
        keys = tuple(it.islice(keys, self.limit))
        logger.info(f'writing {len(keys)} docs to {out_dir}')
        self.doc_graph_aligner.render_level = 0
        out_dir.mkdir(parents=True, exist_ok=True)
        with time(f'aligned {len(keys)}'):
            key: str
            for key in keys:
                try:
                    self._align(out_dir, key)
                except Exception as e:
                    logger.error(f'can not score document {key}: {e}',
                                 stack_info=True, exc_info=True)


@dataclass
class ResultAnalyzer(object):
    """A class that summarizes the results and outputs to be added to the paper.

    """
    CALAMR_METHOD: ClassVar[str] = 'calamr'
    """The name of the CALAMR scoring method column."""

    COMPARE_METHOD: ClassVar[str] = 'score'
    """The name of the compared scoring method column."""

    parser_meta_file: Path = field()
    """A file that describes all the results columns."""

    results_dir: Path = field()
    """Where the results to be read are located (has ``docs`` and ``parse``
    subdirs).

    """
    output_dir: Path = field()
    """The directory to output the summarized results as CSV files."""

    config_dir: Path = field()
    """Where to write the YAML ``datdesc`` table description files."""

    aligns_dir: Path = field()
    """Where the alignment (created by :class:`.AlignmentResultGenerator`)
    output is stored.

    """
    def __post_init__(self):
        self._doc_df = PersistedWork('_doc_df', self)
        self._parse_df = PersistedWork('_parse_df', self)

    def clear(self):
        for pw in [self._doc_df, self._parse_df]:
            pw.clear()

    @persisted('_doc_df')
    def get_doc_df(self) -> pd.DataFrame:
        """"Return a dataframe of the summarized score results."""
        return self._summarize_docs().df

    def _summarize_docs(self) -> DataFrameDescriber:
        doc_dir: Path = self.results_dir / 'docs'
        dfs = []
        names: List[str] = []
        logger.info(f'summarizaing docs in {doc_dir}')
        path: Path
        for path in doc_dir.iterdir():
            df = pd.read_csv(path, index_col=0)
            df = df.drop(columns=['doc_id'])
            d_m = df.mean(axis=0).to_dict()
            d_m = {f'{x[0]}_mean': x[1] for x in d_m.items()}
            d_s = df.sum(axis=0).to_dict()
            d_s = {f'{x[0]}_sum': x[1] for x in d_s.items()}
            dfs.append(d_m | d_s)
            names.append(path.stem)
        df = pd.DataFrame(data=dfs)
        df['corpus'] = names
        return DataFrameDescriber(
            name='calamrdoc',
            df=df,
            meta_path=self.parser_meta_file,
            desc='Alignment Algorithm Documents Summarization Scores')

    @persisted('_parse_df')
    def get_parse_df(self) -> pd.DataFrame:
        """Return a dataframe of all the parser score results."""
        doc_dir: Path = self.results_dir / 'parse'
        dfs = []
        logger.info(f'summarizaing parses in {doc_dir}')
        for csv_path in doc_dir.glob('**/*.csv'):
            corpus: str = csv_path.parent.name
            parser: str = csv_path.stem
            df: pd.DataFrame = pd.read_csv(csv_path, index_col=0)
            df['corpus'] = corpus
            df['parser'] = parser
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def _summarize_parse(self) -> DataFrameDescriber:
        calamr_col: str = 'calamr_agg_aligned_portion'
        rows: List[Tuple[Any, ...]] = []
        dfp: pd.DataFrame = self.get_parse_df()
        dfp = dfp[dfp[calamr_col] > 0].copy()
        dfp['loss_wlk'] = (dfp[calamr_col] - dfp['wlk']).abs()
        dfp['loss_smatch'] = (dfp[calamr_col] - dfp['smatch_f_score']).abs()
        cp_cols: List[str] = 'corpus parser'.split()
        id_cols: List[str] = 'id corpus parser'.split()
        for cp, dfg in dfp.groupby(cp_cols):
            corr: Dict[str, float] = {
                'corr_wlk': dfg[calamr_col].corr(dfg['wlk']),
                'corr_smatch': dfg[calamr_col].corr(dfg['smatch_f_score'])}
            rows.append(dfg[cp_cols].iloc[0].to_dict() |
                        dfg.drop(columns=id_cols).mean().to_dict() |
                        {'count': len(dfg)} |
                        corr)
        df = pd.DataFrame(rows)
        return DataFrameDescriber(
            name='calamrparse',
            df=df,
            meta_path=self.parser_meta_file,
            desc='Alignment Algorithm Parser Scores')

    def report(self):
        """Write all summarized results and ``datdesc`` description YAML files.

        """
        dfd = DataDescriber(describers=(
            self._summarize_docs(),
            self._summarize_parse()))
        dfd.save(self.output_dir, self.config_dir)
