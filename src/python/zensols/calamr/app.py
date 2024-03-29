"""AMR Summarization.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Dict, Any, List, Tuple, ClassVar, Sequence, Iterable
from dataclasses import dataclass, field
import logging
import sys
import traceback
import re
from pathlib import Path
import itertools as it
from io import TextIOWrapper
import pandas as pd
from zensols.util import stdout
from zensols.config import ConfigFactory, DefaultDictable
from zensols.persist import Stash
from zensols.cli import ApplicationError
from zensols.nlp.score import ScoreSet
from zensols.amr import AmrFeatureDocument, Format
from zensols.amr.serial import AmrSerializedFactory
from . import (
    DocumentGraph, DocumentGraphFactory, DocumentGraphAligner, FlowDocumentGraph
)

logger = logging.getLogger(__name__)


@dataclass
class _AlignmentBaseApplication(object):
    doc_graph_aligner: DocumentGraphAligner = field()
    """Create document graphs."""

    doc_graph_factory: DocumentGraphFactory = field()
    """Create document graphs."""

    serialized_factory: AmrSerializedFactory = field()
    """Creates a :class:`.Serialized` from :class:`.AmrDocument`,
    :class:`.AmrSentence` or :class:`.AnnotatedAmrDocument`.

    """
    def _get_output_file(self, output_dir: Path, key: str,
                         output_format: Format) -> Path:
        output_dir = output_dir / key
        self.doc_graph_aligner.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        ext: str = Format.to_ext(output_format)
        return output_dir / f'results.{ext}'

    def _output_align(self, output_format: Format, doc: AmrFeatureDocument,
                      fout: TextIOWrapper, res: FlowDocumentGraph):
        if output_format == Format.txt:
            self.serialized_factory(doc.amr).write(writer=fout)
            res.write(writer=fout)
        elif output_format == Format.json:
            dct = DefaultDictable(
                {'doc': self.serialized_factory(doc.amr).asdict(),
                 'alignment': res.asdict()})
            fout.write(dct.asjson(indent=4))
        elif output_format == Format.csv:
            res.stats_df.to_csv(fout, index=False)
        else:
            raise ApplicationError(
                f'Format not supported: {output_format.name}')

    def _align(self, key: str, output_dir: Path, output_format: Format,
               use_stdout: bool) -> FlowDocumentGraph:
        aligner: DocumentGraphAligner = self.doc_graph_aligner
        logger.info(f'aligning on document {key}')
        fout: TextIOWrapper
        output_file: Path
        if use_stdout:
            fout = sys.stdout
        else:
            output_file = self._get_output_file(output_dir, key, output_format)
            fout = open(output_file, 'w')
        try:
            doc: AmrFeatureDocument = self.anon_doc_stash.get(key)
            if doc is None:
                raise ApplicationError(f"No such key: '{key}', use dump action")
            doc_graph: DocumentGraph = self.doc_graph_factory(doc)
            res: FlowDocumentGraph = aligner.align(doc_graph)
            self._output_align(output_format, doc, fout, res)
            return res
        finally:
            if id(fout) != id(sys.stdout):
                fout.close()
                logger.info(f'wrote: {output_file}')

    def _prep_align(self, output_dir: Path, render_level: int) -> \
            Tuple[Path, bool]:
        assert output_dir is not None
        use_stdout: bool
        if output_dir.name == stdout.STANDARD_OUT_PATH:
            use_stdout = True
        else:
            output_dir = output_dir.expanduser()
            use_stdout = False
        if not use_stdout:
            logger.setLevel(logging.INFO)
        if render_level is not None:
            self.doc_graph_aligner.render_level = render_level
        return output_dir, use_stdout


@dataclass
class CorpusApplication(_AlignmentBaseApplication):
    """AMR graph aligment.

    """
    config_factory: ConfigFactory = field()
    """For prototyping."""

    anon_doc_stash: Stash = field()
    """Contains human annotated AMRs from a small toy corpus or from the AMR 3.0
    Proxy Report corpus.

    """
    results_dir: Path = field()
    """The directory where the output results are written, then read back for
    analysis reporting.

    """
    def write_micro_corpus(self):
        """Write the micro corpus from the JSON created file."""
        from .corpus.micro import MicroSummaryCorpusWriter
        writer: MicroSummaryCorpusWriter = \
            self.config_factory('calamr_micro_corpus')
        corp_file: Path = writer.input_file
        logger.info(f'reading: {corp_file}')
        writer()
        self.clear_doc_cache()

    def add_micro_corpus(self, k: int):
        """Add a CNN/DailyMail corpus entry to the micro corpus file.

        :param k: the kth shortest article

        """
        from zensols.cnndmdb import ApplicationFactory, Corpus, Article
        from .corpus.micro import MicroSummaryCorpusWriter
        import json
        writer: MicroSummaryCorpusWriter = \
            self.config_factory('calamr_micro_corpus')
        corp_file: Path = writer.input_file
        print(f'add the following to {corp_file}')
        corpus: Corpus = ApplicationFactory.get_corpus()
        art: Article = corpus.get_kth_shortest(k)
        data: Dict[str, str] = {
            'body': art.text,
            'summary': ' '.join(art.highlights),
        }
        print(json.dumps(data, indent=4))

    def get_annotated_summary(self, limit: int = None) -> pd.DataFrame:
        """Return a CSV file with a summary of the annotated AMR dataset.

        :param limit: the max of items to process

        """
        rows: List[Tuple[str, int]] = []
        idx: List[str] = []
        limit = sys.maxsize if limit is None else limit
        k: str
        doc: AmrFeatureDocument
        for k, doc in it.islice(self.anon_doc_stash.items(), limit):
            idx.append(k)
            rows.append((doc.token_len, doc.text))
        df = pd.DataFrame(rows, columns='len text'.split(), index=idx)
        df = df.sort_values('len')
        df.index.name = 'id'
        return df

    def dump_annotated(self, limit: int = None, output_dir: Path = None,
                       output_format: Format = Format.csv):
        """Write annotated documents and their keys.

        :param limit: the max of items to process

        :param output_dir: the output directory

        :param output_format: the output format

        """
        if output_dir is None:
            output_dir = self.results_dir
        output_dir = output_dir.expanduser()
        limit = sys.maxsize if limit is None else limit
        fac: AmrSerializedFactory = self.serialized_factory
        docs: Dict[str, Any] = {}
        k: str
        doc: AmrFeatureDocument
        for k, doc in it.islice(self.anon_doc_stash.items(), limit):
            docs[k] = fac(doc.amr).asdict()
        dct = DefaultDictable(docs)
        fname: str = f'annotated.{Format.to_ext(output_format)}'
        output_file: Path = output_dir / fname
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as fout:
            if output_format == Format.txt:
                dct.write(writer=fout)
            elif output_format == Format.csv:
                df: pd.DataFrame = self.get_annotated_summary(limit)
                df.to_csv(fout)
            elif output_format == Format.json:
                fout.write(dct.asjson(indent=4))
            else:
                raise ApplicationError(
                    f'Format not supported: {output_format.name}')
        logger.info(f'wrote: {output_file}')

    def align_corpus(self, keys: str, output_dir: Path = None,
                     output_format: Format = Format.csv,
                     render_level: int = None):
        """Align an annotated AMR document from the corpus.

        :param keys: comma-separated list of dataset keys or file name

        :param output_dir: the output directory

        :param output_format: the output format

        :param render_level: how many graphs to render (0 - 10), higher means
                             more

        """
        success_keys: List[str] = []
        results: List[FlowDocumentGraph] = []
        use_stdout: bool
        output_dir, use_stdout = self._prep_align(output_dir, render_level)
        keys: Sequence[str]
        if keys == 'ALL':
            keys = tuple(self.anon_doc_stash.keys())
        else:
            keys = re.findall(r'[^,\s]+', keys)
        key: str
        for key in keys:
            try:
                fdg: FlowDocumentGraph = self._align(
                    key, output_dir, output_format, use_stdout)
                results.append(fdg)
                success_keys.append(key)
            except Exception as e:
                msg: str = f'Can not align {key}: {e}'
                logger.error(msg)
                if not use_stdout:
                    with open(output_dir / 'errors.txt', 'a') as f:
                        f.write(f'error: {msg}\n')
                        traceback.print_exception(e, file=f)
                        f.write('_' * 80 + '\n')
        if not use_stdout and len(keys) > 1:
            dfs: List[pd.DataFrame] = []
            res: FlowDocumentGraph
            for key, res in zip(success_keys, results):
                df: pd.DataFrame = res.stats_df
                cols = df.columns.tolist()
                df['id'] = key
                df = df[['id'] + cols]
                dfs.append(df)
            df = pd.concat(dfs)
            df_path: Path = output_dir / 'results.csv'
            df.to_csv(df_path, index=False)
            logger.info(f'wrote: {df_path}')

    def clear_doc_cache(self):
        """Clear the annotated feature document stash."""
        self.anon_doc_stash.clear()


@dataclass
class AlignmentApplication(_AlignmentBaseApplication):
    """This application aligns data in files.

    """
    config_factory: ConfigFactory = field()
    """Application configuration factory."""

    def align_file(self, input_file: Path, output_dir: Path = None,
                   output_format: Format = Format.csv,
                   render_level: int = None):
        """Align annotated documents from a file.

        :param input_file: the input JSON file.

        :param output_dir: the output directory

        :param output_format: the output format

        :param render_level: how many graphs to render (0 - 10), higher means
                             more

        """
        from .annotate import AnnotatedAmrFeatureDocumentFactory
        factory: AnnotatedAmrFeatureDocumentFactory = \
            self.config_factory('anon_doc_factory')
        aligner: DocumentGraphAligner = self.doc_graph_aligner
        output_dir, use_stdout = self._prep_align(output_dir, render_level)
        success_keys: List[str] = []
        results: List[FlowDocumentGraph] = []
        dix: int
        doc: AmrFeatureDocument
        for dix, doc in enumerate(factory(input_file)):
            try:
                doc_graph: DocumentGraph = self.doc_graph_factory(doc)
                output_file: Path
                if use_stdout:
                    output_file = stdout.STANDARD_OUT_PATH
                else:
                    output_file: Path = self._get_output_file(
                        output_dir, str(dix), output_format)
                res: FlowDocumentGraph = aligner.align(doc_graph)
                with stdout(output_file, 'w') as fout:
                    self._output_align(output_format, doc, fout, res)
                results.append(res)
                success_keys.append(dix)
            except Exception as e:
                msg: str = f'Can not align {dix}th {doc}: {e}'
                logger.error(msg)
                if not use_stdout:
                    with open(output_dir / 'errors.txt', 'a') as f:
                        f.write(f'error: {msg}\n')
                        traceback.print_exception(e, file=f)
                        f.write('_' * 80 + '\n')
        if not use_stdout and dix > 1:
            dfs: List[pd.DataFrame] = []
            res: FlowDocumentGraph
            for key, res in zip(success_keys, results):
                df: pd.DataFrame = res.stats_df
                cols = df.columns.tolist()
                df['id'] = key
                df = df[['id'] + cols]
                dfs.append(df)
            df = pd.concat(dfs)
            df_path: Path = output_dir / 'results.csv'
            df.to_csv(df_path, index=False)
            logger.info(f'wrote: {df_path}')


@dataclass
class ScorerApplication(object):
    """The application scores AMR documents and sentences.

    """
    config_factory: ConfigFactory = field()
    """For prototyping."""

    score_cachers: Tuple[object] = field()
    """All objects that cache scores."""

    def score_anon_docs(self, limit: int = None, doc_keys: str = None,
                        output_file: Path = Path('scores.csv')) -> ScoreSet:
        """Score annotated documents (config specifies the dataset).

        :param limit: the max of items to process

        :param doc_keys: a comma delimited list of document keys

        :param output_file: where to write the results

        """
        from .score import CalamrScorer
        scorer: CalamrScorer = self.config_factory('calamr_doc_scorer')
        if limit is not None:
            scorer.doc_limit = limit
        if doc_keys is None:
            df = scorer.score_docs()
        else:
            df = scorer.score_docs_by_key(doc_keys.split(','))
        df.to_csv(output_file, index=False)
        logger.info(f'wrote: {output_file}')

    def clear_score_cache(self):
        """Clear all cached score data.

        """
        for cacher in self.score_cachers:
            cacher.clear()
