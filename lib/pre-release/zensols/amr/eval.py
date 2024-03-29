"""Evaluate models.

"""
__author__ = 'Paul Landes'

from typing import Union, Iterable, Any, List, Tuple, Dict
from dataclasses import dataclass, field
import logging
import sys
import itertools as it
from io import TextIOBase, StringIO
from pathlib import Path
from spacy.tokens.doc import Doc
from zensols.util.time import time
from zensols.config import Writable, Configurable
from zensols.persist import persisted, PersistedWork, Stash
from zensols.multi import MultiProcessStash
from zensols.install import Installer
from zensols.nlp import FeatureDocumentParser
from amrlib.evaluate.smatch_enhanced import compute_smatch
from . import AmrDocument, AmrSentence, AmrParser

logger = logging.getLogger(__name__)


@dataclass
class _EvalParseIssue(Writable):
    index: int
    gold: AmrDocument
    pred: AmrDocument
    used: AmrSentence
    error: Exception = field(default=None)

    @property
    def mult_sent_problem(self) -> bool:
        """``True`` if this issue was created from too many sentences so :obj"`pred`
        and :obj:`used` will be populated.  Otherwise :obj:`error` is
        populated.

        """
        return self.used is not None

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_divider(depth, writer, '=', 40)
        self._write_line(f'gold sent: {self.gold.text}', depth, writer)
        if self.mult_sent_problem:
            # issue was too many sentences returned
            pstr = '<none>' if self.pred is None else str(len(self.pred))
            self._write_line('Expecting document to have one sentence ' +
                             f'but got: {pstr}', depth, writer)
            self._write_line('gold:', depth, writer)
            self._write_block(self.gold.graph_string, depth + 1, writer)
            self._write_divider(depth, writer, '=', 20)
            if self.used is None:
                self._write_line('no used set', depth, writer)
            else:
                self._write_line('using:', depth, writer)
                self._write_object(self.used, depth + 1, writer)
            self._write_divider(depth, writer, '=', 20)
            if self.pred is None:
                self._write_line('no prediction', depth, writer)
            else:
                self._write_line('parsed sentences:', depth, writer)
                for sent in self.pred:
                    self._write_object(sent, depth + 1, writer)
                    self._write_divider(depth, writer, width=20)
        if self.error is not None:
            self._write_line(f'error: {self.error}', depth, writer)
        self._write_divider(depth, writer, '=', 40)


@dataclass
class _EvalDataPoint(Writable):
    gold: AmrSentence
    pred: AmrSentence
    issue: _EvalParseIssue = field(default=None)

    @property
    def has_issue(self) -> bool:
        return self.issue is not None

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              dividers: bool = True):
        self._write_line('gold:', depth, writer)
        self._write_object(self.gold, depth + 1, writer)
        if dividers:
            self._write_divider(depth, writer, width=40)
        if self.has_issue:
            self._write_line('issue:', depth, writer)
            self._write_object(self.issue, depth + 1, writer)
        else:
            self._write_line('prediction:', depth, writer)
            self._write_object(self.pred, depth + 1, writer)
        if dividers:
            self._write_divider(depth, writer, width=80)


@dataclass
class AmrCorpusStash(MultiProcessStash):
    source: Union[Path, Installer]
    doc_parser: FeatureDocumentParser
    limit: int = field(default=sys.maxsize)

    def prime(self):
        self.source.install()
        super().prime()

    def _create_data(self) -> Iterable[Any]:
        doc = AmrDocument.from_source(self.source)
        return map(lambda x: (str(x[0]), x[1]),
                   it.islice(zip(range(self.limit), doc), self.limit))

    def _parse(self, index: int, gold: AmrSentence):
        pred_doc: Doc = self.doc_parser.parse_spacy_doc(gold.text)
        pred_amr_doc: AmrDocument = pred_doc._.amr
        issue: _EvalParseIssue = None
        pred_sent: AmrSentence = pred_amr_doc[0]
        if len(pred_amr_doc) != 1:
            pred_sents: List[AmrSentence] = list(pred_amr_doc)
            pred_sents.sort(key=lambda s: len(s.graph.triples), reverse=True)
            if len(pred_sents) > 0:
                pred_sent = pred_sents[0]
            issue = _EvalParseIssue(index, gold, pred_amr_doc, pred_sent)
            issue.write_to_log(logger, logging.WARNING)
        return pred_sent, issue

    def _process(self, chunk: List[Any]) -> Iterable[Tuple[str, Any]]:
        gold: AmrSentence
        for k, gold in chunk:
            pred_sent, issue = None, None
            try:
                pred_sent, issue = self._parse(k, gold)
            except Exception as e:
                issue = _EvalParseIssue(k, gold, None, None, e)
                issue.write_to_log(logger, logging.WARNING)
            yield (k, _EvalDataPoint(gold, pred_sent, issue))

    def get_issue_values(self) -> Iterable[_EvalParseIssue]:
        return filter(lambda dp: dp.has_issue, self.values())

    def get_parsed_values(self) -> Iterable[_EvalParseIssue]:
        return filter(lambda dp: not dp.has_issue, self.values())


@dataclass
class Evaluator(Writable):
    config: Configurable
    doc_parser: FeatureDocumentParser
    vanilla_source: Union[Path, Installer]
    corpus_stash: Stash
    amr_parser: AmrParser
    model_name: str
    temporary_dir: Path

    def __post_init__(self):
        self._source = PersistedWork(
            self.temporary_dir / 'source.dat', self, mkdir=True)

    @property
    @persisted('_source')
    def source(self) -> AmrDocument:
        return AmrDocument.from_source(self.vanilla_source)

    def _get_issues(self) -> Iterable[_EvalParseIssue]:
        return filter(lambda dp: dp.has_issue, self.corpus_stash.values())

    def _get_paths(self) -> Dict[str, Path]:
        model_path: Path = self.amr_parser.model_path
        source_path: Path = AmrDocument.resolve_source(self.vanilla_source)
        parsed_doc_path: Path = self.corpus_stash.delegate.delegate.path
        return {'model path': model_path,
                'source path': source_path,
                'parsed document path': parsed_doc_path}

    def _write_paths(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_dict(self._get_paths(), depth, writer)

    def write_stats(self, depth: int = 0, writer: TextIOBase = sys.stdout,
                    limit: int = sys.maxsize, write_issue: bool = False):
        dps = 0
        issues = 0
        total_docs = len(self.source)
        for dp in it.islice(self.corpus_stash.values(), limit):
            if dp.has_issue:
                if write_issue:
                    self._write_object(dp, depth, writer)
                issues += 1
            dps += 1
        self._write_line(f'found {issues} issues parsed from {dps} ' +
                         f'sentences of a total of {total_docs}',
                         depth, writer)
        self._write_line('file locations:', depth, writer)
        self._write_paths(depth + 1, writer)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self.write_stats(depth, writer)

    def _calc_smatch(self):
        gold = []
        preds = []
        if logger.isEnabledFor(logging.INFO):
            sio = StringIO()
            self._write_paths(writer=sio)
            for ln in sio.getvalue().rstrip().split('\n'):
                logger.info(ln)
            logger.info(
                f'calculating smatch on {len(self.corpus_stash)} graphs...')
        for dp in self.corpus_stash.values():
            if not dp.has_issue:
                gold.append(dp.gold.graph_string_no_comments)
                preds.append(dp.pred.graph_string_no_comments)
        if logger.isEnabledFor(logging.INFO):
            path: Path = AmrDocument.resolve_source(self.vanilla_source)
            logger.info(f'evaluated model {self.model_name} at {path}')
        with time(f'evaluated {len(preds)}/{len(gold)} data points'):
            print(compute_smatch(preds, gold))

    def _test_parse(self):
        sent = 'addition of AG1478 also suppressed ATP-mediated phosphorylation of ERK1/2 and I-?B? (Fig. 4A), indicating the involvement of EGFR activation in these signaling events.'
        pred_doc: Doc = self.doc_parser.parse_spacy_doc(sent)
        pred_amr_doc: AmrDocument = pred_doc._.amr
        for i, sent in enumerate(pred_amr_doc.sents):
            print(i, sent)

    def _tmp(self):
        self._test_parse()
        #self.write_stats(limit=2, write_issue=True)

    def proto(self):
        """Prototype method."""
        {0: self._tmp,
         1: self.write,
         2: self._calc_smatch,
         3: lambda: self.write_stats()
         }[0]()

    def __call__(self):
        self._calc_smatch()
