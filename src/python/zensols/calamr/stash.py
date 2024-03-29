"""Alignment dataframe stash.

"""
__author__ = 'Paul Landes'

from typing import Iterable
from dataclasses import dataclass, field
import logging
import itertools as it
import pandas as pd
from zensols.persist import Stash, ReadOnlyStash, PrimeableStash
from zensols.amr import AmrFeatureDocument
from . import (
    DocumentGraph, DocumentGraphFactory, DocumentGraphAligner, FlowDocumentGraph
)

logger = logging.getLogger(__name__)


@dataclass
class AlignmentDataFrameFactoryStash(ReadOnlyStash, PrimeableStash):
    anon_doc_stash: Stash = field()
    """Contains human annotated AMRs from a small toy corpus or from the AMR 3.0
    Proxy Report corpus.

    """
    doc_graph_aligner: DocumentGraphAligner = field()
    """Create document graphs."""

    doc_graph_factory: DocumentGraphFactory = field()
    """Create document graphs."""

    def load(self, name: str) -> pd.DataFrame:
        logger.info(f'creating alignment {name}')
        doc: AmrFeatureDocument = self.anon_doc_stash[name]
        doc_graph: DocumentGraph = self.doc_graph_factory(doc)
        prev_render_level: int = self.doc_graph_aligner.render_level
        self.doc_graph_aligner.render_level = 0
        try:
            fdg: FlowDocumentGraph = self.doc_graph_aligner.align(doc_graph)
            return fdg.create_align_df(True)
        finally:
            self.doc_graph_aligner.render_level = prev_render_level

    def keys(self) -> Iterable[str]:
        return it.islice(self.anon_doc_stash.keys(), 24)

    def exists(self, name: str) -> bool:
        return name in self.keys()
        #return self.anon_doc_stash.exists(name)

    def prime(self):
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'priming {type(self)}...')
        self.anon_doc_stash.prime()
