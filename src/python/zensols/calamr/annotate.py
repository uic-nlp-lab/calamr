"""Contain a class to add embeddings to AMR feature documents.

"""
__author__ = 'Paul Landes'

from typing import Dict, Tuple
from dataclasses import dataclass, field
import logging
import re
from pathlib import Path
from frozendict import frozendict
from zensols.util import time
from zensols.persist import persisted, PersistedWork, Primeable
from zensols.amr import (
    AmrError, AmrSentence, AmrDocument, AmrFeatureDocument,
    AnnotatedAmrFeatureDocumentStash, AnnotatedAmrDocumentStash,
    AnnotatedAmrFeatureDocumentFactory,
)
from zensols.dataset import SplitKeyContainer
from zensols.deepnlp.transformer import (
    WordPieceFeatureDocumentFactory, WordPieceFeatureDocument
)

logger = logging.getLogger(__name__)


@dataclass
class AddEmbeddingsFeatureDocumentStash(AnnotatedAmrFeatureDocumentStash):
    """Add embeddings to AMR feature documents.  Embedding population is
    disabled by configuring :obj:`word_piece_doc_factory` as ``None``.

    """
    word_piece_doc_factory: WordPieceFeatureDocumentFactory = field(
        default=None)
    """The feature document factory that populates embeddings."""

    def load(self, doc_id: str) -> AmrFeatureDocument:
        doc: AmrFeatureDocument = super().load(doc_id)
        if self.word_piece_doc_factory is not None:
            with time(f'populated embedding of document {doc_id}'):
                wpdoc: WordPieceFeatureDocument = \
                    self.word_piece_doc_factory(doc)
            wpdoc.copy_embedding(doc)
        return doc


@dataclass
class CorpusSplitKeyContainer(SplitKeyContainer, Primeable):
    """Gets the splits of an AMR corpus release.

    """
    anon_doc_stash: AnnotatedAmrDocumentStash = field()
    """A stash used to get :class:`.AnnotatedAmrDocument` keys from the corpus.

    """
    corpus_path: Path = field()
    """The relative path to the files."""

    corpus_file_glob: str = field()
    """The glob pattern to find the corpus files (i.e. proxy)."""

    cache_file: Path = field()
    """File to cache results."""

    def __post_init__(self):
        self._splits = PersistedWork(self.cache_file, self, mkdir=True)

    @property
    def corpus_absolute_path(self) -> Path:
        inst_path: Path = self.anon_doc_stash.installer.get_singleton_path()
        path: Path = (inst_path / self.corpus_path).resolve()
        return path

    @persisted('_splits')
    def _get_keys_by_split(self) -> Dict[str, Tuple[str]]:
        """The splits as a dict with keys as the split name and the sentence IDs
        as the values.

        """
        def map_key(sent: AmrSentence) -> str:
            m: re.Match = stash.id_regexp.match(sent.metadata['id'])
            return m.group(1)

        stash: AnnotatedAmrDocumentStash = self.anon_doc_stash
        splits: Dict[str, Tuple[str]] = {}
        path: Path = self.corpus_absolute_path
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'parsing {path}')
        if not path.is_dir():
            raise AmrError(f'Corpus path does not exist: {path}')
        for corp_file in path.glob(self.corpus_file_glob):
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'parsing file {corp_file}')
            split_name: str = corp_file.parent.name
            doc: AmrDocument = AmrDocument.from_source(corp_file)
            splits[split_name] = tuple(set(map(map_key, doc)))
        return frozendict(splits)

    def prime(self):
        self.anon_doc_stash.installer()

    def clear(self):
        self._splits.clear()


@dataclass
class CalamrAnnotatedAmrFeatureDocumentFactory(
        AnnotatedAmrFeatureDocumentFactory):
    """Adds wordpiece embeddings to
    :class:`~zensols.amr.container.AmrFeatureDocument` instances.

    """
    word_piece_doc_factory: WordPieceFeatureDocumentFactory = field(
        default=None)
    """The feature document factory that populates embeddings."""

    def _populate_embeddings(self, doc: AmrFeatureDocument):
        """Adds the transformer sentinel embeddings to the document."""
        if self.word_piece_doc_factory is not None:
            wpdoc: WordPieceFeatureDocument = self.word_piece_doc_factory(doc)
            wpdoc.copy_embedding(doc)

    def from_dict(self, data: Dict[str, str],
                  doc_id: str = None) -> AmrFeatureDocument:
        fdoc = super().from_dict(data, doc_id)
        if self.word_piece_doc_factory is not None:
            self._populate_embeddings(fdoc)
        return fdoc
