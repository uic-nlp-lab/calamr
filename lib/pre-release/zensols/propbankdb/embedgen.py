"""Creates embeddings for PropBank objects.  This module is used to create the
distribution file's embeddings with :class:`.EmbeddingGenerator`.  The
distribution file's contents include the Framenet object graphs, embeddings and
a metadata files used to configure :obj:`.EmbeddingPopulator.embed_model`, which
is used by the package to populate the embeddings.

"""
__author__ = 'Paul Landes'

from typing import Iterable, Tuple, ClassVar
from dataclasses import dataclass, field
import sys
import logging
import re
import csv
from pathlib import Path
import itertools as it
from itertools import chain
from io import TextIOBase
import torch
from torch import Tensor
from zensols.util import time
from zensols.config import Configurable, IniConfig
from zensols.persist import Stash, persisted, FileTextUtil
from zensols.nlp import FeatureDocument, FeatureDocumentParser
from zensols.deepnlp.embed import WordEmbedModel
from zensols.deepnlp.transformer import TransformerEmbedding
from zensols.deepnlp.transformer import (
    TransformerResource,
    WordPieceFeatureDocument, WordPieceFeatureDocumentFactory,
)
from . import Function, Relation, Roleset, Role, Database
from .embedpop import EmbeddingPopulator

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingGenerator(object):
    """Creates sentence embeddings for PropBank objects (see module docs).

    """
    _NORM_TEXT_REGEX: ClassVar[re.Pattern] = re.compile(
        r'\barg\s*\d*\b', flags=re.IGNORECASE)

    config: Configurable = field()
    """Used to copy application context in to the metadata file included in the
    distribution file.

    """
    doc_parser: FeatureDocumentParser = field()
    """The used for parsing the English in the Framenets."""

    wordpiece_doc_factory: WordPieceFeatureDocumentFactory = field()
    """Used to create the embeddings from the language in the Framenets."""

    populator: EmbeddingPopulator = field()
    """The embedding populator used to create the embedding keys."""

    db: Database = field()
    """Used to get the objects for which to create embeddings."""

    output_dir: Path = field()
    """The directory where the embedding file is written."""

    output_decimals: int = field()
    """The number of decimals of each component in the output embedding."""

    output_limit: int = field(default=sys.maxsize)
    """The maximum number of vocab vectors to write."""

    def _embed(self, text: str) -> Tuple[FeatureDocument, Tensor]:
        doc: FeatureDocument = self.doc_parser(text)
        wpdoc: WordPieceFeatureDocument = self.wordpiece_doc_factory(doc)
        embs: Tensor = wpdoc.embedding
        emb: Tensor = embs.mean(dim=0)
        return wpdoc, emb

    @property
    @persisted('_vec_dim')
    def vector_dimension(self) -> int:
        """Get the output dimension embedding size."""
        return self._embed('the')[1].shape[0]

    def _norm_text(self, s: str) -> str:
        """Normalize sentences by remove all ``argN`` type words, trimming
        front/back space, and normalizing token spacing.

        """
        s = self._NORM_TEXT_REGEX.sub('', s).strip()
        s = ' '.join(s.split()).lower()
        if len(s) > 0:
            return s

    def _get_function_sentences(self) -> Iterable[Tuple[str, str]]:
        func: Function
        for func in self.db.function_persister.get():
            if func.has_description:
                yield(self.populator.function_to_key(func), func.description)

    def _get_relation_sentences(self) -> Iterable[Tuple[str, str]]:
        rel: Relation
        for rel in self.db.relation_persister.get():
            yield(self.populator.relation_to_key(rel), rel.description)

    def _get_roleset_sentences(self) -> Iterable[Tuple[str, str]]:
        """Create the key/sentence pairs."""
        rs_stash: Stash = self.db.roleset_stash
        rs: Roleset
        for rs in rs_stash.values():
            yield (self.populator.roleset_to_key(rs), rs.name)
            role: Role
            for role in rs.roles:
                yield (self.populator.role_to_key(rs, role), role.description)

    def _get_sentences(self) -> Iterable[Tuple[str, str]]:
        return chain.from_iterable(
            [self._get_function_sentences(),
             self._get_relation_sentences(),
             self._get_roleset_sentences()])

    def write_vectors(self, emb_writer: TextIOBase,
                      sent_writer) -> int:
        """Write the vectors to a text sink."""
        n_lines: int = 0
        for key, sent in it.islice(self._get_sentences(), self.output_limit):
            norm: str = self._norm_text(sent)
            if norm is not None:
                doc, emb = self._embed(norm)
                emb = torch.round(emb.type(torch.DoubleTensor),
                                  decimals=self.output_decimals)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{key}: {emb.shape}')
                emb_writer.write(f'{key} ')
                emb_writer.write(' '.join(map(str, emb.tolist())))
                emb_writer.write('\n')
                sent_writer.writerow((key, norm))
                n_lines += 1
        return n_lines

    def _get_config(self, name: str, desc: str, dimension: int) -> IniConfig:
        emb: WordEmbedModel = self.populator.embed_model
        emb.name = name
        emb.desc = desc
        emb.dimension = dimension
        def_sec: str = self.populator.DEFAULT_SECTION
        ver: str = self.config.get_option('version', def_sec)
        config = IniConfig(default_section=self.populator.EMBEDDING_SECTION)
        for attr in 'name desc dimension file_name'.split():
            config.set_option(attr, getattr(emb, attr))
        self.config.copy_sections(config, {'package'})
        config.set_option('version', ver, section=def_sec)
        return config

    def dump(self):
        """Write the embeddings in text format to disk."""
        emb: TransformerEmbedding = self.wordpiece_doc_factory.embed_model
        res: TransformerResource = emb.resource
        model_id: str = res.model_id
        dim: int = self.vector_dimension
        model_id_norm: str = FileTextUtil.normalize_text(model_id)
        config: IniConfig = self._get_config('propbankdb', model_id_norm, dim)
        fname: str = config.get_option(self.populator.FILE_NAME_OPTION)
        emb_output_path: Path = self.output_dir / fname
        ini_output_path: Path = self.output_dir / self.populator.META_FILE
        sent_output_file: Path = self.output_dir / self.populator.SENT_TEXT_FILE
        n_sents: int
        with time('wrote {n_sents} sentences'):
            with open(sent_output_file, 'w') as sf:
                sent_writer = csv.writer(sf)
                sent_writer.writerow('key sent'.split())
                with open(emb_output_path, 'w') as ef:
                    n_sents = self.write_vectors(ef, sent_writer)
        logger.info(f'wrote: {sent_output_file}')
        logger.info(f'wrote: {emb_output_path}')
        config.set_option('vocab_size', n_sents)
        with open(ini_output_path, 'w') as f:
            config.parser.write(f)
        logger.info(f'wrote: {ini_output_path}')

    def _test_function_outupt(self):
        func: Function
        for func in self.db.function_persister.get():
            key = self.populator.function_to_key(func)
            doc: FeatureDocument = self.doc_parser(func.description)
            wpdoc: WordPieceFeatureDocument = self.wordpiece_doc_factory(doc)
            print(key, func.description, func.has_description, '||',
                  wpdoc[0], wpdoc.unknown_count)
            if wpdoc.unknown_count > 0:
                wpdoc.write()

    def _test_relation_output(self):
        rel: Relation
        for rel in self.db.relation_persister.get():
            key = self.populator.relation_to_key(rel)
            doc: FeatureDocument = self.doc_parser(rel.description)
            wpdoc: WordPieceFeatureDocument = self.wordpiece_doc_factory(doc)
            print(key, rel.description, '||', wpdoc[0], wpdoc.unknown_count)
            if wpdoc.unknown_count > 0:
                wpdoc.write()
