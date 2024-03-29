"""Classes that organize document in content in to a hierarchy.

**Terminology**: *token alignments* refer to the sentence index based token
alignments to AMR nodes.  This is not to be confused with alignment edges (aka
graph component alignment edges).

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, ClassVar, Optional, List
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import sys
import logging
import re
from io import TextIOBase
import torch
from torch import Tensor
from zensols.config import Dictable
from zensols.persist import (
    persisted, PersistableContainer, PersistedWork, Stash
)
from zensols.nlp import (
    FeatureToken, FeatureSentence, FeatureDocument, FeatureDocumentParser
)
from zensols.deeplearn import TorchConfig
from zensols.deepnlp.transformer import (
    WordPieceFeatureDocument, WordPieceFeatureDocumentFactory
)
from zensols.amr import AmrError
from zensols.propbankdb import Database, RolesetId, Roleset, Relation

logger = logging.getLogger(__name__)


class ComponentAlignmentError(AmrError):
    pass


@dataclass(init=False, repr=False)
class Role(Dictable):
    """Represents an AMR role, which is a label on the edge an an AMR graph such
    as ``:ARG0-of``.

    """
    label: str = field()
    """The surface name of the role (i.e. `:ARG0-of``)."""

    prefix: str = field()
    """The prefix of the role (i.e. ``ARG`` in ``:ARG0-of``)."""

    is_inverted: bool = field()
    """True if the role is inverted (i.e. has ``of`` in ``:ARG0-of``)."""

    index: Optional[int] = field()
    """The prefix of the role (i.e. ``ARG`` in ``:ARG0-of``)."""

    relation: Optional[Relation] = field()
    """The relation metadata, which has the same label as this role."""

    def __init__(self, label: str):
        m: re.Match = Relation.REGEX.match(label)
        if label[0] != ':':
            raise ComponentAlignmentError(
                f"Expecting role to start with colon, but got: '{label}'")
        self.label = label
        if m is None:
            self.prefix, self.index, self.is_inverted = label[1:], None, False
        else:
            self.prefix, self.index, self.is_inverted = m.groups()
            if self.index is not None:
                self.index = int(self.index)
            self.is_inverted = self.is_inverted is not None
        self.relation = None

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return self.label


@dataclass
class EmbeddingResource(object):
    """Generates embeddings for roles, role sets, text, and feature tokens.

    """
    torch_config: TorchConfig = field()
    """Used to create :obj:`unknown_edge_embedding`"""

    word_piece_doc_factory: FeatureDocumentParser = field(default=None)
    """Used to get single token embeddings for nodes with no token alignments.

    """
    word_piece_document_factory: WordPieceFeatureDocumentFactory = field(
        default=None)
    """Creates word piece data structures that have embeddings."""

    propbank_db: Database = field(default=None)
    """The database used to get propbank and other bank artifcats."""

    @property
    @persisted('_unknown_node_embedding', transient=True)
    def unknown_node_embedding(self) -> Tensor:
        """A zero embedding."""
        emb: Tensor = self.get_token_embedding('[SEP]')
        emb = emb.clone().detach()
        emb.zero_()
        return emb

    @property
    @persisted('_unknown_edge_embedding', transient=True)
    def unknown_edge_embedding(self) -> Tensor:
        """A zero embedding."""
        role_emb_shape: Tuple[int] = self._get_role_vectorizer().shape
        return self.torch_config.ones((role_emb_shape[-1],))

    @property
    @persisted('_roleset_stash', cache_global=True)
    def roleset_stash(self) -> Stash:
        """A stash with :class:`~zensols.propbankdb.domain.RolesetId` as keys and
        :class:`~zensols.propbankdb.domain.Roleset` as values.

        """
        return self.propbank_db.roleset_stash

    def get_word_piece_document(self, text: str) -> WordPieceFeatureDocument:
        """Return a word piece document parsed from ``text``."""
        doc: FeatureDocument = self.word_piece_doc_factory(text)
        return self.word_piece_document_factory(doc)

    def get_token_embedding(self, text: str) -> Tensor:
        """Return the mean of the token embeddings of ``text``."""
        wp_doc: WordPieceFeatureDocument = self.get_word_piece_document(text)
        if str(wp_doc).find('[UNK]') > -1:
            logger.warning(f'oov: {text} -> {wp_doc}')
        embs: Tensor = tuple(map(lambda t: t.embedding, wp_doc.token_iter()))
        return torch.concat(embs, dim=0).mean(dim=0)

    def get_tokens_embedding(self, tokens: Tuple[FeatureToken]) -> Tensor:
        """Return the mean of the embeddings of ``tokens``."""
        embs: Tensor = tuple(map(lambda t: t.embedding, tokens))
        return torch.concat(embs, dim=0).mean(dim=0)

    def get_sentence_tokens_embedding(self, sent: FeatureSentence) -> Tensor:
        """Return the sentence embeddings of ``sent``."""
        return sent.embedding

    def get_role_embedding(self, role: Role) -> Tensor:
        """Return an embedding for a role.  This uses the role's relation's
        embedding if available.  Otherwise, it uses the embedding created fromi
        the role's prefix.

        """
        emb: Tensor
        if role.relation is None:
            logger.warning(f'role {role} has no relation')
            emb = self.get_token_embedding(role.prefix)
        else:
            emb = role.relation.embedding
        return emb

    def get_roleset_embedding(self, roleset_id: Optional[RolesetId],
                              roleset: Roleset) -> Tensor:
        """Return the embedding for ``roleset``.

        :param roleset_id: the role's ID, which is used when ``roleset`` is
                            None

        :param roleset: the role set to use for the embedding if available

        """
        emb: Tensor = None
        if roleset is not None:
            emb = roleset.embedding
        else:
            # for nodes with no token alignment, use the propbank verb
            rsid: RolesetId = roleset_id
            tok = rsid.lemma if rsid.lemma is not None else rsid.label
            emb = self.get_token_embedding(tok)
        return emb


@dataclass
class GraphAttributeContext(Dictable):
    """Contains context data used by nodes and edges of the graph.

    """
    # components
    embedding_resource: EmbeddingResource = field(repr=False)
    """The manager that contains vectorizers that create node and edge
    embeddings.

    """
    propbank_db: Database = field(repr=False)
    """The database used to get propbank and other bank artifacts."""

    # hyperparams
    default_capacity: float = field()
    """The default initial capacity for inter-AMR edges."""

    sink_capacity: float = field()
    """The value to use as the sink terminal node."""

    component_alignment_capacity: float = field()
    """The default initial capacity for source/summary component alignment
    edges.

    """
    doc_capacity: float = field()
    """The bipartitie (between source and summary) capacity value of
    :class:`.DocumentGraphNode`.

    """
    similarity_threshold: float = field()
    """The (range [0, 1]) necessary to allow component alignment edges from the
    source to summary graph..

    """
    default_format_strlen: int = field()
    """The default capacity and flow string formatting label length."""

    # since the module can be reloaded, and thus class redefined, keep a counter
    # data structure that lasts past the instance (and class definition's) life
    @persisted('_graph_attrib_id', cache_global=True)
    def _graph_attrib_id_container(self) -> List[int]:
        return [0]

    def _config_graph_attribute(self, attrib: GraphAttribute):
        cont: List[int] = self._graph_attrib_id_container()
        next_id: int = cont[0] + 1
        attrib._graph_attrib_id = next_id
        cont[0] = next_id

    def to_role(self, role_str: str) -> Role:
        role = Role(role_str)
        match_rel: Relation = None
        rel: Relation
        for rel in self.propbank_db.relation_stash.values():
            if rel.match(role_str):
                match_rel = rel
                break
        if match_rel is None:
            logger.warning(f'missing relation for {role_str}')
        else:
            role.relation = match_rel
        return role


@dataclass(eq=False, repr=False)
class GraphAttribute(PersistableContainer, Dictable, metaclass=ABCMeta):
    """Contains AMR document attribute data added to the :class:`igraph.Graph`.
    This is added as vertexes or edge attribute data.

    """
    _DICTABLE_WRITE_EXCLUDES: ClassVar[str] = {'description'}
    _DICTABLE_ATTRIBUTES: ClassVar[str] = {'description'}

    ATTRIB_TYPE: ClassVar[str] = 'base'
    """The attribute type this class represents."""

    context: GraphAttributeContext = field(repr=False)
    """Contains context data used by nodes and edges of the graph."""

    def __post_init__(self):
        super().__init__()
        # self._graph_attrib_id set by context
        self.context._config_graph_attribute(self)
        self._description = PersistedWork('_description', self)

    @abstractmethod
    def _get_embedding(self) -> Tensor:
        pass

    @abstractmethod
    def _get_description(self) -> str:
        pass

    @property
    @persisted('_description')
    def description(self) -> str:
        """A human readable description that is usually used as the label and
        :meth:`__str__`.

        """
        return self._get_description()

    @property
    def label(self) -> str:
        """Text used when rendering graphs."""
        return self.description

    @property
    def id(self) -> int:
        """The unqiue identifier for this graph attribute."""
        return self._graph_attrib_id

    @property
    def embedding_resource(self) -> EmbeddingResource:
        """Generates embeddings for roles, role sets, text, and feature tokens.

        """
        return self.context.embedding_resource

    @property
    def attrib_type(self) -> str:
        """The attribute type this class represents."""
        return self.ATTRIB_TYPE

    @property
    @persisted('_embedding', transient=True)
    def embedding(self) -> Tensor:
        """The default embedding of the attribute.  Note that some attributes
        have several different embeddings.

        """
        return self._get_embedding()

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(self.__str__(), depth, writer)
        super().write(depth + 1, writer)

    def __hash__(self) -> int:
        return hash(self._graph_attrib_id)

    def __eq__(self, other: GraphAttribute) -> bool:
        return self._graph_attrib_id == other._graph_attrib_id

    def __str__(self) -> str:
        return self.description

    def __repr__(self) -> str:
        return self.__str__()
