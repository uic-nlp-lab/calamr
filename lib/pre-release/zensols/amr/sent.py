"""AMR container classes that fit a document/sentence hierarchy.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Union, Dict, Tuple, List, ClassVar, Set, Iterable, Type
import logging
import sys
from io import TextIOBase
from itertools import chain
import json
from frozendict import frozendict
import penman
from penman.graph import Graph
from penman.tree import Tree
from penman.surface import Alignment, RoleAlignment
from penman.exceptions import DecodeError
from penman.model import Model
import penman.models.amr
import penman.models.noop
from zensols.config import Writable
from zensols.persist import FileTextUtil, PersistableContainer
from amrlib.graph_processing.amr_loading import get_graph_only, split_amr_meta
from amrlib.graph_processing.wiki_remover import wiki_remove_graph
from . import AmrFailure, AmrError

logger = logging.getLogger(__name__)


class AmrSentence(PersistableContainer, Writable):
    """Contains a sentence that contains an AMR graph and a Penman string
    version of the graph.  Instances can be create with a Penman formatted
    string, an already parsed :class:`~penman.graph.Graph` or an
    :class:`.AmrFailure` for any upstream issues.

    These kinds of issues result for situations where downstream APIs expect
    instances of this class, such as in bulk processing situations.  When this
    happens, instance renders with an error message in the AMR metadata.

    """
    # only keep the penman string form of the graph and recreate in memory on
    # demand after unserializing
    _PERSITABLE_TRANSIENT_ATTRIBUTES: ClassVar[Set[str]] = {'_graph'}
    _PERSITABLE_PROPERTIES: ClassVar[Set[str]] = {'graph_string'}

    MAX_SHORT_NAME_LEN: ClassVar[int] = 30
    """Max length of `short_name` property."""

    DEFAULT_MODEL: ClassVar[str] = 'noop'
    """The default :mod:`penman` AMR model to use in the initializer, which is
    one of ``noop`` or ``amr``.  The first does not modify the graph but the
    latter normalizes out inverse relationships such as ``ARG*-of``.

    """
    def __init__(self, data: Union[str, Graph], model: str = None):
        """Initialize based on the kind of data given.

        :param data: either a Penman formatted string graph, an already parsed
                     graph or an :class:`.AmrFailure` for upstream issues

        :param model: the model to use for encoding and decoding

        """
        self._str = None
        self._graph = None
        self._failure = None
        self._model = self.DEFAULT_MODEL if model is None else model
        if data is None:
            # other API components robustly return None's that aren't caught in
            # a few edges cases
            raise AmrError(f'No data given while creating {type(self)}, ' +
                           'which probably means a bad parse')
        elif isinstance(data, str):
            # a penman formatted string
            self._str = data
        elif isinstance(data, Graph):
            # an already parsed penman in memory graph
            self._graph = data
        elif isinstance(data, AmrFailure):
            # upstream issues (see method doc)
            self._graph = Graph()
            self._graph.metadata['snt'] = data.sent
            self._graph.metadata['parse_failure'] = str(data)
            self._failure = data
        else:
            # be paranoid
            raise AmrError(f'Unknown data type: {type(data)}')
        # sanity check on model
        self._get_model()

    def _get_model(self) -> Model:
        model: Model
        if self._model == 'noop':
            model = penman.models.noop.model
        elif self._model == 'amr':
            model = penman.models.amr.model
        else:
            raise AmrError(f'Unknown model: {self._model}')
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating sent with model: {model} ({self._model})')
        return model

    @property
    def is_failure(self) -> bool:
        """Whether the AMR graph failed to be parsed."""
        return 'parse_failure' in self.metadata

    @property
    def text(self) -> str:
        """The text of the natural language form of the sentence."""
        graph: Graph = self.graph
        meta: Dict[str, str] = graph.metadata
        return meta['snt']

    @property
    def tokenized_text(self) -> str:
        """This is useful when it is necessary to force white space tokenization
        to match the already tokenized metadata ('tokens' key).  Examples
        include numbers followed by commas such as dates like ``April 25 ,
        2008``.

        """
        meta: Dict[str, str] = self.metadata
        text: str
        if 'tokens' in meta:
            toks: List[str] = json.loads(meta['tokens'])
            text = ' '.join(toks)
        else:
            text = self.text
        return text

    @property
    def graph(self) -> Graph:
        """The graph as an in memory data structure."""
        if self._graph is None:
            try:
                self._graph = penman.decode(self._str, model=self._get_model())
            except DecodeError as e:
                raise AmrError(f'Could not parse: {e}', self._str) from e
        return self._graph

    @graph.setter
    def graph(self, graph: Graph) -> Graph:
        """The graph as an in memory data structure."""
        self._graph = graph
        # TODO: set self._model
        self.invalidate_graph_string()

    @property
    def graph_string(self) -> str:
        """The graph as a string in Penman format."""
        if self._str is None:
            self._str = penman.encode(self._graph, model=self._get_model())
        return self._str

    @graph_string.setter
    def graph_string(self, graph_string: str):
        """The graph as a string in Penman format."""
        self.invalidate_graph_string(check_graph=False)
        self._str = graph_string
        self._graph = None

    @property
    def graph_only(self) -> str:
        """Like :obj:`graph_string` but without metadata"""
        if not hasattr(self, '_str_graph_only'):
            self._str_graph_only = get_graph_only(self.graph_string)
        return self._str_graph_only

    @property
    def graph_single_line(self) -> str:
        """Like :obj:`graph_only` but return as a single one line string."""
        return ' '.join(map(str.strip, self.graph_only.split('\n')))

    @property
    def tree(self) -> Tree:
        """Return a tree structure of the graph using the top node."""
        return penman.configure(self.graph, model=self._get_model())

    def normalize(self):
        """Normalize the graph string to standard notation per the Penman API.

        """
        if self._str is not None:
            self.graph
        self._str = None

    def invalidate_graph_string(self, check_graph: bool = True):
        """To be called when the graph changes that should be propagated to
        :obj:`graph_string`.

        """
        if check_graph:
            assert self.graph is not None
        self._str = None
        if hasattr(self, '_str_graph_only'):
            del self._str_graph_only

    @property
    def metadata(self) -> Dict[str, str]:
        """The graph metadata as a dict."""
        return dict(self.graph.metadata)

    @metadata.setter
    def metadata(self, metadata: Dict[str, str]):
        """The graph metadata as a dict."""
        if self._str is not None:
            # keep epi data such as alignment in the graph string without having
            # to reparse and re-add by creating the metadata as a string and
            # keeping the graph portion verbatim; we assume the client keeps
            # comment based alignments and epi data in sync
            g = Graph()
            g.metadata.update(metadata)
            meta_lines: List[str] = split_amr_meta(
                penman.encode(g, model=self._get_model()))[0]
            self._str = '\n'.join(meta_lines) + '\n' + get_graph_only(self._str)
        if self._graph is not None:
            self._graph.metadata.clear()
            self._graph.metadata.update(metadata)

    def set_metadata(self, k: str, v: str):
        """Set a metadata value on the graph."""
        self.graph.metadata[k] = v
        self._str = None

    def get_data(self) -> Union[Graph, str]:
        """Return the :obj:`graph` if it is parse, else return the
        :obj:`graph_string`.

        """
        if self._graph is not None:
            return self._graph
        return self._str

    @property
    def short_name(self) -> str:
        """The short name of the sentences, which is the first several words.

        """

        s = self.text[0:self.MAX_SHORT_NAME_LEN]
        return FileTextUtil.normalize_text(s)

    @property
    def has_alignments(self) -> bool:
        """Whether this sentence has any alignments."""
        try:
            next(self.iter_aligns())
            return True
        except StopIteration:
            return False

    def iter_aligns(self, include_types: bool = False) -> \
            Iterable[Tuple[str, str, str], Tuple[int, ...]]:
        """Return an iterator of the alignments of the graph as a tuple.  Each
        iteration is a tuple of triple, the list of alignment indexes, and a
        tuple of bools if the index is a role alignment.

        :param include_types: whether to include types, which is the third
                              element in each tuple, else that element is
                              ``None``

        """
        def filter_aligns(x) -> bool:
            return isinstance(x, (Alignment, RoleAlignment))

        epis: Dict[Tuple[str, str, str], List] = self.graph.epidata
        trip: Tuple[str, str, str]
        for trip in self.graph.triples:
            aligns: Tuple = tuple(filter(filter_aligns, epis[trip]))
            if len(aligns) > 0:
                indices: Iterable[int] = chain.from_iterable(
                    map(lambda a: a.indices, aligns))
                if include_types:
                    types: Tuple[bool, ...] = tuple(
                        map(lambda a: isinstance(a, RoleAlignment), aligns))
                    yield (trip, tuple(indices), types)
                else:
                    yield (trip, tuple(indices), None)

    def remove_wiki_attribs(self):
        """Removes the ``:wiki`` roles from the graph."""
        meta: Dict[str, str] = self.metadata
        self.graph = wiki_remove_graph(self.graph_only)
        self.metadata = meta
        self.invalidate_graph_string()

    def remove_alignments(self):
        """Remove text-to-graph alignments."""
        epis: Dict[Tuple[str, str, str], List] = self.graph.epidata
        epi: List
        for epi in epis.values():
            rms = []
            for i, x in enumerate(epi):
                if isinstance(x, (Alignment, RoleAlignment)):
                    rms.append(i)
            rms.reverse()
            for i in rms:
                del epi[i]
        metadata = self.metadata
        del metadata['alignments']
        self.metadata = metadata

    @property
    def instances(self) -> Dict[str, Tuple[str, str, str]]:
        return frozendict({i.source: tuple(i) for i in self.graph.instances()})

    def clone(self, cls: Type[AmrSentence] = None, **kwargs) -> AmrSentence:
        """Return a deep copy of this instance."""
        cls = self.__class__ if cls is None else cls
        params = dict(data=self.graph_string, model=self._model)
        params.update(kwargs)
        return cls(**params)

    def __eq__(self, other: AmrSentence) -> bool:
        return self.graph == other.graph and self.metadata == other.metadata

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              include_metadata: bool = True, include_stack: bool = False):
        """
        :param include_stack: whether to add the stack trace of the parse of an
                              error occured while trying to do so

        :param include_metadata: whether to add graph metadata to the output
        """
        if self.is_failure:
            parse_failure = self.metadata['parse_failure']
            self._write_line(f'error: {parse_failure}', depth + 2, writer)
            if include_stack and \
               self._failure is not None and \
               self._failure.stack is not None:
                writer.write(self._failure.stack)
        else:
            graph = self.graph_string if include_metadata else self.graph_only
            self._write_block(graph, depth, writer)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.__str__()
