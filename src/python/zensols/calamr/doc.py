"""Document based graph container, factory and strategy classes.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Dict, Tuple, Any, List, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import sys
import warnings as warn
from functools import reduce
from io import TextIOBase
from frozendict import frozendict
import igraph as ig
from igraph import Graph, Vertex, Edge
from zensols.util import time
from zensols.config import ConfigFactory
from zensols.persist import persisted
from zensols.amr import Relation, RelationSet, AmrFeatureDocument
from . import (
    ComponentAlignmentError, GraphAttributeContext, GraphComponent,
    DocumentGraphComponent, GraphNode
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentGraph(GraphComponent):
    """A graph containing the text, text features, AMR Penman graph and igraph.

    """
    config_factory: ConfigFactory = field()
    """The factory used to create the :class:`.GraphAlignmentConstructor`."""

    graph_attrib_context: GraphAttributeContext = field()
    """The context given to all nodees and edges of the graph."""

    doc: AmrFeatureDocument = field()
    """The document that represents the graph."""

    components: Tuple[DocumentGraphComponent] = field()
    """The roots of the trees created by the :class:`.DocumentGraphFactory`.

    """
    original_graph: DocumentGraph = field(default=None)
    """If set, the initial disconnected bipartite component graphs."""

    def __post_init__(self):
        super().__post_init__()
        self._added_vertexes: List[Vertex] = []
        self._added_edges: List[Edge] = []
        self._build()

    def get_containing_component(self, n: GraphNode) -> DocumentGraphComponent:
        """Return the component that conains ``v``."""
        comp: DocumentGraphComponent
        for comp in self.components:
            v: Vertex = comp.node_to_vertex.get(n)
            if v is not None and v in comp.vs:
                return comp

    def _build(self):
        # get each components induced graph
        graphs: Tuple[Graph] = tuple(map(lambda c: c.graph, self.components))
        # union the document graphs in to one igraph instance as disconnected
        # components
        graph: Graph = ig.disjoint_union(graphs)
        # set the component graphs to the union graph
        comp: DocumentGraphComponent
        for comp in self.components:
            comp.graph = graph
        # set our own container graph to the union graph
        self.graph = graph

    def clone(self, reverse_edges: bool = False, **kwargs) -> GraphComponent:
        params = dict(map(lambda a: (a, getattr(self, a)),
                          'config_factory graph_attrib_context doc'.split()))
        params['components'] = tuple(
            map(lambda c: c.clone(reverse_edges), self.components))
        params.update(kwargs)
        params['graph'] = self.graph_instance()
        return super().clone(**params)

    @property
    @persisted('_components_by_name')
    def components_by_name(self) -> Dict[str, DocumentGraphComponent]:
        """Get document graph components by name."""
        return frozendict({c.name: c for c in self.components})

    def _get_stats(self) -> Dict[str, Any]:
        vl = len(self.vs)
        el = len(self.es)
        comp_stats = tuple(map(lambda c: c.stats, self.components))
        vs = reduce(lambda x, y: x + y,
                    map(lambda s: s['vertexes'], comp_stats))
        es = reduce(lambda x, y: x + y,
                    map(lambda s: s['edges'], comp_stats))
        align_diffs: Dict[str, int] = {'vertexes': vl - vs, 'edges': el - es}
        diffs: Dict[str, int] = dict(align_diffs)
        diffs['vertexes'] -= len(self._added_vertexes)
        diffs['edges'] -= len(self._added_edges)
        return {'union': {'vertexes': vl, 'edges': el},
                'align_diffs': align_diffs,
                'diffs': diffs,
                'components': comp_stats}

    def _sanity_check(self, do_print: bool = False, do_assert: bool = True):
        stats: Dict[str, Dict[str, Any]] = self.stats
        diffs = stats['diffs']
        if do_print:
            from pprint import pprint
            pprint(stats)
        if do_assert:
            if not all(map(lambda x: x == 0, diffs.values())):
                raise ComponentAlignmentError(
                    f'Found extraineous vertexes and/or edges: {diffs}')
        return stats

    @property
    def bipartite_relation_set(self) -> RelationSet:
        """The bipartite relations that span components.  This set includes all
        top level relations that are not self contained in any components.

        """
        comp_rel_set: Set[Relation] = set()
        global_rel_set: Set[Relation] = self.doc.relation_set.as_set()
        for comp in self.components:
            comp_rel_set.update(comp.relation_set.as_set())
        return global_rel_set - comp_rel_set

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('doc:', depth, writer)
        comp: DocumentGraphComponent
        for comp in self.components:
            g: Graph = comp.graph
            self._write_line(f'component: (v={len(g.vs)}, e={len(g.es)})',
                             depth + 1, writer)
            self._write_object(comp, depth + 2, writer)


@dataclass
class DocumentGraphDecorator(ABC):
    """A strategy to create a graph from a document structure.

    """
    @abstractmethod
    def decorate(self, component: DocumentGraphComponent):
        """Creates the graph from a :class:`.DocumentNode` root node.

        :param component: the graph to populate from the decorateing process

        """
        pass


@dataclass
class DocumentGraphFactory(ABC):
    """Creates a document graph.  After the document portion of the graph is
    created, the igraph is built and merged using a
    :class:`.DocumentGraphDecorator`.  This igraph has the corresponding
    vertexes and edges associated with the document graph, which includes AMR
    Penman graph and feature document artifacts.

    """
    config_factory: ConfigFactory = field()
    """Used to create a :class:`.DocumentGraphDecorator`."""

    graph_decorators: Tuple[DocumentGraphDecorator] = field()
    """The name of the section that defines a :class:`.DocumentGraphDecorator`
    instance.

    """
    doc_graph_section_name: str = field()
    """The name of a section in the configuration that defines new instances of
    :class:`.DocumentGraph`.

    """
    graph_attrib_context: GraphAttributeContext = field()
    """The context given to all nodees and edges of the graph."""

    @abstractmethod
    def _create(self, root: AmrFeatureDocument) -> \
            Tuple[DocumentGraphComponent]:
        """Create a document node graph from an AMR feature document and return
        the root.

        """
        pass

    def create(self, root: AmrFeatureDocument) -> DocumentGraph:
        """Create a document graph and return it starting from the root note.
        See class docs.

        :param root: the feature document from which to create the graph

        """
        assert isinstance(root, AmrFeatureDocument)
        components: Tuple[DocumentGraphComponent] = self._create(root)
        graph: Graph = GraphComponent.graph_instance()
        decorator: DocumentGraphDecorator
        for decorator in self.graph_decorators:
            component: DocumentGraphComponent
            for component in components:
                with time(f'decorated {component.name}'):
                    decorator.decorate(component)
        doc_graph: DocumentGraph = self.config_factory.new_instance(
            self.doc_graph_section_name,
            graph=graph,
            doc=root,
            components=components)
        # evict to force recreate in the application context on next call
        self.config_factory.clear_instance(self.doc_graph_section_name)
        return doc_graph

    def __call__(self, root: AmrFeatureDocument) -> DocumentGraph:
        """See :meth:`create`."""
        return self.create(root)


# igraph complains about unreachable nodes in the disconnected components of the
# bipartite graph
warn.filterwarnings('ignore', message="Couldn't reach some vertices")
