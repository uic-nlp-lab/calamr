"""Document based graph container, factory and strategy classes.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    ClassVar, List, Iterable, Dict, Tuple, Optional, Union, Any, Sequence
)
from dataclasses import dataclass, field
import logging
import sys
import itertools as it
from io import TextIOBase
from frozendict import frozendict
import igraph as ig
from igraph import Graph, Vertex, Edge
from zensols.config import Writable
from zensols.persist import persisted, PersistedWork
from zensols.amr import RelationSet, AmrFeatureSentence, AmrFeatureDocument
from . import (
    GraphAttribute, GraphNode, GraphEdge, DocumentNode, DocumentGraphNode,
    SentenceGraphNode, ConceptGraphNode, AttributeGraphNode
)

logger = logging.getLogger(__name__)


@dataclass
class GraphComponent(Writable):
    """A container class for an :class:`igraph.Graph`, which also has caching
    data structures for fast access to graph attributes.

    """
    ROOT_ATTRIB_NAME: ClassVar[str] = 'root'
    """The attribute that identifies the root vertex."""

    GRAPH_ATTRIB_NAME: ClassVar[str] = 'ga'
    """The name of the graph attributes on igraph nodes and edges."""

    graph: Graph = field()
    """The graph used for computational manipulation of the synthesized AMR
    sentences.

    """
    def __post_init__(self):
        # sets of Vertex and Edge instances
        self._vs = PersistedWork('_vs', self)
        self._es = PersistedWork('_es', self)
        # dict of GraphNode to iGraph Vertex instances
        self._ntov = PersistedWork('_ntov', self)
        # adjacency list of vectors as list of list of vertex ints
        self._adjacency_list = PersistedWork('_adjacency_list', self)
        self._edges_reversed = False

    @staticmethod
    def graph_instance() -> Graph:
        """Create a new directory nascent graph."""
        return Graph(directed=True)

    @property
    def _graph(self) -> Graph:
        """The :mod:`igraph` graph."""
        return self._igraph

    @_graph.setter
    def _graph(self, graph: Graph):
        """The :mod:`igraph` graph."""
        self._set_graph(graph)

    def _set_graph(self, graph: Graph):
        self._igraph = graph
        self._invalidate()

    def _invalidate(self, reset_subsets: bool = True):
        """Clear cached data structures to force them to be recreated after
        igraph level data has changed.  Graph and edges indexes are reset and
        taken from the current graph.

        """
        if hasattr(self, '_vs'):
            for p in (self._vs, self._es, self._ntov, self._adjacency_list):
                p.clear()
        if reset_subsets:
            self._v_sub = set(self._igraph.vs.indices)
            self._e_sub = set(self._igraph.es.indices)

    def invalidate(self):
        """Clear cached data structures to force them to be recreated after
        igraph level data has changed.

        """
        self._invalidate()

    @staticmethod
    def _resolve_root(org_graph: Graph, org_root: Vertex,
                      new_graph: Graph) -> Tuple[GraphNode, Vertex]:
        ga: str = GraphComponent.GRAPH_ATTRIB_NAME
        # the old root graph node is given by the currently set root vertex
        old_root_doc: GraphNode = org_graph.vs[org_root.index][ga]
        # find the root vertex in the new graph by finding the root graph node
        # since all graph nodes/edges are shared across all graphs
        new_root: Vertex = next(iter(filter(
            lambda v: id(v[ga]) == id(old_root_doc), new_graph.vs)))
        return old_root_doc, new_root

    @staticmethod
    def _reverse_edges(g: Graph, root: Vertex = None) -> Graph:
        """Reverse the direction on all edges in the graph."""
        ga: str = GraphComponent.GRAPH_ATTRIB_NAME
        ra: str = GraphComponent.ROOT_ATTRIB_NAME
        # vertices that make up a component having the root as a vertex
        subgraph_vs: List[int]
        if root is None:
            subgraph_vs = g.vs.indices
        else:
            subgraph_vs = sorted(g.subcomponent(root))
        # edges in the component aka subgraph
        es: Tuple[Edge] = tuple(g.es.select(_incident_in=subgraph_vs))
        # graph edge instances to be set on the new graph
        edge_ctx: List[GraphEdge] = g.es.select(_incident_in=subgraph_vs)[ga]
        # reversed edges
        el: Iterable[Tuple[int, int]] = map(lambda e: (e.target, e.source), es)
        # create the new graph and set the collected data on it
        ng = ig.Graph(el, directed=True)
        ng.es[ga] = edge_ctx
        ng.vs[ga] = g.vs[subgraph_vs][ga]
        ng.vs[ra] = g.vs[subgraph_vs][ra]
        return ng

    def copy_graph(self, reverse_edges: bool = False) -> Graph:
        """Return a copy of the the graph.

        :param reverse_edges: whether to reverse the direction on all edges in
                              the graph

        """
        g: Graph = self.graph
        # vertices that make up a component having the root as a vertex
        copy: Graph = g.subgraph(g.subcomponent(self.root))
        if reverse_edges:
            # find the vertex in the new graph using the old graph and root
            old_root_doc: GraphNode
            new_root: Vertex
            old_root_doc, new_root = self._resolve_root(g, self.root, copy)
            copy = self._reverse_edges(copy, new_root)
        return copy

    def clone(self, reverse_edges: bool = False, **kwargs) -> GraphComponent:
        """Clone an instance and return it.

        :param cls: the type of the new instance

        :param kwargs: arguments to add to as attributes to the clone

        :return: the cloned instance of this instance

        """
        params = kwargs
        if 'graph' not in params:
            params = dict(kwargs)
            params['graph'] = self.copy_graph(reverse_edges=reverse_edges)
        inst: GraphComponent = self.__class__(**params)
        return inst

    def detach(self) -> GraphComponent:
        graph: Graph = self.graph.induced_subgraph(
            self.vs.keys(), 'create_from_scratch')
        params: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                params[k] = v
        params['graph'] = graph
        inst: GraphComponent = self.__class__(**params)
        inst._edges_reversed = self._edges_reversed
        return inst

    @property
    def edges_reversed(self) -> bool:
        """Whether the edge direction in the graph is reversed.  This is
        ``True`` for reverse flow graphs.

        :see: :class:`.summary.ReverseFlowGraphAlignmentConstructor`

        """
        return self._edges_reversed

    @property
    def roots(self) -> Iterable[Vertex]:
        """The roots of the graph, which are usually top level
        :class:`.DocumentNode` instances.

        """
        vs: Dict[Vertex, GraphNode] = self.vs
        sel_params: Dict = {f'{self.ROOT_ATTRIB_NAME}_eq': True}
        try:
            verts: Iterable[Vertex] = self.graph.vs.select(**sel_params)
            return filter(lambda v: v in vs, verts)
        except KeyError:
            pass

    @property
    def root(self) -> Optional[Vertex]:
        """The singular (first found) root of the graph, which is usually the
        top level :class:`.DocumentNode` instance.

        """
        try:
            return next(iter(self.roots))
        except StopIteration:
            pass

    @property
    @persisted('_ntov')
    def node_to_vertex(self) -> Dict[GraphNode, Vertex]:
        """A mapping from graph nodes to vertexes."""
        return frozendict({x[1]: x[0] for x in self.vs.items()})

    @property
    @persisted('_gete')
    def graph_edge_to_edge(self) -> Dict[GraphEdge, Edge]:
        """A mapping from graph nodes to vertexes."""
        return frozendict({x[1]: x[0] for x in self.es.items()})

    @property
    @persisted('_vs')
    def vs(self) -> Dict[Vertex, GraphNode]:
        """The igraph to domain object vertex mapping."""
        g: Graph = self._igraph
        ga: str = GraphComponent.GRAPH_ATTRIB_NAME
        vs = {}
        for i in self._v_sub:
            v: Vertex = g.vs[i]
            vs[v] = v[ga]
        return frozendict(vs)

    @property
    @persisted('_es')
    def es(self) -> Dict[Edge, GraphEdge]:
        """The igraph to domain object edge mapping."""
        g: Graph = self._igraph
        ga: str = GraphComponent.GRAPH_ATTRIB_NAME
        es = {}
        for i in self._e_sub:
            e: Edge = g.es[i]
            es[e]: GraphEdge = e[ga]
        return frozendict(es)

    @classmethod
    def to_node(cls, v: Vertex) -> GraphNode:
        """Narrow a vertex to a node."""
        return v[cls.GRAPH_ATTRIB_NAME]

    @classmethod
    def set_node(cls, v: Vertex, n: GraphNode):
        """Set the graph node data in the igraph vertex."""
        v[cls.GRAPH_ATTRIB_NAME] = n
        return v, n

    @classmethod
    def to_edge(cls, e: Edge) -> GraphEdge:
        """Narrow a vertex to a edge."""
        return e[cls.GRAPH_ATTRIB_NAME]

    @classmethod
    def set_edge(cls, e: Edge, ge: GraphEdge):
        """Set the graph edge data in the igraph edge."""
        e[cls.GRAPH_ATTRIB_NAME] = ge
        return e, ge

    def node_by_id(self, ix: int) -> GraphNode:
        """Return the graph node for the vertex ID."""
        v: Vertex = self.graph.vs[ix]
        return self.to_node(v)

    def edge_by_id(self, ix: int) -> GraphEdge:
        """Return the edge for the vertex ID."""
        v: Vertex = self.graph.es[ix]
        return self.to_edge(v)

    def vertex_ref_by_id(self, ix: int) -> Vertex:
        """Get the :class:`igraph.Vertex` instance by its index."""
        return self.graph.vs[ix]

    def edge_ref_by_id(self, ix: int) -> Edge:
        """Get the :class:`igraph.Edge` instance by its index."""
        return self.graph.es[ix]

    def select_vertices(self, **kwargs) -> Iterable[Vertex]:
        """Return matched graph nodes from an :meth:`igraph.Graph.vs.select`."""
        vs: Dict[Vertex, GraphNode] = self.vs
        c: Vertex
        for v in self.graph.vs.select(**kwargs):
            if v in vs:
                yield v

    def select_edges(self, **kwargs) -> Iterable[Edge]:
        """Return matched graph edges from an :meth:`igraph.Graph.vs.select`."""
        es: Dict[Edge, GraphEdge] = self.es
        e: Edge
        for e in self.graph.es.select(**kwargs):
            if e in es:
                yield e

    @property
    @persisted('_adjacency_list')
    def adjacency_list(self) -> List[List[int]]:
        """"An adjacency list of vertexes based on their relation to each other
        in the graph.  The outer list's index is the source vertex and the inner
        list is that vertex's neighbors.

        **Implementation note**: the list is sub-setted at both the inner and
        outer level for those vertexes in this component.

        """
        def map_verts(x: Tuple[int, List[int]]) -> Tuple[int]:
            if x[0] in verts:
                return tuple(filter(lambda v: v in verts, x[1]))
            else:
                return ()

        mode: str = 'in' if self._edges_reversed else 'out'
        verts = set(map(lambda v: v.index, self.vs.keys()))
        al: List[List[int]] = self.graph.get_adjlist(mode=mode)
        return tuple(map(map_verts, zip(it.count(), al)))

    def delete_edges(self, edge_ids: Sequence[int]):
        """Remove edges by ID from the graph."""
        self.graph.delete_edges(edge_ids)
        self.invalidate()

    @property
    def stats(self) -> Dict[str, Any]:
        """Statistics on the graph, such as vertex and edge counts."""
        return self._get_stats()

    def _get_stats(self) -> Dict[str, Any]:
        return {'vertexes': len(self.vs),
                'edges': len(self.es)}

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('vertexes:', depth, writer)
        v: Vertex
        n: GraphNode
        for v, n in self.vs.items():
            self._write_line(f'v: {v.index}: {n}', depth + 1, writer)
        self._write_line('edges:', depth, writer)
        v: Edge
        for e, n in self.es.items():
            self._write_line(f'v: {e.index}: {n}', depth + 1, writer)

    def __getitem__(self, key: Union[Vertex, Edge]) -> GraphAttribute:
        data: Dict = self.vs if isinstance(key, Vertex) else self.es
        return data[key]

    def __len__(self) -> int:
        """A graph is a set of vertexes and edges."""
        return len(self.vs) + len(self.es)


GraphComponent.graph = GraphComponent._graph


@dataclass
class SentenceEntry(object):
    """Contains the sentence node of a sentence, and the respective concept and
    attribute nodes.

    """
    node: SentenceGraphNode = field(default=None)
    """The sentence node, which is the root of the sentence subgraph."""

    concepts: Tuple[ConceptGraphNode] = field(default=None)
    """The AMR concept nodes of the sentence."""

    attributes: Tuple[AttributeGraphNode] = field(default=None)
    """The AMR attribute nodes of the sentence."""

    @property
    @persisted('_by_variable')
    def concept_by_variable(self) -> Dict[str, ConceptGraphNode]:
        return frozendict({n.variable: n for n in self.concepts})


@dataclass
class SentenceIndex(object):
    """An index of the sentences of a :class:`.DocumentGraphComponent`.

    """
    entries: Tuple[SentenceEntry] = field(default=None)
    """Then entries of the index, each of which is a sentence."""

    @property
    @persisted('_by_sentence')
    def by_sentence(self) -> Dict[AmrFeatureSentence, SentenceEntry]:
        return frozendict({n.node.sent: n for n in self.entries})


@dataclass
class DocumentGraphComponent(GraphComponent):
    """A class containing the root information of the document tree and the
    :class:`igraph.Graph` vertex.  When the :class:`igraph.Graph` is set with
    the :obj:`graph` property, a strongly connected subgraph component is
    induced.  It does this by traversing all reachable verticies and edges from
    the :obj:`root`.  Examples of these induced components include *source* and
    *summary* components of a document AMR graph.

    Instances are created by :class:`.DocumentGraphFactory`.

    """
    root_node: DocumentNode = field()
    """The root of the document tree."""

    sent_index: SentenceIndex = field(default_factory=SentenceIndex)
    """An index of the sentences of a :class:`.DocumentGraphComponent`."""

    def __post_init__(self):
        super().__post_init__()
        self._root_vertex = None

    @property
    def name(self) -> str:
        """Return the name of the AMR document node."""
        return self.root_node.name

    @property
    def root(self) -> Vertex:
        """The roots of the graph, which are usually top level
        :class:`.DocumentNode` instances.

        """
        return self._root_vertex

    @root.setter
    def root(self, vertex: Vertex):
        """The roots of the graph, which are usually top level
        :class:`.DocumentNode` instances.

        """
        self._root_vertex = vertex

    def _set_graph(self, graph: Graph):
        # check for attribute since this is called before __post_init__ when
        # this instance created by the super class dataclass setter
        if hasattr(self, '_root_vertex') and self._root_vertex is not None:
            self._induce_component(self._root_vertex, self.graph, graph)
            self._igraph = graph
        else:
            super()._set_graph(graph)

    def _induce_component(self, root: Vertex, org_graph: Graph, graph: Graph):
        """Reset data structures so that this graph becomes a strongly connected
        partition of the larger graph.

        """
        # new and old graphs
        ng: Graph = graph
        # find the vertex in the new graph using the old graph and root
        old_root_doc: GraphNode
        new_root: Vertex
        old_root_doc, new_root = self._resolve_root(org_graph, root, ng)
        # finds all vertices reachable in the new graph from the root vertex in
        # the same new graph, which gives us the component, which is a subset of
        # the graph nodes if disconnected and yields the new compoennt
        subcomp_verticies: List[int] = ng.subcomponent(new_root)
        # get all edges connected to the new (sub)component we have identified
        # from the vertexes
        subcomp_edges: Iterable[Edge] = ng.es.select(_within=subcomp_verticies)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{old_root_doc} -> {new_root}')
            logger.debug(f'subgraph: {sorted(subcomp_verticies)}')

        # udpate data structures to use the subcomponent
        self._invalidate(reset_subsets=False)
        self._v_sub = set(subcomp_verticies)
        self._e_sub = set(map(lambda e: e.index, subcomp_edges))
        self._root_vertex = new_root

    @property
    def doc_vertices(self) -> Iterable[Vertex]:
        """Get the vertices of :class:`.DocuemntGraphNode`.  This only fetches
        those document nodes that do not branch.

        """
        v: Vertex = self.root
        while True:
            gn: GraphNode = self.to_node(v)
            if not isinstance(gn, DocumentGraphNode):
                break
            yield v
            ns: List[Vertex] = v.neighbors()
            if len(ns) >= 2:
                break
            v = ns[0]

    @property
    def relation_set(self) -> RelationSet:
        """The relations in the contained root node document."""
        doc: AmrFeatureDocument = self.root_node.doc
        return doc.relation_set

    def clone(self, reverse_edges: bool = False, **kwargs) -> GraphComponent:
        params = dict(root_node=self.root_node, sent_index=self.sent_index)
        params.update(kwargs)
        inst = super().clone(reverse_edges, **params)
        inst._edges_reversed = reverse_edges
        inst._induce_component(self._root_vertex, self.graph, inst.graph)
        return inst

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        super().write(depth, writer)
        self._write_object(self.root_node, depth, writer)
        self._write_line('graph:', depth, writer)
