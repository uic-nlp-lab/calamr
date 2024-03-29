"""Populate an :mod:`igraph` from AMR graphs.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, List, ClassVar
from dataclasses import dataclass, field
import logging
from penman import Graph as PGraph
from penman.graph import Instance
from penman.graph import Edge as PEdge
from penman.surface import Alignment, RoleAlignment
from igraph import Graph, Vertex, Edge
from zensols.util import time
from zensols.config import ConfigFactory
from zensols.amr import AmrSentence, AmrFeatureSentence
from . import (
    DocumentNode, DocumentGraphComponent, AmrDocumentNode,
    GraphNode, DocumentGraphNode, ConceptGraphNode, AttributeGraphNode,
    SentenceGraphNode, GraphComponent,
    DocumentGraphEdge, RoleGraphEdge, SentenceGraphEdge,
    DocumentGraphDecorator, SentenceEntry, GraphAttributeContext,
)

logger = logging.getLogger(__name__)


@dataclass
class _Context(object):
    comp: DocumentGraphComponent
    graph_attrib_context: GraphAttributeContext

    @property
    def graph(self) -> Graph:
        return self.comp.graph


@dataclass
class IsomorphDocumentGraphDecorator(DocumentGraphDecorator):
    """Populates a :class:`.igraph.Graph` attributes from a
    :class:`.DocumentGraph` data by adding AMR node and edge information.

    """
    _GA_NAME: ClassVar[str] = GraphComponent.GRAPH_ATTRIB_NAME
    """The name of the graph attriubtes on igraph nodes and edges."""

    config_factory: ConfigFactory = field()
    """The configuration factory used to create a
    :class:`.GraphAttributeContext`.

    """
    graph_attrib_context_name: str = field()
    """The section name of the :class:`.GraphAttributeContext` context given to
    all nodees and edges of the graph.

    """
    def __post_init__(self):
        self._ctx: _Context = None

    def _create_verts(self, n: int) -> Tuple[Vertex]:
        """Create ``n`` vertexes and return them after being added."""
        g: Graph = self._ctx.graph
        start: int = len(g.vs)
        g.add_vertices(n)
        return tuple(map(lambda i: g.vs[start + i], range(n)))

    def _add_edges(self, es: Tuple[Tuple[int, int]]) -> Tuple[Edge]:
        """Add the edges of list ``es`` and return them after being added."""
        g: Graph = self._ctx.graph
        start: int = len(g.es)
        g.add_edges(es)
        return tuple(map(lambda i: g.es[start + i], range(len(es))))

    def _add_sent(self, sent: AmrFeatureSentence, v: Vertex, six: int,
                  sent_entry: SentenceEntry):
        """Add AMR sentence ``sent`` to the graph as a children of ``v``."""
        g: Graph = self._ctx.graph
        asent: AmrSentence = sent.amr
        pg: PGraph = asent.graph
        top: str = pg.top
        epis: Dict[Tuple[str, str, str], List] = pg.epidata
        concepts: List[ConceptGraphNode] = []
        attribs: List[AttributeGraphNode] = []
        c_ixs: Dict[str, int] = {}

        # add AMR concept nodes to the igraph
        ix: int
        i: Instance
        for ix, i in enumerate(pg.instances()):
            # index concept nodes to add them by ID later
            epi: List = epis.get(i)
            aligns = tuple(filter(lambda x: isinstance(x, Alignment), epi))
            concepts.append(ConceptGraphNode(
                context=self._ctx.graph_attrib_context,
                triple=i,
                token_aligns=aligns,
                sent=sent))
            c_ixs[i.source] = ix
        # add vertexes to graph, then set the concept nodes as attributes
        vs: Tuple[Vertex] = self._create_verts(len(concepts))
        for g_node, v_neigh in zip(concepts, vs):
            v_neigh[self._GA_NAME] = g_node

        # connect the concept nodes
        es: List[Tuple[int, int]] = []
        for e in pg.edges():
            # uses indexed concept nodes to dereference vertex IDs
            s = vs[c_ixs[e.source]].index
            t = vs[c_ixs[e.target]].index
            es.append((s, t))
        # add the edges to the igraph
        pe: PEdge
        ge: Edge
        for pe, ge in zip(pg.edges(), self._add_edges(es)):
            # add the AMR roles as edges
            epi: List = epis.get(i)
            aligns = tuple(filter(lambda x: isinstance(x, RoleAlignment), epi))
            ge[self._GA_NAME] = RoleGraphEdge(
                context=self._ctx.graph_attrib_context,
                role=pe.role,
                token_aligns=aligns,
                triple=tuple(pe),
                sent=sent)

        # connect the non-AMR document node to the AMR top (root) sentinel node
        ge = g.add_edge(v.index, vs[c_ixs[top]].index)
        ge[self._GA_NAME] = SentenceGraphEdge(
            context=self._ctx.graph_attrib_context,
            relation='sent-root',
            sent=sent,
            sent_ix=six)

        # add AMR attributes to add as igraph attributes later
        for ix, i in enumerate(pg.attributes()):
            epi: List = epis.get(i)
            aligns = tuple(filter(lambda x: isinstance(x, Alignment), epi))
            attribs.append(AttributeGraphNode(
                context=self._ctx.graph_attrib_context,
                triple=i,
                token_aligns=aligns,
                sent=sent))

        # create the AMR attribute constant nodes
        es: List[Tuple[int, int]] = []
        vsa = self._create_verts(len(attribs))
        # create the AMR edges to the constants
        for g_node, v_neigh in zip(attribs, vsa):
            v_neigh[self._GA_NAME] = g_node
            es.append((vs[c_ixs[g_node.variable]].index, v_neigh.index))

        # add the AMR attribute edges to the igraph
        a: AttributeGraphNode
        ge: Edge
        for a, ge in zip(attribs, self._add_edges(es)):
            ge[self._GA_NAME] = RoleGraphEdge(
                context=self._ctx.graph_attrib_context,
                role=a.role,
                triple=a.triple,
                token_aligns=(),
                sent=sent)

        sent_entry.concepts = tuple(concepts)
        sent_entry.attributes = tuple(attribs)

    def _add_doc_nodes(self, ns: Tuple[GraphNode], v: Vertex) -> Tuple[Vertex]:
        """Add graph nodes ``ns`` as neighbors of ``v``."""
        g: Graph = self._ctx.graph
        vs: Tuple[Vertex] = self._create_verts(len(ns))
        vix: int = v.index
        e_start: int = len(g.es)
        g.add_edges(tuple(map(lambda vc: (vix, vc.index), vs)))
        g_name: GraphNode
        v_neigh: Vertex
        for i, (g_node, v_neigh) in enumerate(zip(ns, vs)):
            v_neigh[self._GA_NAME] = g_node
            g.es[e_start + i][self._GA_NAME] = DocumentGraphEdge(
                context=self._ctx.graph_attrib_context)
        return vs

    def _add_sent_doc_node(self, sent: AmrFeatureSentence,
                           v: Vertex, six: int) -> Tuple[Vertex]:
        ctx: _Context = self._ctx
        g: Graph = ctx.graph
        v_sent: Vertex = g.add_vertex()
        e_sent: Edge = g.add_edge(v.index, v_sent.index)
        sent_node = SentenceGraphNode(
            context=self._ctx.graph_attrib_context,
            sent=sent,
            sent_ix=six)
        sent_edge = SentenceGraphEdge(
            context=self._ctx.graph_attrib_context,
            relation='sent',
            sent=sent,
            sent_ix=six)
        v_sent[self._GA_NAME] = sent_node
        e_sent[self._GA_NAME] = sent_edge
        sent_entry = SentenceEntry(sent_node, None, None)
        self._add_sent(sent, v_sent, six, sent_entry)
        ctx.comp.sent_index.entries.append(sent_entry)

    def _add_doc_node(self, n: DocumentNode, v: Vertex) -> Tuple[Vertex]:
        """Add a document node ``n`` as a neighbor of ``v``."""
        cnds: Tuple[GraphNode] = tuple(map(DocumentGraphNode, n.children))
        cvs: Tuple[Vertex] = self._add_doc_nodes(cnds, v)
        for cdn, cv in zip(n.children, cvs):
            self._add_doc_node(cdn, cv)
        sent: AmrFeatureSentence
        for six, sent in enumerate(n.sents):
            self._add_sent_doc_node(sent, v, six)
        return cvs

    def _decorate(self, comp: DocumentGraphComponent):
        g: Graph = self._ctx.graph
        doc_root: AmrDocumentNode = comp.root_node
        vr: Vertex = g.add_vertex()
        vr[self._GA_NAME] = DocumentGraphNode(
            context=self._ctx.graph_attrib_context,
            doc_node=doc_root)
        self._add_doc_node(doc_root, vr)
        g.vs[GraphComponent.ROOT_ATTRIB_NAME] = False
        vr[GraphComponent.ROOT_ATTRIB_NAME] = True
        comp.root = vr

    def decorate(self, comp: DocumentGraphComponent):
        self._ctx = _Context(
            comp=comp, 
            graph_attrib_context=self.config_factory.new_instance(
                self.graph_attrib_context_name))
        try:
            comp.sent_index.entries = []
            with time(f'decorated {comp.name}'):
                self._decorate(comp)
            comp.sent_index.entries = tuple(comp.sent_index.entries)
            # the "right" thing to do is invalidate here per contract; however,
            # it is done later in a more performant manner
            #
            # comp.invalidate()
        finally:
            self._ctx = None
