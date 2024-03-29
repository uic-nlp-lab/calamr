"""Document graph controller implementations.

"""
__author__ = 'Paul Landes'

from typing import (
    Tuple, Dict, Any, Sequence, List, Set, Type, Optional, Iterable
)
from dataclasses import dataclass, field
import logging
from igraph import Graph, Vertex, Edge
from igraph.cut import Flow as Maxflow
from zensols.util import time
from zensols.persist import persisted, PersistedWork
from zensols.amr import AmrFeatureDocument
from .render.base import GraphRenderer
from . import (
    ComponentAlignmentError,
    GraphNode, GraphEdge, ComponentAlignmentGraphEdge, RoleGraphEdge,
    SentenceGraphEdge, DocumentGraph, GraphComponent, DocumentGraphComponent,
    GraphAlignmentConstructor, DocumentGraphController,
    Flow, FlowDocumentGraphComponent, FlowDocumentGraph,
    ConceptGraphNode, EdgeFlow, Reentrancy, ReentrancySet,
)

logger = logging.getLogger(__name__)


@dataclass
class ConstructDocumentGraphController(DocumentGraphController):
    """Executes the maxflow/min cut algorithm on a document graph.  After its
    :meth:`invoke` method is called, it sets a ``build_graph`` attribute on
    itself, which is the constructed graph provided by :obj:`constructor`.

    """
    constructor: GraphAlignmentConstructor = field()
    """The constructor used to get the source and sink nodes."""

    renderer: GraphRenderer = field()
    """Visually render the graph in to a human understandable presentation."""

    def __post_init__(self):
        self._build_graph = PersistedWork('_build_graph', self)

    def _set_flow_color_style(self, doc_graph: DocumentGraph,
                              weighted_edge: bool = True,
                              partition_nodes: bool = True):
        """Change the graph style so that edge flow gradients are used and
        udpate the concept color per style

        """
        vst: Dict[str, Any] = self.renderer.visual_style.asdict()
        if weighted_edge:
            vst['weighted_edge_flow'] = True
        if partition_nodes and 'concept_color' in vst:
            cc: Dict[str, str] = vst['concept_color']
            comp: DocumentGraphComponent
            for comp in doc_graph.components:
                cc[comp.name] = cc['component'][comp.name]

    @property
    def nascent_graph(self) -> DocumentGraph:
        """The nascent graph initialed used in the :meth:`invoke` call."""
        if not hasattr(self, '_nascent_graph'):
            raise ComponentAlignmentError(f'Not yet invoked: {type(self)}')
        return self._nascent_graph

    @property
    @persisted('_build_graph')
    def build_graph(self) -> DocumentGraph:
        """A cached graph that is built from the constructor."""
        if not hasattr(self, '_doc_graph'):
            raise ComponentAlignmentError(f'Not yet invoked: {type(self)}')
        build_graph: DocumentGraph
        with time('maxflow: clone build graph'):
            build_graph = self._doc_graph.clone(
                reverse_edges=self.constructor.requires_reversed_edges())
        self.constructor.doc_graph = build_graph
        with time('maxflow: constructed build graph'):
            self.constructor.build()
        build_graph.original_graph = self._doc_graph
        return build_graph

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        def map_cap(e: Edge) -> float:
            ge: GraphEdge = GraphComponent.to_edge(e)
            return ge.capacity

        self._doc_graph = doc_graph
        self._nascent_graph = doc_graph
        try:
            with time('constructed flow network graph'):
                build_graph: DocumentGraph = self.build_graph
                self._set_flow_color_style(build_graph)
            return len(doc_graph)
        finally:
            del self._doc_graph

    def reset(self):
        """Clears the cached :obj:`build_graph` instance."""
        super().reset()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'reset: {type(self)}')
        self._build_graph.clear()
        for attr in '_nascent_graph original_graph'.split():
            if hasattr(self, attr):
                delattr(self, attr)


@dataclass
class MaxflowDocumentGraphController(DocumentGraphController):
    """Executes the maxflow/min cut algorithm on a document graph.  After its
    :meth:`invoke` method is called, it sets a ``build_graph`` attribute on
    itself, which is the constructed graph provided by :obj:`constructor`.

    """
    constructor: GraphAlignmentConstructor = field()
    """The constructor used to get the source and sink nodes."""

    def __post_init__(self):
        self.flows: Dict[int, float] = {}

    def _set_flows(self, maxflow: Maxflow, build_graph: DocumentGraph):
        g: Graph = build_graph.graph
        es: Dict[Edge, GraphEdge] = build_graph.es
        updates: int = 0
        eix: int
        flow_val: float
        for eix, flow_val in enumerate(maxflow.flow):
            e: Edge = g.es[eix]
            eg: GraphEdge = es[e]
            if isinstance(eg, RoleGraphEdge):
                prev_flow: Optional[float] = self.flows.get(eg.id)
                if prev_flow is not None:
                    diff: float = abs(prev_flow - flow_val)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'flow diff ({eg.id}):{prev_flow:.3f}->' +
                                     f'{flow_val:.3f} {diff:.3f}')
                    if diff > 0:
                        updates += 1
                else:
                    updates += 1
                # keep previous flows for diffing (update counts)
                self.flows[eg.id] = flow_val
            eg.flow = flow_val
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'add flow {maxflow} to {eg}, cap={eg.capacity}')
        return updates

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        def map_cap(e: Edge) -> float:
            ge: GraphEdge = GraphComponent.to_edge(e)
            return ge.capacity

        s: Vertex = self.constructor.source_flow_node
        t: Vertex = self.constructor.sink_flow_node
        g: Graph = doc_graph.graph
        caps = list(map(map_cap, g.es))
        logger.debug(f'running max flow on {s} -> {t}')
        maxflow: Maxflow
        with time('maxflow: algorithm'):
            maxflow = g.maxflow(s.index, t.index, caps)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('maxflow:')
            logger.debug(f'value: {maxflow.value}')
            logger.debug(f'flow: {maxflow.flow}')
            logger.debug(f'cut: {maxflow.cut}')
            logger.debug(f'partition: {maxflow.partition}')
        with time('maxflow: set max flows'):
            return self._set_flows(maxflow, doc_graph)

    def reset(self):
        """Clears the cached :obj:`build_graph` instance."""
        super().reset()
        self.flows.clear()


@dataclass(frozen=True)
class _PrevAlignment(object):
    source: GraphNode
    target: GraphNode
    edge: ComponentAlignmentGraphEdge


@dataclass
class PrevFlowDocumentGraphController(DocumentGraphController):
    """Keep previous flows to restore previous flows in
    :class:`.RestorePreviousFlowsDocumentGraphController`.

    """
    capture_alignments: bool = field()
    """Whether to track previous alignments."""

    def __post_init__(self):
        self.flows: Dict[int, float] = {}
        self.alignments: List[_PrevAlignment] = set()

    def _capture_prev_flows(self, comp: DocumentGraphComponent):
        ge: GraphEdge
        for ge in comp.es.values():
            if isinstance(ge, (RoleGraphEdge, SentenceGraphEdge)):
                self.flows[ge.id] = ge.flow

    def _capture_alignemnts(self, doc_graph: DocumentGraph):
        e: Edge
        ge: GraphEdge
        for e, ge in doc_graph.es.items():
            if isinstance(ge, ComponentAlignmentGraphEdge):
                src: GraphNode = doc_graph.node_by_id(e.source)
                targ: GraphNode = doc_graph.node_by_id(e.target)
                self.alignments.add(_PrevAlignment(src, targ, ge))

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if comp.name in self.component_names:
                self._capture_prev_flows(comp)
        if self.capture_alignments:
            self._capture_alignemnts(doc_graph)
        return 0

    def reset(self):
        """Clears the cached :obj:`build_graph` instance."""
        super().reset()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'reset: {type(self)}')
        self.flows.clear()
        self.alignments.clear()


@dataclass
class NormFlowDocumentGraphController(PrevFlowDocumentGraphController):
    """Normalizes flow on edges as the flow going through the edge and the total
    number of descendants.  Descendants are counted as the edge's source node
    and all children/descendants of that node.

    This is done recursively to calculate flow per node.  For each call
    recursive iteration, it computes the flow per node of the parent edge(s)
    from the perspective of the nascent graph, (root at top with arrows pointed
    to children underneath).  However, the graph this operates on are the
    reverese flow max flow graphs (flow diretion is taken care of adjacency list
    computed in :class:`.GraphComponent`.

    Since an AMR node can have multiple parents, we keep track of descendants as
    a set rather than a count to avoid duplicate counts when nodes have more
    than one parent.  Otherwise, in multiple parent case, duplicates would be
    counted when the path later converges closer to the root.

    """
    constructor: GraphAlignmentConstructor = field()
    """The instance used to construct the graph passed in the :meth:`invoke`
    method.

    """
    component_names: Set[str] = field()
    """The name of the components to minimize."""

    normalize_mode: str = field(default='fpn')
    """How to normalize nodes (if at all), which is one of:

      * ``fpn``: leaves flow values as they were after the initial flow per node
                 calculation

      * ``norm``: normalize so all values add to one

      * ``vis``: same as ``norm`` but add a ``vis_flow`` attribute to the edges
                 so the original flow is displayed and visualized as the flow
                 color

    """
    def _calc_neigh_flow(self, par: int, neigh: int, neighs: List[int],
                         neigh_desc: Set[int]):
        """Compute the neighbor ``neigh`` flow of parent ``par`` that has (of
        ``par``) neighbors ``neighs`` with descendants ``neigh_desc``.  This is
        then set on the neighbor's edge.

        The neighbors (``neigh`` and ``neighs``) are the children in the nascent
        graph or the parents in the reverse flow graph.

        """
        comp: DocumentGraphComponent = self._comp
        n_descendants: int = len(neigh_desc)
        eid: int = comp.graph.get_eid(neigh, par)
        ge: GraphEdge = comp.edge_by_id(eid)
        flow: float = ge.flow
        fpn: float = flow / n_descendants
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{par} -> {neigh}: desc={n_descendants}, ' +
                         f'n=, e={ge}, f={ge.flow:.4f}->{fpn:.4f}')
        ge.flow = fpn

    def _calc_flow_per_node(self, par: int, visited: Set[int]) -> \
            Tuple[Set[int], float]:
        """See class docs.

        :param par: the parent node from the perspective of the maxflow graph

        :param visited: the set of nodes already visited in this

        :return: all descendants of ``par`` and flow per node as a tuple

        """
        neighs: List[int] = self._alist[par]
        desc: Set[int] = {par}
        tot_fpn: float = 0
        par_visited: bool = par in visited
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{par} -> {neighs}, v={par_visited}, ' +
                         f'graph reversed: {self._comp._edges_reversed}')
        # protect against cycles (i.e. proxy report id=20080125_0121); this
        # condition also satisfies for nodes with more than one parent in the
        # nascent graph (benign)
        if par_visited:
            doc: AmrFeatureDocument = self._comp.root_node.root
            if logger.isEnabledFor(logging.INFO):
                logger.warning('cycle or node with multiple parent ' +
                               f'detected in {par} in {doc}')
        else:
            visited.add(par)
            neigh: int
            # descendants at this level are the immediate children of this node
            for neigh in neighs:
                neigh_desc, tfpn = self._calc_flow_per_node(neigh, visited)
                self._calc_neigh_flow(par, neigh, neighs, neigh_desc)
                desc.update(neigh_desc)
                tot_fpn += tfpn
        return desc, tot_fpn

    def _norm_flow_node(self, tot_fpn: float) -> float:
        comp: DocumentGraphComponent = self._comp
        norm_mode: str = self.normalize_mode
        tot_flow: float = 0
        ge: GraphEdge
        for ge in comp.es.values():
            norm_flow: float = ge.flow / tot_fpn
            if norm_mode == 'norm' or norm_mode == 'vis':
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{ge.flow} / {tot_fpn} = {norm_flow}')
                ge.flow = norm_flow
                if norm_mode == 'vis':
                    ge.vis_flow = norm_flow * tot_fpn
            else:
                raise ComponentAlignmentError(
                    f'Unknown normalization mode: {norm_mode}')
            tot_flow += norm_flow
        return tot_flow

    def _normalize_flow(self, comp: DocumentGraphComponent) -> int:
        """Normalize the flow so that it sums to one."""
        # add sink edge from the component root to the adjacency list to its
        # edge is also computed
        sink: Vertex = self.constructor.sink_flow_node
        self._alist: List[List[int]] = list(comp.adjacency_list)
        self._alist[sink.index] = (comp.root.index,)
        self._comp = comp
        try:
            desc, tfpn = self._calc_flow_per_node(sink.index, set())
            if logger.isEnabledFor(logging.DEBUG):
                logger.info(f'flow/node {tfpn:.3f} on {len(desc)} descendants')
            if self.normalize_mode == 'fpn':
                pass
            elif self.normalize_mode in {'portion', 'norm', 'vis'}:
                one = self._norm_flow_node(tfpn)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'total norm flow: {one}')
            else:
                raise ComponentAlignmentError(
                    f'No such normalization mode: {self.normalize_mode}')
            return len(desc)
        finally:
            del self._alist
            del self._comp

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        updates: int = 0
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if comp.name in self.component_names:
                updates += self._normalize_flow(comp)
        updates += super()._invoke(doc_graph)
        return updates


@dataclass
class FlowSetDocumentGraphController(DocumentGraphController):
    """Set a static flow on components based on name and edges based on class.

    """
    component_names: Set[str] = field(default_factory=set)
    """The components on which to set the flow."""

    match_edge_classes: Set[Type[GraphEdge]] = field(default_factory=set)
    """The edge classes (i.e. :class:`.TerminalGraphEdge`) to set the flow.

    """
    flow: float = field(default=0)
    """The flow to set."""

    def _set_component_flow(self, comp: DocumentGraphComponent) -> int:
        ge: GraphEdge
        for ge in comp.es.values():
            ge.flow = self.flow
        return len(comp.es)

    def _is_match_edge(self, ge: GraphEdge) -> bool:
        match_class: Type
        for match_class in self.match_edge_classes:
            if issubclass(type(ge), match_class):
                return True
        return False

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        updates: int = 0
        # set any component flow
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if comp.name in self.component_names:
                updates += self._set_component_flow(comp)
        ge: GraphEdge
        for e in doc_graph.graph.es:
            ge: GraphEdge = GraphComponent.to_edge(e)
            if self._is_match_edge(ge):
                ge.flow = self.flow
                updates += 1
        return updates


@dataclass
class FlowDiscountDocumentGraphController(DocumentGraphController):
    """Decrease/constrict the capacities by making the sum of the incoming flows
    from the bipartitie edges the value of :obj:`discount_sum`.  The capacities
    are only updated if the sum of the incoming bipartitie edges have a flow
    greater than :obj:`discount_sum`.

    """
    discount_sum: float = field()
    """The capacity sum will be this value (see class docs)."""

    component_names: Set[str] = field(default_factory=set)
    """The name of the components to discount."""

    def _constrict(self, gn: GraphNode, v: Vertex) -> int:
        comp: DocumentGraphComponent = self._comp
        neighs: Sequence[int] = comp.graph.incident(v, mode='in')
        neighs: GraphEdge = map(comp.edge_by_id, neighs)
        neighs: GraphEdge = tuple(filter(
            lambda ge: isinstance(ge, ComponentAlignmentGraphEdge), neighs))
        flow_sum: float = sum(map(lambda ge: ge.flow, neighs))
        squeeze: float = self.discount_sum / flow_sum
        ge: GraphEdge
        for ge in neighs:
            ge.capacity = ge.flow * squeeze
        return len(neighs)

    def _invoke_component(self, par: int) -> int:
        updates: int = 0
        comp: DocumentGraphComponent = self._comp
        neighs: List[int] = self._alist[par]
        for neigh in neighs:
            updates += self._invoke_component(neigh)
        edge_child: int
        for edge_child in comp.graph.incident(par, mode='in'):
            ge: GraphEdge = comp.edge_by_id(edge_child)
            if isinstance(ge, RoleGraphEdge):
                overflow: float = max(0, ge.flow - self.discount_sum)
                if overflow > 0:
                    e: Edge = comp.edge_ref_by_id(edge_child)
                    vid: int = e.source if e.target == par else e.target
                    gn: GraphNode = comp.node_by_id(vid)
                    updates += self._constrict(gn, vid)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'{par} e={ge}, n={gn}: {ge.flow}')
        return updates

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        updates: int = 0
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if comp.name in self.component_names:
                self._comp = comp
                self._alist: List[List[int]] = comp.adjacency_list
                try:
                    updates += self._invoke_component(comp.root.index)
                finally:
                    del self._comp
                    del self._alist
        return updates


@dataclass
class FixReentrancyDocumentGraphController(DocumentGraphController):
    """Fix reentrancies by splitting the flow of the last calculated maxflow as
    the capacity of the outgoing edges in the reversed graph.  This fixes the
    issue edges getting flow starved, then later eliminated in the graph
    reduction steps.

    Subsequently, the maxflow algorithm is rerun if we have at least one
    reentrancy after reallocating the capacit(ies).

    """
    component_names: Set[str] = field()
    """The name of the components to restore."""

    maxflow_controller: MaxflowDocumentGraphController = field()
    """The maxflow component used to recalculate the maxflow ."""

    only_report: bool = field()
    """Whether to only report reentrancies rather than fix them."""

    reentrancy_sets: List[ReentrancySet] = field(default_factory=list)
    """The record of retrances as they were found at the time of the execution
    of this controller.

    """
    def _iterate(self, comp: DocumentGraphComponent, v: Vertex, gn: GraphNode):
        neighs: Tuple[GraphEdge] = tuple(
            filter(lambda ge: isinstance(ge, RoleGraphEdge), map(
                comp.edge_by_id, comp.graph.incident(v, mode='out'))))
        # reentrancies have multiple outgoing edges in the reversed graph
        if len(neighs) > 1:
            # rentrancies are always concept nodes with role edges across only
            assert isinstance(gn, ConceptGraphNode)
            reentrancy = Reentrancy(
                concept_node=gn,
                concept_node_vertex=v.index,
                edge_flows=tuple(map(EdgeFlow, neighs)))
            self._reentrancies.append(reentrancy)
            # we care only about at least one that has no flow
            if not self.only_report and reentrancy.has_zero_flow:
                total_flow: float = reentrancy.total_flow
                new_capacity: float = total_flow / len(neighs)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f'setting cap {new_capacity} on {len(neighs)} edges')
                neigh: GraphEdge
                for neigh in neighs:
                    neigh.capacity = new_capacity

    def _iterate_comp(self, comp: DocumentGraphComponent):
        v: Vertex
        gn: GraphNode
        for v, gn in comp.vs.items():
            self._iterate(comp, v, gn)

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        self._reentrancies: List[Reentrancy] = []
        try:
            comp: DocumentGraphComponent
            for comp in doc_graph.components:
                if comp.name in self.component_names:
                    self._iterate_comp(comp)
            reentrancies: Tuple[Reentrancy] = tuple(self._reentrancies)
            self.reentrancy_sets.append(ReentrancySet(reentrancies))
            if len(reentrancies) > 0:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('rerunning maxflow after finding ' +
                                 f'{len(reentrancies)} reentrancies')
                return self.maxflow_controller.invoke(doc_graph)
            else:
                return 0
        finally:
            del self._reentrancies


@dataclass
class RestorePreviousFlowsDocumentGraphController(DocumentGraphController):
    """Restores the previous set of flows on a component of the graph that was
    clobbered from a previous run of flow going the reverse direction from the
    last controller that changed the flows (norm flow).

    """
    prev_controller: DocumentGraphController = field()
    """The controller that has the previous flows."""

    construct_graph_controller: ConstructDocumentGraphController = field()
    """The controller with the graph to update."""

    component_names: Set[str] = field()
    """The name of the components to restore."""

    def _restore_flow(self, comp: DocumentGraphComponent):
        prev_flows: Dict[int, float] = self.prev_controller.flows
        ge: GraphEdge
        for ge in comp.es.values():
            if isinstance(ge, (RoleGraphEdge, SentenceGraphEdge)):
                prev_flow = prev_flows[ge.id]
                ge.flow = prev_flow

    def _restore_alignment(self, doc_graph: DocumentGraph,
                           alignments: List[_PrevAlignment]):
        def map_align(a: _PrevAlignment) -> Tuple[int, int]:
            if a.source not in n2v or a.target not in n2v:
                raise ComponentAlignmentError(f'missing alignment: {a}')
            src: Vertex = n2v[a.source]
            targ: Vertex = n2v[a.target]
            return (src.index, targ.index)

        g: Graph = doc_graph.graph
        n2v: Dict[GraphNode, Vertex] = doc_graph.node_to_vertex
        es: Tuple[Tuple[int, int]] = tuple(map(map_align, alignments))
        start: int = len(g.es)
        g.add_edges(es)
        edges: Tuple[Edge] = tuple(map(
            lambda i: g.es[start + i], range(len(es))))
        ga_name: str = GraphComponent.GRAPH_ATTRIB_NAME
        e: Edge
        align: _PrevAlignment
        for e, align, ev in zip(edges, alignments, es):
            assert isinstance(align.edge, ComponentAlignmentGraphEdge)
            if align.edge.flow > 0:
                e[ga_name] = align.edge
        doc_graph.invalidate()

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if comp.name in self.component_names:
                self._restore_flow(comp)
        if len(self.prev_controller.alignments) > 0:
            self._restore_alignment(
                self.construct_graph_controller.nascent_graph,
                self.prev_controller.alignments)


@dataclass
class AlignmentCapacitySetDocumentGraphController(DocumentGraphController):
    """Set the capacity on edges if the criteria matches :obj:`min_flow`,
    :obj:`component_names` and :obj:`match_edge_classes`.

    """
    min_capacity: float = field()
    """The minimum capacity to clamping the capacity of a target
    :class:`.GraphEdge` to :obj:`capacity`.

    """
    capacity: float = field()
    """The capacity to set."""

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        updates: int = 0
        ge: GraphEdge
        for ge in doc_graph.es.values():
            if isinstance(ge, ComponentAlignmentGraphEdge):
                if ge.capacity <= self.min_capacity:
                    ge.capacity = self.capacity
                    ge.flow = self.capacity
                    updates += 1
        return updates


@dataclass
class RoleCapacitySetDocumentGraphController(DocumentGraphController):
    """This finds low flow role edges and sets (zeros out) all the capacities of
    all the connected edge alignments recursively for all descendants.  We
    "slough off" entire subtrees (sometimes entire sentences or document nodes)
    for low flow ancestors.

    """
    min_flow: float = field()
    """The minimum amount of flow to trigger setting the capacity of a target
    :class:`.GraphEdge` capacity to :obj:`capacity`.

    """
    capacity: float = field()
    """The capacity (and flow) to set."""

    component_names: Set[str] = field()
    """The name of the components to minimize."""

    def _child_flow(self, comp: DocumentGraphComponent,
                    e: Edge, ge: GraphEdge) -> float:
        """Return the sum of the flow of all children edges of the source
        (parent) node.  If the passed edge's source (parent) node has multiple
        outgoing edges, this might have a different flow than the single passed
        edge.

        """
        child_flow: float = ge.flow
        # get all children edges of the parent of the passed edge; assume
        # reverse flow
        eids: Sequence[int] = comp.graph.incident(e.source, mode='out')
        # this will include the passed edge, but might have others
        if len(eids) > 1:
            child_flow = sum(map(lambda i: comp.edge_by_id(i).flow, eids))
        return child_flow

    def _set_flow_comp(self, comp: DocumentGraphComponent) -> int:
        """Normalize the flow so that it sums to one."""
        def filter_edge(eid: int, ge: GraphEdge):
            return eid not in visited and \
                isinstance(ge, ComponentAlignmentGraphEdge)

        updates: int = 0
        min_flow: float = self.min_flow
        cap: float = self.capacity
        visited: Set[int] = set()
        e: Edge
        ge: GraphEdge
        for e, ge in comp.es.items():
            if isinstance(ge, (RoleGraphEdge, SentenceGraphEdge)) and \
               ge.flow <= min_flow and \
               self._child_flow(comp, e, ge) < min_flow:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'{ge}({e.index}): flow={ge.flow}: {e.source}')
                # assume reverse flow
                for v in comp.graph.dfs(e.source, mode='in')[0]:
                    n: GraphNode = comp.node_by_id(v)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f' {n}({v})')
                    eids: Sequence[int] = comp.graph.incident(v, mode='in')
                    es: Iterable[GraphEdge] = map(
                        lambda eid: (eid, comp.edge_by_id(eid)), eids)
                    es = filter(lambda x: filter_edge(*x), es)
                    for eid, sge in es:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f'  set cap {n}({v}): {eid}({sge})')
                        sge.capacity = cap
                        updates += 1
                    visited.update(eids)
        return updates

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        updates: int = 0
        comp: DocumentGraphComponent
        for comp in doc_graph.components:
            if comp.name in self.component_names:
                updates += self._set_flow_comp(comp)
        return updates


@dataclass
class RemoveAlignsDocumentGraphController(DocumentGraphController):
    """Removes graph component alignment for low capacity links.

    """
    min_capacity: float = field()
    """The graph component alignment edges are removed if their capacities are
    at or below this value.

    """
    def _invoke(self, doc_graph: DocumentGraph) -> int:
        to_remove: List[int] = []
        e: Edge
        ge: GraphEdge
        for e, ge in doc_graph.es.items():
            if isinstance(ge, ComponentAlignmentGraphEdge):
                if ge.capacity <= self.min_capacity:
                    to_remove.append(e.index)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'deleting: {to_remove}')
        doc_graph.delete_edges(to_remove)
        return len(to_remove)


@dataclass
class FlowCollectorDocumentGraphController(DocumentGraphController):
    """Collects statistics on a finished graph that needs no further edits.

    """
    constructor: GraphAlignmentConstructor = field()
    """The instance used to construct the graph passed in the :meth:`invoke`
    method.

    """
    fix_reentrancy_controller: FixReentrancyDocumentGraphController = field()
    """"""

    flow_doc_graph: FlowDocumentGraph = field()
    """The flow document graph to populate with aggregate flow data."""

    component_name: str = field()
    """The name of the components on which to collect stats."""

    add_aligns: bool = field()
    """Whether to add alignment :class:`.Flow`s."""

    add_roles: bool = field()
    """Whether to add AMR role :class:`.Flow`s."""

    def _create_flow(self, doc_graph: DocumentGraph,
                     e: Edge, ge: GraphEdge) -> Flow:
        se: int = e.source
        te: int = e.target
        s: GraphNode = doc_graph.node_by_id(se)
        t: GraphNode = doc_graph.node_by_id(te)
        return Flow(s, t, ge, se, te)

    def _get_root(self, doc_graph: DocumentGraph) -> Flow:
        sink: Vertex = self.constructor.sink_flow_node
        sink_neighs: Sequence[int] = doc_graph.graph.incident(sink, mode='in')
        assert len(sink_neighs) == 1
        e: Edge = doc_graph.edge_ref_by_id(sink_neighs[0])
        ge: GraphEdge = doc_graph.edge_by_id(sink_neighs[0])
        root: Flow = self._create_flow(doc_graph, e, ge)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'sink ({sink.index}): {root}')
        return root

    def _add_edges(self, doc_graph: DocumentGraph, comp: DocumentGraphComponent,
                   flows: List[Flow]) -> int:
        aligns: bool = self.add_aligns
        roles: bool = self.add_roles
        e: Edge
        ge: GraphEdge
        for e, ge in doc_graph.es.items():
            vt: Vertex = doc_graph.vertex_ref_by_id(e.target)
            if vt not in comp.vs:
                continue
            if (aligns and isinstance(ge, ComponentAlignmentGraphEdge)) or \
               (roles and isinstance(ge, (SentenceGraphEdge, RoleGraphEdge))):
                flows.append(self._create_flow(doc_graph, e, ge))

    def _invoke(self, doc_graph: DocumentGraph) -> int:
        comp: DocumentGraphComponent = \
            doc_graph.components_by_name[self.component_name]
        root: Flow = self._get_root(doc_graph)
        flows: List[Flow] = []
        self._add_edges(doc_graph, comp, flows)
        flow_comp = FlowDocumentGraphComponent(
            doc_graph=doc_graph,
            doc_graph_comp=comp,
            root=root,
            flows=tuple(flows),
            reentrancy_sets=tuple(
                self.fix_reentrancy_controller.reentrancy_sets))
        self.flow_doc_graph.add(flow_comp)
        return 0
