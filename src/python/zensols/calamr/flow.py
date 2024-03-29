"""Provides container classes and computes statistics for graph alignments.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import List, Any, ClassVar, Dict, Union, Tuple, Set, Type, Optional
from dataclasses import dataclass, field
import sys
import logging
from io import TextIOBase
import collections
import textwrap as tw
import numpy as np
import pandas as pd
from zensols.persist import persisted, PersistableContainer
from zensols.config import Dictable
from zensols.amr import AmrFeatureDocument
from zensols.datdesc import DataFrameDescriber, DataDescriber
from . import (
    RoleGraphEdge, ReentrancySet, GraphNode, GraphEdge,
    SentenceGraphNode, ConceptGraphNode, AttributeGraphNode, DocumentGraphNode,
    TerminalGraphNode, DocumentGraphComponent, DocumentGraph, SentenceGraphEdge,
    SentenceGraphAttribute,
    ComponentAlignmentGraphEdge, ComponentCorefAlignmentGraphEdge,
)

logger = logging.getLogger(__name__)


_DATA_DESC_META: Tuple[Tuple[str, str]] = (
    ('s_descr', 'source node descriptions such as concept names, attribute constants and sentence text'),
    ('t_descr', 'target node descriptions such as concept names, attribute constants and sentence text'),
    ('s_toks', 'any source node aligned tokens'),
    ('t_toks', 'any target node aligned tokens'),
    ('s_attr', 'source node attribute name give by such as `doc`, `sentence`, `concept`, `attribute`'),
    ('t_attr', 'target node attribute name give by such as `doc`, `sentence`, `concept`, `attribute`'),
    ('s_id', 'source node unique identifier'),
    ('t_id', 'target node unique identifier'),
    ('edge_type', 'whether the edge is an AMR `role` or `alignment`'),
    ('rel_id', 'the coreference relation ID or `null` if the edge is not a corefernce'),
    ('is_bipartite', 'whether relation `rel_id` spans components or `null` if the edge is not a coreference'),
    ('flow', 'the (normalized/flow per node) flow of the edge'),
    ('reentrancy', 'whether the edge participates an AMR reentrancy'),
    ('align_flow', 'the flow sum of the alignment edges for the respective edge'),
    ('align_count', 'the count of incoming alignment edges to the target node'))


@dataclass
class Flow(Dictable):
    """A triple of a source node, target node and connecting edge from the
    graph.  The connecting edge has a flow value associated with it.

    """
    source: GraphNode = field()
    """The starting node in the DAG."""

    target: GraphNode = field()
    """The ending node (arrow head) in the DAG."""

    edge: GraphEdge = field()
    """The edge that connects :obj:`source` and :obj:`target`."""

    source_id: int = field(default=None, repr=False)
    """The :mod:`igraph` ID of :obj:`source`."""

    target_id: int = field(default=None, repr=False)
    """The :mod:`igraph` ID of :obj:`target`."""

    @property
    def edge_type(self) -> str:
        """Whether the edge is an AMR ``role`` or ``align``ment."""
        if isinstance(self.edge, ComponentAlignmentGraphEdge):
            return 'align'
        else:
            return 'role'

    def to_row(self, add_description: bool) -> List[Any]:
        """Create a row from the data in this flow used in
        :meth:`.FlowDocumentGraphComponent.create_df`.

        :param add_description: whether to add ``s_descr`` and ``t_descr`` to
                                the dataframe

        """
        def tok_str(node: GraphNode):
            if isinstance(node, SentenceGraphAttribute):
                return '|'.join(map(lambda t: t.norm, node.tokens))

        row: List[Any]
        if add_description:
            src_toks: str = tok_str(self.source)
            targ_toks: str = tok_str(self.target)
            row = [self.source.description, self.target.description,
                   src_toks, targ_toks]
        else:
            row = []
        rel_id: int = None
        is_bipartite: bool = None
        if isinstance(self.edge, ComponentCorefAlignmentGraphEdge):
            rel_id = self.edge.relation.seq_id
            is_bipartite = self.edge.is_bipartite
        row.extend((self.source_id, self.target_id,
                    self.source.attrib_type, self.target.attrib_type,
                    self.edge_type, rel_id, is_bipartite, self.edge.flow))
        return row

    def __str__(self) -> str:
        return (f'{self.source}[{self.source_id}] -> ' +
                f'{self.target}[{self.target_id}]: ' +
                f'{self.edge.description}')


@dataclass
class FlowDocumentGraphComponent(PersistableContainer, Dictable):
    """Contains all the flows of a :class:`.DocumentComponent`.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES = {'doc_graph', 'doc_graph_comp'}
    _PERSITABLE_PROPERTIES = {
        'n_alignable_nodes', 'df', 'connected_stats', 'stats'}

    DEFAULT_ADD_DESCRIPTION: ClassVar[bool] = False
    """The default for adding description in dataframes.

    :see: :meth:`create_df`

    """
    doc_graph: DocumentGraph = field()
    """The document graph that contains the graph components."""

    doc_graph_comp: DocumentGraphComponent = field(repr=False)
    """The document component from where the flows were aggregated."""

    root: Flow = field()
    """The root flow of the document component, which has the component's
    :class:`.DocumentGraphNode` as the source node and the sink as the target
    node.

    """
    flows: Tuple[Flow] = field()
    """The flows aggregated from the document components."""

    reentrancy_sets: Tuple[ReentrancySet] = field()
    """Concept nodes with multiple parents."""

    def __post_init__(self):
        super().__init__()

    @property
    def root_flow(self) -> float:
        """The flow from the root node to the sink in the reversed graph."""
        return self.root.edge.flow

    @property
    @persisted('_node_counts')
    def node_counts(self) -> Dict[Type[GraphNode], int]:
        """The number of nodes by type."""
        cnts: Dict[Type[GraphNode], int] = {}
        node_type: Type[GraphNode]
        for node_type in (ConceptGraphNode, AttributeGraphNode,
                          SentenceGraphNode, DocumentGraphNode):
            cnts[node_type] = 0
        node: GraphNode
        for node in self.doc_graph_comp.vs.values():
            node_type = type(node)
            cnt: int = cnts[node_type]
            cnts[node_type] = cnt + 1
        return cnts

    @property
    @persisted('_edge_counts')
    def edge_counts(self) -> Dict[Type[GraphEdge], int]:
        """The number of edges by type."""
        cnts: Dict[Type[GraphEdge], int] = {}
        edge_type: Type[GraphEdge]
        # there are no terminal or component alignment edges since the component
        # was taken from the initial graph instance outside the scope of the
        # bipartite graph
        for edge_type in (RoleGraphEdge, SentenceGraphEdge):
            cnts[edge_type] = 0
        edge: GraphEdge
        for edge in self.doc_graph_comp.es.values():
            edge_type = type(edge)
            cnt: int = cnts[edge_type]
            cnts[edge_type] = cnt + 1
        return cnts

    @property
    def n_alignable_nodes(self) -> int:
        """The number of nodes in the component that can take alignment edges.
        Whether those nodes in the count have edges does not effect the result.

        """
        node_counts: Dict[Type[GraphNode], int] = self.node_counts
        return node_counts[ConceptGraphNode] + \
            node_counts[AttributeGraphNode] + \
            node_counts[SentenceGraphNode]

    def create_df(self, add_description: bool = False) -> pd.DataFrame:
        """Return the data in :obj:`flows` and :obj:`root` as a dataframe.  Note
        the terms *source* and *target* refer to the nodes at the ends of the
        directed edge in a **reversed graph**.

          * ``s_descr``: source node descriptions such as concept names,
                         attribute constants and sentence text

          * ``t_descr``: target node of ``s_descr``

          * ``s_toks``: any source node aligned tokens

          * ``t_toks``: any target node aligned tokens

          * ``s_attr``: source node attribute name give by
                        :obj:`.GraphAttribute.attrib_type`, such as ``doc``,
                        ``sentence``, ``concept``, ``attribute``

          * ``t_attr: target node of ``s_attr``

          * ``s_id``: source node :mod:`igraph` ID

          * ``t_id``: target node :mod:`igraph` ID

          * ``edge_type``: whether the edge is an AMR ``role`` or ``alignment``

          * ``rel_id``: the coreference relation ID or ``null`` if the edge is
                        not a corefernce

          * ``is_bipartite``: whether relation ``rel_id`` spans components or
                             ``null`` if the edge is not a coreference

          * ``flow``: the (normalized/flow per node) flow of the edge

          * ``reentrancy``: whether the edge participates an AMR reentrancy

        :param add_description: whether to add ``s_descr`` and ``t_descr`` to
                                the dataframe, which defaults to
                                :obj:`DEFAULT_ADD_DESCRIPTION`

        """
        if add_description is None:
            add_description = self.DEFAULT_ADD_DESCRIPTION
        cols: List[str]
        if add_description:
            cols = 's_descr t_descr s_toks t_toks'.split()
        else:
            cols = []
        cols.extend(('s_id t_id s_attr t_attr edge_type rel_id ' +
                     'is_bipartite flow').split())
        rows: List[Any] = [self.root.to_row(add_description)]
        flow: Flow
        for flow in self.flows:
            rows.append(flow.to_row(add_description))
        df: pd.DataFrame = pd.DataFrame(rows, columns=cols)
        if len(self.reentrancy_sets) > 0:
            rs: ReentrancySet = self.reentrancy_sets[0]
            verts: Set[int] = set(rs.by_vertex.keys())
            df['reentrancy'] = df['s_id'].apply(lambda i: i in verts)
        return df

    def create_align_df(self, add_description: bool = False) -> pd.DataFrame:
        """Add the following columns to the dataframe created by
        :meth:`create_df`:

          * ``align_flow``: the flow sum of the alignment edges for the
                            respective edge

          * ``align_count``: the count of incoming alignment edges to the target
                             node found in :obj:`doc_graph_comp`

        :param add_description: whether to add descriptions and attribute types
                                to the dataframe

        """
        df: pd.DataFrame = self.create_df(add_description=add_description)
        # create alignment edge only dataframe
        adf = df[df['edge_type'] == 'align']
        # create a role only dataframe to zero flows so AMR role edge flow isn't
        # aggregated (summed) with alignment edge flow
        rdf = df[df['edge_type'] == 'role'].copy()
        rdf['flow'] = 0
        # create a dataframe with the counts of incoming alignment on component
        # edges
        join_col: str = 't_id'
        dfc = adf.groupby(join_col)[join_col].agg('count').\
            to_frame().rename(columns={join_col: 'align_count'})
        # aggregate across incoming alignment flow (excludes role edge flow)
        agg_flow = pd.concat((rdf, adf)).groupby(join_col)['flow'].agg('sum').\
            to_frame().rename(columns={'flow': 'align_flow'})
        # create dataframe with aggregate alignment flow and incoming alignment
        # edge (to node) counts
        inflows = agg_flow.merge(dfc, how='left', on=join_col).fillna(0)
        # mixed NAs create floats, so convert back to int
        inflows['align_count'] = inflows['align_count'].astype(int)
        # merge it with the orignal dataframe
        return df.merge(inflows, on=join_col)

    def _get_data_desc_meta_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=tuple(map(lambda t: t[1], _DATA_DESC_META)),
            index=tuple(map(lambda t: t[0], _DATA_DESC_META)),
            columns=['description'])

    def create_data_frame_describer(self) -> DataFrameDescriber:
        """Like :meth:`create_align_df` but includes a human readable
        description of the data.

        """
        doc: AmrFeatureDocument = self.doc_graph_comp.root_node.doc
        sent_norm: str = tw.shorten(doc.norm, width=60)
        return DataFrameDescriber(
            name='alignment flow edges',
            desc=f'the alignment flow and graph data for "{sent_norm}"',
            df=self.create_align_df(True),
            meta=self._get_data_desc_meta_df())

    @property
    @persisted('_df')
    def df(self) -> pd.DataFrame:
        """A cached version of the dataframe created by :meth:`crate_align_df`.

        """
        return self.create_align_df()

    @property
    @persisted('_connected_stats')
    def connected_stats(self) -> Dict[str, Union[int, float]]:
        """The statistics on how well the two graphs are aligned by counting as:

          * ``alignable``: the number of nodes that are eligible for having an
                           alignment (i.e. sentence, concept, and attribute
                           notes)

          * ``aligned``: the number aligned nodes in this :obj:`doc_graph_comp`

          * ``aligned_portion``: the quotient of $aligned / alignable$, which is
                                 a number between $[0, 1]$ representing a score
                                 of how well the two graphs match

        """
        df: pd.DataFrame = self.df
        # no-sink dataframe will scew the score
        nsdf = df[df['t_attr'] != TerminalGraphNode.ATTRIB_TYPE]
        # get the edges that are aligned
        adf = nsdf[(nsdf['edge_type'] == 'align') & (nsdf['align_count'] > 0)]
        # get count of target/component nodes that have at least one alignment
        n_aligned: int = len(adf['t_id'].drop_duplicates())
        # get the count of nodes in this component
        n_alignable: int = self.n_alignable_nodes
        # create the portion of those nodes in the graph connected
        aligned_portion: float = n_aligned / n_alignable
        return {'aligned': n_aligned,
                'alignable': n_alignable,
                'aligned_portion': aligned_portion}

    @property
    @persisted('_stats')
    def stats(self) -> Dict[str, Any]:
        """All statistics/scores available for this instances, which include:

          * ``root_flow``: the flow from the root node to the sink

          * ``connected``: :obj:`connected_stats`

        """
        rs: ReentrancySet
        if len(self.reentrancy_sets) > 0:
            rs = self.reentrancy_sets[0]
        else:
            rs = ReentrancySet()
        return {
            'root_flow': self.root_flow,
            'connected': self.connected_stats,
            'counts': {
                'node': dict(
                    map(lambda c: (c[0].ATTRIB_TYPE, c[1]),
                        self.node_counts.items())),
                'edge': dict(
                    map(lambda c: (c[0].ATTRIB_TYPE, c[1]),
                        self.edge_counts.items())),
            'reentrancies': rs.stats['total'],
            'relations': len(self.doc_graph_comp.relation_set)}}

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_dict(self.stats, depth, writer)


@dataclass
class FlowDocumentGraph(PersistableContainer, Dictable):
    """Contains all the flows of a :class:`.DocumentGraph`.

    **Implementation note**: instances of these are shared in the application
    context since only one is needed, and passing it around between controllers
    and sequencers complicates things more than needed.  For this reason,
    :meth:`reset` is used on a per use basis.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES = {'components'}
    _PERSITABLE_PROPERTIES = {'stats', 'stats_df'}
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {'stats'}

    components: Dict[str, FlowDocumentGraphComponent] = \
        field(default_factory=dict, repr=False)
    """The flow components of the document graph."""

    def __post_init__(self):
        super().__init__()

    def add(self, component: FlowDocumentGraphComponent):
        """Add the component to this instance."""
        self.components[component.doc_graph_comp.name] = component

    def reset(self):
        """Reset all state, which is called for alignment call since this is a
        shared instance in the application context.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'reset: {type(self)}')
        self.components.clear()

    @property
    def doc_graph(self) -> Optional[DocumentGraph]:
        """The document graph taken from one of the :obj:`components`.  Note
        that all :class:`.DocumentGraph` (``doc``) instances are shared between
        all components, which is why we don't care which component contains it.

        """
        try:
            return next(iter(self.components.values())).doc_graph
        except StopIteration:
            # this is raised when an error occured and `reset` was called
            # resulting in no components
            pass

    def create_align_df(self, add_description: bool = None) -> pd.DataFrame:
        """A concatenation of frames created with
        :meth:`.FlowDocumentGraphComponent.create_align_df` with the name of
        each component.

        """
        dfs: List[pd.DataFrame] = []
        name: str
        comp: FlowDocumentGraphComponent
        for name, comp in self.components.items():
            df: pd.DataFrame = comp.create_align_df(add_description)
            df.insert(0, 'name', name)
            dfs.append(df)
        return pd.concat(dfs)

    @property
    @persisted('_df')
    def df(self) -> pd.DataFrame:
        """A cached version of the dataframe created by :meth:`crate_align_df`.

        """
        return self.create_align_df()

    def create_data_describer(self) -> DataFrameDescriber:
        """Like :meth:`create_align_df` but includes a human readable
        description of the data.

        """
        dfds: List[DataFrameDescriber] = []
        name: str
        comp: FlowDocumentGraphComponent
        for name, comp in self.components.items():
            dfd: DataFrameDescriber = comp.create_data_frame_describer()
            dfd.name = name
            dfds.append(dfd)
        return DataDescriber(
            describers=tuple(dfds),
            name='alignment flow edges')

    @property
    @persisted('_stats')
    def stats(self) -> Dict[str, Any]:
        """The statistics with keys as component names and values taken from
        :obj:`.FlowDocumentGraphComponent.stats`.

        """
        cstats: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        astats: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        name: str
        comp: FlowDocumentGraphComponent
        alignable: List[int] = []
        aligned: List[int] = []
        aps: List[float] = []
        flows: List[float] = []
        for name, comp in sorted(self.components.items(), key=lambda x: x[0]):
            stats: Dict[str, Any] = comp.stats
            con: Dict[str, Any] = stats['connected']
            alignable.append(con['alignable'])
            aligned.append(con['aligned'])
            aps.append(con['aligned_portion'])
            flows.append(comp.stats['root_flow'])
            cstats[comp.doc_graph_comp.name] = stats
        alignable: int = sum(alignable)
        aligned: int = sum(aligned)
        aps_sum: float = sum(map(lambda x: 0 if x == 0 else 1 / x, aps))
        fl: int = len(flows)
        nf: bool = (fl == 0)
        aph: float
        if nf:
            aph = np.nan
        elif aps_sum == 0:
            aph = 0
        else:
            aph = len(aps) / aps_sum
        astats['aligned_portion_hmean'] = aph
        astats['mean_flow'] = np.nan if nf else sum(flows) / fl
        astats['tot_alignable'] = np.nan if nf else alignable
        astats['tot_aligned'] = np.nan if nf else aligned
        astats['aligned_portion'] = np.nan if nf else aligned / alignable
        if nf:
            rs = ReentrancySet()
        else:
            sets: List[ReentrancySet] = []
            comp: FlowDocumentGraphComponent
            for comp in self.components.values():
                if len(comp.reentrancy_sets) > 0:
                    sets.append(comp.reentrancy_sets[0])
            if len(sets) > 0:
                rs = ReentrancySet.combine(sets)
            else:
                rs = ReentrancySet()
        astats['reentrancies'] = rs.stats['total']
        if len(cstats) == 0 and nf:
            c = {k: np.nan for k in 'alignable aligned aligned_portion'.split()}
            comp = {
                'connected': c,
                'counts':
                    {'node':
                     dict(map(lambda n: (n.ATTRIB_TYPE, 0),
                              (ConceptGraphNode, AttributeGraphNode,
                               SentenceGraphNode, DocumentGraphNode))),
                     'edge':
                     dict(map(lambda n: (n.ATTRIB_TYPE, 0),
                              (RoleGraphEdge, SentenceGraphEdge)))},
                'root_flow': np.nan,
                'reentrancies': {}}
            cstats = {
                'source': comp,
                'summary': comp}
        bp_rels: int = -1
        doc_graph: DocumentGraph = self.doc_graph
        if doc_graph is not None:
            bp_rels = len(doc_graph.bipartite_relation_set)
        return {
            'components': cstats,
            'agg': astats,
            'bipartite_relations': bp_rels}

    @property
    @persisted('_stats_df')
    def stats_df(self) -> pd.DataFrame:
        """A Pandas dataframe version of :obj:`stats`."""
        stats: Dict[str, Any] = self.stats
        agg: Dict[str, Any] = stats['agg']
        aks: List[str] = ('aligned_portion_hmean mean_flow tot_alignable ' +
                          'tot_aligned aligned_portion reentrancies').split()
        cols: List[str] = 'component text'.split() + \
            list(map(lambda c: f'agg_{c}', aks)) + \
            'root_flow alignable aligned aligned_portion reentrancies'.split()
        agg_row = tuple(map(lambda a: agg[a], aks))
        rows: List[Tuple] = []
        name: str
        cstats: Dict[str, Any]
        for name, cstats in stats['components'].items():
            comp: FlowDocumentGraphComponent = self.components[name]
            doc: AmrFeatureDocument = comp.doc_graph_comp.root_node.doc
            con: Dict[str, Any] = cstats['connected']
            rows.append(
                (name, doc.norm, *agg_row, cstats['root_flow'],
                 con['alignable'], con['aligned'], con['aligned_portion'],
                 cstats['reentrancies'] if 'reentrancies' in cstats else None))
        return pd.DataFrame(rows, columns=cols)

    def clone(self) -> FlowDocumentGraph:
        """Return a clone of this instance to detach it from the shared instance
        (see class docs).

        """
        return self.__class__(components=dict(self.components))

    def __getitem__(self, name: str) -> FlowDocumentGraphComponent:
        return self.components[name]
