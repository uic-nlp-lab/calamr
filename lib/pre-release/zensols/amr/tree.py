"""Penman graph utilities and algorithms.  These classes are currently only used
for debugging and do not have any significant bearing on the overall package.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Any, Union, Dict, List
from dataclasses import dataclass, field
import logging
import collections
from itertools import chain
import json
import copy
from io import StringIO
from pprint import pprint
import penman
from penman import Graph
from penman.tree import Tree
from penman.surface import Alignment, RoleAlignment

logger = logging.getLogger(__name__)


@dataclass
class TreeNavigator(object):
    graph: Graph = field()
    """The graph that will be populated with alignments."""

    strip_alignments: bool = field(default=False)
    """Whether or not to strip alignment tags from the output.  This is set to
    ``True`` in :meth:`get_missing_alignments` for test cases.  However, it's
    might also useful for :meth:`get_node`.

    """
    def __post_init__(self):
        self._nodes: Dict[Tuple[int], Tuple[bool, str, str, str]] = {}

    def _get_role_src_concept(self, node: Tuple):
        """Return the role, source and concept from the tree node."""
        is_role = node[0][0] == ':'
        if is_role:
            # roles have no concepts so we must traverse to another node
            return node[0], None, None
        elif node[1][0][0] == '/':
            # "/<concept>" found as first child node
            return None, node[0], node[1][0]
        else:
            raise ValueError(f'No concept found in <{node}>')

    def _strip_align(self, s: str) -> str:
        """Return the removed the alignment string from ``s``.  This is only
        useful for testing.

        """
        if self.strip_alignments:
            ix = s.find('~')
            if ix > -1:
                s = s[:ix]
        return s

    def _get_node(self, node: Tuple[str, Any], path: Tuple[int],
                  parent: str) -> Tuple[bool, str, str, str]:
        trip: Tuple[bool, str, str, str] = self._nodes.get(path)
        if trip is None:
            trip = self._get_node0(node, path, parent)
            self._nodes[path] = trip
        return trip

    def _get_node0(self, node: Tuple[str, Any], path: Tuple[int],
                   parent: str) -> Tuple[bool, str, str, str]:
        """Return a triple of a node or role edge by recursively traversing the
        tree data structure.

        :param node: the current traversing node

        :param path: used to guide the node being searched for of the remaining
                     subtree

        :param parent: the parent of ``node``

        """
        if logger.isEnabledFor(logging.DEBUG):
            sio = StringIO()
            sio.write(f'traversing node or edge <{node[0]}>, path={path}:\n')
            pprint(node, stream=sio)
            logger.debug(sio.getvalue().strip())

        trip: Tuple[bool, str, str, str]
        plen: int = len(path)
        role_path: bool = plen > 0 and path[0] == 'r'
        role, src, ctup = self._get_role_src_concept(node)
        src: str
        ctup: Tuple[str, str]
        role: str = None if role is None else self._strip_align(role)
        concept: str = None if ctup is None else self._strip_align(ctup[1])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'source: {src}, role: {role}, ' +
                         f'role_path: {role_path}, concept: {concept}')
        if role is not None:
            if isinstance(node[1], str):
                # the alignment is on value
                trip = (role_path, parent, role, self._strip_align(node[1]))
            elif role_path:
                # either the alignment is on the role edge
                trip = (True, parent, role, node[1][0])
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'role alignment found: {trip}')
            else:
                # or traverse to the concept node as its only and first child
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'traversing role: {role}')
                trip = self._get_node0(node[1], path, None)
        elif plen == 0:
            # land on a concept node with the alignment
            trip = (False, src, ':instance', concept)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'concept node found: {trip}')
        elif role_path and len(path) == 1 and ctup[0] == '/':
            # the root node was requested, so return the root triple
            trip = (False, src, ':instance', concept)
        else:
            next_idx: Union[int, str] = path[0]
            child = node[1][1:][next_idx]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'traversing child: {next_idx}, ' +
                             f'children({len(child)}): {str(child)[:60]}')
            trip = self._get_node0(child, path[1:], src)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'return trip: {trip}')
            logger.debug('_' * 40)
        return trip

    def get_node(self, path: Tuple[int, ...]) -> \
            Tuple[Tuple[str, str, str], bool]:
        """Get a triple representing a graph node/edge of the given path.

        :param path: a tuple of 0-based index used to get a node or edge.

        :return: the node/edge triple and ``True`` if a role edge

        """
        tree: penman.tree.Tree = penman.configure(self.graph)
        node: Tuple = self._get_node(tree.nodes()[0], path, None)
        return node[1:], node[0]


@dataclass
class TreePruner(object):
    """Create a subgraph using a tuple found in the graph *configured*
    (:mth:`penman.configure`) as a tree

    """
    graph: Graph = field()
    """The graph to be pruned."""

    keep_root_meta: bool = field(default=True)
    """Whether to keep the original metadata when the query is the root.  When
    this is ``True``, the original :obj:`graph` is returned from
    :meth:`create_sub` when the its ``query`` parameter is the root of
    :obj:`graph`.

    """
    def create_sub(self, query: Tuple[str, str, Any]) -> Graph:
        """Create a subgraph using a tuple found in the graph *configured*
        (:mth:`penman.configure`) as a tree.  Everything starting at ``query``
        and down is included in the resulting graph.

        :param query: a triple found in the contained graph

        :return: a subgraph starting at the ``query`` node

        """
        def filter_align(x):
            """Filter alignments from epidata."""
            return isinstance(x, Alignment) or isinstance(x, RoleAlignment)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'sub query: {query}')

        src: Graph = self.graph
        tree: Tree = penman.configure(src, model=penman.models.noop.model)
        src_epi: Dict[Tuple[str, str, str], List] = src.epidata
        nav = TreeNavigator(src, strip_alignments=True)
        found: Tuple[str, str, Any] = None
        trips: List[Tuple[str, str, str]] = []
        # walk through the AMR graph as a tree
        p: Tuple[int, ...]
        n: Tuple[Union[str, Tuple[Any, ...]], ...]
        for p, n in tree.walk():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'walk: {p}, {n}')
            # create the N-th sybling at each level path on the current walk
            # iteration adding roles paths
            zi_path = tuple(map(lambda x: x - 1 if x > 0 else 'r', p))
            # recursively get the triple in the tree using the N-th sybling path
            trip: Tuple[str, str, str] = nav.get_node(zi_path)[0]
            if trip == query:
                # the triple at the current walk location matches the query
                if p == (0,):
                    # the first iteration of this loop has path = (0,) = ('p',);
                    # the root node matches the query
                    if self.keep_root_meta:
                        graph = self.graph
                    else:
                        graph = copy.deepcopy(self.graph)
                        metadata = {}
                        if 'snt' in graph.metadata:
                            metadata['snt'] = graph.metadata['snt']
                        graph.metadata = metadata
                    return graph
                found = p
                fl = len(found)
            if found is not None and found == p[0:fl]:
                # if we matched the query previously and iterating into the
                # matched node, add those children triples too
                query_ref = (len(p) + 1) == fl and p[-1] == 0
                if not query_ref:
                    trips.append(trip)
        dst = Graph(trips)
        de: Dict[Tuple[str, str, str], List] = collections.defaultdict(list)
        # copy over the token-graph alignments stored in the epigraph
        for trip in dst.triples:
            se = src_epi[trip]
            de[trip].extend(filter(filter_align, se))
        dst.epidata.update(de)
        # create a sentence (snt metadata) using the aligned graph tokens
        tixs = sorted(set(chain.from_iterable(
            map(lambda a: a.indices,
                filter(filter_align, chain.from_iterable(
                    dst.epidata.values()))))))
        toks = json.loads(src.metadata['tokens'])
        dst.metadata['snt'] = ' '.join(map(lambda i: toks[i], tixs))
        return dst
