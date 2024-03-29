"""Includes classes to add alginments to AMR graphs using an ISI formatted
alignment string.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, Union, Dict, Any
from dataclasses import dataclass, field
import logging
import collections
from itertools import chain
from io import StringIO
import penman
from penman import Graph
from penman.surface import Alignment, RoleAlignment
from . import TreeNavigator

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PathAlignment(object):
    """An alignment that contains the path and alignment to node, or an edge for
    role alignments.

    """
    index: int = field()
    """The index of this alignment in the ISI formatted alignment string."""

    path: Tuple[int, ...] = field()
    """The path 0-index path to the node or the edge."""

    alignment_str: str = field()
    """The original unparsed alignment."""

    alignment: Union[Alignment, RoleAlignment] = field()
    """The alignment of the node or edge."""

    triple: Tuple[str, str, str] = field()
    """The triple specifying the node or edge of the alignment."""

    @property
    def is_role(self) -> bool:
        """Whether or not the alignment is a role alignment."""
        return isinstance(self.alignment, RoleAlignment)

    def __str__(self) -> str:
        return (f'{self.index}: {self.alignment_str} -> {self.alignment} @ ' +
                f"{self.triple} (t={'role' if self.is_role else 'alignment'})")

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class AlignmentPopulator(object):
    """Adds alignments from an ISI formatted string.

    """
    graph: Graph = field()
    """The graph that will be populated with alignments."""

    alignment_key: str = field(default='alignments')
    """The key in the graph's metadata with the ISI formatted alignment string.

    """
    def __post_init__(self):
        self._nav = TreeNavigator(self.graph, False)

    def _get_node(self, node: Tuple[str, Any], path: Tuple[int, ...],
                  parent: str) -> Tuple[bool, str, str, str]:
        return self._nav._get_node(node, path, parent)

    def _merge_aligns(self, pas: List[PathAlignment]) -> List[PathAlignment]:
        """Merges nodes in ``pas`` with the same triples in to alignments with multiple
        indices.

        """
        by_trip_role: Dict[Tuple, PathAlignment] = collections.defaultdict(
            lambda: collections.defaultdict(list))
        colls: List[PathAlignment] = []
        pa: PathAlignment
        for pa in pas:
            by_trip_role[pa.triple][pa.is_role].append(pa)
        coll: List[PathAlignment]
        groups = chain.from_iterable(
            map(lambda r: r.values(), by_trip_role.values()))
        for coll in groups:
            if len(coll) > 1:
                aixs = tuple(chain.from_iterable(
                    sorted(map(lambda pa: pa.alignment.indices, coll))))
                coll[0].alignment.indices = aixs
            colls.append(coll[0])
        colls.sort()
        return colls

    def _fix_inverses(self, pas: List[PathAlignment]):
        """The FAA aligner is not consistent in which nodes are reversed for
        ``:<name>-of`` roles.  At least ``part`` and ``location`` have this
        issue.  This is only used when testing.

        """
        epis: Dict[Tuple[str, str, str], List] = self.graph.epidata
        pa: PathAlignment
        for pa in pas:
            trip: Tuple[str, str, str] = pa.triple
            role: str = trip[1]
            if role[0] == ':' and role.endswith('-of') and trip not in epis:
                pa.triple = (trip[2], role[0:-3], trip[0])

    def get_alignments(self) -> Tuple[PathAlignment, ...]:
        """Return the alignments for the graph."""
        graph = self.graph
        insts = {i.source: i for i in graph.instances()}
        assert len(graph.instances()) == len(insts)
        tree: penman.tree.Tree = penman.configure(graph)
        if logger.isEnabledFor(logging.DEBUG):
            sio = StringIO()
            sio.write('graph:\n')
            sio.write(penman.encode(graph, model=penman.models.noop.model))
            sio.write('\nepis:\n')
            for i in graph.epidata.items():
                sio.write(f'{i}\n')
            logger.debug(sio.getvalue().strip())
        aligns: List[str] = graph.metadata[self.alignment_key].split()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'alignments: {aligns}')
        path_aligns: List[PathAlignment] = []
        align: str
        for paix, align in enumerate(aligns):
            ixs, path = align.split('-')
            path = tuple(map(lambda s: s if s == 'r' else (int(s) - 1),
                             path.split('.')))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'search alignment: {align}')
            type_targ_trip = self._get_node(tree.nodes()[0], path[1:], None)
            role_align, targ_trip = type_targ_trip[0], type_targ_trip[1:]
            align_cls = RoleAlignment if role_align else Alignment
            align_inst = align_cls.from_string(f'e.{ixs}')
            pa = PathAlignment(paix, path, align, align_inst, targ_trip)
            path_aligns.append(pa)
        if self._nav.strip_alignments:
            # only necessary when testing
            self._fix_inverses(path_aligns)
        path_aligns = self._merge_aligns(path_aligns)
        return tuple(path_aligns)

    def __call__(self) -> Tuple[PathAlignment, ...]:
        """Add the alignments to the graph using the ISI formatted alignemnt string.

        :return: the alignments added to the graph

        """
        epis: Dict[Tuple[str, str, str], List] = self.graph.epidata
        pas: List[PathAlignment] = self.get_alignments()
        pa: PathAlignment
        for pa in pas:
            epi = epis.get(pa.triple)
            if epi is None:
                epis[pa.triple] = [pa.alignment]
            else:
                epi.append(pa.alignment)
        return pas

    def get_missing_alignments(self) -> Tuple[PathAlignment, ...]:
        """Find all path alignments not in the graph.  This is done by matching against
        the epi mapping.  This is only useful for testing.

        """
        def filter_align(epi: Any) -> bool:
            return isinstance(epi, (RoleAlignment, Alignment))

        missing: List[PathAlignment] = []
        epis: Dict[Tuple[str, str, str], List] = self.graph.epidata
        pas: List[PathAlignment]
        prev_sa = self._nav.strip_alignments
        try:
            self._nav.strip_alignments = True
            pas = self.get_alignments()
        finally:
            self._nav.strip_alignments = prev_sa
        for pa in pas:
            targ_trip: Tuple[str, str, str] = pa.triple
            prev_epis: List = epis.get(targ_trip)
            if prev_epis is None:
                raise ValueError(f'Target not found: {targ_trip}')
            prev_aligns: Tuple[Union[RoleAlignment, Alignment]] = \
                tuple(filter(filter_align, prev_epis))
            if pa.alignment not in prev_aligns:
                missing.append(pa)
        return tuple(missing)
