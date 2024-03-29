"""Utilities such as gradient color generators.

"""
__author__ = 'Paul Landes'

from typing import Tuple
from dataclasses import dataclass, field
from io import StringIO
from colorsys import hls_to_rgb
import textwrap as tw
from zensols.nlp import FeatureToken
from zensols.amr import Relation, AmrFeatureSentence
from .. import (
    GraphNode, GraphEdge, DocumentGraphNode, SentenceGraphNode,
    SentenceGraphAttribute, ComponentCorefAlignmentGraphEdge,
)


class ColorUtil(object):
    @staticmethod
    def rainbow_colors(n: int = 10, light: float = 0.5, end: float = 1.):
        def int_to_rgb(i: int):
            ctup: Tuple[float] = hls_to_rgb(end * i / (n - 1), light, 1)
            r, g, b = tuple(map(lambda x: int(255 * x), ctup))
            return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)

        return tuple(map(int_to_rgb, range(n)))

    @staticmethod
    def gradient_colors(n: int = 10, rgb: float = 0.5, end: float = 1.):
        def int_to_rgb(i: int):
            ctup: Tuple[float] = hls_to_rgb(rgb, (end * i / (n - 1)), 1)
            r, g, b = tuple(map(lambda x: int(255 * x), ctup))
            return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)

        return tuple(map(int_to_rgb, range(n)))


@dataclass
class Formatter(object):
    width: int = field(default=60)

    def node(self, vertex: int, gn: GraphNode, delim: str) -> str:
        title: str = None
        if isinstance(gn, DocumentGraphNode):
            title = gn.doc_node.description.replace('\n', ' ')
            title = delim.join(tw.wrap(title, width=self.width))
        elif isinstance(gn, SentenceGraphAttribute):
            title = self._sent_graph_node(gn)
            if title is not None:
                if delim != '\n':
                    title = title.replace('\n', delim)
        elif isinstance(gn, SentenceGraphNode):
            sent: AmrFeatureSentence = gn.sent
            title = delim.join(tw.wrap(sent.norm, width=self.width))
        vstr: str = f'vertex: {vertex}{delim}type: {gn.attrib_type}'
        if title is None:
            title = vstr
        else:
            title = f'{vstr}{delim}{title}'
        return title

    def edge(self, edge: int, ge: GraphEdge, delim: str) -> str:
        title: str = f"""\
edge: {edge}{delim}\
desc: {ge.description}{delim}\
capacity: {ge.capacity_str()}{delim}\
flow: {ge.flow_str()}"""
        if isinstance(ge, ComponentCorefAlignmentGraphEdge):
            title += (delim + delim.join(self.coref_edge(ge)))
        return title

    def _sent_graph_node(self, dn: SentenceGraphNode) -> str:
        tokens: Tuple[FeatureToken] = dn.tokens
        if len(tokens) > 0:
            sio = StringIO()
            for i, tok in enumerate(tokens):
                if i > 0:
                    sio.write(('_' * 30) + '\n')
                tok.write_attributes(
                    writer=sio,
                    include_type=False,
                    include_none=False,
                    feature_ids='norm ent_ tag_ pos_'.split())
            return sio.getvalue().strip().replace('=', ': ')

    def coref_edge(self, edge: ComponentCorefAlignmentGraphEdge) -> Tuple[str]:
        rel: Relation = edge.relation
        return (f'relation: {repr(rel)}',
                f'bipartite: {edge.is_bipartite}')
