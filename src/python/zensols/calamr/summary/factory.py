"""Classes that organize document in content in to a hierarchy.

"""
__author__ = 'Paul Landes'

from typing import List, Tuple, ClassVar
from dataclasses import dataclass
from igraph import Graph
from zensols.amr import AmrFeatureDocument, AnnotatedAmrDocument
from zensols.amr import AnnotatedAmrSectionDocument
from .. import (
    DocumentNode, AmrDocumentNode, DocumentGraphComponent, DocumentGraphFactory,
    DocumentGraphDecorator,
)


class SummaryConstants(object):
    """Constants used for annotated AMR documents.

    """
    SUMMARY_COMP: ClassVar[str] = 'summary'
    """The summary component name."""

    SOURCE_COMP: ClassVar[str] = 'source'
    """The source component name."""


@dataclass
class AnnotatedSummaryAmrDocumentGraphFactory(DocumentGraphFactory):
    """Creates document graphs from
    :class:`~zensols.amr.annotate.AnnotatedAmrDocument` instances.

    """
    def _create_summary(self, root: AmrFeatureDocument) -> DocumentNode:
        sd: AmrFeatureDocument = root.from_amr_sentences(root.amr.summary.sents)
        ad = AmrDocumentNode(
            context=self.graph_attrib_context,
            name=SummaryConstants.SUMMARY_COMP,
            root=root,
            children=(),
            doc=sd)
        return ad

    def _create_section(self, root: AmrFeatureDocument,
                        sec: AnnotatedAmrSectionDocument) -> DocumentNode:
        td: AmrFeatureDocument = root.from_amr_sentences(sec.section_sents)
        bd: AmrFeatureDocument = root.from_amr_sentences(sec.sents)
        header = AmrDocumentNode(
            context=self.graph_attrib_context,
            name='header',
            root=root,
            children=(),
            doc=td)
        body = AmrDocumentNode(
            context=self.graph_attrib_context,
            name='body-sub',
            root=root,
            children=(),
            doc=bd)
        return DocumentNode(
            context=self.graph_attrib_context,
            name='section',
            root=root,
            children=(header, body))

    def _create_sections(self, root: AmrFeatureDocument) -> DocumentNode:
        secs: List[DocumentNode] = []
        sec: AnnotatedAmrSectionDocument
        secs_node: DocumentNode
        for sec in root.amr.sections:
            sec: DocumentNode = self._create_section(root, sec)
            if sec is not None:
                secs.append(sec)
        return DocumentNode(
            context=self.graph_attrib_context,
            name='body',
            root=root,
            children=tuple(secs))

    def _create(self, root: AmrFeatureDocument) -> \
            Tuple[DocumentGraphComponent]:
        def map_comp(node: DocumentNode) -> DocumentGraphComponent:
            graph: Graph = DocumentGraphComponent.graph_instance()
            return DocumentGraphComponent(graph, node)

        assert isinstance(root, AmrFeatureDocument)
        assert isinstance(root.amr, AnnotatedAmrDocument)
        summary: DocumentNode = self._create_summary(root)
        body: DocumentNode = self._create_sections(root)
        children = tuple(filter(lambda dl: len(dl) > 0, (body, summary)))
        return tuple(map(map_comp, children))


@dataclass
class SummaryDocumentGraphDecorator(DocumentGraphDecorator):
    """A decorator that short cuts the ``section``, ``header`` to make the top
    level ``header`` node the ``body-sub`` that has the sentences.

    """
    def decorate(self, component: DocumentGraphComponent):
        if component.root_node.name == 'body':
            body: DocumentNode = component.root_node['section.body-sub']
            body.name = SummaryConstants.SOURCE_COMP
            component.root_node = body
