"""Extensions of :mod:`zensols.nlp` feature containers.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Type, Iterable, Dict, Tuple, List, ClassVar, Set, Any
from dataclasses import dataclass, field
import sys
from io import TextIOBase
import textwrap as tw
from frozendict import frozendict
from penman.surface import Alignment, RoleAlignment
from penman.graph import Graph
from zensols.persist import persisted, PersistableContainer
from zensols.config import Dictable
from zensols.nlp import (
    LexicalSpan, TextContainer, TokenContainer,
    FeatureToken, FeatureSentence, FeatureDocument,
)
from . import AmrError, AmrSentence, AmrDocument, TreePruner


@dataclass
class ReferenceObject(PersistableContainer, Dictable):
    """A base class reference and relation classes.

    """
    def __post_init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(repr=False, unsafe_hash=True)
class Reference(ReferenceObject):
    """A multi-document coreference target, which points to a node in an AMR
    graph.

    """
    _DICTABLE_ATTRIBUTES: ClassVar[Set[str]] = {'target'}

    sent: AmrFeatureSentence = field(repr=False)
    """The sentence containing the reference."""

    variable: str = field()
    """The variable in the AMR graph."""

    @property
    @persisted('_triple', transient=True)
    def triple(self) -> Tuple[str, str, str]:
        """The AMR tripple of ``(source relation target)`` of the reference."""
        return tuple(self.sent.amr.instances[self.variable])

    @property
    def target(self) -> str:
        """The target of the coreference."""
        return self.triple[2]

    @property
    @persisted('_short', transient=True)
    def short(self):
        """A short string describing the reference."""
        return f'{self.variable} / {self.target}'

    @property
    @persisted('_subtree', transient=True)
    def subtree(self) -> AmrSentence:
        """The subtree of the sentence containing the target as an
        :class:`.AmrFeatureSentence`.

        """
        # only retain the sentence for debugging (write method)
        tp = TreePruner(self.sent.amr.graph, keep_root_meta=False)
        graph: Graph = tp.create_sub(self.triple)
        return AmrSentence(data=graph)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line(self.short, depth, writer)
        self._write_wrap(f'[N]: {self.sent.norm}', depth + 1, writer)
        self._write_line('subtree:', depth + 1, writer)
        self._write_line(f'<{self.subtree.text}>', depth + 2, writer)
        self._write_object(self.subtree, depth + 2, writer)

    def __str__(self) -> str:
        sent_text = tw.shorten(self.sent.text, 70)
        return f'{sent_text}: {self.short}'

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(repr=False)
class Relation(ReferenceObject):
    """A relation makes up a set of references across multuiple sentences of a
    document.  This is what Christopher Manning calls a cluster.

    """
    seq_id: int = field()
    """The sequence identifier of the relation."""

    references: Tuple[Reference, ...] = field()
    """The references for this relation."""

    @property
    @persisted('_by_sent', transient=True)
    def by_sent(self) -> Dict[AmrFeatureSentence, Reference]:
        """An association from sentences to their references."""
        return frozendict({r.sent: r for r in self.references})

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_line('references:', depth, writer)
        for ref in self.references:
            self._write_object(ref, depth + 1, writer)

    def __getitem__(self, i: int) -> Reference:
        return self.references[i]

    def __len__(self) -> int:
        return len(self.references)

    def __eq__(self, other: Relation) -> bool:
        if id(self) == id(other):
            return True
        return self.references == other.references

    def __hash__(self) -> int:
        return hash(self.references)

    def __repr__(self) -> str:
        return ', '.join(map(lambda r: r.short, self.references))


@dataclass(unsafe_hash=True)
class RelationSet(ReferenceObject):
    """All coreference relations for a given document.

    """
    _DICTABLE_WRITABLE_DESCENDANTS = True

    relations: Tuple[Relation, ...] = field()
    """The relations for all documents computed the coreferencer."""

    @persisted('_as_set', transient=True)
    def as_set(self) -> Set[Relation]:
        """A set version of :obj:`relations`."""
        return frozenset(self.relations)

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        self._write_iterable(self.relations, depth, writer, include_index=True)

    def __getitem__(self, i: int) -> Relation:
        return self.relations[i]

    def __len__(self) -> int:
        return len(self.relations)


@dataclass(eq=False, repr=False)
class AmrFeatureSentence(FeatureSentence):
    """A sentence that holds an instance of :class:`.AmrSentence`.

    """
    amr: AmrSentence = field(default=None)
    """The AMR representation of the sentence."""

    def __post_init__(self):
        super().__post_init__()
        if self.spacy_span is not None:
            self.amr = self.spacy_span._.amr

    @property
    def _amr(self) -> AmrSentence:
        return self._amr_val

    @_amr.setter
    def _amr(self, amr: AmrSentence):
        self._amr_val = amr
        # keep spaCy artifact in sync
        if self.spacy_span is not None:
            self.spacy_span._.amr = amr

    @property
    def is_failure(self) -> bool:
        """Whether the AMR graph failed to be parsed."""
        return self.amr.is_failure

    @property
    @persisted('_indexed_alignments', transient=True)
    def indexed_alignments(self) -> \
            Dict[Tuple[str, str, str], Tuple[Tuple[int, FeatureToken]]]:
        """The graph alignments as a triple-to-token dict.  The values are
        tuples 0-index token offset and the feature token pointed to by the
        alignment.

        """
        def filter_aligns(x) -> bool:
            return isinstance(x, (Alignment, RoleAlignment))

        def map_align(i) -> Tuple[int, FeatureToken]:
            tok: FeatureToken = tokens_by_i_sent.get(i)
            if tok is None:
                raise AmrError(f'No such alignment ({i}) in ' +
                               f'<{self.text}>; mapping: <{tokens_by_i_sent}>')
            return i, tok

        tokens_by_i_sent: Dict[int, FeatureToken] = self.tokens_by_i_sent
        trip_to_tok: Dict[Tuple[str, str, str], Tuple[int, FeatureToken]] = {}
        trip: Tuple[str, str, str]
        indices: Tuple[int, ...]
        for trip, indices, _ in self.amr.iter_aligns():
            trip_to_tok[trip] = tuple(map(map_align, indices))
        return frozendict(trip_to_tok)

    @property
    @persisted('_alignments', transient=True)
    def alignments(self) -> Dict[Tuple[str, str, str], Tuple[FeatureToken]]:
        """The tokens only returnd from :obj:`indexed_alignments`."""
        ias = self.indexed_alignments
        als = {x[0]: tuple(map(lambda t: t[1], x[1])) for x in ias.items()}
        return frozendict(als)

    def clone(self, cls: Type = None, **kwargs) -> TokenContainer:
        c = super().clone(cls, **kwargs)
        c.amr = self.amr.clone()
        return c

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              n_tokens: int = 0, include_metadata: bool = False,
              include_original: bool = False, include_normalized: bool = True,
              include_stack: bool = False, include_amr: bool = True):
        """
        :param include_stack: whether to add the stack trace of the parse of an
                              error occured while trying to do so

        :param include_metadata: whether to add graph metadata to the output
        """
        super().write(depth, writer, n_tokens=n_tokens,
                      include_original=include_original,
                      include_normalized=include_normalized)
        if include_amr:
            self.amr.write(depth + 1, writer,
                           include_metadata=include_metadata,
                           include_stack=include_stack)

    def __eq__(self, other: FeatureDocument) -> bool:
        return super().__eq__(other) and self.amr == other.amr

    def __hash__(self) -> int:
        return hash(self.norm)


# keep the dataclass semantics, but allow for a setter to add AMR metadata
AmrFeatureSentence.amr = AmrFeatureSentence._amr


@dataclass(eq=False, repr=False)
class AmrFeatureDocument(FeatureDocument):
    """A feature document that contains an :obj:`amr` graph.

    """
    amr: AmrDocument = field(default=None)
    """The AMR representation of the document."""

    coreference_relations: Tuple[Tuple[Tuple[int, str], ...], ...] = \
        field(default=None)
    """The coreferences tuple sets between the sentences of the document::

      ((<sentence index 1>, <variable 1>),
       (<sentence index 2>, <variable 2>)...)

    """
    def __post_init__(self):
        super().__post_init__()
        if self.spacy_doc is not None:
            self.amr = self.spacy_doc._.amr

    @property
    def _amr(self) -> AmrDocument:
        return self._amr_val

    @_amr.setter
    def _amr(self, amr: AmrDocument):
        self._amr_val = amr
        # keep feature sentences in sync
        self.sync_amr_sents()

    @property
    def _coreference_relations(self) -> Tuple[Tuple[Tuple[int, str], ...], ...]:
        return self.__coreference_relations_val

    @_coreference_relations.setter
    def _coreference_relations(self,
                               v: Tuple[Tuple[Tuple[int, str], ...], ...]):
        self.__coreference_relations_val = v
        if hasattr(self, '_relation_set'):
            self._relation_set.clear()

    def sync_amr_sents(self):
        """Copy :obj:`amr` sentences to each respective
        :obj:`.AmrFeatureSentence.amr`.  This is necessary when then
        :class:`.AmrDocument` is updated with new sentences that need to
        *percolate* down to the feature sentences.

        """
        if self.amr is not None:
            fs: AmrFeatureSentence
            ams: AmrSentence
            # set each cloned feature sentence's AMR sentnce
            for fs, ams in zip(self.sents, self.amr.sents):
                fs.amr = ams

    def _map_coref(self, sents: Tuple[AmrFeatureSentence, ...]) -> \
            Tuple[Tuple[Tuple[int, str], ...], ...]:
        """Create a new value for :obj:`coreference_relations` by adding addin
        each :class:`RelationSet` that encompasses ``sents``. In other words,
        for all relation set ``X``, add ``X`` if ``sents`` is a proper subset of
        the sentences found in ``X``.

        :param sents: the sentences to consider for a new relation set

        :see: :meth:`from_sentences`

        """
        rs: RelationSet = self.relation_set
        doc_cr: List[Tuple[Tuple[int, str], ...], ...] = []
        sents = tuple(sents)
        rel: Relation
        for rel in rs:
            # we already know we don't have enough coverage if we don't have
            # enough sentences to for the relation
            if len(sents) >= len(rel):
                sent_cr: List[Tuple[int, str]] = []
                sent: AmrFeatureSentence
                for six, sent in enumerate(sents):
                    ref: Reference = rel.by_sent.get(sent)
                    if ref is not None:
                        # the provided sentence is also in this relation
                        sent_cr.append((six, ref.variable))
                # check again for sentences missing in this relation
                if len(sent_cr) >= len(rel):
                    doc_cr.append(tuple(sent_cr))
        return tuple(doc_cr)

    def add_coreferences(self, to_populate: AmrFeatureDocument):
        """Add :obj:`coreference_relations` to ``to_populate`` using this
        instance's coreferences.  Note that :meth:`from_sentences`,
        :meth:`from_amr_sentences`, :meth:`get_overlapping_document` and
        meth:`clone` already do this.

        """
        to_populate.coreference_relations = self._map_coref(to_populate.sents)

    def from_sentences(self, sents: Iterable[FeatureSentence],
                       deep: bool = False) -> AmrFeatureDocument:
        """Return a new cloned document using the given sentences.

        :param sents: the sentences to add to the new cloned document

        :param deep: whether or not to clone the sentences

        :see: :meth:`clone`

        :see: :meth:`add_coreferences`

        """
        sents = tuple(sents)
        self._no_amr_clone = True
        try:
            clone = super().from_sentences(sents, deep)
        finally:
            del self._no_amr_clone
        amr_sents = tuple(map(lambda fs: fs.amr, sents))
        clone.amr = self.amr.from_sentences(amr_sents)
        return clone

    def from_amr_sentences(self, amr_sents: Iterable[AmrSentence]) -> \
            AmrFeatureDocument:
        """Like :meth:`from_sentences` from return those a new document with
        :class:`~zensols.nlp.FeatureDocument` sentences sync'd with
        :class:`.AmrSentence`.

        :see: :meth:`add_coreferences`

        """
        a2d = {id(fs.amr): fs for fs in self.sents}
        sents = tuple(map(lambda a: a2d[id(a)], amr_sents))
        return self.from_sentences(sents)

    def get_overlapping_document(self, span: LexicalSpan) -> FeatureDocument:
        doc = super().get_overlapping_document(span)
        sent: FeatureSentence
        for sent in doc:
            os = tuple(self.get_overlapping_sentences(sent.lexspan))
            if len(os) > 0:
                sent.amr = os[0].amr
        doc.amr = self.amr
        return doc

    def clone(self, cls: Type = None, **kwargs) -> TokenContainer:
        clone = super().clone(cls, **kwargs)
        if not hasattr(self, '_no_amr_clone'):
            clone.amr = self.amr.clone()
        self.add_coreferences(clone)
        return clone

    @property
    @persisted('_relation_set', transient=True)
    def relation_set(self) -> RelationSet:
        """The relations in the contained document as a set of relations."""
        crefs: Tuple[Tuple[int, str], ...] = self.coreference_relations
        crefs = () if crefs is None else crefs
        rels: List[Relation] = []
        for rix, rel in enumerate(crefs):
            corefs: Tuple[AmrFeatureSentence, str] = []
            six: int
            var: str
            for six, var in rel:
                entry = Reference(self.sents[six], var)
                corefs.append(entry)
            rels.append(Relation(rix, tuple(corefs)))
        return RelationSet(tuple(rels))

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout,
              n_tokens: int = 0, include_relation_set: bool = False,
              include_original: bool = False, include_normalized: bool = True,
              include_amr: bool = None, sent_kwargs: Dict[str, Any] = {},
              amr_kwargs: Dict[str, Any] = {}):
        if self.amr.__class__.__name__ == 'AnnotatedAmrDocument' and \
           include_amr is None and len(amr_kwargs) == 0:
            include_amr = True
            amr_kwargs = dict(include_amr=False)
        else:
            include_amr = False
        TextContainer.write(self, depth, writer,
                            include_original=include_original,
                            include_normalized=include_normalized)
        self._write_line('sentences:', depth, writer)
        s: AmrFeatureSentence
        for s in self.sents:
            s.write(depth + 1, writer, n_tokens=n_tokens,
                    include_original=include_original,
                    include_normalized=include_normalized,
                    **sent_kwargs)
        if include_relation_set:
            relset: RelationSet = self.relation_set
            if len(relset) > 0:
                self._write_line('relation_sets:', depth, writer)
                self._write_object(relset, depth + 1, writer)
        if include_amr:
            self._write_line('amr:', depth, writer)
            self.amr.write(depth + 1, writer, **amr_kwargs)

    def __eq__(self, other: FeatureDocument) -> bool:
        return super().__eq__(other) and self.amr == other.amr

    def __hash__(self) -> int:
        return hash(self.norm)


# keep the dataclass semantics, but allow for a setter to add AMR metadata
AmrFeatureDocument.amr = AmrFeatureDocument._amr
AmrFeatureDocument.coreference_relations = \
    AmrFeatureDocument._coreference_relations
