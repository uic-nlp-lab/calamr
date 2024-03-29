"""A small serialization framework for :class:`.AmrDocument` and
:class:`.AmrSentence` and other AMR artifcats.

"""
__author__ = 'Paul Landes'

from typing import Dict, Any, Union, List, Set, Type, Sequence
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from collections import OrderedDict
from zensols.config import Dictable
from . import AmrError, AmrSentence, AmrDocument, AnnotatedAmrDocument


class Include(Enum):
    """Indicates what to include at each level.

    """
    document_text = auto()
    sentences = auto()
    sentence_id = auto()
    sentence_text = auto()
    sentence_graph = auto()
    sentence_metadata = auto()
    annotated_document_id = auto()
    annotated_summary = auto()
    annotated_body = auto()
    annotated_sections = auto()


@dataclass
class Serialized(Dictable, metaclass=ABCMeta):
    """A base strategy class that can serialize :class:`.AmrDocument` and
    :class:`.AmrSentence` and other AMR artifcats.

    """
    includes: Set[str] = field()
    """The things to include."""

    @abstractmethod
    def _serialize(self) -> Dict[str, Any]:
        """Implemented to serialize :obj:`container` in to a dictionary."""
        pass

    def _from_dictable(self, recurse: bool, readable: bool,
                       class_name_param: str = None) -> Dict[str, Any]:
        return self._serialize()


@dataclass
class SerializedAmrSentence(Serialized):
    """Serializes instance of :class:`.AmrSentence`.

    """
    sentence: AmrSentence = field()
    """The sentence to serialize."""

    def _serialize(self) -> Dict[str, Any]:
        dct = {}
        if Include.sentence_id in self.includes:
            dct['id'] = self.sentence.metadata.get('id')
        if Include.sentence_text in self.includes:
            dct['text'] = self.sentence.text
        if Include.sentence_graph in self.includes:
            dct['graph'] = self.sentence.graph_only
        if Include.sentence_metadata in self.includes:
            dct['metadata'] = self.sentence.metadata
        return dct


@dataclass
class SerializedAmrDocument(Serialized):
    """Serializes instance of :class:`.AmrDocument`.

    """
    document: AmrDocument = field()
    """The document to serialize."""

    def _serialize(self) -> Dict[str, Any]:
        dct = OrderedDict()
        if Include.document_text in self.includes:
            dct['text'] = self.document.text
        if Include.sentences in self.includes:
            sents: List[Dict[str, Any]] = []
            dct['sentences'] = sents
            sent: AmrSentence
            for sent in self.document:
                ssent = SerializedAmrSentence(
                    sentence=sent,
                    includes=self.includes)
                sents.append(ssent.asdict())
        return dct


@dataclass
class SerializedAnnotatedAmrDocument(SerializedAmrDocument):
    """Serializes instance of :class:`.AnnotatedAmrDocument`.

    """
    def _serialize_doc(self, doc: AmrDocument):
        return SerializedAmrDocument(
            document=doc,
            includes=self.includes).asdict()

    def _serialize(self) -> Dict[str, Any]:
        dct = OrderedDict()
        if Include.annotated_document_id in self.includes:
            dct['id'] = self.document.doc_id
        if Include.annotated_summary in self.includes:
            dct['summary'] = self._serialize_doc(self.document.summary)
        if Include.annotated_body in self.includes:
            dct['body'] = self._serialize_doc(self.document.body)
        if Include.annotated_sections in self.includes:
            docs: List[Dict[str, Any]] = []
            dct['sections'] = docs
            doc: AmrDocument
            for doc in self.document.sections:
                docs.append(self._serialize_doc(doc))
        return dct


@dataclass
class AmrSerializedFactory(Dictable):
    """Creates instances of :class:`.Serialized` from instances of
    :class:`AmrDocument`, :class:`AmrSentence` or :class:`AnnotatedAmrDocument`.
    These can then be used as :class:`~zensols.config.dictable.Dictable`
    instances, specifically with the ``asdict`` and ``asjson`` methods.

    """
    includes: Sequence[Union[Include, str]] = field()

    def __post_init__(self):
        def map_thing(x):
            if isinstance(x, str):
                x = Include.__members__[x]
            return x

        # convert strings to enums for easy app configuration
        self.includes = set(map(map_thing, self.includes))

    def create(self, instance: Union[AmrSentence, AmrDocument]) -> Serialized:
        """Create a serializer from ``container`` (see class docs).

        :param container: he container to be serialized

        :return: an object that can be serialized using ``asdict`` and
                 ``asjson`` method.

        """
        cls: Type
        if isinstance(instance, AnnotatedAmrDocument):
            cls = SerializedAnnotatedAmrDocument
        elif isinstance(instance, AmrDocument):
            cls = SerializedAmrDocument
        elif isinstance(instance, AmrSentence):
            cls = SerializedAmrSentence
        else:
            raise AmrError(f'Unknown serialization type: {type(instance)}')
        return cls(document=instance, includes=self.includes)

    def __call__(self, instance: Union[AmrSentence, AmrDocument]) -> Serialized:
        """See :meth:`create`."""
        return self.create(instance)
