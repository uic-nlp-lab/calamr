from __future__ import annotations
"""Bank domain classes.

"""
__author__ = 'Paul Landes'

from typing import ClassVar, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field, InitVar
from enum import Enum, auto
from abc import ABC, abstractmethod
import sys
import re
from pathlib import Path
from io import TextIOBase
from torch import Tensor
from zensols.util import APIError
from zensols.config import Dictable
from zensols.persist import PersistableContainer
from zensols.cli import ApplicationError


class BankError(ApplicationError):
    """Raised for this package's application errors meant for the command line.
    It will result in a command line error and usage message.

    """


class BankParseError(APIError):
    """Raised for this package's programmatic errors.

    """
    pass


class RoleResource(Enum):
    """The source *bank* of the role.  This has the XML attribute ``resource``.

    """
    verbnet = auto()
    framenet = auto()


class PartOfSpeech(Enum):
    """The part of speech identifier in aliases.

    """
    adverb = 'r'
    preposition = 'p'
    verb = 'v'
    noun = 'n'
    adjective = 'j'
    unknown = '-'


@dataclass
class BankObject(PersistableContainer, Dictable):
    """A base class for all ``*bank`` domain classes.

    """
    _PERSITABLE_TRANSIENT_ATTRIBUTES = {'_embedding'}

    def __post_init__(self):
        super().__init__()

    @property
    def embedding(self) -> Optional[Tensor]:
        """The embedding of this instance if embedding population is on."""
        if hasattr(self, '_embedding'):
            return self._embedding

    @embedding.setter
    def embedding(self, embedding: Tensor):
        self._embedding = embedding

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class BankObjectPopulator(ABC):
    @abstractmethod
    def populate(self, bank_object: BankObject):
        """Populate ``bank_object`` after it was just de-hydrated from the
        database.

        """
        pass


@dataclass(repr=False)
class Function(BankObject):
    """The role function of the role, such as ``PAG`` for ``prototypical
    agent``.  These are taken from the frameset DTD from the AMR 3.0 corpus.

    """
    NO_DESCRIPTION: ClassVar[str] = 'none'
    """A constant used when the function has no description."""

    uid: int = field(default=None, repr=False)
    """A unique identifier of the function."""

    label: str = field(default=None)
    """The label (i.e. ``PAG``)."""

    description: str = field(default='none')
    """The human readable description of the function."""

    group: str = field(default='unknown')
    """The group the function belongs to (i.e. ``spacial``)."""

    @property
    def has_description(self) -> bool:
        """Whether the function has a description."""
        return self.description != self.NO_DESCRIPTION

    def __hash__(self) -> int:
        return hash((self.label, self.description, self.group))

    def __eq__(self, other: Function) -> bool:
        return self.label == other.label and \
            self.description == other.description and \
            self.group == other.group

    def __str__(self) -> str:
        return f'{self.label}: {self.description}'


@dataclass(eq=True, unsafe_hash=True)
class RoleLink(BankObject):
    """Contains links in to other source banks.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    uid: int = field(default=None, repr=False)
    """The database unique identifier."""

    cls: str = field(default=None)
    """The roleset's levine class, which is the ID in to the bank such as
    ``other_cos-45.4``.

    """
    resource: RoleResource = field(default=None)
    """The name of the source bank, such as ``VerbNet``."""

    version: str = field(default=None)
    """The version of the source bank, such as ``verbnet3.3``."""

    name: str = field(default=None)
    """The name of the role, such as ``agent`` or ``cause``."""

    def write(self, depth: int = 0, writer: TextIOBase = sys.stdout):
        dct = self.asdict()
        del dct['cls']
        self._write_line(self.cls, depth, writer)
        self._write_dict(dct, depth + 1, writer)


@dataclass
class RolesetId(BankObject):
    """A role set identifier identifier such as ``see.01`` or ``see-01``.  Note
    the later example is to support AMR formatted nodes.

    :see: :class:`.Roleset`

    :see: `Banarescu et al (2003) <https://aclanthology.org/W13-2322/>`_

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    _REGEX: ClassVar[re.Pattern] = re.compile(r'^(.+)[.-](\d{2})$')
    """The regular expresssion used to parse propbank IDs."""

    label: str = field(default=None)
    """The surface label of the ID (i.e. ``see-01``)."""

    lemma: Optional[str] = field(default=None)
    """The lemma part of the identifier (i.e. ``see`` from :obj:`label`
    ``see-01``).

    """
    index: Optional[int] = field(default=None)
    """The numeric index part of the identifier (i.e. ``1`` from :obj:`label`
    ``see-01``).

    """
    normalize: InitVar[bool] = field(default=True)
    """Whether to normalize the label."""

    def __post_init__(self, normalize: bool):
        if self.label is not None:
            m: re.Match = self._REGEX.match(self.label)
            if m is not None:
                self.lemma = m.group(1)
                self.index = int(m.group(2))
                if normalize:
                    self.label = f'{self.lemma}.{self.index:02}'

    @property
    def is_valid(self) -> bool:
        """Whether this is a valid formatted role set ID, which means it has
        both a :obj:`lemma` and :obj:`index`.

        """
        return self.lemma is not None and self.index is not None

    def __hash__(self) -> int:
        if self.is_valid:
            return hash((self.lemma, self.index))
        else:
            return hash(self.label)

    def __eq__(self, other: RolesetId) -> bool:
        if self.is_valid:
            return self.lemma == other.lemma and self.index == other.index
        else:
            return self.label == other.label

    def __str__(self) -> str:
        return self.label


@dataclass(eq=True, unsafe_hash=True)
class Reification(BankObject):
    """Reifications are a particular kind of transformation that replaces an
    edge relation with a new node and two outgoing edge relations, with one
    inverted.

    """
    concept: RolesetId = field()
    """The concept to add."""

    source_argument: int = field()
    """The source argument used to index."""

    target_argument: int = field()
    """The target argument to create."""


@dataclass(repr=False)
class Relation(BankObject):
    """Represents an AMR relation, which is a label on the edge an an AMR graph
    such as ``:ARG0-of``.  Note that a relation is often referred to as a *role*
    in the context of Penman notation.  However, you can think of an instance of
    role as an edge in am AMR graph as an instance of this class.

    """
    REGEX: ClassVar[re.Pattern] = re.compile(r'^:([^0-9-]+)(\d+)?(?:-(of))?$')
    """The regular expresssion used to parse AMR roles."""

    uid: int = field(default=None, repr=False)
    """The database unique identifier."""

    label: str = field(default=None)
    """The surface name of the relation (i.e. ``ARG`` from ``:ARG0-of``)."""

    type: str = field(default=None)
    """The type of relation (i.e. *general* for ``:ARG`` or *date* for
    ``time``).

    """
    description: str = field(default=None)
    """A somewhat human readable string describing the relation.  This is used
    to create the relation embeddings.

    """
    regex: re.Pattern = field(default=None)
    """A regular expression used to match role instances."""

    reification: Optional[Reification] = field(default=None)
    """The reification of the relation if any exist."""

    def match(self, role_str: str) -> Optional[re.Match]:
        """Return the regular match for this relaion on an AMR role instance
        string.

        """
        return self.regex.match(role_str)

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: RolesetId) -> bool:
        return self.label == other.label

    def __str__(self) -> str:
        return self.label


@dataclass(repr=False)
class Role(BankObject):
    """Defines an argument of the propbank role, which in AMR, has the syntax
    ``:ARG1`` for the second (0-index) second argument.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    uid: int = field(default=None, repr=False)
    """The database unique identifier."""

    description: str = field(default=None)
    """The human readable description of the role, such as (``Cause of
    hardening``).

    """
    function: Function = field(default=None)
    """The function of the role, such as ``PAG`` for ``prototypical agent``."""

    index: str = field(default=None)
    """The index of the role's argument, which is a a number, or an ``M`` for
    common adjuncts that don't qualify for number argument status.
    """
    role_links: Tuple[RoleLink] = field(default=None)
    """Links to the source banks for this role."""

    def __hash__(self) -> int:
        return hash(self.description, self.function, self.index)

    def __eq__(self, other: Function) -> bool:
        return self.function == other.function and \
            self.index == other.index and \
            self.description == other.description

    def __str__(self) -> str:
        return f'{self.function}({self.index}): {self.description}'


@dataclass
class Alias(BankObject):
    """Surface forms of the :obj:`.Roleset.lemma` and their part of speech.

    """
    uid: int = field(default=None, repr=False)
    """A unique identifier of the function."""

    part_of_speech: PartOfSpeech = field(default=None)
    """The part of speech of :obj:`word`."""

    word: str = field(default=None)
    """"Surface forms of the :meth:`.Roleset.lemma`."""


@dataclass(repr=False)
class PropBankArgumentSpan(BankObject):
    """An argument span used in a :class:`.PropBank`."""
    type: str = field()
    """The type (index) of argument (for example, ``ARG0``)."""

    span: Tuple[int, int] = field()
    """The 0-index inclusive token span in form ``(start, end)``.

    """
    token: str = field()
    """The token string selected by :obj:`span`."""


@dataclass(repr=False)
class PropBank(BankObject):
    """The PropBank annotations for the :class:`.Example`, which include token
    spans of the use of arguments.

    """
    UNKONWN_INDEX: ClassVar[int] = -1

    relative_indicies: Tuple[int] = field()
    """The 0-index index of the relative token in the example."""

    relative_tokens: Tuple[str] = field()
    """The relative token in example (for example, ``see``)."""

    argument_spans: Tuple[PropBankArgumentSpan] = field()
    """The spans of the arguments used in the example."""


@dataclass(repr=False)
class Example(BankObject):
    """Examples of the usage of the :class:`.Roleset`.

    """
    uid: int = field(default=None, repr=False)
    """The database unique identifier."""

    name: str = field(default=None)
    """The name, such as (``see-v: ARG0 and ARG1``)."""

    source: str = field(default=None)
    """The source of the example, such as (``ontonotes
    mz/sinorama/10/ectb_1057``).

    """
    text: str = field(default=None)
    """The text of the example, such as (``But recently many people...``)."""

    propbank: Optional[PropBank] = field(default=None)
    """The PropBank annotations for the example, which include token spans of
    the use of arguments.

    """


@dataclass(repr=False)
class Roleset(BankObject):
    """A bank role set entry that contains a grouping of :class:`.Role`s.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    uid: int = field(default=None, repr=False)
    """The database unique identifier."""

    id: RolesetId = field(default=None)
    """The ``*bank`` identifier of the role set, wuch as ``see.01``."""

    name: str = field(default=None)
    """The human readable short description of the role set, such as ``view``.

    """
    aliases: Tuple[Alias] = field(default=None)
    """The surfrace forms of the role set."""

    roles: Tuple[Role] = field(default=None)
    """The roles that define this set."""

    examples: Tuple[Example] = field(default=None)
    """The examples for the roleset."""

    def __hash__(self) -> int:
        return hash((self.id, self.name))

    def __eq__(self, other: RolesetId) -> bool:
        return self.id == other.id and \
            self.name == other.name

    def __str__(self) -> str:
        return f'{self.id}: {self.name}'


@dataclass(eq=True)
class Predicate(BankObject):
    """Contains the role sets for a lemmatized word.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    uid: int = field(default=None, repr=False, compare=False)
    """The database unique identifier."""

    lemma: str = field(default=None)
    """The lemmetized version of the word this role set describes, such as
    ``see``.

    """
    rolesets: Tuple[Roleset] = field(default=None)
    """The associated role sets for this predicate."""


@dataclass(eq=True)
class Frameset(BankObject):
    """Contains all the :class:`.Predicate` definitions from an file.

    """
    _DICTABLE_WRITABLE_DESCENDANTS: ClassVar[bool] = True

    uid: int = field(default=None, repr=False, compare=False)
    """The database unique identifier."""

    path: Path = field(default=None)
    """The file from which the definition was parsed."""

    predicates: Tuple[Predicate] = field(default=None)
    """The the role sets for a lemmatized word."""
