"""Loads parsed XML files to an SQLite database.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Dict, Tuple, Union, Iterable, Optional, List, Any
from dataclasses import dataclass, field
import dataclasses as dc
from abc import ABCMeta, abstractmethod
import logging
import re
import json
from pathlib import Path
from zensols.persist import persisted, Stash
from zensols.db import (
    ConnectionManager, BeanDbPersister, SqliteConnectionManager
)
from zensols.install import Installer
from . import (
    BankObject, BankObjectPopulator, RoleResource, Function, PartOfSpeech,
    PropBankArgumentSpan, PropBank, Example, RolesetId, Role, Alias,
    Roleset, RoleLink, Predicate, Frameset, Relation, Reification,
)

logger = logging.getLogger(__name__)


@dataclass
class InstallerConnectionManager(SqliteConnectionManager):
    """A connection manager that first downloads the distribution SQLite
    Propbankdb file.

    """
    installer: Optional[Installer] = field(default=None)
    """The distribution file resource downloader.  If this is set, it is
    downloaded, uncompressed and :obj:`conn_manager`s ``db_file`` parent path
    set to the resource.

    """
    def __post_init__(self):
        super().__post_init__()
        if self.installer is not None:
            self.installer()
            fname: str = self.db_file.name
            inst_path: Path = self.installer.get_singleton_path() / fname
            if not inst_path.is_file():
                logger.warning(f'missing Frameset SQLite file: {inst_path}')
            else:
                self.db_file = inst_path


@dataclass
class FunctionPersister(BeanDbPersister):
    """Utility persister to access :class:`.Function`s by label and ID.  This is
    a somewhat like a GoF flyweight pattern in that it attempts to minimize the
    memory footprint.

    """
    def __post_init__(self):
        super().__post_init__()
        self._by_id: Dict[int, Function] = None
        self._by_label: Dict[str, Function] = None

    @persisted('_dicts_pw')
    def _dicts(self) -> Tuple[Dict]:
        funcs: Tuple[Function] = super().get()
        by_id = {f.uid: f for f in funcs}
        by_label = {f.label: f for f in funcs}
        return by_id, by_label

    def get(self) -> Tuple[Function]:
        return tuple(self._dicts()[0].values())

    def get_by_id(self, uid: int) -> Function:
        return self._dicts()[0][uid]

    def get_by_label(self, label: str) -> Function:
        return self._dicts()[1][label]


@dataclass
class BankObjectPersister(BeanDbPersister, metaclass=ABCMeta):
    """Utility methods to de-hydrate frame set objects from the datbase.

    """
    _db: Database = field(default=None)
    """The database used to access other persisters."""

    populators: List[BankObjectPopulator] = field(default_factory=list)
    """The list of other populators to invoke on :meth:`_populate`."""

    @abstractmethod
    def _populate(self, bank_object: BankObject):
        """Populate ``bank_object`` after it was just de-hydrated from the
        database.

        """
        pass

    def add_populator(self, pop: BankObjectPopulator):
        self.populators.append(pop)

    def _post_populate(self, bank_object: BankObject):
        pop: BankObjectPopulator
        for pop in self.populators:
            pop.populate(bank_object)

    def get_by_id(self, uid: int) -> BankObject:
        bo: BankObject = super().get_by_id(uid)
        self._populate(bo)
        return bo

    def _get_by_parent_id(self, uid: int) -> Tuple[BankObject]:
        bean_name = self.row_factory.__name__.lower()
        sel_name: str = f'select_{bean_name}_by_par_id'
        bos: Tuple[BankObject] = self.execute_by_name(sel_name, params=(uid,))
        for bo in bos:
            self._populate(bo)
            self._post_populate(bo)
        return bos

    def iterobj(self) -> Iterable[BankObject]:
        """Iterate through the object instances from the database."""
        return map(self.get_by_id, self.get_keys())


@dataclass
class RelationPersister(BankObjectPersister):
    """Utility persister to access :class:`.Relation`s by label and ID.  This is
    a somewhat like a GoF flyweight pattern in that it attempts to minimize the
    memory footprint.

    """
    def __post_init__(self):
        super().__post_init__()
        self._by_id: Dict[int, Relation] = None
        self._by_label: Dict[str, Relation] = None

    def _populate(self, rel: Relation):
        rel.regex = re.compile(rel.regex)

    @persisted('_dicts_pw')
    def _dicts(self) -> Tuple[Dict]:
        def _map_rel(rel: Relation) -> Relation:
            if rel.reification is not None:
                r = Reification(*eval(rel.reification))
                r.concept = RolesetId(r.concept, normalize=False)
                rel.reification = r

        rels: Tuple[Relation] = super().get()
        rel: Relation
        for rel in rels:
            _map_rel(rel)
            self._populate(rel)
            self._post_populate(rel)
        by_id = {f.uid: f for f in rels}
        by_label = {f.label: f for f in rels}
        return by_id, by_label

    def get(self) -> Tuple[Function]:
        return tuple(self._dicts()[0].values())

    def get_by_id(self, uid: int) -> Function:
        return self._dicts()[0][uid]

    def get_by_label(self, label: str) -> Function:
        return self._dicts()[1][label]


@dataclass
class RoleLinkPersister(BankObjectPersister):
    def _populate(self, role_link: RoleLink):
        role_link.resource = RoleResource(role_link.resource)
        self._post_populate(role_link)


@dataclass
class RolePersister(BankObjectPersister):
    def _populate(self, r: Role):
        r.role_links = self._db.role_link_persister._get_by_parent_id(r.uid)
        r.function = self._db.function_persister.get_by_id(r.function)
        self._post_populate(r)


@dataclass
class AliasPersister(BankObjectPersister):
    def _populate(self, alias: Alias):
        alias.part_of_speech = PartOfSpeech(alias.part_of_speech)
        self._post_populate(alias)


@dataclass
class ExamplePersister(BankObjectPersister):
    def _propbank(self, data: Dict[str, Any]) -> PropBank:
        pb = PropBank(**data)
        pb.argument_spans = tuple(
            map(lambda s: PropBankArgumentSpan(**s), pb.argument_spans))
        return pb

    def _populate(self, example: Example):
        if example.propbank is not None:
            example.propbank = self._propbank(json.loads(example.propbank))
        self._post_populate(example)


@dataclass
class RolesetPersister(BankObjectPersister):
    def _populate(self, rs: Roleset):
        rs.roles = self._db.role_persister._get_by_parent_id(rs.uid)
        rs.aliases = self._db.alias_persister._get_by_parent_id(rs.uid)
        rs.id = RolesetId(rs.id)
        rs.examples = self._db.example_persister._get_by_parent_id(rs.uid)
        self._post_populate(rs)

    def get_by_role_id(self, role_id: Union[RolesetId, str]) -> Roleset:
        rs: Roleset = self.execute_singleton_by_name(
            'select_roleset_by_role_id', params=(str(role_id),))
        self._populate(rs)
        return rs


@dataclass
class PredicatePersister(BankObjectPersister):
    def _populate(self, pred: Predicate):
        pred.rolesets = self._db.roleset_persister._get_by_parent_id(pred.uid)
        self._post_populate(pred)

    def get_by_lemma(self, lemma: str) -> Predicate:
        pred: Predicate = self.execute_singleton_by_name(
            'select_predicate_by_lemma', params=(lemma,))
        self._populate(pred)
        return pred


@dataclass
class FramesetPersister(BankObjectPersister):
    def _populate(self, fs: Frameset):
        fs.predicates = self._db.predicate_persister._get_by_parent_id(fs.uid)
        self._post_populate(fs)

    def get_by_file_name(self, file_name: str) -> Frameset:
        fs: Frameset = self.execute_singleton_by_name(
            'select_frameset_by_file_name', params=(file_name,))
        self._populate(fs)
        return fs


@dataclass
class Database(object):
    """A data access object for all frame set data.  This provides access to the
    :mod:`loader`, which parses the XML and loads it in to the database.  It
    also provides methods to *re-hydrate* object instances from the database.

    **Important implementation note:** Stash references need to be obtained from
    this instance rather than directly from the
    :class:`~zensols.config.facbase.ConfigFactory`, otherwise it will not be
    correctly initialized.

    """
    conn_manager: ConnectionManager = field()
    """The relational database (SQLite only for now) connection manager.

    :see: :obj:`installer`

    """
    # persisters
    frameset_persister: FramesetPersister = field()
    """Persists instances of :class:`.Frameset."""

    predicate_persister: PredicatePersister = field()
    """Persists instances of :class:`.Predicate."""

    roleset_persister: RolesetPersister = field()
    """Persists instances of :class:`.Roleset."""

    alias_persister: AliasPersister = field()
    """Persists instances of :class:`.Alias."""

    example_persister: ExamplePersister = field()
    """Persists instances of :class:`.Example."""

    role_persister: RolePersister = field()
    """Persists instances of :class:`.Role"""

    role_link_persister: RoleLinkPersister = field()
    """Persists instances of :class:`.RoleLink."""

    function_persister: FunctionPersister = field()
    """Persists instances of :class:`.Function."""

    relation_persister: RelationPersister = field()
    """Persists instances of :class:`.Relation."""

    # stashes
    roleset_stash: Stash = field()
    """A stash adapatation of :obj:`roleset_persister`."""

    predicate_stash: Stash = field()
    """A stash adapatation of :obj:`predicate_persister`."""

    relation_stash: Stash = field()
    """A stash adapatation of :obj:`relation_persister`."""

    def __post_init__(self):
        # add self reference to all persisters so they can access eachother
        fields: dc.Field = filter(
            lambda f: isinstance(getattr(self, f.name), BeanDbPersister),
            dc.fields(self))
        for f in fields:
            persister: BeanDbPersister = getattr(self, f.name)
            persister._db = self

    def get_frameset(self, file_name: str) -> Frameset:
        """Return the Frameset that was parsed by ``file_name``."""
        return self.frameset_persister.get_by_file_name(file_name)

    def get_framesets(self) -> Iterable[Frameset]:
        """Return an iterable of Framesets."""
        return self.frameset_persister.iterobj()

    def get_predicate(self, lemma: str) -> Predicate:
        """Return the predicate for ``lemma`."""
        return self.predicate_persister.get_by_lemma(lemma)

    def get_predicates(self) -> Iterable[Predicate]:
        """Return an iterable of predicates."""
        return self.predicate_persister.iterobj()
