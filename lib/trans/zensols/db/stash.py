"""Adapt a database centric :class:`.DbPersister` to a general
:class:`~zensols.persist.Stash` container.

"""
__author__ = 'Paul Landes'

from typing import Any, Iterable, Tuple, Optional
from dataclasses import dataclass, field
from zensols.persist import Stash
from . import DBError, BeanDbPersister


@dataclass
class BeanStash(Stash):
    """A stash that uses a backing DB-API backed :class:`BeanDbPersister`.

    """
    persister: BeanDbPersister = field()
    """The delegate bean persister."""

    def load(self, name: str) -> Any:
        return self.persister.get_by_id(int(name))

    def exists(self, name: str) -> bool:
        try:
            name = int(name)
        except ValueError:
            # assume only number IDs
            return False
        return self.persister.exists(name)

    def dump(self, name: str, inst: Any):
        """Since this implementation can let the database auto-increment the
        unique/primary key, beware of "changing" keys.

        :raises DBError: if the key changes after inserted it will raise a
                ``DBError``; for this reason, it's best to pass ``None`` as
                ``name``

        """
        if name is not None:
            id = int(name)
            inst.id = id
        else:
            id = inst.id
        if id is not None and self.exists(id):
            self.persister.update(inst)
        else:
            self.persister.insert(inst)
        if id is not None and inst.id != id:
            raise DBError(f'unexpected key change: {inst.id} != {id}')
        return inst

    def delete(self, name: str):
        self.persister.delete(int(name))

    def keys(self) -> Iterable[str]:
        return map(str, self.persister.get_keys())

    def __len__(self) -> int:
        return self.persister.get_count()


@dataclass
class AlternateKeyBeanStash(BeanStash):
    """A stash that uses another key rather than some unique primary key
    (i.e. rowid for SQLite).  It does this by looking up the alternate key in
    some other column and resolves to the unique primary key.

    The domain and range of the function (:meth:`_key_to_id`) that maps
    alternate keys to unique primary keys ate strings.

    .. document private functions
    .. automethod:: _key_to_id

    """
    key_to_id_name: str = field()
    """The select method SQL name that selects the unique priamry to the
    alterante key.

    """
    keys_name: str = field()
    """The select method SQL name that selects the alternate in :meth:`keys`."""

    def _key_to_id(self, name: str) -> Optional[str]:
        """Maps alternate keys to unique primary keys.

        :param name: the alternate key, which is usually a more client friendly
                     string

        :return: the unique primary key in the database (usually an
                 :class:`int`)

        """
        row: Tuple = self.persister.execute_singleton_by_name(
            self.key_to_id_name, params=(name,),
            row_factory='identity')
        if row is not None:
            return str(row[0])

    def load(self, name: str) -> Any:
        return super().load(self._key_to_id(name))

    def exists(self, name: str) -> bool:
        id: Optional[Any] = self._key_to_id(name)
        return id is not None

    def dump(self, name: str, inst: Any):
        return super().dump(self._key_to_id(name), inst)

    def delete(self, name: str):
        return super().delete(self._key_to_id(name))

    def keys(self) -> Iterable[str]:
        return set(self.persister.execute_by_name(
            self.keys_name, row_factory='identity',
            map_fn=lambda r: r[0]))
