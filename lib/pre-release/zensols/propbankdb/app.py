"""An API to access the frameset database and generate embeddings from them.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import re
from zensols.persist import Stash
from zensols.config import ConfigFactory, Configurable
from zensols.cli import ApplicationError
from . import BankObject, RolesetId, Database

logger = logging.getLogger(__name__)


class Format(Enum):
    """The format of *show* type CLI actions.

    """
    text = auto()
    json = auto()


@dataclass
class Application(object):
    """Access the Frameset database and generate embeddings from them.

    """
    config_factory: ConfigFactory = field()
    """Used to get the metadata configuration from the install."""

    db: Database = field()
    """A data access object for all frame set data."""

    def search_roleset(self, pattern: str):
        """Find role set IDs.

        :param pattern: the regular expression to match role set IDs

        """
        regex: re.Pattern = re.compile(pattern)
        for rid in self.db.roleset_stash.keys():
            if regex.match(rid) is not None:
                print(rid)

    def _show(self, stash: Stash, key: str, format: Format, key_name: str):
        if key not in stash:
            raise ApplicationError(f'{key_name.capitalize()}: {key}')
        else:
            rs: BankObject = stash[key]
            if format == Format.text:
                rs.write()
            elif format == Format.json:
                print(rs.asjson(indent=4))

    def predicate(self, lemma: str, format: Format = Format.text):
        """Dump a role set.

        :param id: the lemma of the predicate (i.e. ``see``)

        :param format: the format of the output

        """
        self._show(self.db.predicate_stash, lemma, format, 'predicate lemma')

    def roleset(self, id: str, format: Format = Format.text):
        """Dump a role set.

        :param id: the role set ID (i.e. ``see.01``)

        :param format: the format of the output

        """
        rid = RolesetId(id)
        self._show(self.db.roleset_stash, str(rid), format, 'role id')

    def info(self) -> Configurable:
        """Print the installed distribution library inforamtion.

        """
        from zensols.propbankdb.embedpop import EmbeddingPopulator
        emb_pop: EmbeddingPopulator = self.config_factory(
            'pbdb_embedding_populator')
        config: Configurable = emb_pop.get_metadata_config(True)
        config.write()
        return config
