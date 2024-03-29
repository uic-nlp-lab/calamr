"""Convenience wrapper for the Python DB-API library, and some specificly for
the SQLite library.

"""
__author__ = 'Paul Landes'

import logging
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from zensols.db import DBError, ConnectionManager

logger = logging.getLogger(__name__)


@dataclass
class SqliteConnectionManager(ConnectionManager):
    """An SQLite connection factory.

    """
    db_file: Path = field()
    """The SQLite database file to read or create."""

    create_db: bool = field(default=True)
    """If ``True``, create the database if it does not already exist.  Otherwise,
    :class:`.DBError` is raised (see :meth:`create`).

    """
    def create(self) -> sqlite3.Connection:
        """Create a connection by accessing the SQLite file.

        :raise DBError: if the SQLite file does not exist (caveat see
                        :`obj:create_db`)

        """
        db_file = self.db_file
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'creating connection to {db_file}')
        created = False
        if not db_file.exists():
            if not self.create_db:
                raise DBError(f'database file {db_file} does not exist')
            if not db_file.parent.exists():
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f'creating sql db directory {db_file.parent}')
                db_file.parent.mkdir(parents=True)
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'creating sqlite db file: {db_file}')
            created = True
        types = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        conn = sqlite3.connect(str(db_file.absolute()), detect_types=types)
        if created:
            logger.info('initializing database...')
            for sql in self.persister.parser.get_init_db_sqls():
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'invoking sql: {sql}')
                conn.execute(sql)
                conn.commit()
        return conn

    def drop(self):
        """Delete the SQLite database file from the file system.

        """
        logger.info(f'deleting: {self.db_file}')
        if self.db_file.exists():
            self.db_file.unlink()
            return True
        return False
