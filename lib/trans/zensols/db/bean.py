"""Basic CRUD utility classes

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import (
    Dict, Any, Tuple, Union, Callable, Iterable, Type, Optional, List
)
from dataclasses import dataclass, field, fields
from abc import abstractmethod, ABC
import logging
import traceback
from pathlib import Path
import pandas as pd
from zensols.persist import resource, chunks
from zensols.db import DynamicDataParser

logger = logging.getLogger(__name__)


class DBError(Exception):
    """"Raised for all :mod:`zensols.db`` related errors.

    """
    pass


class connection(resource):
    """Annotation used to create and dispose of DB-API connections.

    """
    def __init__(self):
        super().__init__('_create_connection', '_dispose_connection')


class _CursorIterator(object):
    """Iterates throw the rows of the database using a cursor.

    """
    def __init__(self, mng: ConnectionManager, conn: Any, cursor: Any):
        """

        :param mng: the connection manager to regulate database resources

        :param conn: the connection to the database

        :param cursor: the cursor to the database

        """
        self._mng = mng
        self._conn = conn
        self._cursor = cursor

    def __iter__(self) -> _CursorIterator:
        return self

    def __next__(self):
        if self._cursor is None:
            raise StopIteration
        try:
            return next(self._cursor)
        except StopIteration:
            try:
                self.dispose()
            finally:
                raise StopIteration

    def dispose(self):
        if self._mng is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('closing cursor iterable')
            self._mng._do_dispose_connection = True
            self._cursor.close()
            self._mng.dispose(self._conn)
            self._mng = None
            self._conn = None
            self._cursor = None


class cursor(object):
    """Iterate through rows of a database.  The connection is automatically
    closed once out of scope.

    Example::

        config_factory: ConfigFactory = ...
        persister: DbPersister = config_factory.instance('person_db_persister')
        with cursor(persister, name='select_people') as c:
            for row in c:
                print(row)

    """
    def __init__(self, persister: DbPersister, sql: str = None,
                 name: str = None, params: Tuple[Any, ...] = ()):
        """Initialize with either ``name`` or ``sql`` (only one should be
        ``None``).

        :param persister: used to execute the SQL and obtain the cursor

        :param sql: the string SQL to execute

        :param name: the named SQL query in the :obj:`.DbPersister.sql_file`

        :param params: the parameters given to the SQL statement (populated
                       with ``?``) in the statement

        """
        self._curiter = persister._execute_iterate(
            sql=sql,
            name=name,
            params=params)

    def __enter__(self) -> Iterable[Any]:
        return self._curiter

    def __exit__(self, cls: Type[Exception], value: Optional[Exception],
                 trace: traceback):
        self._curiter.dispose()


@dataclass
class ConnectionManager(ABC):
    """Instance DB-API connection lifecycle.

    """
    def __post_init__(self):
        self._do_dispose_connection = True

    def register_persister(self, persister: DbPersister):
        """Register the persister used for this connection manager.

        :param persister: the persister used for connection management

        """
        self.persister = persister

    @abstractmethod
    def create(self) -> Any:
        """Create a connection to the database.

        """
        pass

    def dispose(self, conn):
        """Close the connection to the database.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'connection manager: closing {conn}')
        if self._do_dispose_connection:
            conn.close()

    @abstractmethod
    def drop(self):
        """Remove all objects from the database or the database itself.

        For SQLite, this deletes the file.  In database implementations, this
        might drop all objects from the database.  Regardless, it is expected
        that ``create`` is able to recreate the database after this action.

        """
        pass

    def _to_dataframe(self, res: Iterable[Any], cursor: Any) -> pd.DataFrame:
        """Return a Pandas dataframe from the results given by the database.

        :param res: the database results row by row

        :param cursor: the database cursor object, which has a ``description``
                       attribute

        """
        cols = tuple(map(lambda d: d[0], cursor.description))
        return pd.DataFrame(res, columns=cols)

    def execute(self, conn: Any, sql: str, params: Tuple[Any, ...],
                row_factory: Union[str, Callable],
                map_fn: Callable) -> Tuple[Union[dict, tuple, pd.DataFrame]]:
        """Execute SQL on a database connection.

        The ``row_factory`` tells the method how to interpret the row data in
        to an object that's returned.  It can be one of:

            * ``tuple``: tuples (the default)

            * ``identity``: return the unmodified form from the database

            * ``dict``: for dictionaries

            * ``pandas``: for a :class:`pandas.DataFrame`

            * otherwise: a function or class

        Compare this with ``map_fn``, which transforms the data that's given to
        the ``row_factory``.

        :param conn: the connection object with the database

        :param sql: the string SQL to execute

        :param params: the parameters given to the SQL statement (populated
                       with ``?``) in the statement

        :param row_factory: ``tuple``, ``dict``, ``pandas`` or a function

        :param map_fn: a function that transforms row data given to the
                       ``row_factory``

        :see: :meth:`.DbPersister.execute`.

        """
        def dict_row_factory(cursor: Any, row: Tuple[Any, ...]):
            return dict(map(lambda x: (x[1][0], row[x[0]]),
                            enumerate(cursor.description)))

        conn.row_factory = {
            'dict': dict_row_factory,
            'tuple': lambda cursor, row: row,
            'identity': lambda cursor, row: row,
            'pandas': None,
        }.get(
            row_factory,
            lambda cursor, row: row_factory(*row)
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'sql: <{sql}>, params: {params}')
        cur: Any = conn.cursor()
        try:
            res = cur.execute(sql, params)
            if map_fn is not None:
                res = map(map_fn, res)
            if row_factory == 'pandas':
                res = self._to_dataframe(res, cur)
            if conn.row_factory is not None:
                res = tuple(res)
            return res
        finally:
            cur.close()

    def _create_cursor(self, conn: Any, sql: str,
                       params: Tuple[Any, ...]) -> Any:
        """Create a cursor object from connection ``conn``."""
        cur: Any = conn.cursor()
        cur.execute(sql, params)
        return cur

    def execute_no_read(self, conn: Any, sql: str,
                        params: Tuple[Any, ...]) -> int:
        """Return database level information such as row IDs rather than the
        results of a query.  Use this when inserting data to get a row ID.

        :param conn: the connection object with the database

        :param sql: the SQL statement used on the connection's cursor

        :param params: the parameters given to the SQL statement (populated
                       with ``?``) in the statement

        :see: :meth:`.DbPersister.execute_no_read`.

        """
        cur = conn.cursor()
        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'sql: {sql}, params: {params}')
            cur.execute(sql, params)
            conn.commit()
            return cur.lastrowid
        finally:
            cur.close()

    def insert_rows(self, conn: Any, sql: str, rows: Iterable[Any],
                    errors: str, set_id_fn: Callable,
                    map_fn: Callable) -> int:
        """Insert a tuple of rows in the database and return the current row ID.

        :param rows: a sequence of tuples of data (or an object to be
                     transformed, see ``map_fn`` in column order of the SQL
                     provided by the entry :obj:`insert_name`

        :param errors: if this is the string ``raise`` then raise an error on
                       any exception when invoking the database execute,
                       otherwise use ``ignore`` to ignore errors

        :param set_id_fn: a callable that is given the data to be inserted and
                          the row ID returned from the row insert as parameters

        :param map_fn: if not ``None``, used to transform the given row in to a
                       tuple that is used for the insertion

        :return: the ``rowid`` of the last row inserted

        See :meth:`.InsertableBeanDbPersister.insert_rows`.

        """
        cur = conn.cursor()
        try:
            for row in rows:
                if map_fn is not None:
                    org_row = row
                    row = map_fn(row)
                if errors == 'raise':
                    cur.execute(sql, row)
                elif errors == 'ignore':
                    try:
                        cur.execute(sql, row)
                    except Exception as e:
                        logger.error(f'could not insert row ({len(row)})', e)
                else:
                    raise DBError(f'unknown errors value: {errors}')
                if set_id_fn is not None:
                    set_id_fn(org_row, cur.lastrowid)
        finally:
            conn.commit()
            cur.close()
        return cur.lastrowid


@dataclass
class DbPersister(object):
    """CRUDs data to/from a DB-API connection.

    """
    conn_manager: ConnectionManager = field()
    """Used to create DB-API connections."""

    sql_file: Path = field(default=None)
    """The text file containing the SQL statements (see
    :class:`DynamicDataParser`).

    """
    row_factory: Union[str, Type] = field(default='tuple')
    """The default method by which data is returned from ``execute_*`` methods.

    :see: :meth:`execute`.

    """
    def __post_init__(self):
        self.parser = self._create_parser(self.sql_file)
        self.conn_manager.register_persister(self)

    def _create_parser(self, sql_file: Path) -> DynamicDataParser:
        return DynamicDataParser(sql_file)

    @property
    def sql_entries(self) -> Dict[str, str]:
        """Return a dictionary of names -> SQL statements from the SQL file.

        """
        return self.parser.sections

    @property
    def metadata(self) -> Dict[str, str]:
        """Return the metadata associated with the SQL file.

        """
        return self.parser.metadata

    def _create_connection(self):
        """Create a connection to the database.

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('creating connection')
        return self.conn_manager.create()

    def _dispose_connection(self, conn: Any):
        """Close the connection to the database.

        :param conn: the connection to release

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'closing connection {conn}')
        self.conn_manager.dispose(conn)

    def _check_entry(self, name: str):
        if name is None:
            raise DBError('no defined SQL entry for persist function')
        if len(name) == 0:
            raise DBError('non-optional entry not provided')
        if name not in self.sql_entries:
            raise DBError(f"no entry '{name}' found in SQL configuration")

    def _get_entry(self, name: str):
        self._check_entry(name)
        return self.sql_entries[name]

    @connection()
    def execute(self, conn: Any, sql: str, params: Tuple[Any, ...] = (),
                row_factory: Union[str, Callable] = None,
                map_fn: Callable = None) -> \
            Tuple[Union[dict, tuple, pd.DataFrame]]:
        """Execute SQL on a database connection.

        The ``row_factory`` tells the method how to interpret the row data in
        to an object that's returned.  It can be one of:

            * ``tuple``: tuples (the default)
            * ``dict``: for dictionaries
            * ``pandas``: for a :class:`pandas.DataFrame`
            * otherwise: a function or class

        Compare this with ``map_fn``, which transforms the data that's given to
        the ``row_factory``.

        :param sql: the string SQL to execute

        :param params: the parameters given to the SQL statement (populated
                       with ``?``) in the statement

        :param row_factory: ``tuple``, ``dict``, ``pandas`` or a function

        :param map_fn: a function that transforms row data given to the
                       ``row_factory``

        """
        row_factory = self.row_factory if row_factory is None else row_factory
        return self.conn_manager.execute(
            conn, sql, params, row_factory, map_fn)

    def execute_by_name(self, name: str, params: Tuple[Any] = (),
                        row_factory: Union[str, Callable] = None,
                        map_fn: Callable = None):
        """Just like :meth:`execute` but look up the SQL statement to execute on
        the database connection.

        The ``row_factory`` tells the method how to interpret the row data in
        to an object that's returned.  It can be one of:

            * ``tuple``: tuples (the default)
            * ``dict``: for dictionaries
            * ``pandas``: for a :class:`pandas.DataFrame`
            * otherwise: a function or class

        Compare this with ``map_fn``, which transforms the data that's given to
        the ``row_factory``.

        :param name: the named SQL query in the :obj:`sql_file`

        :param params: the parameters given to the SQL statement (populated
                       with ``?``) in the statement

        :param row_factory: ``tuple``, ``dict``, ``pandas`` or a function

        :param map_fn: a function that transforms row data given to the
                       ``row_factory``

        :see: :meth:`execute`

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'name: <{name}>, params: {params}')
        self._check_entry(name)
        sql = self.sql_entries[name]
        return self.execute(sql, params, row_factory, map_fn)

    @connection()
    def _execute_iterate(self, conn: Any, sql: str, name: str,
                         params: Tuple[Any, ...]):
        if sql is None and name is None:
            raise DBError('Both sql string and name can not be None')
        if sql is None:
            self._check_entry(name)
            sql = self.sql_entries[name]
        cur = self.conn_manager._create_cursor(conn, sql, params)
        self.conn_manager._do_dispose_connection = False
        return _CursorIterator(self.conn_manager, conn, cur)

    def execute_singleton_by_name(self, *args, **kwargs):
        """Just like :meth:`execute_by_name` except return only the first item or
        ``None`` if no results.

        """
        res = self.execute_by_name(*args, **kwargs)
        if len(res) > 0:
            return res[0]

    @connection()
    def execute_sql_no_read(self, conn: Any, sql: str,
                            params: Tuple[Any] = ()) -> int:
        """Execute SQL and return the database level information such as row IDs
        rather than the results of a query.  Use this when inserting data to get
        a row ID.

        """
        return self.conn_manager.execute_no_read(conn, sql, params)

    @connection()
    def execute_no_read(self, conn: Any, entry_name: str,
                        params: Tuple[Any] = ()) -> int:
        """Just like :meth:`execute_by_name`, but return database level
        information such as row IDs rather than the results of a query.  Use
        this when inserting data to get a row ID.

        :param entry_name: the key in the SQL file whose value is used as the
                           statement

        :param capture_rowid: if ``True``, return the last row ID from the
                              cursor

        :see: :meth:`execute_sql_no_read`

        """
        self._check_entry(entry_name)
        sql = self.sql_entries[entry_name]
        return self.conn_manager.execute_no_read(conn, sql, params)


@dataclass
class Bean(ABC):
    """A container class like a Java *bean*.

    """
    def get_attr_names(self) -> Tuple[str]:
        """Return a list of string attribute names.

        """
        return tuple(map(lambda f: f.name, fields(self)))

    def get_attrs(self) -> Dict[str, Any]:
        """Return a dict of attributes that are meant to be persisted.

        """
        return {n: getattr(self, n) for n in self.get_attr_names()}

    def get_row(self) -> Tuple[Any]:
        """Return a row of data meant to be printed.  This includes the unique ID of
        the bean (see :meth:`get_insert_row`).

        """
        return tuple(map(lambda x: getattr(self, x), self.get_attr_names()))

    def get_insert_row(self) -> Tuple[Any]:
        """Return a row of data meant to be inserted into the database.  This method
        implementation leaves off the first attriubte assuming it contains a
        unique (i.e. row ID) of the object.  See :meth:`get_row`.

        """
        names = self.get_attr_names()
        return tuple(map(lambda x: getattr(self, x), names[1:]))

    def __eq__(self, other):
        if other is None:
            return False
        if self is other:
            return True
        if self.__class__ != other.__class__:
            return False
        for n in self.get_attr_names():
            if getattr(self, n) != getattr(other, n):
                return False
        return True

    def __hash__(self):
        vals = tuple(map(lambda n: getattr(self, n), self.get_attr_names()))
        return hash(vals)

    def __str__(self):
        return ', '.join(map(lambda x: f'{x}: {getattr(self, x)}',
                             self.get_attr_names()))

    def __repr__(self):
        return self.__str__()


@dataclass
class ReadOnlyBeanDbPersister(DbPersister):
    """A read-only persister that CRUDs data based on predefined SQL given in the
    configuration.  The class optionally works with instances of :class:`.Bean`
    when :obj:`row_factory` is set to the target bean class.

    """
    select_name: str = field(default=None)
    """The name of the SQL entry used to select data/class."""

    select_by_id_name: str = field(default=None)
    """The name of the SQL entry used to select a single row by unique ID."""

    select_exists_name: str = field(default=None)
    """The name of the SQL entry used to determine if a row exists by unique
    ID.

    """

    def get(self) -> list:
        """Return using the SQL provided by the entry identified by :obj:`select_name`.

        """
        return self.execute_by_name(
            self.select_name, row_factory=self.row_factory)

    def get_by_id(self, id: int):
        """Return an object using it's unique ID, which is could be the row ID in
        SQLite.

        """
        rows = self.execute_by_name(
            self.select_by_id_name, params=(id,), row_factory=self.row_factory)
        if len(rows) > 0:
            return rows[0]

    def exists(self, id: int) -> bool:
        """Return ``True`` if there is a object with unique ID (or row ID) in the
        database.  Otherwise return ``False``.

        """
        if self.select_exists_name is None:
            return self.get_by_id(id) is not None
        else:
            cnt = self.execute_by_name(
                self.select_exists_name, params=(id,), row_factory='tuple')
            return cnt[0][0] == 1


@dataclass
class InsertableBeanDbPersister(ReadOnlyBeanDbPersister):
    """A class that contains insert funtionality.

    """
    insert_name: str = field(default=None)
    """The name of the SQL entry used to insert data/class instance."""

    def insert_row(self, *row) -> int:
        """Insert a row in the database and return the current row ID.

        :param row: a sequence of data in column order of the SQL provided by
                    the entry :obj:`insert_name`

        """
        return self.execute_no_read(self.insert_name, params=row)

    @connection()
    def insert_rows(self, conn: Any, rows: Iterable[Any], errors: str = 'raise',
                    set_id_fn: Callable = None, map_fn: Callable = None) -> int:
        """Insert a tuple of rows in the database and return the current row ID.

        :param rows: a sequence of tuples of data (or an object to be
                     transformed, see ``map_fn`` in column order of the SQL
                     provided by the entry :obj:`insert_name`

        :param errors: if this is the string ``raise`` then raise an error on
                       any exception when invoking the database execute

        :param set_id_fn: a callable that is given the data to be inserted and
                          the row ID returned from the row insert as parameters

        :param map_fn: if not ``None``, used to transform the given row in to a
                       tuple that is used for the insertion

        :return: the ``rowid`` of the last row inserted

        """
        entry_name = self.insert_name
        self._check_entry(entry_name)
        sql = self.sql_entries[entry_name]
        return self.conn_manager.insert_rows(
            conn, sql, rows, errors, set_id_fn, map_fn)

    @connection()
    def insert_dataframe(self, conn: Any, df: pd.DataFrame,
                         errors: str = 'raise', set_id_fn: Callable = None,
                         map_fn: Callable = None, chunk_size: int = 100) -> int:
        """Like :meth:`insert_rows` but the data is taken a Pandas dataframe.

        :param df: the dataframe from which the rows are drawn

        :param set_id_fn: a callable that is given the data to be inserted and
                          the row ID returned from the row insert as parameters

        :param map_fn: if not ``None``, used to transform the given row in to a
                       tuple that is used for the insertion

        :param chuck_size: the number of rows inserted at a time, so the number
                           of interactions with the database are at most the row
                           count of the dataframe / ``chunk_size``x

        :return: the ``rowid`` of the last row inserted

        """
        entry_name: str = self.insert_name
        sql: str = self._get_entry(entry_name)
        row_id: int
        rows: List[Tuple[Any, ...]]
        for rows in chunks(df.itertuples(name=None, index=False), chunk_size):
            row_id = self.conn_manager.insert_rows(
                conn, sql, rows, errors, set_id_fn, map_fn)
        return row_id

    def _get_insert_row(self, bean: Bean) -> Tuple[Any]:
        """Factory method to return the bean's insert row parameters."""
        return bean.get_insert_row()

    def insert(self, bean: Bean) -> int:
        """Insert a bean using the order of the values given in
        :meth:`Bean.get_insert_row` as that of the SQL defined with entry
        :obj:`insert_name` given in the initializer.

        """
        row = self._get_insert_row(bean)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'inserting row: {row}')
        curid = self.insert_row(*row)
        bean.id = curid
        return curid

    def insert_beans(self, beans: Iterable[Any], errors: str = 'raise') -> int:
        """Insert a bean using the order of the values given in
        :meth:`Bean.get_insert_row` as that of the SQL defined with entry
        :obj:`insert_name` given in the initializer.

        """
        def map_fn(bean):
            return self._get_insert_row(bean)

        def set_id_fn(bean, id):
            pass

        return self.insert_rows(beans, errors, set_id_fn, map_fn)


@dataclass
class UpdatableBeanDbPersister(InsertableBeanDbPersister):
    """A class that contains the remaining CRUD funtionality the super class
    doesn't have.

    """
    update_name: str = field(default=None)
    """The name of the SQL entry used to update data/class instance(s)."""

    delete_name: str = field(default=None)
    """The name of the SQL entry used to delete data/class instance(s)."""

    def update_row(self, *row: Tuple[Any]) -> int:
        """Update a row using the values of the row with the current unique ID as the
        first element in ``*rows``.

        """
        where_row = (*row[1:], row[0])
        return self.execute_no_read(self.update_name, params=where_row)

    def update(self, bean: Bean) -> int:
        """Update a a bean that using the ``id`` attribute and its attributes as
        values.

        """
        return self.update_row(*bean.get_row())

    def delete(self, id) -> int:
        """Delete a row by ID.

        """
        return self.execute_no_read(self.delete_name, params=(id,))


@dataclass
class BeanDbPersister(UpdatableBeanDbPersister):
    """A class that contains the remaining CRUD funtionality the super class
    doesn't have.

    """
    keys_name: str = field(default=None)
    """The name of the SQL entry used to fetch all keys."""

    count_name: str = field(default=None)
    """The name of the SQL entry used to get a row count."""

    def get_keys(self) -> Iterable[Any]:
        """Return the unique keys from the bean table.

        """
        keys = self.execute_by_name(self.keys_name, row_factory='tuple')
        return map(lambda x: x[0], keys)

    def get_count(self) -> int:
        """Return the number of rows in the bean table.

        """
        if self.count_name is not None:
            cnt = self.execute_by_name(self.count_name, row_factory='tuple')
            return cnt[0][0]
        else:
            # SQLite has a bug that returns one row with all null values
            return sum(1 for _ in self.get_keys())
