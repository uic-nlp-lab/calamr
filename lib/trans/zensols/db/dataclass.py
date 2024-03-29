"""Contains utility classes to persist :class:`dataclasses.dataclass`.

"""
__author__ = 'Paul Landes'

from typing import Type, List, Tuple, Any, ClassVar
from dataclasses import dataclass, field, fields
import dataclasses
import logging
from string import Template
from pathlib import Path
from . import DynamicDataParser, BeanDbPersister, Bean

logger = logging.getLogger(__name__)


class DataClassDynamicDataParser(DynamicDataParser):
    """An SQL data parser that replaces ``${cols}`` in the SQL file with the
    :class:`dataclasses.dataclass` fields.

    :see: :class:`.DataClassDbPersister`

    """
    ID_FIELD: ClassVar[str] = 'id'
    """The name of the column that has the unique identifier of the row/object.

    """
    def __init__(self, dd_path: Path, bean_class: Type):
        super().__init__(dd_path)
        if not dataclasses.is_dataclass(bean_class):
            raise ValueError(f'not a dataclass: {bean_class}')
        cols = map(lambda f: f.name, dataclasses.fields(bean_class))
        cols = ', '.join(filter(lambda c: c != self.ID_FIELD, cols))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'cols: {cols}')
        self.context = {'cols': cols}

    def _map_section_content(self, lines: List[str]) -> str:
        content: str = super()._map_section_content(lines)
        templ = Template(content)
        return templ.substitute(self.context)


@dataclass
class DataClassDbPersister(BeanDbPersister):
    """Persists instances of :class:`dataclasses.dataclass` by narrowing the
    columns from select statements. Instead of ``select *``, use ``select
    ${cols}`` in the SQL resource file.

    :see: :class:`.DataClassDynamicDataParser`

    """
    bean_class: Type[dataclass] = field(default=None)
    """The data class that is CRUD'd for DB operations."""

    def __post_init__(self):
        self.row_factory = self.bean_class
        super().__post_init__()

    def _create_parser(self, sql_file: Path) -> DynamicDataParser:
        return DataClassDynamicDataParser(sql_file, self.bean_class)

    def _get_insert_row(self, bean: Bean) -> Tuple[Any]:
        idf = DataClassDynamicDataParser.ID_FIELD
        return tuple(map(lambda f: getattr(bean, f.name),
                         filter(lambda f: f.name != idf, fields(bean))))
