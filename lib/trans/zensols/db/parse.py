"""Simple parsing of a dynamic definition langauge (DDL) and dynamic
manipulation language (DML) files.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, List, Iterable, ClassVar
import logging
import re
import itertools as it
from pathlib import Path
from zensols.persist import persisted

logger = logging.getLogger(__name__)


class DynamicDataParser(object):
    """Parse a DDL/DML file meant also for prototyping.

    For example the file::

        -- meta=init_sections=create_tables,create_idx
        -- name=create_idx
        create index person_name on person(name);

        -- name=create_tables
        create table person (id int, name text, age int);

    Would have ``create_idx`` and ``create_tables`` as sections and meta data::

        {'init_sections':
         'create_tables,create_idx'}

    """
    COMMENT_PAT: ClassVar[re.Pattern] = re.compile(r'^--.*')
    SEC_START_PAT: ClassVar[re.Pattern] = re.compile(
        r'^-- name=([a-zA-Z0-9_]+)')
    META_PAT: ClassVar[re.Pattern] = re.compile(
        r'^-- meta=([a-zA-Z0-9_]+)=(.+)$')

    def __init__(self, dd_path: Path):
        """Initialize.

        :param dd_path: the path of the file to parse

        """
        self.dd_path = dd_path

    def _map_section_content(self, lines: List[str]) -> str:
        return '\n'.join(lines)

    @persisted('__parse')
    def _parse(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'parsing {self.dd_path}')
        secs: List[str, Tuple[str, List[str]]] = []
        sec_content: List[str] = []
        meta: Dict[str, str] = {}
        with open(self.dd_path) as f:
            line: str
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                if re.match(self.COMMENT_PAT, line):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f'matched comment: {line}')
                    sec_start = re.match(self.SEC_START_PAT, line)
                    meta_match = re.match(self.META_PAT, line)
                    sec_content = []
                    if sec_start is not None:
                        name = sec_start.group(1)
                        secs.append((name, sec_content))
                    elif meta_match is not None:
                        meta[meta_match.group(1)] = meta_match.group(2)
                else:
                    sec_content.append(line)
        sections = {x[0]: self._map_section_content(x[1]) for x in secs}
        return sections, meta

    @property
    def sections(self) -> Dict[str, str]:
        """Return the sections of the file.

        """
        return self._parse()[0]

    @property
    def metadata(self) -> Dict[str, str]:
        """Return the meta data found int he parse object.

        """
        return self._parse()[1]

    def get_init_db_sqls(self) -> Iterable[str]:
        """Return the set of statements that create all DB objects needed to fully
        CRUD.

        """
        init_secs = self.metadata['init_sections']
        secs = init_secs.split(',')
        entries = map(lambda x: self.sections[x], secs)
        sts = map(lambda x: re.split(';[ \t\n]*', x, flags=re.MULTILINE),
                  entries)
        return filter(lambda x: len(x) > 0, it.chain(*sts))
