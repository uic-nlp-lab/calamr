"""Contains the manager classes that invoke the tables to generate.

"""
__author__ = 'Paul Landes'

from typing import List, ClassVar, Set
from dataclasses import dataclass, field
import sys
import logging
from itertools import chain
import re
import yaml
from datetime import datetime
from pathlib import Path
from io import TextIOWrapper
from zensols.persist import persisted
from zensols.config import Writable
from zensols.introspect import ClassImporter
from . import LatexTableError, Table

logger = logging.getLogger(__name__)


@dataclass
class CsvToLatexTable(Writable):
    """Generate a Latex table from a CSV file.

    """
    tables: List[Table] = field()
    """A list of table instances to create Latex table definitions."""

    package_name: str = field()
    """The name Latex .sty package."""

    def _write_header(self, depth: int, writer: TextIOWrapper):
        date = datetime.now().strftime('%Y/%m/%d')
        writer.write("""\\NeedsTeXFormat{LaTeX2e}
\\ProvidesPackage{%(package_name)s}[%(date)s Tables]

""" % {'date': date, 'package_name': self.package_name})
        uses: Set[str] = set(chain.from_iterable(
            map(lambda t: t.uses, self.tables)))
        for use in sorted(uses):
            writer.write(f'\\usepackage{{{use}}}\n')

    def write(self, depth: int = 0, writer: TextIOWrapper = sys.stdout):
        """Write the Latex table to the writer given in the initializer.

        """
        self._write_header(depth, writer)
        for table in self.tables:
            table.write(depth, writer)


@dataclass
class TableFileManager(object):
    """Reads the table definitions file and writes a Latex .sty file of the
    generated tables from the CSV data.

    """
    _FILE_NAME_REGEX: ClassVar[re.Pattern] = re.compile(r'(.+)\.yml')
    _PACKAGE_FORMAT: ClassVar[str] = '{name}'

    table_path: Path = field()
    """The path to the table YAML defintiions file."""

    @property
    @persisted('_package_name')
    def package_name(self) -> str:
        fname = self.table_path.name
        m = self._FILE_NAME_REGEX.match(fname)
        if m is None:
            raise LatexTableError(f'does not appear to be a YAML file: {fname}')
        return self._PACKAGE_FORMAT.format(**{'name': m.group(1)})

    def _fix_path(self, tab: Table):
        """When the CSV path in the table doesn't exist, replace it with a
        relative file from the YAML file if it exists.

        """
        tab_path = Path(tab.path)
        if not tab_path.is_file():
            rel_path = Path(self.table_path.parent, tab_path).resolve()
            if rel_path.is_file():
                tab.path = rel_path

    @property
    def tables(self) -> List[Table]:
        logger.info(f'reading table definitions file {self.table_path}')
        tables: List[Table] = []
        with open(self.table_path) as f:
            content = f.read()
        tdefs = yaml.load(content, yaml.FullLoader)
        for name, td in tdefs.items():
            class_name: str
            if 'type' in td:
                cls_name = td['type'].capitalize() + 'Table'
                del td['type']
            else:
                cls_name = 'Table'
            cls_name = f'zensols.datdesc.{cls_name}'
            td['name'] = name
            inst: Table = ClassImporter(cls_name, reload=False).instance(**td)
            self._fix_path(inst)
            tables.append(inst)
        return tables
