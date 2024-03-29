"""Command line entry point to the application.

"""
__author__ = 'Paul Landes'

from typing import List, Any, Dict, Type
import sys
from zensols.cli import ActionResult, CliHarness
from zensols.cli import ApplicationFactory as CliApplicationFactory
from . import Database


class ApplicationFactory(CliApplicationFactory):
    def __init__(self, *args, **kwargs):
        kwargs['package_resource'] = 'zensols.propbankdb'
        super().__init__(*args, **kwargs)

    @classmethod
    def get_database(cls: Type) -> Database:
        """Get the frameset database."""
        harness: CliHarness = cls.create_harness()
        return harness['pbdb_db']


def main(args: List[str] = sys.argv, **kwargs: Dict[str, Any]) -> ActionResult:
    harness: CliHarness = ApplicationFactory.create_harness(relocate=False)
    harness.invoke(args, **kwargs)
